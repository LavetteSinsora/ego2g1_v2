"""Chunk consumers: how a stream of (H, 26) joint chunks becomes one row per tick.

Ported closely from zh_deploy_inference/examples/unitree_inference/strategies.py
— the stack that drives this arm smoothly — with three deliberate additions,
each flagged [ego2g1] below:

  * an optional `recorder` hook (events.jsonl needs the seams: chunk installs,
    splice indices, infer latencies — the old jitter was only diagnosable
    because these were logged),
  * every infer latency is fed to an optional DelayBudget (latency.py),
  * pop_action never blocks silently: the runner owns starvation via the
    Watchdog instead of a bare spin,
  * a `telemetry()` snapshot on every strategy/buffer for the live dashboard
    (deploy/dashboard.py). PULL-only: telemetry() reads existing state under
    the buffer's existing lock and copies small values; the loop and worker
    bodies gain no code for it.

The five strategies (runner --mode names):

  sync                 one chunk at a time; re-infer when drained. The robot
                       HOLDS during inference (the executor keeps interpolating
                       to the last waypoint). The mode to get smoothness first.
  async                NaiveAsyncBuffer: newest chunk wins, skipping the rows
                       that elapsed during inference.
  temporal_ensembling  every chunk that predicts the current tick votes,
                       exponentially weighted by age.
  temporal_smoothing   linearly blend the unexecuted overlap when a new chunk
                       arrives, after dropping up to max_latency_steps stale rows.
  rtc                  temporal_smoothing consumption + the RTC request fields
                       (prev chunk, predicted delay) attached to each inference.

All buffers blend/aggregate in JOINT space — that is only well-defined because
the policy adapter converted EEF chunks to joints before they got here.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Protocol

import numpy as np


class Policy(Protocol):
    def infer(self, observation: dict) -> dict: ...


def _extract_actions(result: dict) -> np.ndarray:
    if not isinstance(result, dict) or "actions" not in result:
        raise ValueError("Policy response must contain an 'actions' field")
    actions = np.asarray(result["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(
            f"Expected actions with shape [horizon, action_dim], got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned a non-finite action")
    return actions


class SynchronousStrategy:
    def __init__(self, policy: Policy, chunk_size: int, *, recorder=None,
                 budget=None) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._policy = policy
        self._chunk_size = chunk_size
        self._chunk: np.ndarray | None = None
        self._index = 0
        self._recorder = recorder
        self._budget = budget

    def update_observation(self, observation: dict) -> None:
        if self._chunk is None or self._index >= min(self._chunk_size, len(self._chunk)):
            t0 = time.monotonic()
            self._chunk = _extract_actions(self._policy.infer(observation))
            elapsed = time.monotonic() - t0                      # [ego2g1]
            if self._budget is not None:
                self._budget.observe(elapsed)
            if self._recorder is not None:
                # `actions` makes the chunk reconstructable offline
                # (replay_record.py); ~H*26 floats once per chunk.
                self._recorder.log("infer_result", latency=elapsed,
                                   horizon=len(self._chunk), mode="sync",
                                   actions=self._chunk)
            self._index = 0

    def has_action(self) -> bool:
        return self._chunk is not None and self._index < len(self._chunk)

    def telemetry(self) -> dict:
        """Dashboard snapshot. Plain reads only — the loop thread may swap the
        chunk mid-read; a one-poll-stale value is harmless. `inferring` is
        derived, not flagged: sync blocks in update_observation exactly while
        the current chunk is absent or drained."""
        chunk = self._chunk
        index = int(self._index)
        ready = chunk is not None
        horizon = len(chunk) if ready else 0
        return {"mode": "sync", "rtc": False, "ready": ready,
                "horizon": horizon, "index": min(index, horizon),
                "inferring": (not ready) or index >= horizon,
                "pending": False,
                "trigger": horizon or None, "d": None,
                "budget": None if self._budget is None else self._budget.stats()}

    def pop_action(self) -> np.ndarray:
        if not self.has_action():
            raise RuntimeError("No synchronous action is available")
        assert self._chunk is not None
        action = self._chunk[self._index].copy()
        self._index += 1
        return action

    def close(self) -> None:
        pass


class NaiveAsyncBuffer:
    """Use the newest chunk and skip the steps spent waiting for inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunk: np.ndarray | None = None
        self._chunk_start_t = 0
        self._global_t = 0
        self._last_action: np.ndarray | None = None

    def add_chunk(self, actions: np.ndarray, start_timestep: int) -> dict:
        with self._lock:
            skip_steps = min(max(0, self._global_t - start_timestep), len(actions) - 1)
            self._chunk = actions.copy()
            self._chunk_start_t = self._global_t - skip_steps
            return {"skip_steps": skip_steps, "global_t": self._global_t}  # [ego2g1]

    def has_action(self) -> bool:
        with self._lock:
            if self._chunk is None:
                return self._last_action is not None
            return (self._global_t - self._chunk_start_t < len(self._chunk)
                    or self._last_action is not None)

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            if self._chunk is None:
                action = self._last_action
            else:
                index = max(0, self._global_t - self._chunk_start_t)
                action = self._chunk[index] if index < len(self._chunk) else self._last_action
            self._global_t += 1
            if action is not None:
                self._last_action = np.asarray(action).copy()
                return self._last_action.copy()
            return None

    def current_timestep(self) -> int:
        with self._lock:
            return self._global_t

    def telemetry(self) -> dict:
        """[ego2g1] dashboard snapshot; own lock only, small copies."""
        with self._lock:
            ready = self._chunk is not None
            horizon = len(self._chunk) if ready else 0
            index = (int(np.clip(self._global_t - self._chunk_start_t, 0, horizon))
                     if ready else 0)
            return {"ready": ready or self._last_action is not None,
                    "horizon": horizon, "index": index,
                    "global_t": int(self._global_t)}


class TemporalEnsemblingBuffer:
    """Aggregate every chunk that predicts the current global timestep."""

    def __init__(self, exp_weight_m: float) -> None:
        if exp_weight_m < 0:
            raise ValueError("exp_weight_m must be non-negative")
        self._exp_weight_m = exp_weight_m
        self._lock = threading.Lock()
        self._predictions: dict[int, list[tuple[int, np.ndarray]]] = {}
        self._current_t = 0
        self._inference_count = 0
        self._last_action: np.ndarray | None = None

    def add_chunk(self, actions: np.ndarray, start_timestep: int) -> dict:
        with self._lock:
            inference_index = self._inference_count
            self._inference_count += 1
            for offset, action in enumerate(actions):
                timestep = start_timestep + offset
                self._predictions.setdefault(timestep, []).append(
                    (inference_index, action.copy()))
            for timestep in tuple(self._predictions):
                if timestep < max(0, self._current_t - 10):
                    del self._predictions[timestep]
            return {"start_timestep": start_timestep, "global_t": self._current_t}

    def has_action(self) -> bool:
        with self._lock:
            return bool(self._predictions.get(self._current_t)) or self._last_action is not None

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            predictions = sorted(self._predictions.get(self._current_t, ()),
                                 key=lambda item: item[0])
            if predictions:
                actions = np.stack([item[1] for item in predictions])
                weights = np.exp(-self._exp_weight_m
                                 * np.arange(len(actions), dtype=np.float64))
                weights /= weights.sum()
                self._last_action = np.sum(actions * weights[:, None], axis=0)
            action = None if self._last_action is None else self._last_action.copy()
            self._current_t += 1
            return action

    def current_timestep(self) -> int:
        with self._lock:
            return self._current_t

    def telemetry(self) -> dict:
        """[ego2g1] dashboard snapshot. No single active chunk exists here
        (every live chunk votes), so horizon is 0 and `votes` says how many."""
        with self._lock:
            votes = len(self._predictions.get(self._current_t, ()))
            return {"ready": bool(votes) or self._last_action is not None,
                    "horizon": 0, "index": 0,
                    "global_t": int(self._current_t),
                    "votes": votes, "chunks": int(self._inference_count)}


class TemporalSmoothingBuffer:
    """Linearly blend the unexecuted overlap when a new chunk arrives."""

    def __init__(self, max_latency_steps: int, min_smooth_steps: int) -> None:
        if max_latency_steps < 0:
            raise ValueError("max_latency_steps must be non-negative")
        if min_smooth_steps <= 0:
            raise ValueError("min_smooth_steps must be positive")
        self._max_latency_steps = max_latency_steps
        self._min_smooth_steps = min_smooth_steps
        self._lock = threading.Lock()
        self._chunk: deque[np.ndarray] = deque()
        self._steps_since_update = 0
        self._last_action: np.ndarray | None = None
        self._global_t = 0

    def add_chunk(self, actions: np.ndarray, start_timestep: int) -> dict:
        del start_timestep
        with self._lock:
            drop_count = min(self._steps_since_update, self._max_latency_steps)
            # [ego2g1] surfaced: drop_count == max_latency_steps AND the wait
            # was longer means the chunk was LATE — the seam is unguaranteed.
            late = self._steps_since_update > self._max_latency_steps
            info = {"drop_count": drop_count, "late": late,
                    "steps_since_update": self._steps_since_update}
            if drop_count >= len(actions):
                return {**info, "dropped_whole_chunk": True}
            new_actions = [action.copy() for action in actions[drop_count:]]

            if self._chunk:
                old_actions = list(self._chunk)
                if len(old_actions) < self._min_smooth_steps:
                    old_actions.extend(
                        old_actions[-1].copy()
                        for _ in range(self._min_smooth_steps - len(old_actions)))
            elif self._last_action is not None:
                old_actions = [self._last_action.copy()
                               for _ in range(self._min_smooth_steps)]
                self._last_action = None
            else:
                self._chunk = deque(new_actions)
                self._steps_since_update = 0
                return info

            overlap = min(len(old_actions), len(new_actions))
            if overlap == 0:
                combined = new_actions
            else:
                old_actions = old_actions[: len(new_actions)]
                old_weights = np.ones(1) if overlap == 1 else np.linspace(1.0, 0.0, overlap)
                combined = [
                    old_weights[i] * old_actions[i] + (1.0 - old_weights[i]) * new_actions[i]
                    for i in range(overlap)
                ]
                combined.extend(new_actions[overlap:])
            self._chunk = deque(np.asarray(a).copy() for a in combined)
            self._steps_since_update = 0
            return {**info, "overlap": overlap}

    def has_action(self) -> bool:
        with self._lock:
            return bool(self._chunk)

    def pop_action(self) -> np.ndarray | None:
        with self._lock:
            if not self._chunk:
                return None
            if len(self._chunk) == 1:
                self._last_action = self._chunk[0].copy()
            action = self._chunk.popleft()
            self._steps_since_update += 1
            self._global_t += 1
            return action

    def current_timestep(self) -> int:
        with self._lock:
            return self._global_t

    def telemetry(self) -> dict:
        """[ego2g1] dashboard snapshot. horizon/index reconstruct the current
        combined (blended) chunk: `_steps_since_update` rows consumed since the
        last install, `len(_chunk)` still queued."""
        with self._lock:
            remaining = len(self._chunk)
            consumed = int(self._steps_since_update)
            return {"ready": remaining > 0,
                    "horizon": remaining + consumed, "index": consumed,
                    "global_t": int(self._global_t),
                    "steps_since_update": consumed}


class AsyncStrategy:
    """Background inference worker feeding one of the buffers above.

    Ported from zh's AsyncStrategy; [ego2g1] additions: recorder + DelayBudget
    hooks, and worker errors also reach the recorder (a dead worker used to be
    silent until the buffer drained)."""

    def __init__(
        self,
        policy: Policy,
        action_buffer: NaiveAsyncBuffer | TemporalEnsemblingBuffer | TemporalSmoothingBuffer,
        inference_hz: float,
        *,
        rtc: bool = False,
        execute_horizon: int = 0,
        control_hz: float = 0,
        recorder=None,
        budget=None,
        mode: str = "async",           # [ego2g1] telemetry label only
    ) -> None:
        if inference_hz <= 0:
            raise ValueError("inference_hz must be positive")
        if rtc and (execute_horizon <= 0 or control_hz <= 0):
            raise ValueError("RTC requires positive execute_horizon and control_hz")
        self._policy = policy
        self._buffer = action_buffer
        self._period = 1.0 / inference_hz
        self._rtc = rtc
        self._execute_horizon = execute_horizon
        self._control_hz = control_hz
        self._condition = threading.Condition()
        self._observation: dict | None = None
        self._stopping = False
        self._error: BaseException | None = None
        self._previous_chunk: np.ndarray | None = None
        self._delays: deque[float] = deque(maxlen=20)
        self._recorder = recorder
        self._budget = budget
        self._mode = mode
        self._thread = threading.Thread(target=self._inference_loop,
                                        name="ego2g1-inference", daemon=True)
        self._thread.start()

    def update_observation(self, observation: dict) -> None:
        with self._condition:
            self._observation = observation
            self._condition.notify()

    def has_action(self) -> bool:
        self._raise_worker_error()
        return self._buffer.has_action()

    def pop_action(self) -> np.ndarray:
        self._raise_worker_error()
        action = self._buffer.pop_action()
        if action is None:
            raise RuntimeError("No asynchronous action is available")
        return action

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def telemetry(self) -> dict:
        """[ego2g1] dashboard snapshot: the buffer's own snapshot plus the
        strategy-level facts. Reads only — a request-in-flight light would need
        a flag write in the worker body, so it is deliberately not surfaced;
        the DelayBudget stats carry the latency story instead."""
        t = self._buffer.telemetry() if hasattr(self._buffer, "telemetry") else {}
        return {"mode": self._mode, "rtc": self._rtc,
                "inferring": False, "pending": False, "trigger": None,
                "d": self._predicted_delay_steps() if self._rtc else None,
                "worker_dead": self._error is not None,
                "budget": None if self._budget is None else self._budget.stats(),
                **t}

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Inference worker stopped") from self._error

    def _inference_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._observation is not None or self._stopping)
                    if self._stopping:
                        return
                    observation = self._observation
                assert observation is not None

                start_timestep = self._buffer.current_timestep()
                request = dict(observation)
                if self._rtc:
                    request["execute_horizon"] = self._execute_horizon
                    request["enable_rtc"] = True
                    request["inference_delay"] = self._predicted_delay_steps()
                    if self._previous_chunk is not None:
                        request["prev_action_chunk"] = self._previous_chunk

                start_time = time.monotonic()
                actions = _extract_actions(self._policy.infer(request))
                elapsed = time.monotonic() - start_time
                if self._rtc:
                    self._delays.append(elapsed)
                    self._previous_chunk = actions.copy()
                if self._budget is not None:                       # [ego2g1]
                    self._budget.observe(elapsed)
                info = self._buffer.add_chunk(actions, start_timestep) or {}
                if self._recorder is not None:                     # [ego2g1]
                    # `actions` (the converted joint chunk) makes the buffer
                    # reconstructable offline (replay_record.py).
                    self._recorder.log("infer_result", latency=elapsed,
                                       start_timestep=start_timestep,
                                       horizon=len(actions), rtc=self._rtc,
                                       splice=info, actions=actions)

                with self._condition:
                    deadline = start_time + self._period
                    while not self._stopping:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                    if self._stopping:
                        return
        except BaseException as error:
            self._error = error
            if self._recorder is not None:                         # [ego2g1]
                self._recorder.log("worker_error", error=repr(error))

    def _predicted_delay_steps(self) -> int:
        if not self._delays:
            return 0
        return max(0, round(float(np.median(self._delays)) * self._control_hz))


def make_strategy(mode: str, policy, *, chunk_size: int, inference_hz: float = 4.0,
                  exp_weight_m: float = 0.01, max_latency_steps: int = 8,
                  min_smooth_steps: int = 10, control_hz: float = 30.0,
                  rtc_execute_horizon: int | None = None,
                  recorder=None, budget=None):
    """Mirror of zh runner._make_strategy, with the [ego2g1] hooks threaded in."""
    if mode == "sync":
        return SynchronousStrategy(policy, chunk_size=chunk_size,
                                   recorder=recorder, budget=budget)
    if mode == "async":
        return AsyncStrategy(policy, NaiveAsyncBuffer(), inference_hz,
                             recorder=recorder, budget=budget, mode=mode)
    if mode == "temporal_ensembling":
        return AsyncStrategy(policy, TemporalEnsemblingBuffer(exp_weight_m),
                             inference_hz, recorder=recorder, budget=budget,
                             mode=mode)
    if mode == "temporal_smoothing":
        return AsyncStrategy(
            policy, TemporalSmoothingBuffer(max_latency_steps, min_smooth_steps),
            inference_hz, recorder=recorder, budget=budget, mode=mode)
    if mode == "rtc":
        return AsyncStrategy(
            policy, TemporalSmoothingBuffer(max_latency_steps, min_smooth_steps),
            inference_hz, rtc=True,
            execute_horizon=rtc_execute_horizon or chunk_size,
            control_hz=control_hz, recorder=recorder, budget=budget, mode=mode)
    raise ValueError(f"Unsupported inference mode: {mode}")


MODES = ("sync", "async", "temporal_ensembling", "temporal_smoothing", "rtc")
