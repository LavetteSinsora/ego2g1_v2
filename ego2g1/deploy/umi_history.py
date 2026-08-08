"""The deploy-side state-history ring buffer for `umi_eef` mode.

The UMI policy's only proprioception is a window of recent TCP poses expressed
in the current pose's frame, plus the gripper (ego2g1/train/umi_transforms.py).
Training got that window by gathering the dataset at fixed tick offsets; at
deploy it has to be accumulated live, which is what this buffer does.

Three things it gets right, each of which is a bug if done the obvious way:

TIMESTAMPED, NOT PER-ITERATION. The runner paces to `fps` but jitters, and its
idle branch polls at ~20 Hz rather than `fps` (runner.py's `time.sleep(0.05)`).
Indexing "5 entries back" would therefore mean a different amount of real time
depending on what the loop had been doing. Samples carry their own monotonic
timestamp and lags are resolved against it.

STRICT NEAREST-SAMPLE TOLERANCE. A lag is only satisfied by a sample within
`tol_ticks` of the requested time; otherwise it is reported as PADDING. That is
deliberately conservative: `UmiStateHistory` truncates from the stale end only
(because prompt position is what identifies a lag), so a hole means everything
older than it is dropped rather than silently re-labelled as a different lag.
A policy fed a wrong-lag pose is worse than one fed a shorter history — the
short case is trained (that is what `history_len_probs` is for), the wrong case
is not.

CLEARED ON RE-ARM. `DeployRunner._rearm` exists because "the world moved on"
across a pause; a history spanning that gap would read as an enormous velocity
at the exact moment the arm is stationary. `UmiPolicyAdapter.reset()` clears
this buffer for the same reason it resets the causal filters.

The buffer stores ABSOLUTE poses. The anchor-relative composition is done
SERVER-side by the very same `UmiStateHistory` transform training used — see
`ego2g1/deploy/modes/umi_eef.py` for why that is the point rather than an
inefficiency.
"""

from __future__ import annotations

import collections

import numpy as np

from ..core import umi_layout


class PoseHistoryBuffer:
    """Monotonic-timestamped ring of (pose vec9, gripper) samples.

    `push` once per control tick; `sample` resolves a lag grid against the
    newest sample's timestamp.
    """

    def __init__(self, lag_ticks: tuple[int, ...], fps: float, *,
                 tol_ticks: float = 0.75, capacity: int | None = None,
                 max_len: int | None = None):
        if not lag_ticks or lag_ticks[0] != 0:
            raise ValueError(
                f"lag_ticks must start at 0 (the anchor's own tick), got {lag_ticks}")
        if fps <= 0:
            raise ValueError(f"fps={fps} must be > 0")
        self.lag_ticks = tuple(int(l) for l in lag_ticks)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        # 0.75 of a tick: tight enough that a whole dropped tick reads as
        # padding rather than as the neighbouring lag, loose enough to absorb
        # ordinary pacing jitter.
        self.tol_s = float(tol_ticks) * self.dt
        # Hard cap on how many lags are ever reported, on top of whatever the
        # buffer actually holds. `None` = report everything available. Used to
        # ablate the history channel on the real robot the way the training
        # val pools ablate it offline: the cap is expressed by MARKING the
        # surplus lags as padding, so the server's own `UmiStateHistory`
        # truncates exactly as it would at an episode start — one truncation
        # rule, not a second one bolted on here.
        if max_len is not None and not 0 <= max_len <= len(self.lag_ticks):
            raise ValueError(
                f"max_len={max_len} must be in [0, {len(self.lag_ticks)}]")
        self.max_len = max_len
        # enough room for the whole window plus slack for jitter/backlog
        span = max(self.lag_ticks)
        self._buf: collections.deque = collections.deque(
            maxlen=capacity if capacity is not None else max(8, 4 * (span + 1)))

    # --- write ---------------------------------------------------------------

    def push(self, t: float, pose_vec9, gripper: float) -> None:
        """Record the measured state at monotonic time `t`.

        Call at CONTROL rate, from the runner's per-tick observation build --
        not from `adapter.infer`, which the async strategies call at
        `inference_hz` (4 Hz by default). A buffer filled at 4 Hz cannot
        resolve a 3-tick lag grid at all.
        """
        pose = np.asarray(pose_vec9, dtype=np.float64).reshape(-1)
        if pose.shape != (umi_layout.POSE_DIM,):
            raise ValueError(
                f"pose_vec9: expected ({umi_layout.POSE_DIM},), got {pose.shape}")
        if self._buf and t < self._buf[-1][0]:
            raise ValueError(
                f"pose history went backwards in time ({t} < {self._buf[-1][0]}); "
                "the clock must be monotonic")
        self._buf.append((float(t), pose, float(gripper)))

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    # --- read ----------------------------------------------------------------

    def sample(self, t_now: float | None = None) -> dict:
        """Resolve the lag grid -> the request fields the server expects.

        Returns::

            {"observation/pose_history":        (n_lags, 9) absolute vec9,
             "observation/gripper_history":     (n_lags, 1),
             "observation/pose_history_is_pad": (n_lags,) bool,
             "history_len":                     int}

        `history_len` is the number of LEADING real lags -- what the server
        will actually keep, since truncation is from the stale end. It is
        returned for telemetry/recording; the server derives its own from the
        pad mask and does not trust this number.

        Lag 0 resolves to the newest sample, and its time defines the grid --
        NOT the caller's `t_now`. Anchoring on the sample itself is what keeps
        the grid consistent with the pose the action chunk is anchored at, even
        if the caller's clock has advanced since the push.
        """
        n = len(self.lag_ticks)
        poses = np.zeros((n, umi_layout.POSE_DIM), dtype=np.float32)
        grips = np.zeros((n, 1), dtype=np.float32)
        is_pad = np.ones(n, dtype=bool)
        if not self._buf:
            return {"observation/pose_history": poses,
                    "observation/gripper_history": grips,
                    "observation/pose_history_is_pad": is_pad,
                    "history_len": 0}

        times = np.asarray([s[0] for s in self._buf], dtype=np.float64)
        t0 = times[-1] if t_now is None else float(t_now)
        for j, lag in enumerate(self.lag_ticks):
            want = t0 - lag * self.dt
            k = int(np.argmin(np.abs(times - want)))
            if abs(times[k] - want) > self.tol_s:
                continue                      # stays padding
            poses[j] = self._buf[k][1]
            grips[j, 0] = self._buf[k][2]
            is_pad[j] = False

        # Truncation is FAR-END only, so the usable window is the leading run
        # of real lags. Anything after a hole is dropped -- see the module
        # docstring on why that is the conservative direction.
        length = int(np.argmax(is_pad)) if bool(is_pad.any()) else n
        if self.max_len is not None:
            length = min(length, self.max_len)
        is_pad[length:] = True
        poses[length:] = 0.0
        grips[length:] = 0.0
        return {"observation/pose_history": poses,
                "observation/gripper_history": grips,
                "observation/pose_history_is_pad": is_pad,
                "history_len": length}
