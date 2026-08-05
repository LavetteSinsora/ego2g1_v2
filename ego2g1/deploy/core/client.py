"""Websocket client to `python -m ego2g1.serve`. Ported from the old deploy's
client.py (third_party/openpi/ego2g1/deploy/client.py); DelayBudget moved to
latency.py.

The request is the plain openpi obs dict, optionally carrying the RTC prefix:

    {"observation/image": HWC uint8,
     "observation/state": float32 — (30,) FK proprio for relative_eef/joint
                          checkpoints, (56,) hand-major relation vector for
                          relation_eef ones (the server's own transforms
                          decide; the client just forwards what the adapter
                          built),
     "prompt": str,
     "prev_chunk": (H, 30) float32,   # optional: re-anchored leftover (RTC,
                                      # relative_eef only)
     "d": int,                        # optional: inference delay, ticks
     "n_prefix": int}                 # how many prev_chunk rows are REAL

and the reply is {"actions": (H, action_dim), "policy_timing": {...},
"rtc": {...}} — action_dim 30 for relative_eef, 14 for relation_eef.

The client does NOT choose the sampler. It always sends the prefix when it has
one; the checkpoint's stamp decides guided vs pinned RTC. That is what makes a
future rtc_training retrain a server-side swap.

The image is resized to the model's 224x224 HERE, on the wire, not in the
camera (which keeps handing out the raw frame — that is what check/recording
must see). Full head frames are ~920 KB at 640x480 per inference; on the robot
LAN that is free, through a tunnel it is the dominant latency term, and latency
is `d`. Resizing first costs 150 KB. Doing it twice is safe only because the
target matches the server's ResizeImages(224, 224) exactly (resize_with_pad
early-returns on an already-sized image) — resize to anything else and the
server letterboxes the letterbox.
"""

import logging
import time

import numpy as np


class PolicyClient:
    """Thin wrapper: connect, read the layout out of the handshake, infer."""

    def __init__(self, host: str, port: int, *, api_key: str | None = None,
                 resize: tuple[int, int] | None = (224, 224)):
        # Imported here, not at module scope, so the module stays importable
        # (and testable) without the transport.
        from openpi_client import websocket_client_policy

        self._resize = None if resize is None else (int(resize[0]), int(resize[1]))
        self._ws = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port, api_key=api_key
        )
        meta = self._ws.get_server_metadata()
        self.metadata = meta

        # Config-free client: the model's layout comes from the server (which
        # would otherwise drag JAX onto the robot PC).
        cfg = meta.get("ego2g1")
        if cfg is None:
            raise RuntimeError(
                "server did not advertise ego2g1 metadata — is it running "
                "`python -m ego2g1.serve`? Stock openpi serve_policy.py cannot "
                "serve these checkpoints correctly."
            )
        self.hands = tuple(cfg["hands"])
        self.action_horizon = int(cfg["action_horizon"])
        self.action_dim = int(cfg["action_dim"])
        self.fps = int(cfg["fps"])
        self.control_mode = str(cfg.get("control_mode", "relative_eef"))
        self.rtc_training = bool(cfg["rtc_training"])
        self.rtc = dict(cfg["rtc"])

        stamp = meta.get("ego2g1_stamp", {})
        logging.info("policy: horizon=%d dim=%d fps=%d hands=%s mode=%s",
                     self.action_horizon, self.action_dim, self.fps, self.hands,
                     self.control_mode)
        logging.info("checkpoint config hash: %s", stamp.get("ego2g1_config_hash"))
        logging.info("RTC: %s (checkpoint rtc_training=%s)", self.rtc, self.rtc_training)

    def _prepare_image(self, image):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if self._resize is None:
            return image
        from openpi_client import image_tools

        return np.ascontiguousarray(
            image_tools.resize_with_pad(image, *self._resize), dtype=np.uint8
        )

    def infer(self, image, state, prompt, *, prev_chunk=None, d: int = 0,
              n_prefix: int | None = None) -> dict:
        obs = {
            "observation/image": self._prepare_image(image),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": prompt,
        }
        if prev_chunk is not None:
            obs["prev_chunk"] = np.asarray(prev_chunk, dtype=np.float32)
            obs["d"] = int(d)
            # How many rows are real. The rest is zero padding, and a zero vec9
            # is not a pose — the server must know where to stop.
            obs["n_prefix"] = int(len(prev_chunk) if n_prefix is None else n_prefix)

        t0 = time.monotonic()
        out = self._ws.infer(obs)
        out["client_latency_s"] = time.monotonic() - t0
        return out
