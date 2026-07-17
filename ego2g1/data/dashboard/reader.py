"""Two data sources, one interface.

`WorkdirReader` reads the per-stage npz outputs under
`<work_dir>/<episode>/` (schemas in SPEC.md); `DatasetReader`
reads the final LeRobot dataset (parquet via pyarrow + mp4 via imageio —
lerobot itself is never imported). Both return an `EpisodeRecord` covering
one SOURCE episode:

- workdir: T = every control tick of the recording, per-tick bad masks and
  the s004 sub-episode boundaries included;
- dataset: the source episode's sub-episodes (one LeRobot episode each) are
  concatenated back together; ticks the filters dropped are simply absent
  (`tick_orig` keeps the original tick index so gaps stay visible), and the
  bad masks are empty — the dataset only contains good ticks.

All poses are vec9 flange poses in the PELVIS frame (SPEC.md s002_01 /
s003_state conventions).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

from ..config import PipelineConfig

SIDES = (("left", "l"), ("right", "r"))
FILTER_NAMES = ["bad_cam"] + [f"bad_{f}_{p}" for f in
                              ("gap", "vel", "ik", "clear", "hand_blocked",
                               "hand_contact", "hand_residual")
                              for p in ("l", "r")]


def _ep_sort_key(name):
    """episode_10 after episode_9, not after episode_1."""
    return [int(p) if p.isdigit() else p for p in name.replace("_", " ").split()]


@dataclasses.dataclass
class SubEp:
    start: int          # tick index into the record arrays
    end: int            # exclusive
    real_end: bool


@dataclasses.dataclass
class EpisodeRecord:
    name: str
    source: str                       # "workdir" | "dataset"
    fps: float
    horizon: int                      # chunk length H
    ticks_ns: np.ndarray              # (T,) i64
    tick_orig: np.ndarray             # (T,) i32 original recording tick index
    pose: dict                        # side -> (T,9) f32, pelvis frame
    hand: dict                        # side -> (T,6) f32
    state: np.ndarray                 # (T,30) f32
    arm_qpos: np.ndarray              # (T,14) f32
    masks: dict                       # filter name -> (T,) bool (True=bad); {} for dataset
    subeps: list                      # [SubEp]
    S: np.ndarray                     # (4,4) placement (Pico world -> robot world)
    B: dict                           # side -> (4,4) wrist->flange alignment
    config_hash: str
    b_mode: str
    filter_stats: dict                # filter -> bad tick count (recording total)
    ticks_total: int                  # recording length before filtering
    ticks_kept: int
    meta: dict = dataclasses.field(default_factory=dict)
    _images: object = None            # backend-specific image provider

    @property
    def T(self):
        return len(self.ticks_ns)

    def subep_of(self):
        """(T,) int: sub-episode index per tick, -1 outside every sub-episode."""
        out = np.full(self.T, -1, dtype=np.int32)
        for i, se in enumerate(self.subeps):
            out[se.start:se.end] = i
        return out

    # -- images: dedup key + payload (jpeg bytes or HxWx3 uint8) per tick
    def cam_key(self, t):
        return self._images.key(t)

    def tick_image(self, t):
        """jpeg bytes (workdir) or RGB uint8 array (dataset)."""
        return self._images.image(t)

    def close(self):
        if self._images is not None:
            self._images.close()


# --------------------------------------------------------------- workdir

class _Hdf5Images:
    """Human egocentric jpegs, lazily read from the source HDF5 via the s001
    cam_match indexing (images are not copied into the work dir)."""

    def __init__(self, source_path, cam_match):
        self.source_path = source_path
        self.cam_match = cam_match
        self._f = None

    def key(self, t):
        return int(self.cam_match[t])

    def image(self, t):
        if self._f is None:
            import h5py
            self._f = h5py.File(self.source_path, "r")
            self._ds = self._f["camera/images_left_jpeg"]
        return bytes(self._ds[int(self.cam_match[t])])

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


class WorkdirReader:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.work_dir = Path(self.cfg.work_dir)

    def episodes(self):
        """Episode names with a complete stage set (through s004)."""
        out = []
        for d in self.work_dir.iterdir() if self.work_dir.exists() else []:
            if d.name.startswith("_"):
                continue
            if all((d / f"{s}.npz").exists() for s in
                   ("s001", "s002_01", "s002_02", "s003_placement",
                    "s003_state", "s004")):
                out.append(d.name)
        return sorted(out, key=_ep_sort_key)

    def _load(self, ep, stage):
        from .. import io
        # io.load_stage needs an ep_path-like name only
        return io.load_stage(self.cfg, ep, stage)

    def load(self, ep_name) -> EpisodeRecord:
        cfg = self.cfg
        s001, m001 = self._load(ep_name, "s001")
        s0201, _ = self._load(ep_name, "s002_01")
        s0202, _ = self._load(ep_name, "s002_02")
        plc, _ = self._load(ep_name, "s003_placement")
        st, _ = self._load(ep_name, "s003_state")
        s004, m004 = self._load(ep_name, "s004")
        bcal, mb = self._load(self._global_key(), "b_calib")

        # stale-cache guard: the dashboard's config must be the pipeline's
        mpath = self.work_dir / ep_name / "s004.manifest.json"
        man = json.loads(mpath.read_text())
        if man.get("stage_hash") != cfg.stage_hash("s004"):
            print(f"  WARNING: {ep_name} s004 stage hash "
                  f"{man.get('stage_hash')} != current config "
                  f"{cfg.stage_hash('s004')} — work dir was produced with a "
                  f"different config")

        T = len(s001["ticks_ns"])
        pose = {s: s0201[f"pose_{p}"].astype(np.float32) for s, p in SIDES}
        hand = {s: s0202[f"hand_cmds_{p}"].astype(np.float32) for s, p in SIDES}
        state = np.concatenate(
            [np.concatenate([st[f"state_eef_{s[0]}"], hand[s]], axis=1)
             for s in cfg.hands], axis=1).astype(np.float32)

        masks = {k: s004[k].astype(bool) for k in FILTER_NAMES + ["bad_any"]
                 if k in s004}
        subeps = [SubEp(int(a), int(b), bool(r)) for a, b, r in
                  zip(s004["subep_start"], s004["subep_end"],
                      s004["subep_real_end"])]

        return EpisodeRecord(
            name=ep_name, source="workdir", fps=cfg.fps,
            horizon=cfg.action_horizon,
            ticks_ns=s001["ticks_ns"].astype(np.int64),
            tick_orig=np.arange(T, dtype=np.int32),
            pose=pose, hand=hand, state=state,
            arm_qpos=st["arm_qpos"].astype(np.float32),
            masks=masks, subeps=subeps,
            S=plc["S"].astype(np.float64),
            B={"left": bcal["B_left"], "right": bcal["B_right"]},
            config_hash=cfg.config_hash, b_mode=mb.get("mode", "?"),
            filter_stats=m004.get("bad_counts", {}),
            ticks_total=int(m004.get("ticks_total", T)),
            ticks_kept=int(m004.get("ticks_kept", T)),
            meta={"source_file": m001.get("source", "?"),
                  "source_episode": m001.get("episode", ep_name),
                  "ik_pos_cm": {s: st[f"ik_pos_cm_{p}"] for s, p in SIDES},
                  "ik_ori_deg": {s: st[f"ik_ori_deg_{p}"] for s, p in SIDES}},
            _images=_Hdf5Images(m001["source"], s001["cam_match"]),
        )

    @staticmethod
    def _global_key():
        return None    # io.load_stage(cfg, None, stage) -> _global


# --------------------------------------------------------------- dataset

class _VideoImages:
    """Per-tick RGB frames decoded from the LeRobot per-episode mp4s."""

    def __init__(self, mp4_paths, lengths):
        # tick t -> (file index, frame index)
        self.mp4_paths = mp4_paths
        self.of = np.repeat(np.arange(len(lengths)), lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self.fi = np.arange(int(np.sum(lengths))) - starts[self.of]
        self._cache_idx = None
        self._cache = None

    def key(self, t):
        return int(t)   # every tick has its own decoded frame

    def image(self, t):
        i = int(self.of[t])
        if self._cache_idx != i:
            import imageio
            with imageio.get_reader(str(self.mp4_paths[i])) as rdr:
                self._cache = [np.asarray(f) for f in rdr]
            self._cache_idx = i
        return self._cache[int(self.fi[t])]

    def close(self):
        self._cache = None


def _parquet_columns(path):
    import pyarrow.parquet as pq
    tab = pq.read_table(path)
    out = {}
    for name in tab.column_names:
        col = tab.column(name).combine_chunks()
        try:
            arr = np.stack([np.asarray(v) for v in col.to_pylist()])
        except Exception:
            arr = np.asarray(col.to_pylist())
        out[name] = arr
    return out


def _pick(cols, *names):
    """First present column among naming variants (pose.left / pose_left)."""
    for n in names:
        if n in cols:
            return cols[n]
    raise KeyError(f"none of {names} in parquet columns {sorted(cols)}")


class DatasetReader:
    def __init__(self, root=None, cfg: PipelineConfig | None = None):
        cfg = cfg or PipelineConfig()
        self.root = Path(root) if root else Path(cfg.output_root) / cfg.repo_id
        sidecar = self.root / "extraction_meta.json"
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        if not sidecar.exists():
            raise FileNotFoundError(f"missing sidecar {sidecar} — was the "
                                    f"dataset written by s005?")
        self.sidecar = json.loads(sidecar.read_text())
        self.cfg_dict = self.sidecar.get("config", {})
        info = self.root / "meta" / "info.json"
        self.info = json.loads(info.read_text()) if info.exists() else {}

    def _source_stem(self, entry):
        return entry.get("source_episode", "?").split("/")[-1]

    def episodes(self):
        stems = {self._source_stem(e) for e in
                 self.sidecar.get("episodes", {}).values()}
        return sorted(stems, key=_ep_sort_key)

    def _find(self, sub, pattern):
        hits = sorted((self.root / sub).rglob(pattern))
        return hits[0] if hits else None

    def load(self, ep_name) -> EpisodeRecord:
        entries = sorted(
            ((int(i), e) for i, e in self.sidecar.get("episodes", {}).items()
             if self._source_stem(e) == ep_name),
            key=lambda kv: kv[1].get("tick_start", kv[0]))
        if not entries:
            raise FileNotFoundError(
                f"{ep_name} not in dataset sidecar (have "
                f"{', '.join(self.episodes())})")

        fps = float(self.cfg_dict.get("control_hz",
                                      self.info.get("fps", 30.0)))
        horizon = int(self.cfg_dict.get("action_horizon", 50))
        hands = tuple(self.cfg_dict.get("hands", ("left", "right")))

        parts, mp4s, lengths, subeps, tick_orig = [], [], [], [], []
        off = 0
        for idx, entry in entries:
            pq_path = self._find("data", f"episode_{idx:06d}.parquet")
            if pq_path is None:
                raise FileNotFoundError(
                    f"parquet for episode index {idx} not found under "
                    f"{self.root / 'data'}")
            cols = _parquet_columns(pq_path)
            n = len(next(iter(cols.values())))
            parts.append(cols)
            lengths.append(n)
            mp4 = self._find("videos", f"episode_{idx:06d}.mp4")
            mp4s.append(mp4)
            subeps.append(SubEp(off, off + n, bool(entry.get(
                "episode_real_end", entry.get("real_end", True)))))
            t0 = int(entry.get("tick_start", 0))
            tick_orig.append(np.arange(t0, t0 + n, dtype=np.int32))
            off += n

        def col(*names):
            return np.concatenate(
                [_pick(p, *names) for p in parts]).astype(np.float32)

        pose = {s: col(f"pose.{s}", f"pose_{s}") for s in ("left", "right")}
        hand = {s: col(f"hand.{s}", f"hand_{s}") for s in ("left", "right")}
        state = col("state", "observation.state", "observation_state")
        arm_qpos = col("arm_qpos")
        T = off

        e0 = entries[0][1]
        S = np.asarray(e0.get("S", np.eye(4)), dtype=np.float64)
        B = {"left": np.asarray(e0.get("B_left", np.eye(4))),
             "right": np.asarray(e0.get("B_right", np.eye(4)))}
        fstats = e0.get("filter_stats", {})
        if any(m is None for m in mp4s):
            raise FileNotFoundError(
                f"mp4 missing for {ep_name} under {self.root / 'videos'}")

        return EpisodeRecord(
            name=ep_name, source="dataset", fps=fps, horizon=horizon,
            ticks_ns=(np.arange(T, dtype=np.int64) * round(1e9 / fps)),
            tick_orig=np.concatenate(tick_orig),
            pose=pose, hand=hand, state=state, arm_qpos=arm_qpos,
            masks={}, subeps=subeps, S=S, B=B,
            config_hash=self.sidecar.get(
                "config_hash", self.cfg_dict.get("_derived", {}).get(
                    "config_hash", "?")),
            b_mode=str(self.cfg_dict.get("b_alignment", "?")),
            filter_stats=fstats,
            ticks_total=sum(int(e.get("tick_end", 0)) -
                            int(e.get("tick_start", 0)) for _, e in entries)
                        or T,
            ticks_kept=T,
            meta={"source_file": e0.get("source_file", "?"),
                  "source_episode": e0.get("source_episode", ep_name),
                  "hands": hands,
                  "lerobot_episodes": [i for i, _ in entries]},
            _images=_VideoImages(mp4s, lengths),
        )


def open_reader(source, dataset_root=None, cfg=None):
    """source: 'auto' | 'dataset' | 'workdir' -> (reader, resolved source)."""
    cfg = cfg or PipelineConfig()
    root = Path(dataset_root) if dataset_root else \
        Path(cfg.output_root) / cfg.repo_id
    if source == "auto":
        source = "dataset" if (root / "extraction_meta.json").exists() \
            else "workdir"
        print(f"source auto -> {source} "
              f"({root if source == 'dataset' else cfg.work_dir})")
    if source == "dataset":
        return DatasetReader(root, cfg), "dataset"
    return WorkdirReader(cfg), "workdir"
