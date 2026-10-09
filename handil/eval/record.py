"""Record MP4 video of a rollout.

Reimplemented rather than importing the source repo's `SceneRecorder`: that one
depends on `cv2` + its `kinova_common.video.H264Writer`, while `imageio-ffmpeg`
bundles its own encoder and needs no system package. The camera names and the
fps/size defaults are kept identical so the output is comparable to the
`--trial-log` videos hand-control produces.

The robocasa scene defines three cameras: `overview` (3rd person), `wrist`
(tracking 3rd-person view of the gripper) and `viewer`.

Recording is best-effort and never load-bearing: if the GL context or the
encoder cannot be created, the rollout still runs and the caller is told.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CAMERAS = ("overview", "wrist")
DEFAULT_FPS = 30.0
DEFAULT_SIZE = (1280, 960)


@dataclass
class Recording:
    paths: dict[str, Path]
    n_frames: int
    fps: float


class RolloutRecorder:
    """Captures frames from named cameras and writes one MP4 per camera.

    Frame capture is driven by the caller and throttled on *sim* time, so a
    rollout stepping 12 physics substeps per control step still yields a
    real-time-looking video.
    """

    def __init__(self, model, out_dir: Path, name: str,
                 cameras=DEFAULT_CAMERAS, fps: float = DEFAULT_FPS,
                 size=DEFAULT_SIZE):
        import mujoco as mj

        self.enabled = False
        self.fps = float(fps)
        self._w, self._h = int(size[0]), int(size[1])
        self._writers: dict[str, object] = {}
        self._renderer = None
        self._last_t: float | None = None
        self._n = 0
        self.paths: dict[str, Path] = {}

        self._cams = []
        for cam in cameras:
            cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cam)
            if cid < 0:
                print(f"[record] camera {cam!r} not in the scene — skipping")
            else:
                self._cams.append((cam, int(cid)))
        if not self._cams:
            print("[record] no valid cameras — recording disabled")
            return

        try:
            import imageio.v2 as imageio

            self._renderer = mj.Renderer(model, height=self._h, width=self._w)
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for cam, _ in self._cams:
                p = out_dir / f"{name}_{cam}.mp4"
                # macro_block_size=None keeps the exact requested resolution
                # instead of silently rounding it to a multiple of 16.
                self._writers[cam] = imageio.get_writer(
                    p, fps=self.fps, macro_block_size=None,
                    codec="libx264", quality=7)
                self.paths[cam] = p
            self.enabled = True
        except Exception as e:                       # GL context, codec, ...
            print(f"[record] disabled ({type(e).__name__}: {e})")
            self._teardown()

    def capture(self, data, force: bool = False):
        """Render one frame per camera, throttled to `fps` on sim time."""
        if not self.enabled:
            return
        t = float(data.time)
        if not force and self._last_t is not None and \
                (t - self._last_t) < (1.0 / self.fps):
            return
        self._last_t = t
        for cam, cid in self._cams:
            w = self._writers.get(cam)
            if w is None:
                continue
            self._renderer.update_scene(data, camera=cid)
            w.append_data(np.asarray(self._renderer.render()))
        self._n += 1

    def close(self) -> Recording:
        rec = Recording(paths=dict(self.paths), n_frames=self._n, fps=self.fps)
        self._teardown()
        return rec

    def _teardown(self):
        for w in self._writers.values():
            try:
                w.close()
            except Exception:
                pass
        self._writers.clear()
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self.enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._teardown()
        return False
