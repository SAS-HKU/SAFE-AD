"""Disk-backed numerical-teacher snapshots for multi-recording PINN training."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np


CACHE_VERSION = 1
GRID_FIELDS = (
    "R",
    "Q",
    "vx",
    "vy",
    "D",
    "occ_mask",
    "dist_nearest",
)
OPTIONAL_GRID_FIELDS = (
    "R_prev",
    "R_t",
    "R_t_prev",
    "road_mask",
    "Q_vehicle",
    "Q_occlusion",
    "Q_merge",
    "Q_behavior_refinement",
    "Q_terminal",
)
SCALAR_FIELDS = (
    "t",
    "dt",
    "frame_id",
    "N_agents",
    "truck_presence",
    "occlusion_score",
    "selection_mass",
    "ego_x",
    "ego_y",
    "ego_ax",
    "ego_v_lat",
    "ego_heading",
    "ego_vx",
    "ego_vy",
    "ego_trackId",
    "ego_world_x",
    "ego_world_y",
    "ego_world_heading",
    "ego_yaw_rate",
)


def _stable_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _source_fingerprint(data_dir: Path, recording_id: str) -> list[dict[str, Any]]:
    records = []
    for suffix in ("tracks.csv", "tracksMeta.csv", "recordingMeta.csv"):
        path = data_dir / f"{recording_id}_{suffix}"
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        records.append(
            {
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return records


def _safe_remove_cache_dir(path: Path, cache_root: Path) -> None:
    resolved = path.resolve()
    root = cache_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to remove cache outside {root}: {resolved}") from exc
    if resolved == root:
        raise RuntimeError(f"Refusing to remove cache root: {root}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _replace_cache_dir(source: Path, target: Path, attempts: int = 12) -> None:
    """Atomically publish a cache, tolerating transient Windows file locks."""
    last_error = None
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.20 * (attempt + 1))
    raise PermissionError(
        f"Unable to publish completed cache {source} -> {target} after {attempts} attempts"
    ) from last_error


def write_snapshot_cache(
    *,
    snapshots: Sequence[dict[str, Any]],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Write one recording as independently memory-mappable ``.npy`` arrays."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if not snapshots:
        raise ValueError("Cannot cache an empty snapshot sequence")

    x_grid = np.asarray(x_grid, dtype=np.float32)
    y_grid = np.asarray(y_grid, dtype=np.float32)
    shape = tuple(np.asarray(snapshots[0]["R"]).shape)
    expected_shape = (len(y_grid), len(x_grid))
    if shape != expected_shape:
        raise ValueError(f"Snapshot shape {shape} does not match grid {expected_shape}")

    np.save(output / "x_grid.npy", x_grid)
    np.save(output / "y_grid.npy", y_grid)

    present_optional = [key for key in OPTIONAL_GRID_FIELDS if key in snapshots[0]]
    for key in (*GRID_FIELDS, *present_optional):
        target = np.lib.format.open_memmap(
            output / f"{key}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(snapshots), *shape),
        )
        for index, snapshot in enumerate(snapshots):
            if key == "dist_nearest" and key not in snapshot:
                value = np.full(shape, 1000.0, dtype=np.float32)
            elif key == "occ_mask" and key not in snapshot:
                value = np.zeros(shape, dtype=np.float32)
            else:
                value = np.asarray(snapshot[key], dtype=np.float32)
            if value.shape != shape:
                raise ValueError(
                    f"Snapshot {index} field {key!r} has shape {value.shape}; expected {shape}"
                )
            target[index] = value
        target.flush()
        del target

    scalar_payload: dict[str, np.ndarray] = {}
    for key in SCALAR_FIELDS:
        default = -1.0 if key in {"frame_id", "ego_trackId"} else 0.0
        scalar_payload[key] = np.asarray(
            [float(snapshot.get(key, default)) for snapshot in snapshots],
            dtype=np.float32,
        )
    np.savez(output / "scalars.npz", **scalar_payload)

    manifest = {
        "cache_version": CACHE_VERSION,
        "complete": True,
        "n_snapshots": len(snapshots),
        "grid_shape": list(shape),
        "grid_fields": list(GRID_FIELDS),
        "optional_grid_fields": present_optional,
        "scalar_fields": list(SCALAR_FIELDS),
        **metadata,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def cache_teacher_recording(
    *,
    data_root: str | Path,
    dataset: str,
    recording_id: str,
    cache_root: str | Path,
    max_seconds: float,
    warmup_seconds: float,
    perception_range: float = 80.0,
    selection_mode: str = "soft_topk",
    top_k: int = 5,
    threshold_ratio: float = 0.15,
    store_source_components: bool = True,
    rebuild: bool = False,
) -> Path:
    """Build or restore one recording cache with strict provenance matching."""
    from pinn_risk_field import ExiDLoader

    data_root = Path(data_root)
    data_dir = data_root / dataset
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"{dataset}_{recording_id}"

    spec = {
        "cache_version": CACHE_VERSION,
        "dataset": str(dataset),
        "recording_id": str(recording_id),
        "max_seconds": float(max_seconds),
        "warmup_seconds": float(warmup_seconds),
        "perception_range": float(perception_range),
        "selection_mode": str(selection_mode),
        "top_k": int(top_k),
        "threshold_ratio": float(threshold_ratio),
        "store_source_components": bool(store_source_components),
        "source_files": _source_fingerprint(data_dir, str(recording_id)),
    }
    signature = _stable_signature(spec)

    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") and manifest.get("signature") == signature:
            return output
        raise RuntimeError(
            f"Stale or incompatible cache at {output}. Use --rebuild-cache explicitly."
        )

    building = cache_root / f".{output.name}.building-{os.getpid()}"
    _safe_remove_cache_dir(building, cache_root)
    loader = ExiDLoader(
        data_dir=str(data_dir),
        recording_id=str(recording_id),
        max_seconds=float(max_seconds),
        warmup_seconds=float(warmup_seconds),
        perception_range=float(perception_range),
        selection_mode=str(selection_mode),
        top_k=int(top_k),
        threshold_ratio=float(threshold_ratio),
        store_source_components=bool(store_source_components),
    )
    snapshots = loader.load()
    write_snapshot_cache(
        snapshots=snapshots,
        x_grid=loader.x_grid,
        y_grid=loader.y_grid,
        output_dir=building,
        metadata={**spec, "signature": signature},
    )
    if output.exists():
        if not rebuild:
            _safe_remove_cache_dir(building, cache_root)
            raise FileExistsError(output)
        _safe_remove_cache_dir(output, cache_root)
    _replace_cache_dir(building, output)
    return output


class CachedRecording(Sequence):
    """Lazy, read-only view of one cached recording."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise RuntimeError(f"Incomplete snapshot cache: {self.path}")
        if int(self.manifest.get("cache_version", -1)) != CACHE_VERSION:
            raise RuntimeError(f"Unsupported snapshot cache version: {self.path}")

        self.x_grid = np.load(self.path / "x_grid.npy", mmap_mode="r")
        self.y_grid = np.load(self.path / "y_grid.npy", mmap_mode="r")
        keys = [*self.manifest["grid_fields"], *self.manifest.get("optional_grid_fields", [])]
        self._fields = {
            key: np.load(self.path / f"{key}.npy", mmap_mode="r") for key in keys
        }
        with np.load(self.path / "scalars.npz") as scalars:
            self._scalars = {key: np.asarray(scalars[key]) for key in scalars.files}
        self.recording_id = str(self.manifest.get("recording_id", self.path.name))
        self.dataset = str(self.manifest.get("dataset", "unknown"))
        self.times = np.asarray(self._scalars["t"], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.manifest["n_snapshots"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        snapshot = {key: values[index] for key, values in self._fields.items()}
        snapshot.update({key: float(values[index]) for key, values in self._scalars.items()})
        snapshot["frame_id"] = int(round(snapshot["frame_id"]))
        snapshot["recording_id"] = self.recording_id
        snapshot["dataset"] = self.dataset
        snapshot["sequence_end"] = index == len(self) - 1
        return snapshot


class CachedSnapshotCollection(Sequence):
    """Concatenated lazy recordings with explicit temporal boundaries."""

    def __init__(self, recording_paths: Sequence[str | Path]):
        if not recording_paths:
            raise ValueError("At least one recording cache is required")
        self.recordings = [CachedRecording(path) for path in recording_paths]
        self.x_grid = np.asarray(self.recordings[0].x_grid)
        self.y_grid = np.asarray(self.recordings[0].y_grid)
        for recording in self.recordings[1:]:
            if not np.array_equal(recording.x_grid, self.x_grid) or not np.array_equal(
                recording.y_grid, self.y_grid
            ):
                raise ValueError("All cached recordings must use the same teacher grid")

        self._offsets = np.cumsum([0, *[len(recording) for recording in self.recordings]])
        self.times = np.concatenate([recording.times for recording in self.recordings])
        self.valid_next_indices = np.asarray(
            [
                int(self._offsets[r_index] + local_index)
                for r_index, recording in enumerate(self.recordings)
                for local_index in range(max(0, len(recording) - 1))
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rec_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local_index = int(index - self._offsets[rec_index])
        return self.recordings[rec_index][local_index]

    @property
    def recording_keys(self) -> list[str]:
        return [f"{recording.dataset}:{recording.recording_id}" for recording in self.recordings]

    @property
    def signatures(self) -> list[str]:
        return [str(recording.manifest.get("signature", "")) for recording in self.recordings]
