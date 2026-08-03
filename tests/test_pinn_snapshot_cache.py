from __future__ import annotations

import numpy as np

from rl.risk.pinn_snapshot_cache import CachedSnapshotCollection, write_snapshot_cache


def _snapshots(recording_id: str, n: int, shape=(3, 4)):
    rows = []
    for index in range(n):
        base = np.full(shape, float(index + 1), dtype=np.float32)
        rows.append(
            {
                "recording_id": recording_id,
                "t": 0.1 * index,
                "dt": 0.1,
                "frame_id": index,
                "R": base,
                "Q": base * 0.1,
                "vx": base * 0.0,
                "vy": base * 0.0,
                "D": base * 0.2,
                "occ_mask": base * 0.0,
                "dist_nearest": base * 2.0,
                "N_agents": 2.0,
                "ego_x": 1.0,
                "ego_y": 2.0,
            }
        )
    return rows


def test_cached_collection_preserves_recording_boundaries(tmp_path):
    x = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    y = np.linspace(-2.0, 2.0, 3, dtype=np.float32)
    paths = []
    for recording_id, n in (("01", 3), ("02", 2)):
        path = tmp_path / f"inD_{recording_id}"
        write_snapshot_cache(
            snapshots=_snapshots(recording_id, n),
            x_grid=x,
            y_grid=y,
            output_dir=path,
            metadata={"dataset": "inD", "recording_id": recording_id, "signature": recording_id},
        )
        paths.append(path)

    collection = CachedSnapshotCollection(paths)
    assert len(collection) == 5
    assert collection.recording_keys == ["inD:01", "inD:02"]
    assert collection.valid_next_indices.tolist() == [0, 1, 3]
    assert collection[2]["sequence_end"] is True
    assert collection[3]["t"] == 0.0
    np.testing.assert_allclose(collection[4]["R"], np.full((3, 4), 2.0))
