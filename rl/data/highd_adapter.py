"""
highD dataset → exiD-compatible reader shim.
============================================

highD uses a different column schema from exiD (simpler: straight 3-lane
Autobahn, no curvature). This adapter reads ``<rid>_tracks.csv``,
``<rid>_tracksMeta.csv`` and ``<rid>_recordingMeta.csv`` from highD and
emits the same (tracks, tracks_meta, rec_meta) shape that
``tracks_import.read_from_csv`` returns for exiD, so
``rl/data/historical_extractor.py`` can reuse its entire pipeline.

Synthesised columns
-------------------
* ``heading`` (deg)          — 0 for drivingDirection=2 (+x traffic),
                                180 for drivingDirection=1 (-x traffic).
                                highD lanes are straight so this is exact.
* ``xCenter``, ``yCenter``   — bbox centre (highD ``x,y`` are top-left).
* ``laneChange``             — 1 on frames where ``laneId`` changes from
                                the previous frame, 0 otherwise.
* ``latLaneCenterOffset``    — y minus lane-centre y, where lane centres
                                come from recordingMeta's
                                ``upperLaneMarkings`` / ``lowerLaneMarkings``
                                string (semicolon-separated y values).

``xVelocity`` and ``yVelocity`` are kept as-is; the extractor rotates them
into the ego frame using the synthesised ``heading`` so direction-1 and
direction-2 traffic both end up with ``+ve`` longitudinal velocity.
"""

from __future__ import annotations

import os
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd


def _parse_lane_centres(markings_str: str) -> np.ndarray:
    """Parse a '8.51;12.59;16.43' lane-marking string into centre y-coords.

    Returns an array of length (n_markings - 1); centre i is the midpoint
    between marking i and marking i+1.
    """
    if not isinstance(markings_str, str) or not markings_str.strip():
        return np.zeros((0,), dtype=np.float32)
    ys = np.array([float(v) for v in markings_str.split(";") if v.strip()],
                  dtype=np.float32)
    if len(ys) < 2:
        return np.zeros((0,), dtype=np.float32)
    return 0.5 * (ys[:-1] + ys[1:])


def _build_lane_centre_map(rec_meta: Dict) -> Dict[int, float]:
    """Map laneId → lane-centre y.

    highD laneId convention (from the dataset docs): for a 2+3-lane
    recording, laneIds on the upper (driving-direction=1, -x traffic)
    side are the low numbers (e.g. 2,3), and laneIds on the lower side
    (direction=2, +x traffic) are the high numbers (e.g. 5,6,7). We
    don't know the exact mapping without more metadata, so we simply
    concatenate upper-lane centres then lower-lane centres in order
    and index by laneId - 2 (the smallest valid laneId). This gives
    a stable, recording-specific mapping that is good enough for the
    `latLaneCenterOffset` feature — the extractor already tolerates
    missing values by falling back to 0.
    """
    upper = _parse_lane_centres(rec_meta.get("upperLaneMarkings", ""))
    lower = _parse_lane_centres(rec_meta.get("lowerLaneMarkings", ""))
    centres = np.concatenate([upper, lower])
    # highD laneId starts at 2 (IDs 0 and 1 are reserved for off-road).
    # Map laneId 2 → centres[0], laneId 3 → centres[1], ...
    return {int(lid + 2): float(c) for lid, c in enumerate(centres)}


def read_highd_csvs(tracks_csv: str,
                    tracks_meta_csv: str,
                    rec_meta_csv: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """Return (tracks, tracks_meta, rec_meta) in exiD-compatible shape."""
    tdf = pd.read_csv(tracks_csv)
    mdf = pd.read_csv(tracks_meta_csv)
    rdf = pd.read_csv(rec_meta_csv)

    # Older highD releases ship the track identifier column as ``id`` rather
    # than ``trackId``. Normalise so downstream code can rely on a single
    # name. (recordingMeta.csv also has a stray ``trackId`` column that is
    # actually the recording id — we don't use it, but rename for safety.)
    if "id" in tdf.columns and "trackId" not in tdf.columns:
        tdf = tdf.rename(columns={"id": "trackId"})
    if "id" in mdf.columns and "trackId" not in mdf.columns:
        mdf = mdf.rename(columns={"id": "trackId"})
    rec_meta_row = rdf.iloc[0].to_dict()

    # Build metadata dicts keyed exiD-style
    drive_dir_by_id = dict(zip(mdf["trackId"].astype(int),
                               mdf["drivingDirection"].astype(int)))

    lane_centre_map = _build_lane_centre_map(rec_meta_row)

    # Per-track dicts.  Sort by frame to ensure time ordering.
    tdf = tdf.sort_values(["trackId", "frame"], kind="stable")

    # Pre-extract width/height so we can compute bbox centres from the
    # top-left (x, y) that highD stores.
    wh_by_id = dict(zip(mdf["trackId"].astype(int),
                        zip(mdf["width"].astype(float),
                            mdf["height"].astype(float))))

    tracks: List[Dict] = []
    for tid, g in tdf.groupby("trackId", sort=True):
        tid = int(tid)
        w, h = wh_by_id.get(tid, (0.0, 0.0))
        drive_dir = int(drive_dir_by_id.get(tid, 2))

        x_raw = g["x"].to_numpy(dtype=np.float32)
        y_raw = g["y"].to_numpy(dtype=np.float32)
        vx    = g["xVelocity"].to_numpy(dtype=np.float32)
        vy    = g["yVelocity"].to_numpy(dtype=np.float32)
        lane  = g["laneId"].to_numpy(dtype=np.int64)
        frame = g["frame"].to_numpy(dtype=np.int64)

        xc = x_raw + 0.5 * float(w)
        yc = y_raw + 0.5 * float(h)

        # heading (deg): 180 for -x traffic (direction 1), 0 for +x (direction 2)
        heading_deg = np.full_like(xc, 180.0 if drive_dir == 1 else 0.0,
                                   dtype=np.float32)

        # laneChange: 1 on the frame of a transition, else 0
        lane_change = np.zeros_like(lane, dtype=np.int64)
        if len(lane) > 1:
            lane_change[1:] = (lane[1:] != lane[:-1]).astype(np.int64)

        # latLaneCenterOffset: y minus lane-centre y (0 for missing ids)
        lat_offset = np.zeros_like(yc, dtype=np.float32)
        for i, lid in enumerate(lane):
            c = lane_centre_map.get(int(lid))
            if c is not None:
                lat_offset[i] = yc[i] - c

        tracks.append({
            "trackId":    tid,
            "frame":      frame,
            "xCenter":    xc,
            "yCenter":    yc,
            "xVelocity":  vx,
            "yVelocity":  vy,
            "heading":    heading_deg,      # degrees, matches exiD
            "laneChange": lane_change,
            "latLaneCenterOffset": lat_offset,
        })

    # tracks_meta list of dicts (exiD has "trackId" and "class")
    tracks_meta = []
    for _, row in mdf.iterrows():
        d = row.to_dict()
        d["trackId"] = int(d["trackId"])
        d["class"]   = str(d.get("class", "car")).lower()  # 'Car' → 'car'
        tracks_meta.append(d)

    rec_meta = {
        "frameRate": float(rec_meta_row.get("frameRate", 25.0)),
        "locationId": int(rec_meta_row.get("locationId", -1)),
        "speedLimit": float(rec_meta_row.get("speedLimit", -1)),
    }

    return tracks, tracks_meta, rec_meta
