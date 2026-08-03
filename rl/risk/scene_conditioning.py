"""
Shared scene-conditioning utilities for PINN training and inference.

This module centralizes:
  - relevance-based agent selection
  - context summaries for context-enabled PINNs
  - trainer-side source / diffusion / decay refinements

The helpers are deliberately numpy-only so they can be imported from the
historical trainer, the synthetic highway trainer, runtime inference, and
visualization scripts without extra dependencies.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _sigmoid(x: float) -> float:
    z = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-z))


def _heading_unit(vehicle: dict) -> tuple[float, float]:
    if 'heading' in vehicle:
        return float(math.cos(vehicle['heading'])), float(math.sin(vehicle['heading']))
    vx = float(vehicle.get('vx', 0.0))
    vy = float(vehicle.get('vy', 0.0))
    speed = math.hypot(vx, vy)
    if speed > 1e-6:
        return vx / speed, vy / speed
    return 1.0, 0.0


def _project_relative(ego: dict, vehicle: dict) -> tuple[float, float, float]:
    hx, hy = _heading_unit(ego)
    dx = float(vehicle['x']) - float(ego['x'])
    dy = float(vehicle['y']) - float(ego['y'])
    delta_x = dx * hx + dy * hy
    delta_y = -dx * hy + dy * hx
    rel_v = ((float(vehicle.get('vx', 0.0)) - float(ego.get('vx', 0.0))) * hx +
             (float(vehicle.get('vy', 0.0)) - float(ego.get('vy', 0.0))) * hy)
    return delta_x, delta_y, rel_v


def _is_occluding_candidate(ego: dict, vehicle: dict) -> bool:
    if vehicle.get('class', 'car') != 'truck':
        return False
    delta_x, delta_y, _ = _project_relative(ego, vehicle)
    distance = math.hypot(delta_x, delta_y)
    return delta_x > 0.0 and abs(delta_y) < 6.0 and distance <= 70.0


def _lane_bucket(delta_y: float) -> str:
    ay = abs(float(delta_y))
    if ay < 2.75:
        return 'same'
    if ay < 6.5:
        return 'adjacent'
    return 'outer'


def _mandatory_tags(record: dict) -> set[str]:
    tags: set[str] = set()
    dx = float(record['delta_x'])
    dy = float(record['delta_y'])
    closing = float(record['closing'])
    bucket = _lane_bucket(dy)

    if record['is_occluding']:
        tags.add('occluding_truck')
    if bucket == 'same' and dx > -10.0:
        tags.add('same_lane_anchor')
        if dx >= 0.0:
            tags.add('same_lane_lead')
    if bucket == 'same' and dx < 0.0 and closing > 0.5 and abs(dx) < 25.0:
        tags.add('rear_closing')
    if bucket == 'adjacent' and dx > -15.0:
        tags.add('adjacent_anchor')
    return tags


def score_agent_relevance(
    ego: dict,
    vehicle: dict,
    perception_range: float = 80.0,
    is_occluding: bool | None = None,
) -> dict:
    """
    Compute a soft relevance score and the associated relative features.
    """
    delta_x, delta_y, rel_v_parallel = _project_relative(ego, vehicle)
    distance = math.hypot(delta_x, delta_y)
    closing = max(0.0, -rel_v_parallel)
    ttc = float(np.clip(max(delta_x, 0.0) / max(closing, 0.1), 0.0, 8.0))
    if is_occluding is None:
        is_occluding = _is_occluding_candidate(ego, vehicle)

    gate = _sigmoid((float(perception_range) - distance) / 5.0) if math.isfinite(perception_range) else 1.0
    score = gate * (
        0.35 * math.exp(-abs(delta_x) / 35.0) +
        0.20 * math.exp(-abs(delta_y) / 3.5) +
        0.20 * math.exp(-ttc / 3.0) +
        0.10 * float(np.clip(closing / 10.0, 0.0, 1.0)) +
        0.10 * float(vehicle.get('class', 'car') == 'truck') +
        0.05 * float(is_occluding)
    )
    return {
        'score': float(score),
        'distance': float(distance),
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
        'closing': float(closing),
        'ttc': float(ttc),
        'is_occluding': bool(is_occluding),
    }


def compute_dist_nearest_field(
    X: np.ndarray,
    Y: np.ndarray,
    agents: Iterable[dict],
    fill_value: float = 1000.0,
) -> np.ndarray:
    pts = list(agents)
    if not pts:
        return np.full(X.shape, float(fill_value), dtype=np.float32)
    axy = np.array([[float(v['x']), float(v['y'])] for v in pts], dtype=np.float32)
    dx = X[:, :, np.newaxis] - axy[:, 0]
    dy = Y[:, :, np.newaxis] - axy[:, 1]
    return np.sqrt(dx**2 + dy**2).min(axis=2).astype(np.float32)


def select_agents(
    ego: dict,
    vehicles: Iterable[dict],
    perception_range: float = 80.0,
    selection_mode: str = 'soft_topk',
    top_k: int = 5,
    threshold_ratio: float = 0.15,
) -> dict:
    """
    Select the most relevant surrounding agents and summarize their context.
    """
    records = []
    for vehicle in vehicles:
        if vehicle is None or vehicle.get('id') == ego.get('id'):
            continue
        rel = score_agent_relevance(ego, vehicle, perception_range=perception_range)
        tags = _mandatory_tags(rel)
        if math.isfinite(perception_range) and rel['distance'] > perception_range and selection_mode == 'all':
            continue
        records.append({'vehicle': vehicle, 'tags': tags, **rel})

    if not records:
        return {
            'selected_agents': [],
            'relevance_weights': np.zeros(0, dtype=np.float32),
            'N_agents_selected': 0.0,
            'dist_nearest_selected': None,
            'truck_presence': 0.0,
            'occlusion_score': 0.0,
            'mass_retained': 1.0,
        }

    total_mass = max(sum(r['score'] for r in records), 1e-8)
    records.sort(key=lambda r: r['score'], reverse=True)

    if selection_mode == 'all':
        kept = records
    else:
        peak = max(records[0]['score'], 1e-8)
        kept_ids = set()
        kept = []

        def _take(rec):
            vid = rec['vehicle'].get('id')
            if vid in kept_ids:
                return
            kept.append(rec)
            kept_ids.add(vid)

        # Always preserve structural agents that define the occlusion / lane-critical geometry.
        for tag in ('occluding_truck', 'same_lane_lead', 'same_lane_anchor', 'adjacent_anchor', 'rear_closing'):
            tagged = [r for r in records if tag in r['tags']]
            if not tagged:
                continue
            if tag in ('same_lane_anchor', 'adjacent_anchor'):
                # Keep the closest anchor for each side/bucket.
                tagged.sort(key=lambda r: (abs(r['delta_y']), abs(r['delta_x'])))
                _take(tagged[0])
            else:
                tagged.sort(key=lambda r: (-r['score'], abs(r['delta_x'])))
                _take(tagged[0])

        # Fill the remaining budget with the strongest soft scores.
        budget = max(int(top_k), len(kept))
        for rec in records:
            if rec['score'] < threshold_ratio * peak and rec['vehicle'].get('id') not in kept_ids:
                continue
            _take(rec)
            if len(kept) >= budget:
                break

        if not kept:
            kept = [records[0]]

    weights = np.array([max(r['score'], 1e-8) for r in kept], dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-8)

    selected_agents = []
    for rank, (rec, weight) in enumerate(zip(kept, weights), start=1):
        agent = dict(rec['vehicle'])
        agent['relevance_score'] = float(rec['score'])
        agent['relevance_weight'] = float(weight)
        agent['selection_rank'] = rank
        agent['is_occluding'] = bool(rec['is_occluding'])
        agent['selection_tags'] = tuple(sorted(rec.get('tags', ())))
        selected_agents.append(agent)

    return {
        'selected_agents': selected_agents,
        'relevance_weights': weights,
        'N_agents_selected': float(len(selected_agents)),
        'dist_nearest_selected': None,
        'truck_presence': float(any(a.get('class', 'car') == 'truck' for a in selected_agents)),
        'occlusion_score': float(sum(
            a.get('relevance_weight', 0.0) for a in selected_agents if a.get('is_occluding', False)
        )),
        'mass_retained': float(sum(r['score'] for r in kept) / total_mass),
    }


def summarize_selected_agents(
    ego: dict,
    vehicles: Iterable[dict],
    X: np.ndarray | None = None,
    Y: np.ndarray | None = None,
    perception_range: float = 80.0,
    selection_mode: str = 'soft_topk',
    top_k: int = 5,
    threshold_ratio: float = 0.15,
) -> dict:
    result = select_agents(
        ego,
        vehicles,
        perception_range=perception_range,
        selection_mode=selection_mode,
        top_k=top_k,
        threshold_ratio=threshold_ratio,
    )
    selected = result['selected_agents']
    if X is not None and Y is not None:
        result['dist_nearest_selected'] = compute_dist_nearest_field(
            X, Y, selected, fill_value=perception_range if math.isfinite(perception_range) else 1000.0
        )
    elif selected:
        result['dist_nearest_selected'] = float(min(
            math.hypot(float(v['x']) - float(ego['x']), float(v['y']) - float(ego['y']))
            for v in selected
        ))
    else:
        result['dist_nearest_selected'] = float(perception_range if math.isfinite(perception_range) else 1000.0)
    return result


def refine_source_field(
    Q_base: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    agents: Iterable[dict],
) -> np.ndarray:
    """
    Add trainer-only source refinements for class asymmetry and braking wakes.
    """
    Q = np.asarray(Q_base, dtype=np.float32).copy()
    for agent in agents:
        weight = float(agent.get('relevance_weight', 1.0))
        cls = agent.get('class', 'car')
        hx, hy = _heading_unit(agent)
        dX = X - float(agent['x'])
        dY = Y - float(agent['y'])
        dX_rot = hx * dX + hy * dY
        dY_rot = -hy * dX + hx * dY

        sigma_par = 8.0
        sigma_perp = 2.5
        amp = 0.0
        if cls == 'truck':
            sigma_par *= 1.5
            sigma_perp *= 1.2
            amp += 0.35

        accel = float(agent.get('a', 0.0))
        if accel < -0.3:
            sigma_par *= 1.2 + min(abs(accel), 3.0) * 0.15
            amp += 0.20 + 0.10 * min(abs(accel), 3.0)

        if amp <= 0.0:
            continue

        gaussian = np.exp(-0.5 * (dX_rot**2 / max(sigma_par**2, 1e-6) +
                                   dY_rot**2 / max(sigma_perp**2, 1e-6)))
        Q += (weight * amp * gaussian).astype(np.float32)
    return Q


def refine_diffusion_field(
    D_base: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    agents: Iterable[dict],
    occ_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Add trainer-only diffusion refinements for trucks, braking wakes, and
    occlusion-aware spreading.
    """
    D = np.asarray(D_base, dtype=np.float32).copy()
    if occ_mask is not None:
        D += 0.15 * np.asarray(occ_mask, dtype=np.float32)

    for agent in agents:
        weight = float(agent.get('relevance_weight', 1.0))
        hx, hy = _heading_unit(agent)
        dX = X - float(agent['x'])
        dY = Y - float(agent['y'])
        dX_rot = hx * dX + hy * dY
        dY_rot = -hy * dX + hx * dY

        if agent.get('class', 'car') == 'truck':
            wake = np.exp(-0.5 * (np.maximum(dX_rot, 0.0)**2 / (14.0**2) + dY_rot**2 / (4.0**2)))
            D += (0.75 * weight * wake).astype(np.float32)

        accel = float(agent.get('a', 0.0))
        if accel < -0.3:
            local = np.exp(-0.5 * (dX_rot**2 / (10.0**2) + dY_rot**2 / (3.5**2)))
            D += ((0.35 + 0.20 * min(abs(accel), 3.0)) * weight * local).astype(np.float32)

    return D


def compute_geometry_decay_field(
    vx: np.ndarray,
    vy: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    cfg,
    occ_mask: np.ndarray | None = None,
    enable_geometry: bool = False,
) -> np.ndarray:
    """
    Build the trainer-side decay field used by the solver-consistent residual.
    """
    speed = np.sqrt(np.asarray(vx, dtype=np.float32)**2 + np.asarray(vy, dtype=np.float32)**2)
    lam = float(cfg.lambda_decay) + speed / max(float(getattr(cfg, 'L_decay', 25.0)), 1e-6)

    if hasattr(cfg, 'sponge_length') and float(getattr(cfg, 'sponge_length', 0.0)) > 0.0:
        x_max = float(np.asarray(X)[:, -1].mean())
        sponge_len = float(cfg.sponge_length)
        x_start = x_max - sponge_len
        w_sponge = np.clip((X - x_start) / max(sponge_len, 1e-6), 0.0, 1.0)
        lam = lam + float(getattr(cfg, 'lambda_sponge', 0.0)) * w_sponge**2

    if occ_mask is not None:
        lam = lam + 0.15 * np.asarray(occ_mask, dtype=np.float32)

    if enable_geometry:
        y_min = float(np.min(Y))
        y_max = float(np.max(Y))
        lane_width = float(getattr(cfg, 'lane_width', max((y_max - y_min) / 4.0, 1.0)))
        dist_edge = np.minimum(Y - y_min, y_max - Y)
        b_edge = 1.0 - np.clip(dist_edge / max(2.0 * lane_width, 1e-6), 0.0, 1.0)
        merge_mask = np.zeros_like(X, dtype=np.float32)
        if hasattr(cfg, 'merge_x_start') and hasattr(cfg, 'merge_x_end'):
            merge_mask = ((X >= float(cfg.merge_x_start)) & (X <= float(cfg.merge_x_end))).astype(np.float32)
        lam = lam + 0.08 * b_edge.astype(np.float32) + 0.12 * merge_mask

    return lam.astype(np.float32)


def finite_difference_samples(
    field: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample first-order central differences from a 2-D field at integer indices.
    """
    x_l = np.clip(xi - 1, 0, field.shape[1] - 1)
    x_r = np.clip(xi + 1, 0, field.shape[1] - 1)
    y_b = np.clip(yi - 1, 0, field.shape[0] - 1)
    y_t = np.clip(yi + 1, 0, field.shape[0] - 1)
    fx = (field[yi, x_r] - field[yi, x_l]) / max(2.0 * dx, 1e-6)
    fy = (field[y_t, xi] - field[y_b, xi]) / max(2.0 * dy, 1e-6)
    return fx.astype(np.float32), fy.astype(np.float32)
