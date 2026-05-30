"""
pinn_scene_compare.py
=====================
Side-by-side scene overlay: Numerical DRIFT risk field (left) vs PINN
surrogate (right), drawn on the real orthophoto background with actual
vehicle bounding boxes.

Matches the visual style and figsave logic of drift_dataset_visualization.py:
  - Top-level config variables (no argparse)
  - Same pixel-space rendering (bboxVis, _vis_scale_down, _cfg_X_vis/Y_vis)
  - Same viewport logic (VIEW_X/Y clamp around ego)
  - Frame naming: {i}.png  (not frame_0000.png)
  - Single figure reused per frame (fig.clf() each iteration)
  - Progress bar via progress.Bar

Layout per frame
----------------
  ┌──────────────────────────┬──────────────────────────┐
  │  Numerical DRIFT  (PDE)  │  PINN Surrogate  (θ)     │
  │  + orthophoto + vehicles │  + orthophoto + vehicles │
  └──────────────────────────┴──────────────────────────┘

Usage
-----
  Edit DATASET_DIR, RECORDING_ID, N_t below, then run:
      python pinn_scene_compare.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import scienceplots          # noqa: F401
from scipy.ndimage import gaussian_filter as _gf
from scipy.stats import spearmanr
from progress.bar import Bar

DREAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DREAM_ROOT)

import torch
from tracks_import import read_from_csv
from config import Config as cfg
from pde_solver import (create_vehicle as drift_create_vehicle,
                        compute_total_Q, compute_velocity_field,
                        compute_diffusion_field)
from Integration.drift_interface import DRIFTInterface
from pinn_risk_field import (Normalizer, FlatSampleCache, RiskFieldNet,
                              PINNTrainer)
from rl.risk.scene_conditioning import (
    summarize_selected_agents,
    refine_source_field,
    refine_diffusion_field,
)

# ===========================================================================
# PARAMETERS — edit these
# ===========================================================================

DATASET_DIR    = r"c:\field_modeling\data\rounD"
DATASET = "rounD"
RECORDING_ID   = "02"          # zero-padded recording number, e.g. "00", "18"
EGO_TRACK_ID   = 37          # None = auto-select longest car track
EGO_MIN_FRAMES = 60            # minimum track length for ego candidate

MODEL_PATH     = r"c:\DREAM_final\pinn_inD_all.pt"          # None = auto-find pinn_inD_all.pt etc.

dt             = 0.1           # simulation step [s]
N_t            = 200           # number of steps to render
WARMUP_S       = 3.0           # DRIFT warm-up duration [s]

DRIFT_CELL_M   = 2.0           # DRIFT grid cell size [m]
SCENE_MARGIN   = 60.0          # margin beyond track bbox [m]

# Risk visualisation (same as drift_dataset_visualization.py)
RISK_ALPHA        = 0.24
RISK_CMAP         = "jet"
RISK_LEVELS       = 80
RISK_VMAX         = 2.0
RISK_MIN_VIS      = 0.08
RISK_SMOOTH_SIGMA = 2.0
RISK_ALPHA_GAMMA  = 0.8

# Viewport half-extents around ego [m]
VIEW_X = 50.0
VIEW_Y = 28.0

VIS_SCALE_DOWN    = None       # None = auto-detect
VIS_SCALE_PRESETS = {"exid": 6.0, "ind": 12.0, "round": 10.0}

rec      = f"{int(RECORDING_ID):02d}"
save_dir = os.path.join(DREAM_ROOT, f"figsave_PINN_compare_{rec}_DATASET-{DATASET}")
os.makedirs(save_dir, exist_ok=True)

# ===========================================================================
# DEFAULT CFG CENTRE — PINN training used this to shift world coords
# ===========================================================================
_DEFAULT_CFG_X_MID = (-150.0 + 255.2) / 2.0     #  52.6
_DEFAULT_CFG_Y_MID = (-225.2 + -45.3) / 2.0     # -135.25

# Module-level vars set during initialization
_ortho_px_m     = 1.0
_vis_scale_down = 1.0
_cfg_X_vis      = None
_cfg_Y_vis      = None
_ox             = 0.0    # world → PINN coordinate offset x
_oy             = 0.0    # world → PINN coordinate offset y
_track_x_all    = None
_track_y_all    = None


# ===========================================================================
# HELPERS
# ===========================================================================

def _infer_vis_scale(tracks, bg_img, dataset_dir, manual=None):
    if manual is not None and float(manual) > 0.0:
        return float(manual)
    ds_key = os.path.basename(os.path.normpath(dataset_dir)).lower()
    if ds_key in VIS_SCALE_PRESETS:
        return float(VIS_SCALE_PRESETS[ds_key])
    if bg_img is None:
        return 1.0
    try:
        x_vis = np.concatenate([np.asarray(t["xCenterVis"]).ravel() for t in tracks])
        y_vis = np.concatenate([np.asarray(t["yCenterVis"]).ravel() for t in tracks])
        h, w  = bg_img.shape[:2]
        s = max(1.0,
                float(np.nanmax(x_vis)) / max(1.0, 0.98 * w),
                float(np.nanmax(y_vis)) / max(1.0, 0.98 * h))
        return round(s * 2.0) / 2.0 if s > 2.0 else float(s)
    except Exception:
        return 1.0


def build_drift_vehicles(frame_idx, ego_track_id, tracks, tracks_meta, class_map):
    """
    Return (surrounding_vehicles, ego_vehicle, surr_tids) as DRIFT dicts for frame_idx.
    Ego is the dataset track with trackId == ego_track_id.
    surr_tids: list of dataset trackIds in the same order as surrounding_vehicles.
    """
    surrounding, ego_dict = [], None
    surr_tids = []   # dataset trackId for each surrounding vehicle (same order)
    vid = 1
    for tm in tracks_meta:
        tid = tm["trackId"]
        if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]):
            continue
        tr  = tracks[tid]
        fi  = frame_idx - tm["initialFrame"]
        cls = class_map.get(tid, "car")

        x     = float(tr["xCenter"][fi])
        y     = float(tr["yCenter"][fi])
        psi   = math.radians(float(tr["heading"][fi]))
        lon_v = float(tr["lonVelocity"][fi])
        vx_g  = lon_v * math.cos(psi)
        vy_g  = lon_v * math.sin(psi)
        lon_a = float(tr["lonAcceleration"][fi]) if "lonAcceleration" in tr else 0.0

        # Skip nearly-stationary non-vehicles (pedestrians, parked)
        if abs(lon_v) < 0.3 and cls not in ("car", "truck", "van"):
            continue

        vclass = "truck" if cls in ("truck", "van") else "car"
        v = drift_create_vehicle(vid=vid, x=x, y=y, vx=vx_g, vy=vy_g, vclass=vclass)
        v["heading"] = psi
        v["a"]       = lon_a

        if tid == ego_track_id:
            ego_dict = v
        else:
            surrounding.append(v)
            surr_tids.append(tid)
            vid += 1

    return surrounding, ego_dict, surr_tids


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _ego_behavior_targets(track, fi, frame_rate):
    n = len(track["xCenter"])
    i0 = max(fi - 1, 0)
    i1 = min(fi + 1, n - 1)
    dt_local = max((i1 - i0) / float(frame_rate), 1e-3)

    if "lonAcceleration" in track:
        a_human = float(track["lonAcceleration"][fi])
    else:
        v0 = float(track["lonVelocity"][i0])
        v1 = float(track["lonVelocity"][i1])
        a_human = (v1 - v0) / dt_local

    x0 = float(track["xCenter"][i0]); x1 = float(track["xCenter"][i1])
    y0 = float(track["yCenter"][i0]); y1 = float(track["yCenter"][i1])
    psi = math.radians(float(track["heading"][fi]))
    dx = (x1 - x0) / dt_local
    dy = (y1 - y0) / dt_local
    v_lat = -dx * math.sin(psi) + dy * math.cos(psi)
    return float(a_human), float(v_lat)


def _field_features_at_ego(risk_field, x, y, lookahead_m=20.0):
    ix = int(np.argmin(np.abs(cfg.x - x)))
    iy = int(np.argmin(np.abs(cfg.y - y)))
    ix = max(0, min(ix, cfg.nx - 1))
    iy = max(0, min(iy, cfg.ny - 1))

    ix_l = max(ix - 1, 0)
    ix_r = min(ix + 1, cfg.nx - 1)
    iy_b = max(iy - 1, 0)
    iy_t = min(iy + 1, cfg.ny - 1)
    ix_20 = int(np.argmin(np.abs(cfg.x - (x + lookahead_m))))
    ix_20 = max(0, min(ix_20, cfg.nx - 1))

    r_ego = float(risk_field[iy, ix])
    r_20m = float(risk_field[iy, ix_20])
    grad_x = float((risk_field[iy, ix_r] - risk_field[iy, ix_l]) / max(2.0 * cfg.dx, 1e-6))
    grad_y = float((risk_field[iy_t, ix] - risk_field[iy_b, ix]) / max(2.0 * cfg.dy, 1e-6))
    g_long = grad_x + 0.5 * (r_20m - r_ego) / max(lookahead_m, 1e-6)
    return {
        "r_ego": r_ego,
        "r_20m": r_20m,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "g_long": float(g_long),
        "g_lat": grad_y,
    }


def _background_stats(risk_field, dist_field, radius_m=10.0):
    mask = np.asarray(dist_field, dtype=np.float32) > float(radius_m)
    if not np.any(mask):
        mask = np.ones_like(risk_field, dtype=bool)
    bg_mean = float(np.mean(risk_field[mask]))
    d2x = np.gradient(np.gradient(risk_field, cfg.dx, axis=1), cfg.dx, axis=1)
    d2y = np.gradient(np.gradient(risk_field, cfg.dy, axis=0), cfg.dy, axis=0)
    lap = d2x + d2y
    lap_energy = float(np.mean((lap[mask]) ** 2))
    return bg_mean, lap_energy


# ===========================================================================
# VISUALIZATION
# ===========================================================================

def draw_frame_pinn_compare(i, frame_idx, tracks, tracks_meta, class_map,
                             bg_img,
                             risk_field_num, risk_at_ego_num,
                             risk_field_pinn, risk_at_ego_pinn,
                             agent_info=None):
    """
    Two-panel pixel-space rendering — matches draw_frame_drift_overlay() style.
      Panel 0: Numerical solver
      Panel 1: PINN Surrogate

    Uses global _ortho_px_m, _vis_scale_down, _cfg_X_vis, _cfg_Y_vis,
    _track_x_all, _track_y_all.
    """
    fig = plt.gcf()
    fig.clf()

    # Shared vmax derived from numerical field (same scale on both panels)
    vmax = RISK_VMAX
    if risk_field_num is not None:
        R_sm0 = _gf(risk_field_num, sigma=RISK_SMOOTH_SIGMA)
        nz = R_sm0[R_sm0 > RISK_MIN_VIS]
        if nz.size > 50:
            vmax = float(np.percentile(nz, 95))
        vmax = max(vmax, RISK_MIN_VIS + 1e-3)

    panels = [
        (risk_field_num,  risk_at_ego_num,
         r"Numerical solver"),
        (risk_field_pinn, risk_at_ego_pinn,
         r"PINN Surrogate  $\hat{\mathcal{R}}_\theta$"),
    ]

    # ── Ego pixel position for viewport ──────────────────────────────────────
    if EGO_TRACK_ID is not None:
        ego_tm = tracks_meta[EGO_TRACK_ID]
        ego_tr = tracks[EGO_TRACK_ID]
        if ego_tm["initialFrame"] <= frame_idx <= ego_tm["finalFrame"]:
            fi_ego = frame_idx - ego_tm["initialFrame"]
            ex_px = float(ego_tr["xCenterVis"][fi_ego]) / _vis_scale_down
            ey_px = float(ego_tr["yCenterVis"][fi_ego]) / _vis_scale_down
        else:
            ex_px = float(np.mean(_track_x_all) / (_ortho_px_m * _vis_scale_down))
            ey_px = float(np.mean(-_track_y_all) / (_ortho_px_m * _vis_scale_down))
    else:
        ex_px = float(np.mean(_track_x_all) / (_ortho_px_m * _vis_scale_down))
        ey_px = float(np.mean(-_track_y_all) / (_ortho_px_m * _vis_scale_down))

    view_x_px  = VIEW_X / (_ortho_px_m * _vis_scale_down)
    view_y_px  = VIEW_Y / (_ortho_px_m * _vis_scale_down)
    x0_vp, x1_vp       = ex_px - view_x_px, ex_px + view_x_px
    y_top_vp, y_bot_vp = ey_px - view_y_px, ey_px + view_y_px

    if bg_img is not None:
        h_bg, w_bg = bg_img.shape[:2]
        if x0_vp < 0:
            x1_vp -= x0_vp;  x0_vp = 0.0
        if x1_vp > (w_bg - 1):
            x0_vp -= (x1_vp - (w_bg - 1));  x1_vp = float(w_bg - 1)
        if y_top_vp < 0:
            y_bot_vp -= y_top_vp;  y_top_vp = 0.0
        if y_bot_vp > (h_bg - 1):
            y_top_vp -= (y_bot_vp - (h_bg - 1));  y_bot_vp = float(h_bg - 1)
        x0_vp    = max(0.0, x0_vp);  x1_vp    = min(float(w_bg - 1), x1_vp)
        y_top_vp = max(0.0, y_top_vp); y_bot_vp = min(float(h_bg - 1), y_bot_vp)

    # ── Draw each panel ───────────────────────────────────────────────────────
    for panel_idx, (rf, rae, panel_title) in enumerate(panels):
        ax = fig.add_subplot(1, 2, panel_idx + 1)
        ax.cla()

        # 1) Orthophoto background
        if bg_img is not None:
            ax.imshow(bg_img, origin="upper", zorder=0)
        else:
            ax.set_facecolor("#111111")

        # 2) Risk overlay in pixel space
        if rf is not None and _cfg_X_vis is not None and _cfg_Y_vis is not None:
            R_sm = _gf(rf, sigma=RISK_SMOOTH_SIGMA)
            Rn   = ((np.clip(R_sm, RISK_MIN_VIS, vmax) - RISK_MIN_VIS)
                    / max(vmax - RISK_MIN_VIS, 1e-9))
            Rn   = np.power(np.clip(Rn, 0.0, 1.0), RISK_ALPHA_GAMMA)
            R_masked = np.ma.masked_less_equal(Rn, 0.0)
            if np.ma.count(R_masked) > 0:
                ax.contourf(_cfg_X_vis, _cfg_Y_vis, R_masked,
                            levels=np.linspace(0.02, 1.0, RISK_LEVELS),
                            cmap=RISK_CMAP, alpha=RISK_ALPHA,
                            zorder=2, antialiased=True)

        # 3) Vehicles — keep the same visibility on both panels.
        for tm in tracks_meta:
            tid = tm["trackId"]
            if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]):
                continue
            tr   = tracks[tid]
            fi   = frame_idx - tm["initialFrame"]
            cls_ = class_map.get(tid, "car")
            is_ego = (tid == EGO_TRACK_ID)
            fc = "#F4511E" if is_ego else (
                 "#FF8C00" if cls_ in ("truck", "van") else "#AED6F1")
            ec  = "red" if is_ego else "black"
            lw  = 1.0   if is_ego else 0.5
            av  = 0.95  if is_ego else 0.82
            z   = 5     if is_ego else 4
            ls  = "-"

            if tr.get("bboxVis") is not None:
                bbox = np.asarray(tr["bboxVis"][fi], dtype=float) / _vis_scale_down
                poly = plt.Polygon(bbox, closed=True, facecolor=fc, edgecolor=ec,
                                   linewidth=lw, linestyle=ls,
                                   alpha=av, zorder=z)
                ax.add_patch(poly)
            else:
                cx = float(tr["xCenterVis"][fi]) / _vis_scale_down
                cy = float(tr["yCenterVis"][fi]) / _vis_scale_down
                circ = plt.Circle((cx, cy),
                                  radius=max(1.4, 2.4 / _vis_scale_down),
                                  facecolor=fc, edgecolor=ec,
                                  linewidth=lw, linestyle=ls,
                                  alpha=av, zorder=z)
                ax.add_patch(circ)

        # 4) Viewport
        ax.set_xlim(x0_vp, x1_vp)
        ax.set_ylim(y_bot_vp, y_top_vp)   # pixel Y increases downward
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # 5) Risk badge (bottom-right) + selector summary badge (bottom-left, PINN panel only)
        rc = ("red"    if rae > 1.5 else
              "orange" if rae > 0.5 else "lime")
        ax.text(0.985, 0.035, f"R={rae:.2f}",
                transform=ax.transAxes, ha="right", va="bottom",
                color=rc, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

        if panel_idx == 1 and agent_info is not None:
            n_in  = agent_info["n_in"]
            n_tot = agent_info["n_tot"]
            pr    = agent_info["perc_range"]
            pr_str = f"  ≤{pr:.0f}m" if pr is not None else "  (all)"
            ac    = "lime" if n_in == n_tot else "orange"
            ax.text(0.015, 0.035, f"Selected: {n_in}/{n_tot}{pr_str}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    color=ac, fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

        ax.set_title(f"{panel_title}  |  t={i * dt:.1f} s  frame={frame_idx}",
                     fontsize=9, fontweight="bold")

        # 6) Colorbar
        _sm = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=RISK_MIN_VIS, vmax=vmax),
            cmap=plt.colormaps[RISK_CMAP])
        _sm.set_array([])
        cbar = fig.colorbar(_sm, ax=ax, fraction=0.018, pad=0.005)
        if panel_idx == 0:
            cbar.set_label(f"Risk  (vmax={vmax:.2f})", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    plt.savefig(os.path.join(save_dir, f"{i}.png"), dpi=150,
                bbox_inches="tight")


# ===========================================================================
# INITIALIZATION
# ===========================================================================

print("=" * 70)
print(f"PINN vs Numerical DRIFT  |  {RECORDING_ID}  |  N_t={N_t}  dt={dt}")
print("=" * 70)

# ── Load dataset ─────────────────────────────────────────────────────────────
tracks_file      = os.path.join(DATASET_DIR, f"{rec}_tracks.csv")
tracks_meta_file = os.path.join(DATASET_DIR, f"{rec}_tracksMeta.csv")
rec_meta_file    = os.path.join(DATASET_DIR, f"{rec}_recordingMeta.csv")

print(f"Loading recording {rec} from {DATASET_DIR} ...")
tracks, tracks_meta, recording_meta = read_from_csv(
    tracks_file, tracks_meta_file, rec_meta_file,
    include_px_coordinates=True)

_ortho_px_m = float(recording_meta["orthoPxToMeter"])
frame_rate   = float(recording_meta["frameRate"])
frame_stride = max(1, round(dt / (1.0 / frame_rate)))   # e.g. 25 Hz → stride 2 for dt=0.08

class_map    = {tm["trackId"]: tm["class"] for tm in tracks_meta}
_track_x_all = np.concatenate([t["xCenter"] for t in tracks])
_track_y_all = np.concatenate([t["yCenter"] for t in tracks])

print(f"  Tracks: {len(tracks)}  |  frameRate={frame_rate} Hz  "
      f"frame_stride={frame_stride}  (dt={dt} s)")

# ── Background image ──────────────────────────────────────────────────────────
bg_path = os.path.join(DATASET_DIR, f"{rec}_background.png")
bg_img  = None
img_h, img_w = 0, 0
if os.path.exists(bg_path):
    _raw   = cv2.imread(bg_path)
    bg_img = cv2.cvtColor(_raw, cv2.COLOR_BGR2RGB)
    img_h, img_w = bg_img.shape[:2]
    print(f"  Background: {img_w}x{img_h} px")
else:
    print(f"  [WARN] Background not found at {bg_path}")

_vis_scale_down = _infer_vis_scale(tracks, bg_img, DATASET_DIR, VIS_SCALE_DOWN)
print(f"  ortho={_ortho_px_m:.6f} m/px  vis_scale_down={_vis_scale_down:.2f}")

# ── Expand DRIFT grid to full scene (same as drift_dataset_visualization.py) ─
cfg.x_min = float(np.min(_track_x_all)) - SCENE_MARGIN
cfg.x_max = float(np.max(_track_x_all)) + SCENE_MARGIN
cfg.y_min = float(np.min(_track_y_all)) - SCENE_MARGIN
cfg.y_max = float(np.max(_track_y_all)) + SCENE_MARGIN
cfg.nx    = int((cfg.x_max - cfg.x_min) / DRIFT_CELL_M) + 2
cfg.ny    = int((cfg.y_max - cfg.y_min) / DRIFT_CELL_M) + 2
cfg.dx    = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
cfg.dy    = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
cfg.x     = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
cfg.y     = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)   # world coordinates

_cfg_X_vis = cfg.X / (_ortho_px_m * _vis_scale_down)    # pixel space
_cfg_Y_vis = -cfg.Y / (_ortho_px_m * _vis_scale_down)
print(f"[DRIFT Grid] x=[{cfg.x_min:.0f},{cfg.x_max:.0f}]  "
      f"y=[{cfg.y_min:.0f},{cfg.y_max:.0f}]  "
      f"({cfg.nx}×{cfg.ny}={cfg.nx * cfg.ny // 1000}k cells)")

# ── PINN coordinate offset ────────────────────────────────────────────────────
# ExiDLoader training used: x_pinn = x_world - ox, where ox = median(x) - cfg_default_centre
# At inference we apply the same shift so the PINN input is in-distribution.
_ox = float(np.median(_track_x_all)) - _DEFAULT_CFG_X_MID
_oy = float(np.median(_track_y_all)) - _DEFAULT_CFG_Y_MID
# Grid in PINN coordinate space (for predict_field_from_arrays)
_X_pinn = cfg.X - _ox
_Y_pinn = cfg.Y - _oy
print(f"[PINN offset] ox={_ox:.1f}  oy={_oy:.1f}")

# ── Ego selection ─────────────────────────────────────────────────────────────
if EGO_TRACK_ID is None:
    _best_tid, _best_len = None, 0
    for tm in tracks_meta:
        if class_map.get(tm["trackId"], "") in ("car", "van"):
            if tm["numFrames"] > _best_len:
                _best_len = tm["numFrames"]
                _best_tid = tm["trackId"]
    EGO_TRACK_ID = _best_tid

ego_meta  = tracks_meta[EGO_TRACK_ID]
ego_track = tracks[EGO_TRACK_ID]
ego_fi0   = ego_meta["initialFrame"]
print(f"  Ego trackId={EGO_TRACK_ID}  class={class_map.get(EGO_TRACK_ID)}  "
      f"frames={ego_meta['numFrames']}")

# ── Load PINN model ───────────────────────────────────────────────────────────
print("\nLoading PINN model ...")
model_path = MODEL_PATH
if model_path is None:
    ds_name = os.path.basename(os.path.normpath(DATASET_DIR))
    for cand in [f"pinn_{ds_name}_all.pt",
                 f"pinn_{ds_name}_{rec}.pt",
                 "pinn_risk_field.pt"]:
        fp = os.path.join(DREAM_ROOT, cand)
        if os.path.isfile(fp):
            model_path = fp
            break
if model_path is None or not os.path.isfile(model_path):
    raise FileNotFoundError(
        "No PINN model found. Train first:\n"
        "  python pinn_risk_field.py --dataset inD --recording all")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt   = torch.load(model_path, map_location=device, weights_only=False)

norm = Normalizer.__new__(Normalizer)
norm.ranges       = ckpt["norm_ranges"]
norm.lambda_decay = cfg.lambda_decay
norm.tau          = cfg.tau

_HIDDEN       = int(ckpt.get("hidden",       128))
_DEPTH        = int(ckpt.get("depth",        6))
_USE_RFF      = bool(ckpt.get("use_rff",     False))
_RFF_FEATURES = int(ckpt.get("rff_features", 64))
_RFF_SCALE    = float(ckpt.get("rff_scale",  10.0))
_USE_CONTEXT  = bool(ckpt.get("use_context", False))
# Perception range used during training.
# Old models (pre-filter) have no key → 0.0 → no filter at inference.
# New models store the actual range → apply same filter at inference.
_PERC_RANGE   = float(ckpt.get("perception_range", 0.0))
_SELECTION_MODE = str(ckpt.get("selection_mode", "soft_topk"))
_TOP_K = int(ckpt.get("top_k", 5))
_THRESH_RATIO = float(ckpt.get("threshold_ratio", 0.15))

_n_cache_cols = len(FlatSampleCache.KEYS)   # 10 with context features
dummy_cache = FlatSampleCache.__new__(FlatSampleCache)
dummy_cache.x_grid = cfg.x
dummy_cache.y_grid = cfg.y
dummy_cache.times  = np.array([0.0, 1.0])
dummy_cache._buf   = np.zeros((2, _n_cache_cols), dtype=np.float32)
dummy_cache._N     = 2

trainer = PINNTrainer(snapshots=[], norm=norm, interp=dummy_cache,
                      hidden=_HIDDEN, depth=_DEPTH,
                      use_rff=_USE_RFF, rff_features=_RFF_FEATURES,
                      rff_scale=_RFF_SCALE,
                      use_context=_USE_CONTEXT,
                      device=device)
trainer.model.load_state_dict(ckpt["model_state"])
trainer.model.eval()
arch_tag = (f"RFF(feat={_RFF_FEATURES},scale={_RFF_SCALE:.0f})"
            if _USE_RFF else "no-RFF")
ctx_tag = "+ctx" if _USE_CONTEXT else ""
print(f"  PINN loaded from {model_path}  "
      f"({_HIDDEN}×{_DEPTH}, {arch_tag}{ctx_tag}), device={device}")
print(f"  selector={_SELECTION_MODE}  top_k={_TOP_K}  thr={_THRESH_RATIO:.2f}")

# ── DRIFT warm-up ─────────────────────────────────────────────────────────────
# DRIFTInterface is created AFTER cfg is scene-adapted, so it uses the correct grid.
print("\nDRIFT warm-up ...")
drift = DRIFTInterface()

_surr_init, _ego_v_init, _ = build_drift_vehicles(
    ego_fi0, EGO_TRACK_ID, tracks, tracks_meta, class_map)
if _ego_v_init is None:
    fi0 = ego_fi0 - ego_meta["initialFrame"]
    ex0 = float(ego_track["xCenter"][fi0])
    ey0 = float(ego_track["yCenter"][fi0])
    psi0 = math.radians(float(ego_track["heading"][fi0]))
    ev0  = max(float(ego_track["lonVelocity"][fi0]), 0.5)
    _ego_v_init = drift_create_vehicle(vid=0, x=ex0, y=ey0,
                                       vx=ev0 * math.cos(psi0),
                                       vy=ev0 * math.sin(psi0),
                                       vclass="car")
    _ego_v_init["heading"] = psi0

drift.warmup(_surr_init + [_ego_v_init], _ego_v_init,
             dt=dt, duration=WARMUP_S, substeps=3)
print()

# ===========================================================================
# MAIN RENDER LOOP
# ===========================================================================

print(f"Rendering {N_t} frames → {save_dir}/ ...")
risk_at_ego_num_list  = []
risk_at_ego_pinn_list = []
n_agents_in_list   = []   # N agents included in PINN per step
n_agents_tot_list  = []   # total agents visible per step
selection_mass_list = []
selected_tid_weight_hist = []
human_acc_list = []
human_vlat_list = []
g_long_num_list = []
g_long_pinn_list = []
g_lat_num_list = []
g_lat_pinn_list = []
bg_mean_num_list = []
bg_mean_pinn_list = []
bg_lap_num_list = []
bg_lap_pinn_list = []
max_frame_all = max(tm["finalFrame"] for tm in tracks_meta)

bar = Bar(max=N_t - 1)
plt.figure(figsize=(22, 7))

for i in range(N_t):
    bar.next()
    frame_idx = ego_fi0 + i * frame_stride

    if frame_idx > ego_meta["finalFrame"] or frame_idx > max_frame_all:
        print(f"\n[WARN] End of recording at step {i}.")
        break

    fi_ego = frame_idx - ego_meta["initialFrame"]
    ex     = float(ego_track["xCenter"][fi_ego])
    ey     = float(ego_track["yCenter"][fi_ego])
    epsi   = math.radians(float(ego_track["heading"][fi_ego]))
    ev     = float(ego_track["lonVelocity"][fi_ego])
    a_human, v_lat_human = _ego_behavior_targets(ego_track, fi_ego, frame_rate)

    surr_vehicles, ego_drift_v, surr_tids = build_drift_vehicles(
        frame_idx, EGO_TRACK_ID, tracks, tracks_meta, class_map)
    if ego_drift_v is None:
        ego_drift_v = drift_create_vehicle(
            vid=0, x=ex, y=ey,
            vx=ev * math.cos(epsi), vy=ev * math.sin(epsi),
            vclass="car")
        ego_drift_v["heading"] = epsi

    # ── Numerical DRIFT step ──────────────────────────────────────────────────
    risk_field_num  = drift.step(surr_vehicles, ego_drift_v, dt=dt, substeps=3)
    risk_at_ego_num = float(drift.get_risk_cartesian(ex, ey))
    risk_at_ego_num_list.append(risk_at_ego_num)

    # ── PINN inference ────────────────────────────────────────────────────────
    # Apply the same selection logic that the new PINN trainers use.
    _perc = _PERC_RANGE if _PERC_RANGE > 0 else float('inf')
    ctx = summarize_selected_agents(
        ego=ego_drift_v,
        vehicles=surr_vehicles,
        X=cfg.X,
        Y=cfg.Y,
        perception_range=_perc,
        selection_mode=_SELECTION_MODE,
        top_k=_TOP_K,
        threshold_ratio=_THRESH_RATIO,
    )
    all_vehicles = ctx['selected_agents']
    selected_tid_weights = {}
    for veh, tid in zip(surr_vehicles, surr_tids):
        for sel in all_vehicles:
            if veh['id'] == sel['id']:
                selected_tid_weights[tid] = float(sel.get('relevance_weight', 0.0))
                break
    included_tids = {
        tid for v, tid in zip(surr_vehicles, surr_tids)
        if any(v['id'] == sel['id'] for sel in all_vehicles)
    }
    excluded_tids = set(surr_tids) - included_tids
    n_in  = len(included_tids)
    n_tot = len(surr_vehicles)

    # Console log (every 20 steps to avoid spam)
    if i % 20 == 0:
        in_ids  = sorted(included_tids - {EGO_TRACK_ID})
        ex_ids  = sorted(excluded_tids)
        print(f"  [step {i:3d}] agents in PINN: {n_in}/{n_tot} "
              f"| included surr tids={in_ids} | excluded tids={ex_ids}")

    Q, _, _, occ = compute_total_Q(all_vehicles, ego_drift_v, cfg.X, cfg.Y)
    Q = refine_source_field(Q, cfg.X, cfg.Y, all_vehicles)
    vx_f, vy_f, *_ = compute_velocity_field(all_vehicles, ego_drift_v, cfg.X, cfg.Y)
    D_f = compute_diffusion_field(occ, cfg.X, cfg.Y, all_vehicles, ego_drift_v)
    D_f = refine_diffusion_field(D_f, cfg.X, cfg.Y, all_vehicles, occ_mask=occ)

    # Context features (only needed when model was trained with --use_context)
    _N_agents_frame = int(ctx['N_agents_selected'])
    _dist_nearest_frame = ctx.get('dist_nearest_selected', None) if _USE_CONTEXT else None

    # Query PINN in training coordinate space (world - offset)
    t_sim = WARMUP_S + i * dt
    risk_field_pinn = trainer.predict_field_from_arrays(
        _X_pinn, _Y_pinn, t_sim, Q, vx_f, vy_f, D_f,
        N_agents=_N_agents_frame, dist_nearest=_dist_nearest_frame)

    # Risk at ego: nearest grid cell
    i_y = int(np.argmin(np.abs(cfg.y - ey)))
    i_x = int(np.argmin(np.abs(cfg.x - ex)))
    i_y = max(0, min(i_y, cfg.ny - 1))
    i_x = max(0, min(i_x, cfg.nx - 1))
    risk_at_ego_pinn = float(risk_field_pinn[i_y, i_x])
    risk_at_ego_pinn_list.append(risk_at_ego_pinn)
    n_agents_in_list.append(n_in)
    n_agents_tot_list.append(n_tot)
    selection_mass_list.append(float(ctx.get('mass_retained', 1.0)))
    selected_tid_weight_hist.append(selected_tid_weights)
    human_acc_list.append(a_human)
    human_vlat_list.append(v_lat_human)

    feat_num = _field_features_at_ego(risk_field_num, ex, ey)
    feat_pinn = _field_features_at_ego(risk_field_pinn, ex, ey)
    g_long_num_list.append(feat_num["g_long"])
    g_long_pinn_list.append(feat_pinn["g_long"])
    g_lat_num_list.append(feat_num["g_lat"])
    g_lat_pinn_list.append(feat_pinn["g_lat"])

    bg_mean_num, bg_lap_num = _background_stats(risk_field_num, ctx['dist_nearest_selected'])
    bg_mean_pinn, bg_lap_pinn = _background_stats(risk_field_pinn, ctx['dist_nearest_selected'])
    bg_mean_num_list.append(bg_mean_num)
    bg_mean_pinn_list.append(bg_mean_pinn)
    bg_lap_num_list.append(bg_lap_num)
    bg_lap_pinn_list.append(bg_lap_pinn)

    # ── Draw frame ────────────────────────────────────────────────────────────
    agent_info = dict(included=included_tids, excluded=excluded_tids,
                      n_in=n_in, n_tot=n_tot,
                      perc_range=_perc if math.isfinite(_perc) else None)
    draw_frame_pinn_compare(
        i, frame_idx, tracks, tracks_meta, class_map, bg_img,
        risk_field_num,  risk_at_ego_num,
        risk_field_pinn, risk_at_ego_pinn,
        agent_info=agent_info)

bar.finish()
print()
print(f"Rendering complete — {N_t} frames saved to {save_dir}/")

# ── Summary metrics plot ──────────────────────────────────────────────────────
if len(risk_at_ego_num_list) > 1:
    _t = np.arange(len(risk_at_ego_num_list)) * dt
    with plt.style.context(["science", "no-latex"]):
        fig_m, ax_m = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
        ax_m.plot(_t, risk_at_ego_num_list,  color="C3", lw=1.4,
                  label="Numerical DRIFT")
        ax_m.plot(_t, risk_at_ego_pinn_list, color="C5", lw=1.4, ls="--",
                  label=r"PINN $\hat{\mathcal{R}}_\theta$")
        ax_m.fill_between(_t, risk_at_ego_num_list,  alpha=0.15, color="C3")
        ax_m.fill_between(_t, risk_at_ego_pinn_list, alpha=0.15, color="C5")
        ax_m.set_xlabel("t [s]")
        ax_m.set_ylabel("Risk at ego $R$")
        ax_m.set_title(
            f"DRIFT vs PINN Risk at Ego  |  rec {rec}  track {EGO_TRACK_ID}",
            fontsize=10)
        ax_m.legend(fontsize=8)
        ax_m.grid(True, lw=0.4, alpha=0.4)
        fig_m.savefig(os.path.join(save_dir, "risk_at_ego.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_m)
    print(f"Risk-at-ego plot → {save_dir}/risk_at_ego.png")

# ── Agent selection summary plot ──────────────────────────────────────────────
if len(n_agents_in_list) > 1:
    _t = np.arange(len(n_agents_in_list)) * dt
    _n_in  = np.array(n_agents_in_list,  dtype=float)
    _n_tot = np.array(n_agents_tot_list, dtype=float)
    _frac  = np.where(_n_tot > 0, _n_in / _n_tot, 1.0)

    with plt.style.context(["science", "no-latex"]):
        fig_a, (ax_a1, ax_a2) = plt.subplots(
            2, 1, figsize=(10, 5), constrained_layout=True,
            gridspec_kw={"height_ratios": [2, 1]})

        # Top: absolute counts
        ax_a1.plot(_t, _n_tot, color="C0", lw=1.2, label="Total agents visible")
        ax_a1.plot(_t, _n_in,  color="C2", lw=1.4, label="Agents selected for PINN")
        ax_a1.fill_between(_t, _n_in, _n_tot, alpha=0.20, color="C3",
                           label="Excluded agents")
        ax_a1.set_ylabel("Agent count")
        ax_a1.set_title(
            f"Agent selection  |  rec {rec}  ego={EGO_TRACK_ID}"
            + (f"  perc_range={_PERC_RANGE:.0f} m" if _PERC_RANGE > 0 else "  (no filter)"),
            fontsize=9)
        ax_a1.legend(fontsize=7, ncol=3)
        ax_a1.grid(True, lw=0.4, alpha=0.4)
        ax_a1.set_xlim(_t[0], _t[-1])

        # Bottom: inclusion fraction
        ax_a2.plot(_t, _frac * 100, color="C2", lw=1.2)
        ax_a2.axhline(100, color="grey", lw=0.6, ls="--")
        ax_a2.set_ylim(0, 110)
        ax_a2.set_xlabel("t [s]")
        ax_a2.set_ylabel("% included")
        ax_a2.set_xlim(_t[0], _t[-1])
        ax_a2.grid(True, lw=0.4, alpha=0.4)

        fig_a.savefig(os.path.join(save_dir, "agent_selection.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_a)
    print(f"Agent selection plot → {save_dir}/agent_selection.png")

    # Print per-step summary to console
    print("\n--- Agent selection summary ---")
    print(f"  Mean included:  {_n_in.mean():.1f} / {_n_tot.mean():.1f}  "
          f"({_frac.mean()*100:.1f}%)")
    print(f"  Min  included:  {int(_n_in.min())} / {int(_n_tot.max())}  "
          f"(step {int(np.argmin(_n_in))})")
    print(f"  Steps w/ excluded agents: "
          f"{int((_n_in < _n_tot).sum())} / {len(_n_in)}")

if len(human_acc_list) > 5:
    _acc = np.asarray(human_acc_list, dtype=float)
    _vlat = np.asarray(human_vlat_list, dtype=float)
    _gln = np.asarray(g_long_num_list, dtype=float)
    _glp = np.asarray(g_long_pinn_list, dtype=float)
    _gyn = np.asarray(g_lat_num_list, dtype=float)
    _gyp = np.asarray(g_lat_pinn_list, dtype=float)

    rho_long_num = _safe_spearman(-_gln, _acc)
    rho_long_pinn = _safe_spearman(-_glp, _acc)
    rho_lat_num = _safe_spearman(-_gyn, _vlat)
    rho_lat_pinn = _safe_spearman(-_gyp, _vlat)

    with plt.style.context(["science", "no-latex"]):
        fig_b, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        plots = [
            (axs[0, 0], -_gln, _acc, f"Numerical long  rho={rho_long_num:.2f}",
             r"$-g_{\mathrm{long}}$", r"$a_{\mathrm{human}}$ [m/s$^2$]", "C3"),
            (axs[0, 1], -_glp, _acc, f"PINN long  rho={rho_long_pinn:.2f}",
             r"$-g_{\mathrm{long}}$", r"$a_{\mathrm{human}}$ [m/s$^2$]", "C5"),
            (axs[1, 0], -_gyn, _vlat, f"Numerical lat  rho={rho_lat_num:.2f}",
             r"$-g_{\mathrm{lat}}$", r"$v_{\mathrm{lat,human}}$ [m/s]", "C3"),
            (axs[1, 1], -_gyp, _vlat, f"PINN lat  rho={rho_lat_pinn:.2f}",
             r"$-g_{\mathrm{lat}}$", r"$v_{\mathrm{lat,human}}$ [m/s]", "C5"),
        ]
        for ax, xx, yy, title, xl, yl, color in plots:
            ax.scatter(xx, yy, s=12, alpha=0.55, c=color, edgecolors="none")
            ax.axhline(0.0, color="grey", lw=0.6, ls="--")
            ax.axvline(0.0, color="grey", lw=0.6, ls="--")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(True, lw=0.4, alpha=0.35)
        fig_b.savefig(os.path.join(save_dir, "behavior_alignment.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_b)
    print(f"Behavior alignment plot → {save_dir}/behavior_alignment.png")
    print("\n--- Behavior alignment summary ---")
    print(f"  Spearman(-g_long, a_human):  numerical={rho_long_num:.3f}  pinn={rho_long_pinn:.3f}")
    print(f"  Spearman(-g_lat,  v_lat):    numerical={rho_lat_num:.3f}  pinn={rho_lat_pinn:.3f}")

if len(selected_tid_weight_hist) > 1:
    _t = np.arange(len(selected_tid_weight_hist)) * dt
    tid_score = {}
    for wmap in selected_tid_weight_hist:
        for tid, wt in wmap.items():
            tid_score[tid] = tid_score.get(tid, 0.0) + float(wt)
    top_tids = [tid for tid, _ in sorted(tid_score.items(), key=lambda kv: kv[1], reverse=True)[:8]]
    if top_tids:
        heat = np.zeros((len(top_tids), len(selected_tid_weight_hist)), dtype=float)
        for j, wmap in enumerate(selected_tid_weight_hist):
            for i_tid, tid in enumerate(top_tids):
                heat[i_tid, j] = float(wmap.get(tid, 0.0))

        _mass = np.asarray(selection_mass_list, dtype=float) * 100.0
        with plt.style.context(["science", "no-latex"]):
            fig_s, (ax_s1, ax_s2) = plt.subplots(
                2, 1, figsize=(11, 6), constrained_layout=True,
                gridspec_kw={"height_ratios": [1, 2]}
            )
            ax_s1.plot(_t, _mass, color="C2", lw=1.3, label="Retained relevance mass")
            ax_s1.axhline(85.0, color="grey", lw=0.7, ls="--", label="85% target")
            ax_s1.set_ylabel("Mass [%]")
            ax_s1.set_title(f"Selection timeline  |  rec {rec}  ego={EGO_TRACK_ID}", fontsize=9)
            ax_s1.grid(True, lw=0.4, alpha=0.35)
            ax_s1.legend(fontsize=7)

            im = ax_s2.imshow(
                heat, aspect="auto", interpolation="nearest", origin="lower",
                extent=[_t[0], _t[-1], -0.5, len(top_tids) - 0.5], cmap="viridis"
            )
            ax_s2.set_yticks(range(len(top_tids)))
            ax_s2.set_yticklabels([str(tid) for tid in top_tids], fontsize=7)
            ax_s2.set_xlabel("t [s]")
            ax_s2.set_ylabel("Selected trackId")
            ax_s2.set_title("Selected-agent weights over time", fontsize=9)
            cbar = fig_s.colorbar(im, ax=ax_s2, fraction=0.02, pad=0.015)
            cbar.ax.set_ylabel("weight", rotation=90)
            fig_s.savefig(os.path.join(save_dir, "selection_timeline.png"),
                          dpi=150, bbox_inches="tight")
            plt.close(fig_s)
        print(f"Selection timeline plot → {save_dir}/selection_timeline.png")

if len(bg_mean_num_list) > 1:
    _t = np.arange(len(bg_mean_num_list)) * dt
    _bg_num = np.asarray(bg_mean_num_list, dtype=float)
    _bg_pinn = np.asarray(bg_mean_pinn_list, dtype=float)
    _lap_num = np.asarray(bg_lap_num_list, dtype=float)
    _lap_pinn = np.asarray(bg_lap_pinn_list, dtype=float)

    with plt.style.context(["science", "no-latex"]):
        fig_r, (ax_r1, ax_r2) = plt.subplots(2, 1, figsize=(10, 5.5), constrained_layout=True)
        ax_r1.plot(_t, _bg_num, color="C3", lw=1.3, label="Numerical")
        ax_r1.plot(_t, _bg_pinn, color="C5", lw=1.3, ls="--", label="PINN")
        ax_r1.set_ylabel("Background mean risk")
        ax_r1.set_title(f"Background redundancy  |  rec {rec}  ego={EGO_TRACK_ID}", fontsize=9)
        ax_r1.grid(True, lw=0.4, alpha=0.35)
        ax_r1.legend(fontsize=7)

        ax_r2.plot(_t, _lap_num, color="C3", lw=1.3, label="Numerical")
        ax_r2.plot(_t, _lap_pinn, color="C5", lw=1.3, ls="--", label="PINN")
        ax_r2.set_xlabel("t [s]")
        ax_r2.set_ylabel("Background Laplacian energy")
        ax_r2.grid(True, lw=0.4, alpha=0.35)
        fig_r.savefig(os.path.join(save_dir, "background_redundancy.png"),
                      dpi=150, bbox_inches="tight")
        plt.close(fig_r)
    print(f"Background redundancy plot → {save_dir}/background_redundancy.png")
    print("\n--- Background redundancy summary ---")
    print(f"  Mean background risk:        numerical={np.nanmean(_bg_num):.4f}  pinn={np.nanmean(_bg_pinn):.4f}")
    print(f"  Mean background lap energy:  numerical={np.nanmean(_lap_num):.4e}  pinn={np.nanmean(_lap_pinn):.4e}")
