"""
PINN Risk Adapter
=================
Runtime inference wrapper around the trained RiskFieldNet (pinn_risk_field.py).

Role in the RL architecture
----------------------------
                              traffic state
                                    │
                                    ▼
             pde_solver.compute_total_Q / compute_velocity_field /
             compute_diffusion_field
                                    │ Q, vx, vy, D  (grid arrays)
                                    ▼
                         PINNRiskAdapter.query_points()
                                    │ R̂, ∂R̂/∂x, ∂R̂/∂y  (at query pts)
                                    ▼
                       RL observation builder  →  policy  →  action
                                    │                            │
                                    └────── risk cost ───────────┘

Key design choices
------------------
* The PINN was trained on `config.py` grid coordinates (the IDEAM curved-road
  scenario: x ∈ [-150, 255] m, y ∈ [-225, -45] m).  The RL highway environment
  uses different coordinates.  We handle this by remapping the simulation
  coordinates into the PINN's training domain before building the input tensor.
  This is an approximation — for best accuracy, retrain the PINN on the target
  scenario.  See `inference_x_range` / `inference_y_range` constructor args.

* Gradients ∂R̂/∂x, ∂R̂/∂y are computed via PyTorch autograd.  This is only
  called for the ego position (2 extra backward passes) and is fast (~0.5 ms).

* Fallback: if no checkpoint path is given, or the file is absent, all queries
  return zero risk.  This lets the RL environment start before a model is trained.
"""

import os, sys
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from rl.risk.scene_conditioning import summarize_selected_agents
from rl.utils.timing import get_timer, CATEGORY_PINN_OUTPUT


# ---------------------------------------------------------------------------
# Lazy import of PINN classes — only needed when a checkpoint is loaded
# ---------------------------------------------------------------------------
def _import_pinn():
    from pinn_risk_field import RiskFieldNet
    return RiskFieldNet


# ---------------------------------------------------------------------------
# Default inference coordinate range for the highway RL environment
# (matches config_highway.py: x in [-10, 1000], y in [-3, 14])
# ---------------------------------------------------------------------------
_HIGHWAY_X_RANGE = (-10.0, 1000.0)
_HIGHWAY_Y_RANGE = (-3.0,  14.0)


class PINNRiskAdapter:
    """
    Thin wrapper for trained PINN inference at runtime.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to a .pt file saved by PINNTrainer.save().
        If None or the file does not exist, returns zero risk (fallback mode).
    device : str
        Torch device string, e.g. 'cpu' or 'cuda'.
    inference_x_range : (float, float)
        The x-coordinate bounds of the runtime simulation domain.
        Used to remap simulation x → normalised [-1, 1] instead of
        the training-domain bounds stored in the checkpoint.
    inference_y_range : (float, float)
        Same for y.
    t_clip : float or None
        Clip simulation time to this maximum before normalising.
        Prevents out-of-range queries when RL episodes are long.
        None → use checkpoint's t_max.
    """

    def __init__(self,
                 checkpoint_path: str = None,
                 device: str = 'cpu',
                 inference_x_range: tuple = _HIGHWAY_X_RANGE,
                 inference_y_range: tuple = _HIGHWAY_Y_RANGE,
                 t_clip: float = None,
                 time_mode: str = 'clip'):

        self.device = device
        self.inference_x_range = inference_x_range
        self.inference_y_range = inference_y_range
        self._t_clip = t_clip
        if time_mode not in {'clip', 'error'}:
            raise ValueError("time_mode must be 'clip' or 'error'")
        self.time_mode = str(time_mode)

        self._model = None
        self._norm_ranges = None
        self._available = False
        self._use_context = False
        self._use_spatial_context = False
        self._use_distance_rbf = False
        self._perception_range = _HIGHWAY_X_RANGE[1]
        self._checkpoint_metadata = {}
        self._training_time_range = None
        self._coordinate_mode = 'runtime_scaled'

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load(checkpoint_path)
        elif checkpoint_path:
            print(f"[PINNAdapter] WARNING: checkpoint not found: {checkpoint_path}")
            print("  → falling back to zero-risk mode")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if a PINN model is loaded and ready."""
        return self._available

    def warmup(self, *, compute_gradient: bool = True) -> None:
        """Materialize model kernels outside the measured control loop."""
        if not self._available:
            return

        def _midpoint(key: str, fallback: float = 0.0) -> float:
            value = self._norm_ranges.get(key)
            if value is None:
                return float(fallback)
            return 0.5 * (float(value[0]) + float(value[1]))

        x = _midpoint("x")
        y = _midpoint("y")
        kwargs = {}
        if self._use_context:
            kwargs = {
                "N_agents": np.asarray([_midpoint("N_agents")], dtype=np.float32),
                "dist_nearest": np.asarray(
                    [_midpoint("dist_nearest", self._perception_range)], dtype=np.float32
                ),
            }
            if self._use_spatial_context:
                kwargs.update(
                    dist_dx=np.asarray([0.0], dtype=np.float32),
                    dist_dy=np.asarray([0.0], dtype=np.float32),
                )
        self.query_arrays(
            xs=np.asarray([x], dtype=np.float32),
            ys=np.asarray([y], dtype=np.float32),
            t=float(self._training_time_range[0] if self._training_time_range else 0.0),
            Q=np.asarray([_midpoint("Q")], dtype=np.float32),
            vx=np.asarray([_midpoint("vx")], dtype=np.float32),
            vy=np.asarray([_midpoint("vy")], dtype=np.float32),
            D=np.asarray([_midpoint("D")], dtype=np.float32),
            compute_gradient=compute_gradient,
            **kwargs,
        )

    def query_points(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        t: float,
        Q_grid: np.ndarray,
        vx_grid: np.ndarray,
        vy_grid: np.ndarray,
        D_grid: np.ndarray,
        sim_cfg,
        N_agents: int = 0,
        dist_nearest: np.ndarray | float | None = None,
        vehicles: list | None = None,
        ego_vehicle: dict | None = None,
        selection_mode: str = 'soft_topk',
        top_k: int = 5,
        threshold_ratio: float = 0.15,
        compute_gradient: bool = False,
    ) -> dict:
        """
        Query PINN risk at arbitrary (x, y) points in simulation space.

        Steps:
          1. Interpolate Q, vx, vy, D from simulation grids at query points.
          2. Normalise all inputs using checkpoint ranges
             (with x, y overridden to inference_x/y_range).
          3. Forward pass → R̂ values.
          4. Optionally compute ∂R̂/∂x, ∂R̂/∂y via autograd at query points.

        Parameters
        ----------
        xs, ys   : 1-D float arrays of query x, y positions [m]
        t        : current simulation time [s]
        Q_grid   : 2-D array (ny, nx) — source term
        vx_grid  : 2-D array (ny, nx) — x-velocity field
        vy_grid  : 2-D array (ny, nx) — y-velocity field
        D_grid   : 2-D array (ny, nx) — diffusion field
        sim_cfg  : Config object with grid arrays (x, y) and bounds
        N_agents : number of surrounding vehicles (context feature, optional)
        compute_gradient : if True, also return dR/dx, dR/dy at each query point

        Returns
        -------
        dict with:
            'R'     : (N,) float32 risk values
            'grad_x': (N,) float32 dR/dx  (zeros if compute_gradient=False)
            'grad_y': (N,) float32 dR/dy
        """
        xs = np.asarray(xs, dtype=np.float32).ravel()
        ys = np.asarray(ys, dtype=np.float32).ravel()
        N = len(xs)

        if not self._available:
            return {
                'R':      np.zeros(N, dtype=np.float32),
                'grad_x': np.zeros(N, dtype=np.float32),
                'grad_y': np.zeros(N, dtype=np.float32),
            }

        # Step 1: interpolate fields at query points
        from scipy.interpolate import RegularGridInterpolator
        x_g = sim_cfg.x    # 1-D (nx,)
        y_g = sim_cfg.y    # 1-D (ny,)

        pts = np.column_stack([ys, xs])   # (N, 2)  — (y, x) convention

        def _interp(grid):
            fi = RegularGridInterpolator(
                (y_g, x_g), grid,
                method='linear', bounds_error=False, fill_value=0.0)
            return fi(pts).astype(np.float32)

        Q_pts  = _interp(Q_grid)
        vx_pts = _interp(vx_grid)
        vy_pts = _interp(vy_grid)
        D_pts  = _interp(D_grid)

        # Step 2: resolve optional context features for 9-D checkpoints
        N_agents_pts, dist_nearest_pts = self._resolve_context(
            xs=xs,
            ys=ys,
            sim_cfg=sim_cfg,
            N_agents=N_agents,
            dist_nearest=dist_nearest,
            vehicles=vehicles,
            ego_vehicle=ego_vehicle,
            selection_mode=selection_mode,
            top_k=top_k,
            threshold_ratio=threshold_ratio,
        )
        dist_dx_pts = dist_dy_pts = None
        if self._use_spatial_context:
            if vehicles is not None and ego_vehicle is not None:
                ctx = summarize_selected_agents(
                    ego=ego_vehicle,
                    vehicles=vehicles,
                    X=None,
                    Y=None,
                    perception_range=float(self._perception_range),
                    selection_mode=selection_mode,
                    top_k=top_k,
                    threshold_ratio=threshold_ratio,
                )
                dist_dx_pts, dist_dy_pts = self._dist_direction_points(
                    xs, ys, ctx['selected_agents']
                )
            elif dist_nearest is not None and np.asarray(dist_nearest).ndim == 2:
                dn_grid = np.asarray(dist_nearest, dtype=np.float32)
                dn_dy_grid, dn_dx_grid = np.gradient(
                    dn_grid, float(sim_cfg.y[1] - sim_cfg.y[0]),
                    float(sim_cfg.x[1] - sim_cfg.x[0]),
                )
                dist_dx_pts = _interp(dn_dx_grid)
                dist_dy_pts = _interp(dn_dy_grid)
            else:
                dist_dx_pts = np.zeros(N, dtype=np.float32)
                dist_dy_pts = np.zeros(N, dtype=np.float32)

        model_xs, model_ys, heading = self._model_coordinates(
            xs, ys, ego_vehicle=ego_vehicle
        )
        if self._coordinate_mode == 'ego_local':
            ego_vx = float((ego_vehicle or {}).get('vx', 0.0))
            ego_vy = float((ego_vehicle or {}).get('vy', 0.0))
            vx_pts, vy_pts = self._rotate_world_to_local(
                vx_pts - ego_vx, vy_pts - ego_vy, heading
            )
            if dist_dx_pts is not None and dist_dy_pts is not None:
                dist_dx_pts, dist_dy_pts = self._rotate_world_to_local(
                    dist_dx_pts, dist_dy_pts, heading
                )

        result = self.query_arrays(
            xs=model_xs,
            ys=model_ys,
            t=t,
            Q=Q_pts,
            vx=vx_pts,
            vy=vy_pts,
            D=D_pts,
            N_agents=N_agents_pts,
            dist_nearest=dist_nearest_pts,
            dist_dx=dist_dx_pts,
            dist_dy=dist_dy_pts,
            compute_gradient=compute_gradient,
        )
        if self._coordinate_mode == 'ego_local':
            result['R'] = np.where(
                self._training_domain_mask(model_xs, model_ys), result['R'], 0.0
            ).astype(np.float32)
            if compute_gradient:
                gx, gy = self._rotate_local_to_world(
                    result['grad_x'], result['grad_y'], heading
                )
                inside = self._training_domain_mask(model_xs, model_ys)
                result['grad_x'] = np.where(inside, gx, 0.0).astype(np.float32)
                result['grad_y'] = np.where(inside, gy, 0.0).astype(np.float32)
        return result

    def query_arrays(
        self,
        *,
        xs: np.ndarray,
        ys: np.ndarray,
        t: float,
        Q: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        D: np.ndarray,
        N_agents: np.ndarray | float | None = None,
        dist_nearest: np.ndarray | float | None = None,
        dist_dx: np.ndarray | float | None = None,
        dist_dy: np.ndarray | float | None = None,
        compute_gradient: bool = False,
    ) -> dict:
        """Query values when coefficient fields are already sampled.

        This avoids constructing four ``RegularGridInterpolator`` objects for
        policy descriptors and avoids a redundant interpolation pass for
        full-grid visualization.  All arrays must have one value per query
        point.
        """
        xs = np.asarray(xs, dtype=np.float32).reshape(-1)
        ys = np.asarray(ys, dtype=np.float32).reshape(-1)
        if xs.shape != ys.shape:
            raise ValueError("xs and ys must have identical shapes")
        n = xs.size
        if not self._available:
            return {
                'R': np.zeros(n, dtype=np.float32),
                'grad_x': np.zeros(n, dtype=np.float32),
                'grad_y': np.zeros(n, dtype=np.float32),
            }

        def _points(values, name: str) -> np.ndarray:
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 0:
                return np.full(n, float(arr), dtype=np.float32)
            arr = arr.reshape(-1)
            if arr.size != n:
                raise ValueError(f"{name} has {arr.size} values; expected {n}")
            return arr

        Q_pts = _points(Q, 'Q')
        vx_pts = _points(vx, 'vx')
        vy_pts = _points(vy, 'vy')
        D_pts = _points(D, 'D')
        N_agents_pts = None if N_agents is None else _points(N_agents, 'N_agents')
        dist_nearest_pts = (
            None if dist_nearest is None else _points(dist_nearest, 'dist_nearest')
        )
        dist_dx_pts = None if dist_dx is None else _points(dist_dx, 'dist_dx')
        dist_dy_pts = None if dist_dy is None else _points(dist_dy, 'dist_dy')

        t_val = self._resolve_time(float(t))
        inp_np = self._build_input_np(
            xs, ys, t_val, Q_pts, vx_pts, vy_pts, D_pts,
            N_agents_pts=N_agents_pts,
            dist_nearest_pts=dist_nearest_pts,
            dist_dx_pts=dist_dx_pts,
            dist_dy_pts=dist_dy_pts,
        )

        sync_cuda = (str(self.device) != 'cpu' and _TORCH_OK and torch.cuda.is_available())
        if compute_gradient:
            with get_timer().measure(CATEGORY_PINN_OUTPUT, sync_cuda=sync_cuda):
                return self._query_with_grad(inp_np, xs, ys)
        with get_timer().measure(CATEGORY_PINN_OUTPUT, sync_cuda=sync_cuda):
            inp_t = torch.tensor(inp_np, dtype=torch.float32, device=self.device)
            self._model.eval()
            with torch.no_grad():
                R_raw = self._model(inp_t).squeeze(-1).cpu().numpy()
            R_vals = R_raw * self._R_scale
            return {
                'R': R_vals.astype(np.float32),
                'grad_x': np.zeros(n, dtype=np.float32),
                'grad_y': np.zeros(n, dtype=np.float32),
            }

    def query_grid(
        self,
        *,
        X: np.ndarray,
        Y: np.ndarray,
        t: float,
        Q: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        D: np.ndarray,
        vehicles: list | None = None,
        ego_vehicle: dict | None = None,
        selection_mode: str = 'soft_topk',
        top_k: int = 5,
        threshold_ratio: float = 0.15,
    ) -> np.ndarray:
        """Predict a complete field without re-interpolating grid inputs."""
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if X.shape != Y.shape:
            raise ValueError("X and Y must have identical shapes")

        n_agents = None
        dist_nearest = None
        dist_dx = dist_dy = None
        if self._use_context:
            if vehicles is not None and ego_vehicle is not None:
                ctx = summarize_selected_agents(
                    ego=ego_vehicle,
                    vehicles=vehicles,
                    X=X,
                    Y=Y,
                    perception_range=float(self._perception_range),
                    selection_mode=selection_mode,
                    top_k=top_k,
                    threshold_ratio=threshold_ratio,
                )
                n_agents = np.full(X.size, float(ctx['N_agents_selected']), dtype=np.float32)
                dist_nearest = np.asarray(
                    ctx['dist_nearest_selected'], dtype=np.float32
                ).reshape(-1)
            else:
                n_agents = np.zeros(X.size, dtype=np.float32)
                dist_nearest = np.full(
                    X.size, float(self._perception_range), dtype=np.float32
                )
            if self._use_spatial_context:
                dn_grid = dist_nearest.reshape(X.shape)
                grid_dy = float(np.mean(np.diff(Y[:, 0]))) if Y.shape[0] > 1 else 1.0
                grid_dx = float(np.mean(np.diff(X[0, :]))) if X.shape[1] > 1 else 1.0
                dist_dy_grid, dist_dx_grid = np.gradient(dn_grid, grid_dy, grid_dx)
                dist_dx = dist_dx_grid.reshape(-1).astype(np.float32)
                dist_dy = dist_dy_grid.reshape(-1).astype(np.float32)

        xs_world = X.reshape(-1)
        ys_world = Y.reshape(-1)
        model_xs, model_ys, heading = self._model_coordinates(
            xs_world, ys_world, ego_vehicle=ego_vehicle
        )
        vx_values = np.asarray(vx, dtype=np.float32).reshape(-1)
        vy_values = np.asarray(vy, dtype=np.float32).reshape(-1)
        if self._coordinate_mode == 'ego_local':
            ego_vx = float((ego_vehicle or {}).get('vx', 0.0))
            ego_vy = float((ego_vehicle or {}).get('vy', 0.0))
            vx_values, vy_values = self._rotate_world_to_local(
                vx_values - ego_vx, vy_values - ego_vy, heading
            )
            if dist_dx is not None and dist_dy is not None:
                dist_dx, dist_dy = self._rotate_world_to_local(
                    dist_dx, dist_dy, heading
                )

        result = self.query_arrays(
            xs=model_xs,
            ys=model_ys,
            t=t,
            Q=np.asarray(Q, dtype=np.float32).reshape(-1),
            vx=vx_values,
            vy=vy_values,
            D=np.asarray(D, dtype=np.float32).reshape(-1),
            N_agents=n_agents,
            dist_nearest=dist_nearest,
            dist_dx=dist_dx,
            dist_dy=dist_dy,
            compute_gradient=False,
        )
        if self._coordinate_mode == 'ego_local':
            result['R'] = np.where(
                self._training_domain_mask(model_xs, model_ys), result['R'], 0.0
            ).astype(np.float32)
        return np.asarray(result['R'], dtype=np.float32).reshape(X.shape)

    def query_ego(
        self,
        ego_x: float,
        ego_y: float,
        t: float,
        Q_grid: np.ndarray,
        vx_grid: np.ndarray,
        vy_grid: np.ndarray,
        D_grid: np.ndarray,
        sim_cfg,
    ) -> tuple:
        """
        Convenience method: query risk + gradient at a single ego position.

        Returns
        -------
        (R_ego, grad_x, grad_y) : floats
        """
        result = self.query_points(
            [ego_x], [ego_y], t,
            Q_grid, vx_grid, vy_grid, D_grid, sim_cfg,
            compute_gradient=True,
        )
        return (float(result['R'][0]),
                float(result['grad_x'][0]),
                float(result['grad_y'][0]))

    def query_risk_features(
        self,
        ego_x: float,
        ego_y: float,
        t: float,
        Q_grid: np.ndarray,
        vx_grid: np.ndarray,
        vy_grid: np.ndarray,
        D_grid: np.ndarray,
        sim_cfg,
        lane_centers: list,
        current_lane: int,
        lookahead_m: float = 20.0,
        N_agents: int = 0,
        dist_nearest: np.ndarray | float | None = None,
        vehicles: list | None = None,
        ego_vehicle: dict | None = None,
        selection_mode: str = 'soft_topk',
        top_k: int = 5,
        threshold_ratio: float = 0.15,
    ) -> dict:
        """
        Return the 8 PINN risk features used in the RL observation.

        Features
        --------
        r_ego      : R̂ at ego position
        r_5m       : R̂ 5 m ahead, same lane
        r_10m      : R̂ 10 m ahead, same lane
        r_20m      : R̂ 20 m ahead, same lane
        grad_x     : ∂R̂/∂x at ego  (forward risk gradient)
        grad_y     : ∂R̂/∂y at ego  (lateral risk gradient)
        r_left     : R̂ at (ego_x + 10, left_lane_y)    — 0 if no left lane
        r_right    : R̂ at (ego_x + 10, right_lane_y)   — 0 if no right lane
        """
        curr_y = lane_centers[current_lane]
        left_y  = lane_centers[current_lane - 1] if current_lane > 0          else None
        right_y = lane_centers[current_lane + 1] if current_lane < len(lane_centers) - 1 else None

        # Build all query points in one batch pass (more efficient than separate calls)
        xs = [ego_x, ego_x + 5.0, ego_x + 10.0, ego_x + 20.0]
        ys = [curr_y, curr_y,      curr_y,        curr_y       ]
        if left_y  is not None: xs.append(ego_x + 10.0); ys.append(left_y)
        if right_y is not None: xs.append(ego_x + 10.0); ys.append(right_y)

        result = self.query_points(
            xs, ys, t,
            Q_grid, vx_grid, vy_grid, D_grid, sim_cfg,
            N_agents=N_agents,
            dist_nearest=dist_nearest,
            vehicles=vehicles,
            ego_vehicle=ego_vehicle,
            selection_mode=selection_mode,
            top_k=top_k,
            threshold_ratio=threshold_ratio,
            compute_gradient=True,
        )
        R = result['R']
        idx = 4   # first adjacency index (if any)

        r_left  = float(R[idx])   if left_y  is not None else 0.0
        r_right = float(R[idx + (1 if left_y is not None else 0)]) \
                  if right_y is not None else 0.0

        return {
            'r_ego'   : float(R[0]),
            'r_5m'    : float(R[1]),
            'r_10m'   : float(R[2]),
            'r_20m'   : float(R[3]),
            'grad_x'  : float(result['grad_x'][0]),
            'grad_y'  : float(result['grad_y'][0]),
            'r_left'  : r_left,
            'r_right' : r_right,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, path: str):
        if not _TORCH_OK:
            print("[PINNAdapter] torch not available — fallback mode")
            return
        try:
            RiskFieldNet = _import_pinn()
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            nr   = ckpt['norm_ranges']
            self._norm_ranges = dict(nr)
            self._training_time_range = tuple(float(v) for v in nr['t'])
            self._checkpoint_metadata = dict(ckpt.get('metadata', {}))
            self._coordinate_mode = str(
                self._checkpoint_metadata.get('coordinate_mode', 'runtime_scaled')
            )
            if self._coordinate_mode not in {'runtime_scaled', 'native', 'ego_local'}:
                raise ValueError(
                    f"Unsupported PINN coordinate mode: {self._coordinate_mode!r}"
                )

            # Legacy checkpoints linearly remap the runtime grid into the
            # training bounds. Native and ego-local checkpoints retain the
            # coordinates on which they were calibrated.
            if self._coordinate_mode == 'runtime_scaled':
                self._norm_ranges['x'] = self.inference_x_range
                self._norm_ranges['y'] = self.inference_y_range

            # Set t_clip
            self._t_clip = self._t_clip or float(nr['t'][1])

            # Risk output scale (denormalise network output → raw R value)
            self._R_scale = float(nr['R'][1])

            # Infer architecture from state dict when not stored explicitly
            sd = ckpt['model_state']
            n_layers = sum(1 for k in sd if k.startswith('layers.') and k.endswith('.weight'))
            in_0  = sd['layers.0.weight'].shape[1]   # input dim of first layer
            hidden = sd['layers.0.weight'].shape[0]  # units in first hidden layer
            depth  = n_layers

            use_rff     = ckpt.get('use_rff', False) or False
            rff_features = ckpt.get('rff_features', 64)
            rff_scale    = ckpt.get('rff_scale', 10.0)
            use_context  = ckpt.get('use_context', False) or False
            use_spatial_context = bool(ckpt.get('use_spatial_context', False))
            use_distance_rbf = bool(ckpt.get('use_distance_rbf', False))
            rff_include_raw = bool(ckpt.get('rff_include_raw', False))
            output_bias_init = float(ckpt.get('output_bias_init', 0.0))

            self._model = RiskFieldNet(
                hidden=hidden, depth=depth,
                use_rff=use_rff, rff_features=rff_features, rff_scale=rff_scale,
                use_context=use_context,
                use_spatial_context=use_spatial_context,
                use_distance_rbf=use_distance_rbf,
                rff_include_raw=rff_include_raw,
                output_bias_init=output_bias_init,
            ).to(self.device)
            self._model.load_state_dict(sd)
            self._model.eval()

            self._use_context = use_context
            self._use_spatial_context = use_spatial_context
            self._use_distance_rbf = use_distance_rbf
            self._perception_range = float(ckpt.get('perception_range', _HIGHWAY_X_RANGE[1]))
            self._available = True
            print(f"[PINNAdapter] Loaded {os.path.basename(path)}  "
                  f"(h={hidden} d={depth} rff={use_rff} ctx={use_context})")
            print(f"  inference domain: x{self.inference_x_range} y{self.inference_y_range} "
                  f"t_clip={self._t_clip:.0f}s  R_scale={self._R_scale:.1f} "
                  f"coordinates={self._coordinate_mode}")
        except Exception as e:
            print(f"[PINNAdapter] Load failed: {e} → fallback mode")
            self._available = False

    @property
    def checkpoint_metadata(self) -> dict:
        return dict(self._checkpoint_metadata)

    @property
    def training_time_range(self) -> tuple[float, float] | None:
        return self._training_time_range

    @property
    def coordinate_mode(self) -> str:
        return str(self._coordinate_mode)

    def runtime_domain_mask(self, X, Y, *, ego_vehicle=None) -> np.ndarray:
        """Return cells covered by the checkpoint's calibrated spatial domain."""
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        model_x, model_y, _heading = self._model_coordinates(
            X.reshape(-1), Y.reshape(-1), ego_vehicle=ego_vehicle
        )
        return self._training_domain_mask(model_x, model_y).reshape(X.shape)

    def _model_coordinates(self, xs, ys, *, ego_vehicle=None):
        """Map runtime positions into the coordinate frame used in training."""
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        if self._coordinate_mode != 'ego_local':
            return xs, ys, 0.0
        if ego_vehicle is None:
            # Validation caches are already expressed in the ego-local frame.
            return xs, ys, 0.0
        heading = float(ego_vehicle.get('heading', 0.0))
        c = float(np.cos(heading))
        s = float(np.sin(heading))
        dx = xs - float(ego_vehicle['x'])
        dy = ys - float(ego_vehicle['y'])
        return (
            (c * dx + s * dy).astype(np.float32),
            (-s * dx + c * dy).astype(np.float32),
            heading,
        )

    @staticmethod
    def _rotate_world_to_local(vx, vy, heading: float):
        c = float(np.cos(heading))
        s = float(np.sin(heading))
        vx = np.asarray(vx, dtype=np.float32)
        vy = np.asarray(vy, dtype=np.float32)
        return (
            (c * vx + s * vy).astype(np.float32),
            (-s * vx + c * vy).astype(np.float32),
        )

    @staticmethod
    def _rotate_local_to_world(vx, vy, heading: float):
        c = float(np.cos(heading))
        s = float(np.sin(heading))
        vx = np.asarray(vx, dtype=np.float32)
        vy = np.asarray(vy, dtype=np.float32)
        return (
            (c * vx - s * vy).astype(np.float32),
            (s * vx + c * vy).astype(np.float32),
        )

    def _training_domain_mask(self, xs, ys) -> np.ndarray:
        x_lo, x_hi = self._norm_ranges['x']
        y_lo, y_hi = self._norm_ranges['y']
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        return (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)

    def _resolve_time(self, t: float) -> float:
        if self._t_clip is None:
            return float(t)
        if t <= float(self._t_clip) + 1e-9:
            return float(t)
        if self.time_mode == 'error':
            raise ValueError(
                f"PINN query time {t:.3f}s exceeds the trained deployment "
                f"horizon {float(self._t_clip):.3f}s"
            )
        return float(self._t_clip)

    def _norm1(self, val, key: str) -> np.ndarray:
        """Normalise values to [-1, 1] using stored range for key."""
        lo, hi = self._norm_ranges[key]
        return np.asarray(2.0 * (val - lo) / max(hi - lo, 1e-8) - 1.0, dtype=np.float32)

    def _build_input_np(self, xs, ys, t_val, Q_pts, vx_pts, vy_pts, D_pts,
                        N_agents_pts=None, dist_nearest_pts=None,
                        dist_dx_pts=None, dist_dy_pts=None) -> np.ndarray:
        """Build normalised numpy input array for the PINN."""
        N = len(xs)
        t_arr = np.full(N, t_val, dtype=np.float32)
        cols = [
            self._norm1(xs, 'x'),
            self._norm1(ys, 'y'),
            self._norm1(t_arr, 't'),
            self._norm1(Q_pts, 'Q'),
        ]
        if self._use_context:
            if N_agents_pts is None:
                N_agents_pts = np.zeros(N, dtype=np.float32)
            if dist_nearest_pts is None:
                dist_nearest_pts = np.full(N, _HIGHWAY_X_RANGE[1], dtype=np.float32)
            cols.append(self._norm1(N_agents_pts, 'N_agents'))
            cols.append(self._norm1(dist_nearest_pts, 'dist_nearest'))
            if self._use_distance_rbf:
                for scale in (2.0, 6.0, 15.0):
                    cols.append(
                        np.asarray(
                            2.0 * np.exp(-dist_nearest_pts / scale) - 1.0,
                            dtype=np.float32,
                        )
                    )
            if self._use_spatial_context:
                if dist_dx_pts is None:
                    dist_dx_pts = np.zeros(N, dtype=np.float32)
                if dist_dy_pts is None:
                    dist_dy_pts = np.zeros(N, dtype=np.float32)
                cols.append(self._norm1(dist_dx_pts, 'dist_dx'))
                cols.append(self._norm1(dist_dy_pts, 'dist_dy'))
        cols.extend([
            self._norm1(vx_pts, 'vx'),
            self._norm1(vy_pts, 'vy'),
            self._norm1(D_pts, 'D'),
        ])
        return np.column_stack(cols)   # (N, 7)

    def _dist_nearest_points(self, xs, ys, agents, fill_value: float) -> np.ndarray:
        if not agents:
            return np.full(len(xs), fill_value, dtype=np.float32)
        axy = np.array([[float(v['x']), float(v['y'])] for v in agents], dtype=np.float32)
        dx = xs[:, None] - axy[:, 0]
        dy = ys[:, None] - axy[:, 1]
        return np.sqrt(dx**2 + dy**2).min(axis=1).astype(np.float32)

    def _dist_direction_points(self, xs, ys, agents) -> tuple[np.ndarray, np.ndarray]:
        """Gradient of distance-to-nearest-agent at arbitrary query points."""
        if not agents:
            zeros = np.zeros(len(xs), dtype=np.float32)
            return zeros, zeros.copy()
        axy = np.array([[float(v['x']), float(v['y'])] for v in agents], dtype=np.float32)
        dx = xs[:, None] - axy[:, 0]
        dy = ys[:, None] - axy[:, 1]
        distances = np.sqrt(dx**2 + dy**2)
        nearest = np.argmin(distances, axis=1)
        rows = np.arange(len(xs))
        denom = np.maximum(distances[rows, nearest], 1e-6)
        return (
            (dx[rows, nearest] / denom).astype(np.float32),
            (dy[rows, nearest] / denom).astype(np.float32),
        )

    def _resolve_context(self, xs, ys, sim_cfg, N_agents=0, dist_nearest=None,
                         vehicles=None, ego_vehicle=None, selection_mode='soft_topk',
                         top_k=5, threshold_ratio=0.15):
        if not self._use_context:
            return None, None

        N = len(xs)
        fill_value = float(getattr(self, '_perception_range', _HIGHWAY_X_RANGE[1]))

        if vehicles is not None and ego_vehicle is not None:
            ctx = summarize_selected_agents(
                ego=ego_vehicle,
                vehicles=vehicles,
                X=None,
                Y=None,
                perception_range=fill_value,
                selection_mode=selection_mode,
                top_k=top_k,
                threshold_ratio=threshold_ratio,
            )
            selected = ctx['selected_agents']
            N_agents_pts = np.full(N, float(ctx['N_agents_selected']), dtype=np.float32)
            dist_nearest_pts = self._dist_nearest_points(xs, ys, selected, fill_value=fill_value)
            return N_agents_pts, dist_nearest_pts

        N_agents_pts = np.full(N, float(N_agents), dtype=np.float32)
        if dist_nearest is None:
            dist_nearest_pts = np.full(N, fill_value, dtype=np.float32)
        else:
            dn_arr = np.asarray(dist_nearest, dtype=np.float32)
            if dn_arr.ndim == 0:
                dist_nearest_pts = np.full(N, float(dn_arr), dtype=np.float32)
            elif dn_arr.shape[0] == N:
                dist_nearest_pts = dn_arr.reshape(-1).astype(np.float32)
            else:
                from scipy.interpolate import RegularGridInterpolator
                pts = np.column_stack([ys, xs])
                fi = RegularGridInterpolator(
                    (sim_cfg.y, sim_cfg.x), dn_arr,
                    method='linear', bounds_error=False, fill_value=fill_value
                )
                dist_nearest_pts = fi(pts).astype(np.float32)
        return N_agents_pts, dist_nearest_pts

    def _query_with_grad(self, inp_np: np.ndarray, xs, ys) -> dict:
        """
        Forward pass with gradient computation for ∂R/∂x and ∂R/∂y.

        The gradient is computed with respect to the *raw* (unnormalised)
        coordinates using the chain rule:
            ∂R/∂x_raw = ∂R/∂x̂ · (2 / (x_max - x_min))
        """
        N = inp_np.shape[0]
        self._model.eval()

        # Create raw-coordinate tensors with grad tracking
        x_raw = torch.tensor(xs, dtype=torch.float32, requires_grad=True,
                             device=self.device)
        y_raw = torch.tensor(ys, dtype=torch.float32, requires_grad=True,
                             device=self.device)

        # Compute normalised x, y as differentiable ops
        x_lo, x_hi = self._norm_ranges['x']
        y_lo, y_hi = self._norm_ranges['y']
        xn = 2.0 * (x_raw - x_lo) / max(x_hi - x_lo, 1e-8) - 1.0
        yn = 2.0 * (y_raw - y_lo) / max(y_hi - y_lo, 1e-8) - 1.0

        # Non-differentiable columns (already normalised numpy arrays)
        rest = torch.tensor(inp_np[:, 2:], dtype=torch.float32, device=self.device)
        # inp_np columns: [x̂, ŷ, t̂, Q̂, v̂x, v̂y, D̂] → rest = cols 2..6

        inp_t = torch.stack([xn, yn], dim=-1)
        inp_t = torch.cat([inp_t, rest], dim=-1)   # (N, 7)

        R_raw = self._model(inp_t).squeeze(-1)      # (N,)

        # Sum for scalar backward
        ones = torch.ones_like(R_raw)
        grad_x_norm, = torch.autograd.grad(
            R_raw, x_raw, grad_outputs=ones,
            create_graph=False, retain_graph=True)
        grad_y_norm, = torch.autograd.grad(
            R_raw, y_raw, grad_outputs=ones,
            create_graph=False, retain_graph=False)

        R_vals = R_raw.detach().cpu().numpy() * self._R_scale
        # Convert normalised-space gradient back to raw-space gradient
        # ∂R_raw/∂x = (R_scale) · ∂R̂/∂x_raw   (chain through denorm)
        gx = grad_x_norm.detach().cpu().numpy() * self._R_scale
        gy = grad_y_norm.detach().cpu().numpy() * self._R_scale

        return {
            'R':      R_vals.astype(np.float32),
            'grad_x': gx.astype(np.float32),
            'grad_y': gy.astype(np.float32),
        }


# ---------------------------------------------------------------------------
# Convenience: load the best available checkpoint from the repo root
# ---------------------------------------------------------------------------

_PREFERRED_CHECKPOINTS = [
    'pinn_highway.pt',     # trained on synthetic config_highway.py domain  ← prefer
    'pinn_multi_all.pt',   # trained on real exiD datasets (domain mismatch for RL highway)
    'pinn_inD_00.pt',
    'pinn_risk_field.pt',
]

def load_best_available(repo_root: str = None,
                        device: str = 'cpu',
                        inference_x_range: tuple = _HIGHWAY_X_RANGE,
                        inference_y_range: tuple = _HIGHWAY_Y_RANGE) -> PINNRiskAdapter:
    """
    Load the best available PINN checkpoint from the repo.

    Tries checkpoints in order of preference; falls back to zero-risk mode
    if none are found.
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    for name in _PREFERRED_CHECKPOINTS:
        path = os.path.join(repo_root, name)
        if os.path.exists(path):
            return PINNRiskAdapter(
                checkpoint_path=path,
                device=device,
                inference_x_range=inference_x_range,
                inference_y_range=inference_y_range,
            )

    print("[PINNAdapter] No checkpoint found → zero-risk fallback")
    return PINNRiskAdapter()
