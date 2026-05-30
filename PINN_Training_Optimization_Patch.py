# PATCH: Enhanced Training Strategy for Smoother Risk Fields
# ============================================================
# Apply these fixes to your existing pinn_risk_field.py

import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# FIX 1: RANDOM FOURIER FEATURES for Spectral Representation
# ==============================================================================
# Replaces plain normalization with learned sinusoidal encoding
# This addresses "spectral bias" — helps network learn spatial variations

class RandomFourierFeatures(nn.Module):
    """
    Map [x, y, t, ...] → Fourier features to help network learn high-freq patterns
    
    Input:  (batch, input_dim)
    Output: (batch, 2 * n_features) via [sin(B @ x), cos(B @ x)]
    where B is a fixed random matrix (samples from standard Gaussian)
    """
    
    def __init__(self, input_dim, n_features=64, scale=1.0):
        super().__init__()
        # Fixed random projection matrix (not learned)
        self.register_buffer(
            'B', 
            torch.randn(input_dim, n_features) * scale
        )
        self.n_features = n_features
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
        Returns:
            features: (batch, 2 * n_features)
        """
        proj = x @ self.B  # (batch, n_features)
        return torch.cat([torch.sin(2 * np.pi * proj), 
                         torch.cos(2 * np.pi * proj)], dim=-1)

class EnhancedNeuralRiskField(nn.Module):
    """
    PINN network with:
    - Random Fourier Features for spectral representation
    - Tanh activations (smooth, good for PDE learning)
    - Skip connections at midpoint
    - Normalization-aware input handling
    """
    
    def __init__(self, input_dim=7, hidden=128, depth=6, use_rff=True, rff_scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.use_rff = use_rff
        
        if use_rff:
            self.rff = RandomFourierFeatures(input_dim, n_features=64, scale=rff_scale)
            rff_dim = 2 * 64  # [sin, cos] for each frequency
            net_input_dim = rff_dim
        else:
            net_input_dim = input_dim
        
        # Build network
        layers = []
        prev_dim = net_input_dim
        
        for i in range(depth):
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.Tanh())
            
            # Skip connection at midpoint: concatenate RFF output
            if use_rff and i == depth // 2:
                # Next layer input will be hidden + rff_dim
                prev_dim = hidden + rff_dim
            else:
                prev_dim = hidden
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Enforce R ≥ 0
        
        self.network = nn.Sequential(*layers)
        self.rff_dim = rff_dim if use_rff else 0
    
    def forward(self, x, y, t, Q=None, vx=None, vy=None, D=None):
        """
        Evaluate risk field at (x, y, t)
        
        Args:
            x, y, t: Position and time (shape: (...,))
            Q, vx, vy, D: Additional normalized features (or None to ignore)
        
        Returns:
            R: Risk field (shape: (...,), all ≥ 0)
        """
        # Stack all inputs
        inputs = [x, y, t]
        if Q is not None:
            inputs.append(Q)
        if vx is not None:
            inputs.append(vx)
        if vy is not None:
            inputs.append(vy)
        if D is not None:
            inputs.append(D)
        
        # Handle batches
        inputs = torch.stack(inputs, dim=-1)  # (..., input_dim)
        original_shape = inputs.shape[:-1]
        
        # Flatten for network
        if len(original_shape) > 1:
            batch_size = np.prod(original_shape)
            inputs_flat = inputs.reshape(batch_size, -1)
        else:
            inputs_flat = inputs
        
        # Apply RFF if enabled
        if self.use_rff:
            rff_out = self.rff(inputs_flat)  # (batch, rff_dim)
            net_input = rff_out
        else:
            net_input = inputs_flat
        
        # Forward through network (with skip at midpoint)
        x_hidden = net_input
        for i, layer in enumerate(self.network):
            x_hidden = layer(x_hidden)
            
            # Skip connection: concatenate RFF at midpoint
            if (self.use_rff and i == len(self.network) // 2 and 
                isinstance(layer, nn.Tanh)):
                x_hidden = torch.cat([x_hidden, rff_out], dim=-1)
        
        R = x_hidden.squeeze(-1)
        
        if len(original_shape) > 1:
            R = R.reshape(original_shape)
        
        return R


# ==============================================================================
# FIX 2: SMOOTHNESS REGULARIZATION via Laplacian Penalty
# ==============================================================================
# Forces the risk field to be smooth — penalizes high curvature

class SmoothnessRegularizer:
    """
    Penalize ||∇²R|| (Laplacian) to encourage smooth fields
    
    Physics motivation: The diffusion term in the PDE naturally creates smoothness,
    but the network may learn sharp features. This term enforces that preference.
    """
    
    def __init__(self, field_net, eps=1e-3):
        self.field_net = field_net
        self.eps = eps
    
    def compute_laplacian_penalty(self, x, y, t):
        """
        Compute ||∇²R||² at query point (x, y, t)
        
        ∇²R ≈ (R_xx + R_yy + R_tt) where subscripts denote second derivatives
        """
        eps = self.eps
        
        # Evaluate at ±eps in each direction
        R_center = self.field_net(x, y, t)
        
        # Second derivative in x
        R_xp = self.field_net(x + eps, y, t)
        R_xm = self.field_net(x - eps, y, t)
        R_xx = (R_xp - 2 * R_center + R_xm) / (eps ** 2)
        
        # Second derivative in y
        R_yp = self.field_net(x, y + eps, t)
        R_ym = self.field_net(x, y - eps, t)
        R_yy = (R_yp - 2 * R_center + R_ym) / (eps ** 2)
        
        # Second derivative in t
        R_tp = self.field_net(x, y, t + eps)
        R_tm = self.field_net(x, y, t - eps)
        R_tt = (R_tp - 2 * R_center + R_tm) / (eps ** 2)
        
        # Laplacian penalty
        laplacian = R_xx + R_yy + R_tt
        penalty = (laplacian ** 2).mean()
        
        return penalty
    
    def compute_gradient_magnitude_penalty(self, x, y, t):
        """
        Alternative: penalize ||∇R||² (gradient magnitude)
        Less aggressive than Laplacian, but still encourages smoothness
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        R = self.field_net(x, y, t)
        R_sum = R.sum()
        
        R_sum.backward(create_graph=True)
        
        grad_x = x.grad
        grad_y = y.grad
        grad_t = t.grad
        
        gradient_mag = (grad_x ** 2 + grad_y ** 2 + grad_t ** 2).mean()
        
        return gradient_mag


# ==============================================================================
# FIX 3: REBALANCED LOSS WEIGHTS for Physics Dominance
# ==============================================================================
# The key insight: L_phys should strongly constrain the field
# L_data should guide the solution to match numerical solver
# The ratio should reflect physics importance

class ImprovedLossWeights:
    """
    Strategy for loss weight selection:
    
    - L_data: Fitting to numerical solver (supervised signal)  → weight = 1.0
    - L_phys: PDE residual (unsupervised physics constraint)   → weight = 1.0–5.0 (INCREASE)
    - L_ic:   Initial condition (weak)                         → weight = 0.1
    - L_bc:   Boundary condition (weak)                        → weight = 0.1
    - L_smooth: Laplacian smoothness (new)                     → weight = 0.1–0.5
    
    Recommended config:
    """
    
    @staticmethod
    def get_config():
        return {
            'L_data': 1.0,        # Keep as baseline
            'L_phys': 3.0,        # INCREASED from 0.1 — physics must be prioritized
            'L_ic': 0.1,          # Initial condition
            'L_bc': 0.1,          # Boundary condition
            'L_smooth': 0.2,      # NEW: smoothness penalty
        }


# ==============================================================================
# FIX 4: ENHANCED TRAINING LOOP with Smoothness
# ==============================================================================

class EnhancedPINNTrainer:
    """
    Updated training loop incorporating smoothness regularization
    """
    
    def __init__(self, field_net, optimizer, loss_weights):
        self.field_net = field_net
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.smoother = SmoothnessRegularizer(field_net, eps=1e-3)
    
    def compute_loss(self, batch):
        """
        Compute total loss with smoothness term
        
        Args:
            batch: {
                'x', 'y', 't': coordinates
                'R_target': numerical solver output
                'Q', 'vx', 'vy', 'D': features
            }
        """
        x, y, t = batch['x'], batch['y'], batch['t']
        R_target = batch.get('R_target', None)
        Q = batch.get('Q', None)
        vx = batch.get('vx', None)
        vy = batch.get('vy', None)
        D = batch.get('D', None)
        
        # Predict
        R_pred = self.field_net(x, y, t, Q, vx, vy, D)
        
        losses = {}
        
        # 1. Data loss: fit to numerical solver
        if R_target is not None:
            losses['L_data'] = torch.nn.functional.mse_loss(R_pred, R_target)
        else:
            losses['L_data'] = torch.tensor(0.0)
        
        # 2. Physics loss (PDE residual) — placeholder, implement your PDE
        losses['L_phys'] = self._compute_pde_residual(x, y, t, Q, vx, vy, D)
        
        # 3. Initial/boundary conditions (weak, placeholder)
        losses['L_ic'] = torch.tensor(0.0)
        losses['L_bc'] = torch.tensor(0.0)
        
        # 4. SMOOTHNESS loss (NEW)
        n_smooth_samples = min(len(x), 50)  # Sample for efficiency
        smooth_loss = 0.0
        for _ in range(n_smooth_samples):
            idx = torch.randint(0, len(x), (1,))
            smooth_loss += self.smoother.compute_laplacian_penalty(
                x[idx], y[idx], t[idx]
            )
        losses['L_smooth'] = smooth_loss / n_smooth_samples
        
        # Total loss
        loss_total = (self.loss_weights['L_data'] * losses['L_data'] +
                     self.loss_weights['L_phys'] * losses['L_phys'] +
                     self.loss_weights['L_ic'] * losses['L_ic'] +
                     self.loss_weights['L_bc'] * losses['L_bc'] +
                     self.loss_weights['L_smooth'] * losses['L_smooth'])
        
        return loss_total, losses
    
    def _compute_pde_residual(self, x, y, t, Q, vx, vy, D):
        """Placeholder for PDE residual — implement your telegrapher equation"""
        return torch.tensor(0.0)  # TODO: implement


# ==============================================================================
# FIX 5: INTELLIGENT AGENT FILTERING
# ==============================================================================
# Only pass 4-5 most critical agents to network, pre-filter by topology

class AgentFilter:
    """
    Select critical agents based on:
    1. Distance to ego (closer = higher priority)
    2. Size (truck > car)
    3. Relative velocity (high difference = high threat)
    4. Collision course (trajectory overlap with ego's path)
    """
    
    def __init__(self, ego_id, ego_vehicle_size='car'):
        self.ego_id = ego_id
        self.ego_size = 'truck' if ego_vehicle_size == 'truck' else 'car'
    
    def filter_agents(self, ego_state, all_agents, max_agents=5, 
                     distance_threshold=100.0):
        """
        Select top-k agents by threat score
        
        Args:
            ego_state: {'x': float, 'y': float, 'vx': float, 'vy': float}
            all_agents: List[{'x', 'y', 'vx', 'vy', 'size', 'track_id'}]
            max_agents: Maximum number to select
            distance_threshold: Ignore agents beyond this distance (m)
        
        Returns:
            filtered_agents: List of dicts with threat scores, sorted by threat desc
        """
        ego_x, ego_y = ego_state['x'], ego_state['y']
        ego_vx, ego_vy = ego_state['vx'], ego_state['vy']
        ego_v = np.sqrt(ego_vx**2 + ego_vy**2)
        
        threat_scores = []
        
        for agent in all_agents:
            if agent['track_id'] == self.ego_id:
                continue
            
            # Feature 1: Distance
            dx = agent['x'] - ego_x
            dy = agent['y'] - ego_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > distance_threshold:
                continue
            
            distance_score = 1.0 / (1.0 + dist / 10.0)  # 0–1, closer=higher
            
            # Feature 2: Size (vehicle type)
            size_score = 1.5 if agent['size'] == 'truck' else 1.0
            
            # Feature 3: Relative velocity magnitude
            agent_v = np.sqrt(agent['vx']**2 + agent['vy']**2)
            rel_v = abs(ego_v - agent_v)
            velocity_score = 1.0 + rel_v / 5.0  # 0–?, higher=larger threat
            
            # Feature 4: Collision course (trajectory overlap in next T seconds)
            T_lookahead = 3.0  # seconds
            
            # Predict positions
            ego_x_future = ego_x + ego_vx * T_lookahead
            ego_y_future = ego_y + ego_vy * T_lookahead
            agent_x_future = agent['x'] + agent['vx'] * T_lookahead
            agent_y_future = agent['y'] + agent['vy'] * T_lookahead
            
            # Future distance
            future_dx = agent_x_future - ego_x_future
            future_dy = agent_y_future - ego_y_future
            future_dist = np.sqrt(future_dx**2 + future_dy**2)
            
            # Score: low future distance = collision course
            collision_score = 1.0 / (1.0 + future_dist / 10.0)  # 0–1
            
            # Composite threat score
            threat_score = distance_score * velocity_score * size_score * (1.0 + collision_score)
            
            threat_scores.append({
                'agent': agent,
                'threat_score': threat_score,
                'distance': dist,
                'collision_likelihood': collision_score,
            })
        
        # Sort by threat descending
        threat_scores.sort(key=lambda x: x['threat_score'], reverse=True)
        
        # Return top-k with full details
        return threat_scores[:max_agents]
    
    def format_for_network(self, ego_state, critical_agents, max_agents=5,
                          normalize=True):
        """
        Convert filtered agents to network input features
        
        Args:
            ego_state: ego position and velocity
            critical_agents: output from filter_agents()
            normalize: whether to normalize to [-1, 1]
        
        Returns:
            features: (n_agents, 7) tensor [x, y, vx, vy, size_code, rel_vx, rel_vy]
        """
        n_agents = len(critical_agents)
        features = np.zeros((max_agents, 7))
        
        ego_x, ego_y = ego_state['x'], ego_state['y']
        ego_vx, ego_vy = ego_state['vx'], ego_state['vy']
        
        for i, entry in enumerate(critical_agents):
            agent = entry['agent']
            
            # Relative position
            rel_x = agent['x'] - ego_x
            rel_y = agent['y'] - ego_y
            
            # Relative velocity
            rel_vx = agent['vx'] - ego_vx
            rel_vy = agent['vy'] - ego_vy
            
            # Size encoding (car=0, truck=1)
            size_code = 1.0 if agent['size'] == 'truck' else 0.0
            
            features[i] = [
                rel_x,      # Relative position x
                rel_y,      # Relative position y
                agent['vx'], # Agent absolute velocity x
                agent['vy'], # Agent absolute velocity y
                size_code,   # Vehicle type
                rel_vx,      # Relative velocity x
                rel_vy,      # Relative velocity y
            ]
        
        # Normalize if requested
        if normalize:
            # Simple clip normalization
            features = np.clip(features, -100, 100) / 100.0
        
        return torch.FloatTensor(features)


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

def main_with_improvements():
    """
    How to integrate these fixes into your training pipeline
    """
    
    # 1. Create network with RFF
    field_net = EnhancedNeuralRiskField(
        input_dim=7,
        hidden=256,
        depth=8,
        use_rff=True,      # Enable Random Fourier Features
        rff_scale=10.0     # Frequency scale
    )
    
    # 2. Optimizer
    optimizer = torch.optim.Adam(field_net.parameters(), lr=1e-3)
    
    # 3. Improved loss weights
    loss_weights = ImprovedLossWeights.get_config()
    print(f"Loss weights: {loss_weights}")
    
    # 4. Trainer with smoothness
    trainer = EnhancedPINNTrainer(field_net, optimizer, loss_weights)
    
    # 5. Agent filtering example
    agent_filter = AgentFilter(ego_id=1, ego_vehicle_size='car')
    
    # 6. Example usage in training loop
    ego_state = {'x': 0.0, 'y': 0.0, 'vx': 25.0, 'vy': 0.0}
    all_agents = [
        {'x': 10.0, 'y': 0.5, 'vx': 22.0, 'vy': 0.0, 'size': 'car', 'track_id': 2},
        {'x': -15.0, 'y': -2.0, 'vx': 20.0, 'vy': 0.0, 'size': 'truck', 'track_id': 3},
        {'x': 50.0, 'y': 3.0, 'vx': 24.0, 'vy': 0.0, 'size': 'car', 'track_id': 4},
        # ... more agents
    ]
    
    # Filter to top 5 critical agents
    critical_agents = agent_filter.filter_agents(
        ego_state, all_agents, max_agents=5, distance_threshold=100.0
    )
    
    print(f"Filtered to {len(critical_agents)} critical agents")
    for entry in critical_agents:
        print(f"  Agent {entry['agent']['track_id']}: threat={entry['threat_score']:.3f}, "
              f"collision_likelihood={entry['collision_likelihood']:.3f}")
    
    # Convert to network input
    agent_features = agent_filter.format_for_network(ego_state, critical_agents)
    print(f"Agent features shape: {agent_features.shape}")
    print(f"Agent features:\n{agent_features}")

if __name__ == '__main__':
    main_with_improvements()
