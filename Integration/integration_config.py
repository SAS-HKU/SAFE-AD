"""
DRIFT-IDEAM Integration Configuration
======================================
Central configuration for all integration levels with A/B testing support.
"""

from dataclasses import dataclass, asdict
import json
import os

@dataclass
class IntegrationConfig:
    """Configuration for DRIFT-IDEAM integration."""

    # ==== FEATURE FLAGS ====
    enable_decision_veto: bool = True
    enable_mpc_cost: bool = True
    enable_cbf_modulation: bool = True

    # ==== MPC COST INTEGRATION ====
    mpc_risk_weight: float = 0.5  # Weight for risk term in cost function
    # Cost scaling: 0.0 = disabled, 0.5 = moderate, 1.0+ = aggressive

    # ==== DECISION VETO ====
    decision_risk_threshold: float = 1.5  # Max risk to allow lane change
    # Threshold: 1.0 = conservative, 1.5 = moderate, 2.0+ = permissive

    # ==== CBF MODULATION ====
    cbf_alpha: float = 0.3  # Modulation strength (0-1)
    cbf_max_scale: float = 2.0  # Max ellipse expansion
    # Risk value at which CBF/headway scale reaches maximum.
    # Sparse scenarios: 1.5–2.0.  Dense traffic: 6.0–10.0.
    cbf_risk_normalization: float = 1.5

    # ==== HEADWAY MODULATION ====
    headway_beta: float = 0.4  # Following distance scaling
    headway_max_scale: float = 2.0

    # ==== DRIFT PDE PARAMETERS ====
    drift_dt: float = 0.1  # PDE timestep
    drift_substeps: int = 3  # Sub-stepping for stability
    warmup_duration: float = 5.0  # Cold-start warmup [s]

    # ==== EXPERIMENTAL MODES ====
    mode: str = "full"  # "baseline", "decision_only", "mpc_only", "cbf_only", "full"

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, filepath):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))

    def apply_mode(self):
        """Apply experimental mode by setting feature flags."""
        if self.mode == "baseline":
            self.enable_decision_veto = False
            self.enable_mpc_cost = False
            self.enable_cbf_modulation = False
        elif self.mode == "decision_only":
            self.enable_decision_veto = True
            self.enable_mpc_cost = False
            self.enable_cbf_modulation = False
        elif self.mode == "mpc_only":
            self.enable_decision_veto = False
            self.enable_mpc_cost = True
            self.enable_cbf_modulation = False
        elif self.mode == "cbf_only":
            self.enable_decision_veto = False
            self.enable_mpc_cost = False
            self.enable_cbf_modulation = True
        elif self.mode == "full":
            self.enable_decision_veto = True
            self.enable_mpc_cost = True
            self.enable_cbf_modulation = True

        return self


# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

PRESETS = {
    "baseline": IntegrationConfig(mode="baseline"),

    "conservative": IntegrationConfig(
        mode="full",
        mpc_risk_weight=1.0,
        decision_risk_threshold=1.0,
        cbf_alpha=0.8,
        cbf_max_scale=2.5,
        cbf_risk_normalization=1.5,   # sparse scenario (≤5 vehicles)
    ),

    "balanced": IntegrationConfig(
        mode="full",
        mpc_risk_weight=0.5,
        decision_risk_threshold=1.5,
        cbf_alpha=0.6,
        cbf_max_scale=2.5,
        cbf_risk_normalization=1.5,   # sparse scenario (≤5 vehicles)
    ),

    "permissive": IntegrationConfig(
        mode="full",
        mpc_risk_weight=0.2,
        decision_risk_threshold=2.0,
        cbf_alpha=0.3,
        cbf_risk_normalization=1.5,
    ),

    # Dense-traffic preset — use when N_surrounding > ~8.
    # Risk fields aggregate linearly, so thresholds and the CBF saturation
    # reference must scale proportionally to preserve planner functionality.
    "dense": IntegrationConfig(
        mode="full",
        mpc_risk_weight=0.08,          # ×6 reduction vs balanced
        decision_risk_threshold=6.0,   # ×4 increase vs balanced
        cbf_alpha=0.25,                # reduced to prevent permanent max-scale
        cbf_max_scale=2.0,
        cbf_risk_normalization=8.0,    # CBF saturates at aggregate risk ≈ 8
    ),
}


def get_preset(name):
    """
    Get a preset configuration.

    Args:
        name: Preset name ("baseline", "conservative", "balanced", "permissive")

    Returns:
        IntegrationConfig instance

    Raises:
        ValueError: If preset name is unknown
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("DRIFT-IDEAM Integration Configuration")
    print("=" * 60)

    # Show all presets
    for preset_name in PRESETS.keys():
        config = get_preset(preset_name)
        config.apply_mode()
        print(f"\n{preset_name.upper()}:")
        print(f"  MPC Risk Weight: {config.mpc_risk_weight}")
        print(f"  Decision Threshold: {config.decision_risk_threshold}")
        print(f"  CBF Alpha: {config.cbf_alpha}")
        print(f"  Flags: decision={config.enable_decision_veto}, "
              f"mpc={config.enable_mpc_cost}, cbf={config.enable_cbf_modulation}")

    # Test save/load
    print("\n" + "=" * 60)
    print("Testing save/load...")
    test_config = get_preset("balanced")
    test_config.apply_mode()
    test_path = "test_config.json"
    test_config.save(test_path)
    loaded_config = IntegrationConfig.load(test_path)
    print(f"Saved and loaded successfully: {test_path}")
    os.remove(test_path)
    print("Test config file removed.")
