"""
DRIFT-IDEAM Benchmarking Framework
===================================
Compare baseline IDEAM vs. PRIDEAM with various integration levels.

Provides metrics computation, A/B testing, and statistical comparison.
"""

import numpy as np
from dataclasses import dataclass, asdict
import json
import os
import time
from typing import List, Dict, Optional


@dataclass
class BenchmarkMetrics:
    """Metrics for a single simulation run."""

    # ==== SAFETY METRICS ====
    min_distance: float  # Minimum inter-vehicle distance [m]
    ttc_violations: int  # Count of TTC < 2.0s
    risk_integral: float  # Integral of risk over time
    max_risk: float  # Maximum risk encountered
    collision: bool  # Whether collision occurred

    # ==== EFFICIENCY METRICS ====
    mean_velocity: float  # Average velocity [m/s]
    velocity_std: float  # Velocity variance
    progress: float  # Total distance traveled [m]
    lane_changes: int  # Number of lane changes executed

    # ==== COMFORT METRICS ====
    max_accel: float  # Max longitudinal acceleration [m/s^2]
    max_jerk: float  # Max jerk [m/s^3]
    mean_abs_accel: float  # Mean |a|
    mean_abs_steer: float  # Mean |delta|

    # ==== DECISION METRICS ====
    lane_changes_blocked: int  # Number of DRIFT veto events
    mpc_failures: int  # MPC solver failures

    # ==== TIMING METRICS ====
    total_time: float  # Simulation wall-clock time [s]
    mean_mpc_time: float  # Average MPC solve time [ms]
    mean_drift_time: float  # Average DRIFT step time [ms]

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_simulation_data(cls, data: Dict):
        """
        Compute metrics from simulation data arrays.

        Args:
            data: Dict containing simulation data arrays:
                - S_obs: Minimum distances over time
                - risk: Risk values at ego over time
                - vel: Velocities over time
                - s: Longitudinal positions over time
                - path_record: Path indices over time
                - acc: Accelerations over time
                - steer: Steering angles over time
                - lane_changes_blocked: Count of DRIFT veto events
                - mpc_failures: Count of MPC solver failures
                - wall_clock_time: Total simulation time [s]
                - mpc_solve_times: List of MPC solve times [s]
                - drift_step_times: List of DRIFT step times [s]

        Returns:
            BenchmarkMetrics instance
        """
        # Safety metrics
        s_obs = data.get('S_obs', [])
        min_dist = np.min(s_obs) if len(s_obs) > 0 else np.inf

        # TTC violations (assuming threshold of 2.0s)
        ttc = data.get('ttc', [])
        ttc_violations = np.sum(np.array(ttc) < 2.0) if len(ttc) > 0 else 0

        # Risk metrics
        risk = data.get('risk', np.zeros(1))
        risk_integral = float(np.trapz(risk, dx=0.1)) if len(risk) > 0 else 0.0
        max_risk = float(np.max(risk)) if len(risk) > 0 else 0.0

        # Collision detection
        collision = bool(min_dist < 1.0)

        # Efficiency metrics
        vel = data.get('vel', [])
        mean_vel = float(np.mean(vel)) if len(vel) > 0 else 0.0
        vel_std = float(np.std(vel)) if len(vel) > 0 else 0.0

        s_data = data.get('s', [])
        progress = float(s_data[-1] - s_data[0]) if len(s_data) > 1 else 0.0

        path_record = data.get('path_record', [])
        lane_changes = int(np.sum(np.diff(path_record) != 0)) if len(path_record) > 1 else 0

        # Comfort metrics
        acc = data.get('acc', [])
        max_accel = float(np.max(np.abs(acc))) if len(acc) > 0 else 0.0

        jerk = np.diff(acc) / 0.1 if len(acc) > 1 else []
        max_jerk = float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0.0

        mean_abs_accel = float(np.mean(np.abs(acc))) if len(acc) > 0 else 0.0

        steer = data.get('steer', [])
        mean_abs_steer = float(np.mean(np.abs(steer))) if len(steer) > 0 else 0.0

        # Decision metrics
        lc_blocked = int(data.get('lane_changes_blocked', 0))
        mpc_fails = int(data.get('mpc_failures', 0))

        # Timing metrics
        total_time = float(data.get('wall_clock_time', 0.0))

        mpc_times = data.get('mpc_solve_times', [])
        mean_mpc_time = float(np.mean(mpc_times) * 1000) if len(mpc_times) > 0 else 0.0

        drift_times = data.get('drift_step_times', [])
        mean_drift_time = float(np.mean(drift_times) * 1000) if len(drift_times) > 0 else 0.0

        return cls(
            min_distance=float(min_dist),
            ttc_violations=int(ttc_violations),
            risk_integral=float(risk_integral),
            max_risk=float(max_risk),
            collision=bool(collision),
            mean_velocity=float(mean_vel),
            velocity_std=float(vel_std),
            progress=float(progress),
            lane_changes=int(lane_changes),
            max_accel=float(max_accel),
            max_jerk=float(max_jerk),
            mean_abs_accel=float(mean_abs_accel),
            mean_abs_steer=float(mean_abs_steer),
            lane_changes_blocked=int(lc_blocked),
            mpc_failures=int(mpc_fails),
            total_time=float(total_time),
            mean_mpc_time=float(mean_mpc_time),
            mean_drift_time=float(mean_drift_time),
        )


class BenchmarkRunner:
    """Run comparative benchmarks between configurations."""

    def __init__(self, output_dir="c:/DREAM/benchmarks"):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_comparison(self, scenario_configs: List[Dict], n_trials=10):
        """
        Run A/B test comparing multiple configurations.

        Args:
            scenario_configs: List of dicts with:
                {
                    'name': str (configuration name),
                    'config': IntegrationConfig,
                    'scenario_params': dict (scenario-specific parameters)
                }
            n_trials: Number of trials per config

        Returns:
            results: Dict mapping config name to list of BenchmarkMetrics

        Example:
            from Integration.integration_config import get_preset

            configs = [
                {'name': 'Baseline', 'config': get_preset('baseline'), 'scenario_params': {}},
                {'name': 'PRIDEAM', 'config': get_preset('balanced'), 'scenario_params': {}},
            ]
            results = runner.run_comparison(configs, n_trials=10)
        """
        results = {}

        for sc in scenario_configs:
            name = sc['name']
            config = sc['config']

            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")

            trial_metrics = []

            for trial in range(n_trials):
                print(f"  Trial {trial+1}/{n_trials}...", end=' ')

                # Run simulation with this config
                # NOTE: User must implement run_simulation_with_config()
                # or modify this to call their simulation script
                sim_data = self._run_single_trial(config, sc['scenario_params'], trial)

                # Compute metrics
                metrics = BenchmarkMetrics.from_simulation_data(sim_data)
                trial_metrics.append(metrics)

                print(f"Done (risk={metrics.max_risk:.2f}, vel={metrics.mean_velocity:.1f})")

            results[name] = trial_metrics

        # Save results
        self._save_results(results)

        # Print summary
        self._print_summary(results)

        return results

    def _run_single_trial(self, config, scenario_params, trial_idx):
        """
        Run a single simulation trial.

        NOTE: This is a placeholder. User should implement this by:
        1. Modifying emergency_test_prideam.py to accept config as argument
        2. Returning data dict with all required metrics arrays

        Args:
            config: IntegrationConfig instance
            scenario_params: Dict of scenario parameters
            trial_idx: Trial index (for random seed)

        Returns:
            data: Dict with simulation data arrays

        Raises:
            NotImplementedError: This method must be implemented by user
        """
        # TODO: Implement simulation runner
        # Example:
        # from emergency_test_prideam import run_simulation
        # data = run_simulation(config, seed=trial_idx, **scenario_params)
        # return data

        raise NotImplementedError(
            "BenchmarkRunner._run_single_trial() must be implemented.\n"
            "Modify emergency_test_prideam.py to:\n"
            "1. Accept IntegrationConfig as argument\n"
            "2. Return data dict with metrics arrays\n"
            "3. Import and call from here."
        )

    def _save_results(self, results):
        """
        Save benchmark results to disk.

        Args:
            results: Dict mapping config name to list of BenchmarkMetrics
        """
        timestamp = int(time.time())
        save_path = os.path.join(self.output_dir, f"results_{timestamp}.json")

        # Convert to serializable format
        serializable = {}
        for name, metrics_list in results.items():
            serializable[name] = [m.to_dict() for m in metrics_list]

        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to: {save_path}")

    def _print_summary(self, results):
        """
        Print statistical summary of results.

        Args:
            results: Dict mapping config name to list of BenchmarkMetrics
        """
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        for name, metrics_list in results.items():
            print(f"\n{name}:")
            print("-" * 60)

            # Aggregate statistics
            min_dists = [m.min_distance for m in metrics_list]
            mean_vels = [m.mean_velocity for m in metrics_list]
            max_risks = [m.max_risk for m in metrics_list]
            risk_integrals = [m.risk_integral for m in metrics_list]
            lc_blocked = [m.lane_changes_blocked for m in metrics_list]

            print(f"  Safety:")
            print(f"    Min Distance: {np.mean(min_dists):.2f} ± {np.std(min_dists):.2f} m")
            print(f"    Max Risk: {np.mean(max_risks):.2f} ± {np.std(max_risks):.2f}")
            print(f"    Risk Integral: {np.mean(risk_integrals):.2f} ± {np.std(risk_integrals):.2f}")

            print(f"  Efficiency:")
            print(f"    Mean Velocity: {np.mean(mean_vels):.2f} ± {np.std(mean_vels):.2f} m/s")

            print(f"  Decision:")
            print(f"    Lane Changes Blocked: {np.mean(lc_blocked):.1f} ± {np.std(lc_blocked):.1f}")

    def compare_two(self, results, name1, name2):
        """
        Detailed comparison between two configurations.

        Args:
            results: Dict from run_comparison()
            name1: First configuration name
            name2: Second configuration name

        Prints:
            Detailed statistical comparison
        """
        if name1 not in results or name2 not in results:
            print(f"Error: {name1} or {name2} not found in results")
            return

        metrics1 = results[name1]
        metrics2 = results[name2]

        print("\n" + "="*80)
        print(f"DETAILED COMPARISON: {name1} vs {name2}")
        print("="*80)

        # Helper to compute stats
        def stats(values):
            return np.mean(values), np.std(values)

        # Compare each metric
        metrics_to_compare = [
            ('min_distance', 'Min Distance [m]', lambda m: m.min_distance),
            ('max_risk', 'Max Risk', lambda m: m.max_risk),
            ('risk_integral', 'Risk Integral', lambda m: m.risk_integral),
            ('mean_velocity', 'Mean Velocity [m/s]', lambda m: m.mean_velocity),
            ('lane_changes_blocked', 'LC Blocked', lambda m: m.lane_changes_blocked),
            ('max_accel', 'Max Accel [m/s²]', lambda m: m.max_accel),
        ]

        for metric_key, metric_label, extractor in metrics_to_compare:
            vals1 = [extractor(m) for m in metrics1]
            vals2 = [extractor(m) for m in metrics2]

            mean1, std1 = stats(vals1)
            mean2, std2 = stats(vals2)

            diff = mean2 - mean1
            pct_change = (diff / mean1 * 100) if mean1 != 0 else 0

            print(f"\n{metric_label}:")
            print(f"  {name1}: {mean1:.3f} ± {std1:.3f}")
            print(f"  {name2}: {mean2:.3f} ± {std2:.3f}")
            print(f"  Difference: {diff:+.3f} ({pct_change:+.1f}%)")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def run_baseline_vs_prideam(n_trials=10):
    """
    Run standard baseline vs. PRIDEAM comparison.

    Args:
        n_trials: Number of trials per configuration

    Returns:
        results: Dict mapping config name to list of BenchmarkMetrics

    Example:
        results = run_baseline_vs_prideam(n_trials=10)
    """
    from Integration.integration_config import get_preset

    runner = BenchmarkRunner()

    configs = [
        {
            'name': 'Baseline',
            'config': get_preset('baseline'),
            'scenario_params': {'initial_seed': 42}
        },
        {
            'name': 'PRIDEAM-Balanced',
            'config': get_preset('balanced'),
            'scenario_params': {'initial_seed': 42}
        },
        {
            'name': 'PRIDEAM-Conservative',
            'config': get_preset('conservative'),
            'scenario_params': {'initial_seed': 42}
        },
    ]

    return runner.run_comparison(configs, n_trials)


def load_results(filepath):
    """
    Load previously saved benchmark results.

    Args:
        filepath: Path to results JSON file

    Returns:
        results: Dict mapping config name to list of BenchmarkMetrics
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = {}
    for name, metrics_dicts in data.items():
        results[name] = [BenchmarkMetrics(**m) for m in metrics_dicts]

    return results


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("DRIFT-IDEAM Benchmarking Framework")
    print("=" * 60)
    print()
    print("This framework provides tools for comparing IDEAM baseline")
    print("vs. PRIDEAM with different integration levels.")
    print()
    print("To use:")
    print("1. Implement _run_single_trial() in BenchmarkRunner")
    print("2. Or call run_baseline_vs_prideam() after implementation")
    print()
    print("Example:")
    print("  from Integration.benchmark import run_baseline_vs_prideam")
    print("  results = run_baseline_vs_prideam(n_trials=10)")
