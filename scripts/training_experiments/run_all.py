"""
Run all 8 model training configurations.

This script executes train.py with each experiment config and saves:
- Training logs to results/{exp_name}.log
- Metrics CSV to results/{exp_name}_metrics.csv
"""

import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "training_experiments" / "results"


def run_all_experiments():
    """Run all experiment configurations."""
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get all config files
    config_files = sorted(CONFIGS_DIR.glob("*.yaml"))

    # Filter to run only specific experiment(s)
    config_files = [f for f in config_files if f.stem == "embedder_encoder_only"]

    print(f"Starting all experiments...")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Found {len(config_files)} configurations\n")

    for config_file in config_files:
        exp_name = config_file.stem
        log_file = RESULTS_DIR / f"{exp_name}.log"
        metrics_file = RESULTS_DIR / f"{exp_name}_metrics.csv"

        print("=" * 80)
        print(f"Running: {exp_name}")
        print(f"Config: {config_file}")
        print(f"Log: {log_file}")
        print(f"Metrics: {metrics_file}")
        print("=" * 80)

        # Run training
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--config", str(config_file),
            "--log-file", str(log_file),
            "--metrics-file", str(metrics_file)
        ]

        try:
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
            print(f"✓ {exp_name} completed successfully\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ {exp_name} failed with return code {e.returncode}\n")
        except Exception as e:
            print(f"✗ {exp_name} failed with error: {e}\n")

    print("=" * 80)
    print("All experiments complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()
