"""Compatibility entry point for the corrected naturalistic PINN figure.

The former revision montage mixed ego-local cache images with quantitative
error panels.  It has been superseded by the orthophoto-calibrated renderer;
planning statistics are generated separately from paired closed-loop episodes.
"""

from rl.plot_dataset_pinn_orthophoto import main


if __name__ == "__main__":
    print(
        "plot_pinn_revision_evidence is deprecated; running "
        "plot_dataset_pinn_orthophoto instead."
    )
    raise SystemExit(main())
