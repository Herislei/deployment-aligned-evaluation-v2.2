# Reproducibility package - Research Paper v2.2

**Paper:** Deployment-Aligned Evaluation with Temporal and Geographic Holdouts: A Cross-Case Empirical Study of Forecasting and Semantic Segmentation  
**Author:** Herislei Pimentel  
**Date:** 28 September 2026

This package preserves the computational workflow used to generate the empirical results reported in the manuscript.

## Included files
- `Herislei_Pimentel_Deployment_Aligned_Evaluation_Reproducibility_v2.2.ipynb`: deterministic notebook with preserved execution outputs.
- `Herislei_Pimentel_Deployment_Aligned_Evaluation_Reproducibility_v2.2.py`: plain-Python export of the same analytical workflow.

## Public data
- UCI Bike Sharing dataset (Capital Bikeshare hourly observations).
- Inria Aerial Image Labeling dataset.

The notebook downloads the public data at runtime. Part 2 is computationally intensive and a CUDA GPU is strongly recommended.

## Reproducibility controls
The segmentation workflow fixes seed 42 for Python, NumPy, PyTorch and CUDA; disables CuDNN benchmarking; requests deterministic CuDNN behavior and deterministic algorithms where supported; fixes the training DataLoader generator; uses `num_workers=0`; and resets the PyTorch RNG immediately before model initialization.

The author-defined Inria split is:
- Train: Austin, Chicago, Kitsap
- Validation: Tyrol-W
- Held-out labeled evaluation: Vienna

Patch starts are 0, 1496, 2992, and 4488 pixels on each axis for 5000 x 5000 tiles, producing 16 non-overlapping 512 x 512 patches per tile (about 16.8% area coverage).

## Important limits
This package reproduces the reported fixed configurations but does not establish that their hyperparameters were selected through a pre-registered search protocol. It also does not contain the additional experiments recommended for a stronger journal submission: random-vs-structured split counterfactuals, rotating held-out cities, and multi-seed uncertainty analysis. Exact package versions from the original 2026 Colab runtime were not archived.
