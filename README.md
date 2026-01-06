# IFCR – Inverted Fault Counter Regressor

This repository contains a general-purpose implementation of the **IFCR (Inverted Fault Counter Regressor)** algorithm I developed during my research on predictive maintenance.

The goal of IFCR is to generate a *distance-to-next-fault* counter from any dataset containing an event or fault column. This counter can then be used as a supervised-learning target for regression models, including non-sequential methods such as decision trees.

----------------------------------------------------

## What IFCR does

- Reads raw sensor or event data  
- Sorts and aligns rows by timestamp  
- Detects fault events from a binary indicator column  
- Reconstructs a counter that resets to 0 on each fault and grows when moving backwards in time  
- Outputs a usable regression label for supervised ML models (e.g. Remaining Useful Life–like targets)

The algorithm is designed to adapt to different datasets and error semantics, making it a general tool for predictive-maintenance scenarios.

----------------------------------------------------
## Implementations

There are two implementations available:

1. build_ifcr_counter (recommended)

A simple, clean implementation suitable for most use cases.
This version focuses on clarity and is ideal for users who want a lightweight and readable function.

2. build_ifcr_counter_segmented

A version closer to the original research prototype.
This implementation follows the step-by-step logic used in the initial experimental pipeline, including segmentation of the dataset and a backward walk inside each segment.

Both functions produce the same type of output (an inverted fault counter), but the second is more explicit in how intermediate steps are constructed.

----------------------------------------------------
## Related publication

Amorelli Y. et al. (2025) — *Predictive Maintenance for Water Supply Networks:  
Advanced Expert System Models for Enhanced Water Resource Management and Monitoring.*

IEEE International Workshop on Metrology for Living Environment, Venice 2025.  
First Author. Published on IEEE Xplore.  
DOI: https://doi.org/10.1109/MetroLivEnv64961.2025.11107070

----------------------------------------------------
## Repository structure

```text
ifcr/
    __init__.py
    ifcr.py               # main IFCR implementations
examples/
    data_example.csv
    simple_example.py     # minimal usage example
README.md
LICENSE
