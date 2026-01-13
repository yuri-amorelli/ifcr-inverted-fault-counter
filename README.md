# IFCR – Inverted Fault Counter Regressor

(A practical labeling tool for predictive maintenance pipelines)

This repository contains a general-purpose implementation of IFCR (Inverted Fault Counter Regressor), a lightweight and practical labeling algorithm developed to support predictive maintenance workflows on real-world industrial IoT data.

IFCR addresses a common problem in operational datasets: failures are often logged explicitly, but labels suitable for supervised regression models (e.g. Remaining Useful Life–like targets) are missing.
The algorithm transforms raw fault or event logs into a distance-to-next-fault counter that can be directly used as a regression target, including with non-sequential models such as decision trees.

----------------------------------------------------

## What IFCR does

- Reads raw sensor or event data
- Sorts and aligns rows by timestamp
- Detects fault events from a binary indicator column
- Reconstructs a counter that resets to 0 on each fault and grows when moving backwards in time
- Outputs a usable regression label for supervised ML models (e.g. RUL-like targets)

The algorithm adapts to different dataset structures and fault semantics, making it suitable for a wide range of predictive maintenance applications.

----------------------------------------------------
## Implementations

Two implementations are provided:

1. build_ifcr_counter (recommended)

A clean and concise implementation suitable for most use cases.
This version prioritizes clarity and maintainability, making it ideal for integration into existing data pipelines and production workflows.

2. build_ifcr_counter_segmented

A version closer to the original research prototype.
It follows the step-by-step logic used in the initial experimental pipeline, including dataset segmentation and backward traversal within each segment.

Both implementations produce the same type of output (an inverted fault counter); the second is simply more explicit in how intermediate steps are constructed.

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
