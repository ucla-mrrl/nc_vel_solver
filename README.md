# nc_vel_solver
Nonconvex Velocity Solver for PC-MRI Data

# About
This code is meant to accompany the reconstruction described in "Velocity Reconstruction with Non-Convex Optimization for Low-VENC Phase Contrast MRI"

The velocity solver itself is written in C++, accompanying code to set up the example datasets and run the solver is written in Python.

# Installation

The solver itself is the contained in src/solver.cpp, which should compile with the includes provided in /src/include

To run the full example in python, some binaries are supplied, but they may need to be recompiled depending on your operating system.  To do so run:
```
python setup.py build_ext --inplace
```
and the example script should be able to be run.  The python and c++ code was tested an ran using an Anaconda install of Python 3.6 on Windows 10 and MacOS.

# Running

The recon itself is the function reg_v2 which is in solver.cpp

For the entire python example, run solver_example.ipynb in a ipython/jupyter notebook