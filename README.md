Dynamic Boolean Modelling of Regulatory Networks.

Python code for the analysis of the dynamics of regulatory networks for bacterial efflux pump acrAB. Modules 'matplotlib' & 'graphviz' are used for visualisation with python.

## Requirements

Before running, ensure you have access to:
- Python (version 3.7.4 used here)
  - Required packages:
    - numpy
    - pandas
    - random
    - sys
    - os
    - graphviz
    - matplotlib
    - matplotlib.pyplot
    - math
    - scipy.cluster.hierarchy
    - scipy.cluster
    - datetime
    - operator
    - module_rk (available on Github repository)


## Simulation of regulatory network dynamics

The analysis proceeds through two files (1) & (2), requiring module (3) for execution:
1) heatmaps.py
2) timeseries.py
3) module_rk.py

--- File 1 ---
** heatmaps.py -- applies energy-dependent Boolean modelling framework to regulatory networks to gather state space data i.e. the accessible transitions for each network state.

--- File 2 ---
** timeseries.py -- applies energy-dependent Boolean modelling framework to regulatory networks to gather time evolution data  for the mean activation of each network component.

--- File 3 ---
** module_rk.py -- self-defined functions for executing tasks in (1) & (2).


The figures in the corresponding manuscript are produced by:
Fig 1 -- (schematic plot)
Fig 2 -- timeseries.py
Fig 3 -- timeseries.py
Fig 4 -- heatmaps.py
Fig S1 -- (schematic plot)
Fig S2 -- timeseries.py
Fig S3 -- timeseries.py
Fig S4 -- timeseries.py
Fig S5 -- timeseries.py
Fig S6 -- heatmaps.py

To run the model code with chosen regulatory architecture, update method, update rule and energy levels, use:

In the directory where the two python scripts are located:
```sh
mkdir input-data
```

Create initial condition and regulatory architecture comma-separted values files.
   - Initial condition files: If left empty, the script runs through all possible global states (2^M, with M = number of elements in regulatory architecture).
   - Regulatory architecture files: Split into two indivudual files, a node-node and node-edge regulation file respectively.

For simulation, run:
```sh
python <path-to-file>/<filename.py>
```

Enter requested inputs. Script will execute

If there are any questions regarding the included files, email ryan.mathbio@gmail.com.
