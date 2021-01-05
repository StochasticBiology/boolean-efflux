# Dynamic Boolean Modelling of Regulatory Networks.

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


## Github files

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
  - Fig 1 -- (schematic plot)
  - Fig 2 -- timeseries.py 
  - Fig 3 -- timeseries.py
  - Fig 4 -- heatmaps.py
  - Fig S1 -- (schematic plot)
  - Fig S2 -- timeseries.py
  - Fig S3 -- timeseries.py
  - Fig S4 -- timeseries.py
  - Fig S5 -- timeseries.py
  - Fig S6 -- heatmaps.py

## Simulation of regulatory network dynamics

The below steps describe the steps to run the model code with chosen regulatory architecture, update method, update rule and energy levels.

Create a clone of Github files locally on your computer through method (i) or (ii):
- (i) Download directly using the 'Code' --> 'Download' buttons.
- (ii) Open Terminal (on Mac). Change the current working directory to the location where you want the cloned directory. Type git clone, and then paste the HTTPS clone URL found from clicking the 'Code' button. Press Enter. A directory named 'boolean-efflux' will now be found in the specified location. It will look like this:<br/>
```sh 
cd <user-specified-location> 
git clone https://github.com/StochasticBiology/boolean-efflux.git 
```

In the 'boolean-efflux' directory, the 'input-data' sub-directory homes the information about the regulatory network(s) that are considered. The files used for *E. coli* and *Salmonella* in the manuscript are included in this repository.

For new regulatory networks, create initial condition and regulatory architecture comma-separted values files, using the example files in the 'input-data' directory for layout help.
   - Initial condition file: If left empty, the script runs through all possible global states (2^M, with M = number of elements in regulatory architecture).
   - Regulatory architecture files: Split into two indivudual files, a node-node and node-edge regulation file respectively.

For simulation, run:
```sh
python <path-to-file>/<filename.py>
```

File will request inputs for the regulatory network name and information on the stress signal. It will look like this for simulating the timeseries evolution of *E. coli* with a stress implemented from timestep 10 to 12 for 2000 runs:
```sh
Enter regulatory network name:ecoli
Enter signal start and end values (separate by a single space e.g. no signal would be entered as 0 0):10 12
Enter total simulations:2000
```

Script will execute and outputs can be found within 'boolean-efflux' directory.

If there are any questions regarding the included files, email ryan.mathbio@gmail.com.
