# Dynamic Boolean Modelling of Regulatory Networks.

Python code to analyse the dynamics of regulatory networks for bacterial efflux pump acrAB. Module 'matplotlib' is used for visualisation with python.

## Requirements

Before running, ensure you have access to:
- Python (version 3.7.4 used here)
  - Required packages for data:
    - numpy
    - pandas
    - random
    - csv
    - sys
    - os
    - math
    - datetime
    - argparse
    - module_rk (available on Github repository)
  - Required packages for producing figures:
    - matplotlib
    - matplotlib.pyplot
    - scipy.cluster.hierarchy
    - scipy.cluster
    - matplotlib.ticker
    - module_rk (available on Github repository)

## Github files

The analysis proceeds through two files (1), (2), (4) & (5) requiring module (3) for execution:
1) heatmaps.py
2) timeseries.py
3) module_rk.py
4) timeseries-multi-stress.py
5) timeseries-vary-energy.py

--- File 1 ---<br/>
** heatmaps.py -- applies energy-dependent Boolean modelling framework to regulatory networks to gather state space data i.e. the accessible transitions for each network state.

--- File 2 ---<br/>
** timeseries.py -- applies energy-dependent Boolean modelling framework to regulatory networks to gather time evolution data for the mean activation of each network component.

--- File 3 ---<br/>
** module_rk.py -- self-defined functions for executing tasks in (1) & (2).

--- File 4 ---<br/>
** timeseries-multi-stress.py -- similarly to File 2, except multiple stress periods are now applied.

--- File 5 ---<br/>
** timeseries-vary-energy.py -- similarly to File 2, except the energy level is not constant and can be increased or decreased within the simulation.

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
  - Fig S7 -- timeseries-multi-stress.py
  - Fig S8 -- timeseries-vary-energy.py

## Simulation of regulatory network dynamics

--- Getting Started ---

The below steps describe the steps to run the model code with chosen regulatory architecture. To start a local copy of the repository needs to be retrieved.

Create a clone of Github files locally on your computer through method (i) or (ii):
- (i) Download directly using the 'Code' --> 'Download' buttons.
- (ii) Open Terminal (on Mac). Change the current working directory to the location where you want the cloned directory. Type git clone, and then paste the HTTPS clone URL found from clicking the 'Code' button. Press Enter. A directory named 'boolean-efflux' will now be found in the specified location. It will look like this:<br/>
```sh
cd <user-specified-location>
git clone https://github.com/StochasticBiology/boolean-efflux.git
Cloning into 'boolean-efflux'...
```

--- Code ---

Invoke heatmaps.py with
```sh
python <path-to-file>/heatmaps.py [motif] [signal_status] [length_index]
```

Invoke timeseries.py with
```sh
python <path-to-file>/timeseries.py [motif] [signal_status] [signal_0] [signal_length] [length_index]
```

Invoke timeseries-multi-stress.py with
```sh
python <path-to-file>/timeseries-multi-stress.py [motif] [length_index]
```

Invoke timeseries-vary-energy.py with
```sh
python <path-to-file>/timeseries-vary-energy.py [motif] [signal_status] [signal_start] [signal_length] [length_index] [direction] [switchpoint]
```

for example
./scripts/timeseries.py ecoli True 15 1 3

Notes:
 - When executing timeseries scripts, if signal_status = False, values for signal_0 and signal_length are to be set as 0.
 - To output figures for 'timeseries-vary-energy.py', the prerequisite requirements are the same simulations parameters but with constant energy level (using 'timeseries.py').


--- Command-line arguments ---

In addition to invoking each python script, they take a subset of the following additional command-line parameters:

[motif] -- motif being considered.

[signal_status] -- a boolean input (i.e. True or False) setting whether the stressor is active or inactive in simulations.

[length_index] -- an integer initialising the length of the simulation. Script will run for 2*10^([length index]) steps, so 2 = 200, 3 = 2000 etc.

[signal_0] -- an integer initialising the starting time-step of a stressor.

[signal_length] -- an integer initialising the length of the stressor.

[direction] -- a string (i.e. increase or decrease) initialising whether the energy level increase from low to high, or decreases from high to low in timeseries-vary-energy.py

[switchpoint] -- an integer initialising the time-step to change the energy level.


--- Data and input files ---

In the 'boolean-efflux' directory, the 'input-data' sub-directory homes the information about the regulatory network(s) that are considered. The files used for *E. coli* and *Salmonella* in the manuscript are included in this repository.

For new regulatory networks, 3 comma-separated values files (.csv) are required:
1) Initial condition file [filename format: &lt;motif&gt;-ICs.csv]: File contains the selected starting state(s) of the networks that are to be simulated. If no row information is supplied below the column headers, all possible global states (2^M, with M = number of elements in regulatory architecture) are simulated.

| Gene_1 | Gene_2 | ... | Gene_M |
| :-: | :-: | :-: | :-: |
| 1 | 0 | ... | 1 |
| 0 | 0 | ... | 1 |
| ... | ... | ... | ... |
| 0 | 0 | ... | 0 |

2) Regulatory architecture (node-node) [filename format: &lt;motif&gt;-regulation-nodes.csv]: File contains all node-node regulation within the regulatory archtecture.

|start node|end node|regulation|
| :-: | :-: | :-: |
| Gene_1 | Gene_2 | 1 |
| Gene_2 | Gene_2 | 1 |
|...| ...|... |
| Gene_M | Gene_1 | -1 |

3) Regulatory architecture signal regulation (node-edge) [filename format: &lt;motif&gt;-regulation-edges.csv]: File contains all regulation performed by the stressor/signal that is targetting edges within the wiring diagram.

|regulator|target edge start|target edge end| regulation|
| :-: | :-: | :-: | :-: |
| signal | Gene_1 | Gene_2 | -1 |
| signal | Gene_2 | Gene_2 | 1 |
| signal |...| ...|... |
| signal | Gene_M | Gene_1 | -1 |

See provided files in the 'input-data' directory for further layout help. Files (1)-(3) need to be formatted in the above layout and file format for scripts to be executed.

--- Output files ---

The output of heatmaps.py includes all unique transition at each energy level and the probability of that transition happening (&lt;motif&gt;-hm-&lt;signal_status&gt;-count-energy=&lt;energy_levelx100&gt;.csv); the converted data in a format for plotting the heatmap (&lt;motif&gt;-hm-&lt;signal_status&gt;-data-energy=&lt;energy_levelx100&gt;.csv); and the corresponding heatmap figure (&lt;motif&gt;-hm-&lt;signal_status&gt;-energy=&lt;energy_levelx100&gt;.pdf).

The output of each timeseries script includes the mean, standard deviation (std) and coefficient of variation (cv) for each network components at all time-steps and energy levels, starting from each prescribed initial state of the network. The corresponding figure (&lt;motif&gt;-timeOutput-&lt;initial_state&gt;-energy=&lt;energy_levelx100&gt;.pdf) displays time evolution dynamics.

Each script also outputs a summary file (network-summary.txt) containing information about the motif and Boolean modelling features.

Outputs from succesfully executed scripts will be found within the 'output' directory within 'boolean-efflux'.

If there are any questions regarding the included files, email ryan.mathbio@gmail.com
