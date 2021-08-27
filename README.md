# Dynamic Boolean Modelling of Regulatory Networks.

Python code to analyse the dynamics of regulatory networks for bacterial efflux pump acrAB. The heatmap scripts are generalised to work with any regulatory network. For plots displaying the evolution of the network, minor tweaks are required in functions 'timeFigOther' and 'timeFigEfflux' in 'timeseries.py' to plot all components (currently, the script plots efflux behaviour, and remaining network components, as two figures).

## Requirements

Before running, ensure you have access to:
- Python (version 3.8.2 used in the connected paper)
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
    - netInfo (available on Github repository)
  - Required packages for visualisation:
    - matplotlib (inc. sub-packages matplotlib.pyplot, matplotlib.patches, matplotlib.ticker, matplotlib.colors)
    - scipy.cluster (inc. scipy.cluster.hierarchy)
    - seaborn

## Github files

The analysis proceeds through two files (1) & (2), requiring module (3) for execution:
1) heatmaps.py
2) timeseries.py
3) netInfo.py

--- File 1 ---<br/>
** heatmaps.py -- applies energy-dependent Boolean modelling framework to regulatory networks to gather state space data i.e. the accessible transitions for each network state.

--- File 2 ---<br/>
** timeseries.py -- applies energy-dependent Boolean modelling framework to regulatory networks to gather time evolution data for the mean activation of each network component and simulated single-cell behaviour.

--- File 3 ---<br/>
** netInfo.py -- self-defined functions for executing tasks in (1) & (2).

The figures in the corresponding manuscript are produced by:
  - Fig 1 -- (schematic plot)
  - Fig 2 -- timeseries.py
  - Fig 3 -- timeseries.py
  - Fig 4 -- heatmaps.py
  - Fig 5 -- timeseries.py
  - Fig S1 -- (schematic plot)
  - Fig S2 -- timeseries.py
  - Fig S3 -- timeseries.py
  - Fig S4 -- timeseries.py
  - Fig S5 -- heatmaps.py
  - Fig S6 -- timeseries.py
  - Fig S7 -- timeseries.py
  - Fig S8 -- timeseries.py

## Simulation of regulatory network dynamics

--- Getting Started ---

The below steps describe the steps to run the model code with chosen regulatory architecture. To start a local copy of the repository needs to be retrieved.

Create a clone of Github files locally on your computer through method (i) or (ii):
- (i) Download directly using the 'Code' --> 'Download' buttons.
- (ii) Open Terminal (on Mac). Change the current working directory to the location where you want the cloned directory. Type git clone, and then paste the HTTPS clone URL found from clicking the 'Code' button. Press Enter. A directory named 'boolean-efflux' will now be found in the specified location. It will look similar to this:<br/>
```sh
cd <user-specified-location>
git clone https://github.com/StochasticBiology/boolean-efflux.git
Cloning into 'boolean-efflux'...
```

--- Code ---

Invoke heatmaps.py with
```sh
python <path-to-file>/heatmaps.py [motif] [updating_method] [signal_status] [length_index]
```

Invoke timeseries.py with
```sh
python <path-to-file>/timeseries.py [motif] [updating_method] [signal_start] [signal_end] [length_index]
```

for example
./scripts/timeseries.py ecoli timeseries.py ecoli async 125 25 1

Note:
 - For stress-free simulations, argument [signal_status] is to be set as 'False' in heatmaps.py, and [signal_start] = [signal_end] = 0 in timeseries.py.


--- Command-line arguments ---

In addition to invoking each python script, they take a subset of the following additional command-line parameters:

[motif] -- motif being considered.

[updating_method] -- asynchronous or synchronous.

[signal_status] -- a boolean input (i.e. True or False) setting whether the stressor is active or inactive in simulations.

[length_index] -- an integer initialising the length of the simulation. Script will run for 2*10^([length index]) steps, so 2 = 200, 3 = 2000 etc.

[signal_start] -- an integer initialising the starting time-step of a stressor period.

[signal_end] -- an integer initialising the end of the stressor period.


--- Data and input files ---

In the 'boolean-efflux' directory, the 'input-data' sub-directory homes the information about the regulatory network(s) that are considered. The files used for *E. coli* and *Salmonella* in the manuscript are included in this repository.

For new regulatory networks, 3 comma-separated values files (.csv) are required:
1) Initial condition file [filename format: <motif>-ICs.csv]: File contains the selected starting state(s) of the networks that are to be simulated. If no row information is supplied below the column headers, all possible global states (2^M, with M = number of elements in regulatory architecture) are simulated.

| Gene_1 | Gene_2 | ... | Gene_M |
| :-: | :-: | :-: | :-: |
| 1 | 0 | ... | 1 |
| 0 | 0 | ... | 1 |
| ... | ... | ... | ... |
| 0 | 0 | ... | 0 |

2) Regulatory architecture (node-node) [filename format: <motif>-regulation-nodes.csv]: File contains all node-node regulation within the regulatory architecture and the interactions/regulation that is coupled. The coupled interactions are linked by the specifying the row number it is coupled with. Below, for example, the first two rows are coupled and the final row is not coupled with any interaction.

|start node|end node|regulation|grouped regulation|
| :-: | :-: | :-: | :-: |
| Gene_1 | Gene_2 | 1 | 1 |
| Gene_2 | Gene_2 | 1 | 0 |
|...| ...|... |... |
| Gene_M | Gene_1 | -1 |  |

3) Regulatory architecture signal regulation (node-edge) [filename format: <motif>-regulation-edges.csv]: File contains all regulation performed by the stressor that is targeting edges within the wiring diagram.

|regulator|target edge start|target edge end| regulation|
| :-: | :-: | :-: | :-: |
| signal | Gene_1 | Gene_2 | -1 |
| signal | Gene_2 | Gene_2 | 1 |
| signal |...| ...|... |
| signal | Gene_M | Gene_1 | -1 |

See provided files in the 'input-data' directory for further layout help. Files (1)-(3) need to be formatted in the above layout and file format for scripts to be executed.

--- Output files ---

The output of heatmaps.py includes all unique transitions at each energy level and the probability of that transition happening in a time-step (<motif>-hm-<signal_status>-count-energy=<energy_levelx100>.csv). The data is then converted and plotted (<motif>-hm-<signal_status>-data-energy=<energy_levelx100>.csv and <motif>-hm-<signal_status>-energy=<energy_levelx100>.pdf respectively).

The output of each timeseries script includes the mean, standard deviation (std) and coefficient of variation (cv) for all network components at all time-steps and energy levels, starting from each prescribed initial state of the network. The corresponding pdf figure displays time evolution dynamics. The timeseries.py script also outputs simulated single-cell behaviour, through qualitative and quantitative figures.

Each script also outputs a summary file (network-summary.txt) containing information about the motif and Boolean modelling features.

Outputs from succesfully executed scripts will be found within the 'output' directory within 'boolean-efflux'.

If there are any questions regarding the included files, email ryankerr8@gmail.com
