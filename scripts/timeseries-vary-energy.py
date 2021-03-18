# -*- coding: utf-8 - *-  # telling Python the source file you've saved is utf-8
"""
Script calculates mean "on" per time-step for each regulatory network component.
Outputs data and figures (time series).
"""

# # Import modules
import numpy as np
import pandas as pd
import csv
import os
import datetime as dt
import module_rk as rk
import matplotlib.pyplot as plt
import argparse

# # Model Inputs
parser = argparse.ArgumentParser()
parser.add_argument("motif", type=str)  # Regulatory motifs to loop through
parser.add_argument("signal_status", type=str)
parser.add_argument("signal_0", type=int)
parser.add_argument("signal_length", type=int)
parser.add_argument("length_index", type=int)
parser.add_argument("direction", type=str)
parser.add_argument("switchpoint", type=int)
args, unknown = parser.parse_known_args()
print(args)

print('Preamble')
# # Model Inputs
# Number of simulations per energy level
number_sims = 2*10**args.length_index
# set number time-steps per simulation
total_time = 40
# signal features
if args.signal_status == "True":
    signal = True
if args.signal_status == "False":
    signal = False
signal_range = rk.signal_range(signal, args.signal_0, args.signal_length)

# # energy parameters
direction = args.direction
energy_min = 0.1
energy_max = 1
switchpoint = args.switchpoint

# input and output locations
pathIn, pathOut = rk.pathOut(__file__)
# Create directory for script outputs
motif_dir = '%s/outputs/%s' % (pathOut, args.motif)
if signal == False:
    dir_out = '%s/timeseries-variable-energy-%s-no-signal' % (motif_dir, direction)
else:
    dir_out = '%s/timeseries-variable-energy-%s-signalStart-%.0f-signalEnd-%.0f' % \
        (motif_dir, direction, args.signal_0, signal_range[-1])

create_directories = [motif_dir, dir_out]
for dirName in create_directories:
    # print(dirName)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

# signal regulation & unique nodes
signal_node_edge, nodes_unique = rk.unique_nodes(pathIn, args.motif)
# split unique nodes into groups
nodes_grn, nodes_ghost, nodes_signal, total_nodes, number_ghost, number_signal, number_grn = \
    rk.node_groups(nodes_unique)
# unique edges targeted by signal
signal_node_edge_targets = rk.signal_targets_unique(signal_node_edge)
# additional information about network nodes
nodeOrder_list, df_nodeLabels, signal_labels = rk.node_labels(nodes_grn, nodes_signal, nodes_ghost)
# Total number of global states (size of state space)
totalStates = 2 ** number_grn

# # Use GRN data to fill interaction matrix
interactions = rk.interaction_matrix(pathIn, args.motif, total_nodes, nodeOrder_list)

# # Import ICs
ics = rk.import_ICs(pathIn, args.motif, nodes_grn)

# # Output summary file of all args.motif data
rk.summary_file(dir_out, args.motif, total_nodes, number_grn, nodes_grn, number_ghost,
                nodes_ghost, number_signal, nodes_signal, totalStates,
                ics, interactions, signal_node_edge, df_nodeLabels)

# # Dataframe column headers
# headers for time-evolution network behaviour dataframe
df_raw_head = ['energy_sim', 'initial_condition', 'time_step', 'signal_state']
df_raw_head.extend(list(nodes_grn))
# headers for mean node activation dataframe
df_stats_head = list(nodes_grn)
df_stats_head[:0] = ['time_step', 'signal_state']

# # create empty array, pre-allocate size and data type
index_length = (total_time+1)*number_sims*len(ics)
array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.int64)
index_number = 0

print('Simulations.')
# # run numerous simulations per energy level
for energy_sim in range(0, number_sims):
    if energy_sim % 50000 == 0:
        # # Similation progress file
        progressFile = open('%s/simulation-progress.txt' % (pathOut), 'a')
        line = '%s [vary enery]: %s, signal length %.0f, energy simulation %s' \
            % (dt.datetime.now().strftime('%H:%M:%S'), args.motif, len(signal_range), energy_sim)
        progressFile.write("%s\n" % line)
        # # Close progress file
        progressFile.close()

    # # Reduce matrices & dataframes using probability-based condition
    low_energy_matrix = rk.submatrix(interactions, rk.energy_prob_function(energy_min))
    high_energy_matrix = rk.submatrix(interactions, rk.energy_prob_function(energy_max))
    # print('Low energy sub-interaction matrix:\n%s' % str(low_energy_matrix))

    # # Loop through all initial conditions
    for globalState_0 in ics:
        # # Loop through full time range (per simulation)
        for timestep in range(0, total_time):
            if timestep == 0:
                # # integer conversion
                globalState_0_int = int(globalState_0, 2)

                # # modify state with signal & ghost nodes states on the end
                globalState_signal = rk.extend_state(globalState_0, len(nodes_signal), 0)
                globalState_extended = rk.extend_state(globalState_signal, len(nodes_ghost), 1)

                # # Add t=0 data to array
                array_raw_data[index_number] = rk.add_row(
                    energy_sim, globalState_0_int, 0, 0, globalState_0)
                index_number += 1

            """
            NEW ADDITION
            varying energy over simulation time period
            """
            # # fix probability threshold
            energy, submatrix = rk.discrete_energy_time_function(
                direction, low_energy_matrix, high_energy_matrix, energy_min,
                energy_max, switchpoint, timestep)
            # print('Sub-interaction matrix:\n%s' % str(submatrix))

            """ OG """
            # # Modify signal node state if in pulse range
            globalState_extended = \
                rk.node_state_change(globalState_extended, len(nodes_grn), '1') \
                if (timestep) in signal_range else globalState_extended[:]
            # check state of signal
            signal_state = globalState_extended[len(nodes_grn):len(nodes_grn)+1]

            # # Update interaction matrix with signal regulation
            # print('Signal State = %.0f' % int(signal_state))
            if signal_state == '1':
                submatrix_signal = rk.signal_node_edge_regulation_matrix(
                    submatrix, signal_node_edge, signal_node_edge_targets, nodeOrder_list, globalState_extended)
            else:
                submatrix_signal = submatrix.copy()
            # print(submatrix_signal)

            # # Updating of global state - Update N nodes
            N = np.random.poisson(number_grn + number_signal)
            # print(N)
            if N == 0:
                globalState_extended = globalState_extended[:]
                globalState = globalState_extended[:len(nodes_grn)]
            else:
                globalState, globalState_extended = rk.update_state(globalState_extended, number_grn, number_signal,
                                                                    submatrix_signal, total_nodes,  N)

            if timestep not in signal_range[:-1]:  # list(map(lambda x: x+1, signal_range[:-1]))
                signal_state = globalState_extended[len(nodes_grn):len(nodes_grn)+1]
            else:
                signal_state = '1'
            # # # Print data for time-step
            # print('%.0f, %s, %s, %s' %
            #       (timestep, signal_state, globalState, globalState_extended))

            # # Add data to array
            array_raw_data[index_number] = rk.add_row(
                energy_sim, globalState_0_int, timestep+1, signal_state, globalState)
            index_number += 1
# print(array_raw_data)

# # Create a Pandas DataFrame from the Numpy array of data & export as csv file
df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:], index=range(
    0, index_length), columns=df_raw_head)
# full_filename = rk.export_df('timeseries-data-variable-energy', dir_out, args.motif, df_raw_data)

# # Calculate mean "on" and standard deviation for each time-step
for globalState_0 in ics:
    # # integer conversion
    globalState_0_int = int(globalState_0, 2)
    # globalState_0 = rk.get_bin(globalStateInt_0, len(nodes_grn))
    df_sub_raw1 = df_raw_data[df_raw_data.initial_condition ==
                              globalState_0_int].reset_index(drop=True)  # .astype(float)
    # print(df_sub_raw1)

    # # initiate dataframes for mean, std & cv
    df_mean = pd.DataFrame(index=[], columns=df_stats_head)
    df_std = pd.DataFrame(index=[], columns=df_stats_head)
    df_cv = pd.DataFrame(index=[], columns=df_stats_head)

    for time_iter in range(0, total_time+1):
        df_sub_raw = df_sub_raw1[df_sub_raw1.time_step ==
                                 time_iter].astype(float).reset_index(drop=True)
        # print(df_sub_raw)

        stat_summary = df_sub_raw.describe(exclude=[np.object])
        # print(stat_summary)
        mean = stat_summary.iloc[1, 2:len(df_raw_head)].tolist()
        df_mean.loc[len(df_mean)] = mean
        # print(mean)
        std = stat_summary.iloc[2, 3:len(df_raw_head)].tolist()
        std.insert(0, time_iter)
        df_std.loc[len(df_std)] = std
        # print(std)

    # # Export mean "on" per time-step dataframe
    df_mean = df_mean.fillna(0)
    df_mean_filename = 'timeseries-%s-mean' % (globalState_0)
    rk.export_df(df_mean_filename, dir_out, args.motif, df_mean)

    # # Export standard deviation per time-step dataframe
    df_std = df_std.fillna(0)
    df_std_filename = 'timeseries-%s-std' % (globalState_0)
    rk.export_df(df_std_filename, dir_out, args.motif, df_std)

    # # Calculate the cv (=standard deviation/mean) for each time-step and datframe column
    df_cv = df_std.div(df_mean)
    df_cv['time_step'] = df_std['time_step']

    # # Export cv per time-step dataframe
    # df_cv = df_cv.fillna(0)
    df_cv_filename = 'timeseries-%s-cv' % (globalState_0)
    rk.export_df(df_cv_filename, dir_out, args.motif, df_cv)

# # Produce figure(s)
print('Producing figures.')

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


def time_evolution_fig(networkGlobalState_0, uniqueNodes, numberNodes, fig_timeSteps, df, df_error, df_low, df_high):
    """
    Alternative visualization modules:
    Visualization with Seaborn: https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
    """
    fig, axs = plt.subplots(2, 2, figsize=(3, 2.5), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.19, 'wspace': 0.08})
    ax = axs.ravel()

    # # set all text to arial font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.facecolor'] = '[1,1,1]'  # E5E5E5, [0.756,0.756,0.756,0.4]

    # fig.suptitle("Global state initial condition: %s" % str(networkGlobalState_0), fontsize=16)
    fig_subplotTitles = list(uniqueNodes)
    subplot_rows, subplot_cols = rk.time_evolution_subplot(numberNodes)
    x_label_subplots = [((subplot_rows-1)*subplot_cols)+n for n in range(0, subplot_cols)]
    y_label_subplots = [(n*subplot_cols) for n in range(0, subplot_rows)]

    for i in range(0, numberNodes):
        ax[i].set_title('%s' % fig_subplotTitles[i], fontsize=8, style='italic', pad=1.5)

        # # Vary energy data
        ax[i].plot(df.iloc[:, 0], df.iloc[:, i+2], '-', color='#0023D1',
                   linewidth=1, zorder=2)  # ax[i].step / ax[i].plot

        # # Low energy data
        ax[i].plot(df_low.iloc[:, 0], df_low.iloc[:, i+2], '-', color='g',
                   linewidth=1.2, alpha=0.95, zorder=0)

        # # High energy data
        ax[i].plot(df_high.iloc[:, 0], df_high.iloc[:, i+2], '-', color='m',
                   linewidth=1.2, alpha=0.95, zorder=1)

        ax[i].set_xlim(-0.5, fig_timeSteps+0.5)
        ax[i].set_ylim(-0.02, 1.02)

        ax[i].xaxis.set_major_locator(MultipleLocator(10))
        ax[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax[i].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i].set_yticks(np.linspace(0, 1, 5))  # , minor=True)
        ax[i].tick_params(axis="both", labelsize=7, length=1.5, pad=1)

        if i in y_label_subplots:
            ax[i].set_ylabel("Mean Activation", size=8, labelpad=2)

        if i in x_label_subplots:
            ax[i].set_xlabel("Time-step", size=8, labelpad=2)

        ax[i].grid(True, which='both', linestyle='--', linewidth='0.1')
    return fig


if signal == False:
    dir = '%s/outputs/%s/timeseries-no-signal' % (pathOut, args.motif)
else:
    dir = '%s/outputs/%s/timeseries-signalStart-%.0f-signalEnd-%.0f' % \
        (pathOut, args.motif, args.signal_0, signal_range[-1])
if os.path.exists(dir):
    # # loop through 0 to totalGlobalStates-1 in chronological/random order
    for globalState_0 in ics:
        # # Import varying energy data
        df_mean_filename = '%s/%s-timeseries-%s-mean.csv' % (dir_out, args.motif, globalState_0)
        df_mean = pd.read_csv(df_mean_filename)
        df_std_filename = '%s/%s-timeseries-%s-std.csv' % (dir_out, args.motif, globalState_0)
        df_std = pd.read_csv(df_std_filename)

        # # Import fixed energy data
        if signal == False:
            df_low_mean_filename = '%s/outputs/%s/timeseries-no-signal/ecoli-timeseries-%s-energy=10-mean.csv' \
                % (pathOut, args.motif, globalState_0)
            df_low_mean = pd.read_csv(df_low_mean_filename)

            # # Import high energy data
            df_high_mean_filename = '%s/outputs/%s/timeseries-no-signal/ecoli-timeseries-%s-energy=100-mean.csv' \
                % (pathOut, args.motif, globalState_0)
            df_high_mean = pd.read_csv(df_high_mean_filename)
        else:
            # df_low_mean_filename = '%s/outputs/%s/timeseries-signalStart-%.0f-signalEnd-%.0f/ecoli-timeseries-mean-%s-10.csv' \
            df_low_mean_filename = '%s/outputs/%s/timeseries-signalStart-%.0f-signalEnd-%.0f/ecoli-timeseries-%s-energy=10-mean.csv' \
                % (pathOut, args.motif, args.signal_0, signal_range[-1], globalState_0)
            df_low_mean = pd.read_csv(df_low_mean_filename)

            # # Import high energy data
            # df_high_mean_filename = '%s/outputs/%s/timeseries-signalStart-%.0f-signalEnd-%.0f/ecoli-timeseries-mean-%s-100.csv' \
            df_high_mean_filename = '%s/outputs/%s/timeseries-signalStart-%.0f-signalEnd-%.0f/ecoli-timeseries-%s-energy=100-mean.csv' \
                % (pathOut, args.motif, args.signal_0, signal_range[-1], globalState_0)
            df_high_mean = pd.read_csv(df_high_mean_filename)

        """Plot figures"""
        fig_timeSteps = total_time

        # # Plot figure of time-evolution subplots
        fig_timeEvolution = time_evolution_fig(
            globalState_0, nodes_grn, len(nodes_grn), fig_timeSteps,
            df_mean, df_std, df_low_mean, df_high_mean)  # , df_binary_count)

        # # Save time-evolution figure
        fig_timeEvolutionFilename = "%s/%s-timeOutput-%s.pdf" % (
            dir_out, args.motif, globalState_0)
        fig_timeEvolution.savefig(fig_timeEvolutionFilename, format='pdf',
                                  transparent=False, bbox_inches='tight')
        plt.close(fig_timeEvolution)
else:
    print('Required directories for full figure do not exist.')
