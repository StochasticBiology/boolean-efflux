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
import sys
import datetime as dt
import module_rk as rk
import matplotlib.pyplot as plt
import argparse

print('Preamble')
# # Model Inputs
parser = argparse.ArgumentParser()
parser.add_argument("motif", type=str)  # Regulatory motifs to loop through
parser.add_argument("length_index", type=int)
args, unknown = parser.parse_known_args()
print(args)

print('Preamble')
# # Model Inputs
# Energy availability levels to scan through - can use set levels or a linspace.
energy_levels = np.array([0.1, 0.5, 1])
# Number of simulations per energy level
number_sims = 2*10**args.length_index
# set number time-steps per simulation
total_time = 70
# signal features
signal = True
signal_0 = 15
signalLength = 10
signal_range1 = rk.signal_range(signal, int(signal_0), int(signalLength))
signal_range2 = rk.signal_range(signal, int(40), int(signalLength))
signal_range3 = rk.signal_range(signal, int(52), int(signalLength))
signal_range = np.concatenate([signal_range1, signal_range2, signal_range3])

# input and output locations
pathIn, pathOut = rk.pathOut(__file__)
# create directory for script outputs
motif_dir = '%s/outputs/%s' % (pathOut, args.motif)
if signal == False:
    sys.exit("\n\nError: Signal state set as 'false'.")
else:
    dir_out = '%s/timeseries-multistress' % (motif_dir)
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

print('Simulations.')
# # Loop through each energy level
for energy in energy_levels:
    # # fix probability threshold
    prob_threshold = rk.energy_prob_function(energy)

    # # create empty array, pre-allocate size and data type
    index_length = (total_time+1)*number_sims*len(ics)
    array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.int64)
    index_number = 0

    # # run numerous simulations per energy level
    for energy_sim in range(0, number_sims):
        if energy_sim % 50000 == 0:
            # # Similation progress file
            progressFile = open('%s/simulation-progress.txt' % (pathOut), 'a')
            line = '%s [multi-stress]: %s, signal length %.0f, energy level %.2f, energy simulation %s' \
                % (dt.datetime.now().strftime('%H:%M:%S'), args.motif, len(signal_range),
                   energy, energy_sim)
            progressFile.write("%s\n" % line)
            # # Close progress file
            progressFile.close()

        # # Reduce matrices & dataframes using probability-based condition
        submatrix = rk.submatrix(interactions, prob_threshold)
        # print('Sub-interaction matrix:\n%s' % str(submatrix))

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
    df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:],
                               index=range(0, index_length), columns=df_raw_head)
    # full_filename = rk.export_df('timeseries-data-energy=%.0f' % (float(energy)*100), dir_out,
    #                              args.motif, df_raw_data)

    # # Calculate mean "on" and standard deviation for each time-step
    for globalState_0 in ics:
        # # integer conversion
        globalState_0_int = int(globalState_0, 2)
        df_sub_raw1 = df_raw_data[df_raw_data.initial_condition ==
                                  globalState_0_int].reset_index(drop=True)

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
        df_mean_filename = 'timeseries-%s-energy=%.0f-mean' % (globalState_0, float(energy)*100)
        rk.export_df(df_mean_filename, dir_out, args.motif, df_mean)

        # # Export standard deviation per time-step dataframe
        df_std = df_std.fillna(0)
        df_std_filename = 'timeseries-%s-energy=%.0f-std' % (globalState_0, float(energy)*100)
        rk.export_df(df_std_filename, dir_out, args.motif, df_std)

        # # Calculate the cv (=standard deviation/mean) for each time-step and datframe column
        df_cv = df_std.div(df_mean)
        df_cv['time_step'] = df_std['time_step']

        # # Export cv per time-step dataframe
        # df_cv = df_cv.fillna(0)
        df_cv_filename = 'timeseries-%s-energy=%.0f-cv' % (globalState_0, float(energy)*100)
        rk.export_df(df_cv_filename, dir_out, args.motif, df_cv)

print('Producing figures.')
# # Loop through energy steps and produce figures
for energy in energy_levels:
    # # loop through 0 to totalGlobalStates-1 in chronological/random order
    for globalState_0 in ics:
        # # Import mean "on" per time-step dataframe
        df_mean_filename = '%s/%s-timeseries-%s-energy=%.0f-mean.csv' % (
            dir_out, args.motif, globalState_0, float(energy)*100)
        df_mean = pd.read_csv(df_mean_filename)

        # # Import standard deviation per time-step dataframe
        df_std_filename = '%s/%s-timeseries-%s-energy=%.0f-std.csv' % (
            dir_out, args.motif, globalState_0, float(energy)*100)
        df_std = pd.read_csv(df_std_filename)

        # # Plot figures
        fig_timeSteps = total_time
        fig_timeEvolution = rk.time_evolution_fig(
            globalState_0, nodes_grn, len(nodes_grn), fig_timeSteps,
            df_mean, df_std)  # , df_binary_count)
        # Save time-evolution figure
        fig_timeEvolutionFilename = "%s/%s-timeOutput-%s-energy=%.0f.pdf" % (
            dir_out, args.motif, globalState_0, float(energy)*100)
        fig_timeEvolution.savefig(fig_timeEvolutionFilename, format='pdf',
                                  transparent=False, bbox_inches='tight')
        plt.close(fig_timeEvolution)
