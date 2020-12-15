"""
Script calculates mean "on" per time-step for each regulatory network component.
Outputs data and figures (time series).
"""

"""
Import modules
"""
import numpy as np
import pandas as pd
import csv
import os
import sys
import datetime as dt
from operator import truediv
import module_rk as rk
import matplotlib.pyplot as plt

"""
Model Inputs
"""
# # Regulatory motifs to loop through
allMotifs = ["ecoli"]
# # Update method: 'sync' OR 'async'
update_method = 'async'
# # update rule: 0 = 'blue', 1 = 'red'
update_rule = 0
# # Levels of energy availability to scan through - can use set values or a linspace.
energy_partition = np.array([0.1, 0.5, 1])
# # Number of simulations per energy level
number_sims = 2*10**2
# # set number time-steps per simulation
total_time = 41
# # start & end time-point of signal activation (start, end)
signal_range, sig_start, sig_end = rk.signal_range(15, 15)


import pdb

"""
Path to directory locations
"""
path_file = os.path.abspath(__file__)
# # Directory for scripts
path_scripts = os.path.dirname(path_file)
# # Directory for script inputs: wiring diagram information & ICs
path_input = '%s/input-data' % os.path.dirname(path_scripts)
# # Directory for script output files & directories
path_out = os.path.dirname(path_scripts)

"""
Data accumulation
"""
for motifName in allMotifs:

    # # Create directory for script outputs
    dir_out = rk.create_dir(update_rule, sig_start, sig_end, motifName,
                            len(signal_range), update_method, path_out)

    # # Calculate pre-simulation info and data
    # node-node and node-edge data
    node_node_data, node_edge_data = rk.regulation_data(path_input, motifName)
    # unique nodes
    signal_node_edge, nodes_unique = rk.unique_nodes(node_node_data, node_edge_data)
    # split unique nodes into groups
    nodes_grn, nodes_ghost, nodes_signal = rk.node_groups(nodes_unique)
    # unique edges targeted by signal
    signal_node_edge_targets = rk.signal_targets_unique(signal_node_edge)
    # calculate size of node groups
    number_grn, number_ghost, number_signal = rk.node_total(
        nodes_grn, nodes_ghost, nodes_signal, nodes_unique)
    # additional information about network nodes
    nodeOrder_list, df_nodeLabels, signal_labels = rk.node_labels(
        nodes_grn, nodes_signal, nodes_ghost)
    # Total number of nodes in wiring diagram
    total_nodes = number_grn + number_ghost + number_signal
    # Total number of global states (size of state space)
    totalStates = 2 ** number_grn
    # Node order
    node_labels_filename = 'node_order_labels'
    rk.export_df(node_labels_filename, dir_out, motifName, df_nodeLabels)

    # # Use GRN data to fill interaction matrix
    interaction_matrix = rk.interaction_matrix(path_input, motifName, total_nodes, nodeOrder_list)

    # # Import ICs
    ics = rk.import_ICs(path_input, motifName, nodes_grn)

    # # Output summary file of all motif data
    rk.summary_file(dir_out, motifName, total_nodes, number_grn, nodes_grn, number_ghost,
                    nodes_ghost, number_signal, nodes_signal, totalStates, nodeOrder_list,
                    ics, interaction_matrix, signal_node_edge, df_nodeLabels)

    # # Dataframe column headers
    # headers for time-evolution network behaviour dataframe
    df_raw_head = ['energy_sim', 'initial_condition', 'time_step', 'signal_state']
    df_raw_head.extend(list(nodes_grn))
    # headers for mean node activation dataframe
    df_stats_head = list(nodes_grn)
    df_stats_head[:0] = ['time_step', 'signal_state']

    # # Loop through each energy level
    for energy in energy_partition:
        # # fix probability threshold
        prob_threshold = rk.energy_prob_function(energy)

        # # create empty array, pre-allocate size and data type
        index_length = total_time*number_sims*len(ics)
        array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.int64)
        index_number = 0

        # # display simulation information
        print('\n%s: Motif: %s, Update rule: %s, Signal range: [%s, %s], Energy = %.2f, Energy Simulation = 0' %
              (dt.datetime.now().strftime('%H:%M:%S'), motifName, update_rule, sig_start, sig_end, energy))

        # # run numerous simulations per energy level
        for energy_sim in range(0, number_sims):
            # # display simulation information
            print('%s: Motif: %s, Update rule: %s, Signal range: [%s, %s], Energy = %.2f, Energy Simulation = %s'
                  % (dt.datetime.now().strftime('%H:%M:%S'), motifName, update_rule, sig_start, sig_end,
                     energy, energy_sim+1)) if (energy_sim+1) % 50000 == 0 else 'nothing'

            # # Reduce matrices & dataframes using probability-based condition
            sub_matrix = rk.sub_matrix(interaction_matrix, prob_threshold)
            # print('Sub-interaction matrix:\n%s' % str(sub_matrix))

            # # Loop through all initial conditions
            for globalState_0 in ics:
                # # integer conversion
                globalState_0_int = int(globalState_0, 2)

                # # modify state with signal & ghost nodes states on the end
                globalState_signal = rk.extend_state(globalState_0, len(nodes_signal), 0)
                globalState_extended = rk.extend_state(globalState_signal, len(nodes_ghost), 1)

                # # Add t=0 data to array
                array_raw_data[index_number] = rk.add_row(
                    energy_sim, globalState_0_int, 0, 0, globalState_0)
                index_number += 1

                # # Loop through full time range (per simulation)
                # print('\nTime-step, Signal state, global state, global state extended')
                # print('0, 0, %s, %s' % (globalState_0, globalState_extended))
                for timestep in range(1, total_time):
                    # # Modify signal node state if in pulse range
                    globalState_extended = \
                        rk.node_state_change(globalState_extended, len(nodes_grn), '1') \
                        if timestep in signal_range else globalState_extended[:]
                    # check state of signal
                    signal_state = globalState_extended[len(nodes_grn):len(nodes_grn)+1]

                    # # Update interaction matrix with signal regulation
                    # print('Signal State = %.0f' % int(signal_state))
                    if signal_state == '1':
                        sub_matrix_signal = rk.signal_node_edge_regulation_matrix(
                            sub_matrix, signal_node_edge, signal_node_edge_targets, nodeOrder_list, globalState_extended)
                    else:
                        sub_matrix_signal = sub_matrix.copy()
                    # print(sub_matrix_signal)

                    # # Updating of global state - Update N nodes
                    if update_method == 'async':
                        N = np.random.poisson(number_grn + number_signal)
                    elif update_method == 'sync':
                        N = number_grn + number_signal
                    else:
                        sys.exit("Error: Update method not known.")
                    # print(N)

                    if N == 0:
                        globalState_extended = globalState_extended[:]
                        globalState = globalState_extended[:len(nodes_grn)]
                    else:
                        globalState, globalState_extended = \
                            rk.update_state(globalState_extended, update_method, number_grn, number_signal,
                                            sub_matrix_signal, total_nodes, update_rule,  N)

                    if timestep not in signal_range[:-1]:
                        signal_state = globalState_extended[len(nodes_grn):len(nodes_grn)+1]
                    # # # Print data for time-step
                    # print('%.0f, %s, %s, %s' %
                    #       (timestep, signal_state, globalState, globalState_extended))

                    # # Add data to array
                    array_raw_data[index_number] = rk.add_row(
                        energy_sim, globalState_0_int, timestep, signal_state, globalState)
                    index_number += 1
        # print(array_raw_data)

        # # Create a Pandas DataFrame from the Numpy array of data & export as csv file
        df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:],
                                   index=range(0, index_length), columns=df_raw_head)
        full_filename = rk.export_df('timeseries-data-%.0f' % (float(energy)*100), dir_out,
                                     motifName, df_raw_data)

        # # Calculate mean "on" and standard deviation for each time-step
        for globalState_0 in ics:
            # # integer conversion
            globalState_0_int = int(globalState_0, 2)
            # globalState_0 = rk.get_bin(globalStateInt_0, len(nodes_grn))
            df_sub_raw1 = df_raw_data[df_raw_data.initial_condition ==
                                      globalState_0_int].reset_index(drop=True)  # .astype(float)
            # print(df_sub_raw1)

            # # dataframe of mean activtion per time-step
            df_mean = pd.DataFrame(index=[], columns=df_stats_head)
            # # dataframe of standard deviation for activtion per time-step
            df_std = pd.DataFrame(index=[], columns=df_stats_head)
            # # dataframe of cv
            df_cv = pd.DataFrame(index=[], columns=df_stats_head)

            for time_iter in range(0, total_time):
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
            df_mean_filename = 'timeseries-mean-%.0f-%s' % (float(energy)*100, globalState_0)
            rk.export_df(df_mean_filename, dir_out, motifName, df_mean)

            # # Export standard deviation per time-step dataframe
            df_std = df_std.fillna(0)
            df_std_filename = 'timeseries-std-%.0f-%s' % (float(energy)*100, globalState_0)
            rk.export_df(df_std_filename, dir_out, motifName, df_std)

            # # Calculate the cv (=standard deviation/mean) for each time-step and datframe column
            df_cv = df_std.div(df_mean)
            df_cv['time_step'] = df_std['time_step']

            # # Export cv per time-step dataframe
            # df_cv = df_cv.fillna(0)
            df_cv_filename = 'timeseries-cv-%.0f-%s' % (float(energy)*100, globalState_0)
            rk.export_df(df_cv_filename, dir_out, motifName, df_cv)

"""
Produce figure(s)
"""
for motifName in allMotifs:
    print('Producing figures.')

    # # Create directory for script outputs
    dir_out = rk.create_dir(update_rule, sig_start, sig_end, motifName,
                            len(signal_range), update_method, path_out)

    # # Calculate pre-simulation info and data
    # node-node and node-edge data
    node_node_data, node_edge_data = rk.regulation_data(path_input, motifName)
    # unique nodes
    signal_node_edge, nodes_unique = rk.unique_nodes(node_node_data, node_edge_data)
    # split unique nodes into groups
    nodes_grn, nodes_ghost, nodes_signal = rk.node_groups(nodes_unique)
    # unique edges targeted by signal
    signal_node_edge_targets = rk.signal_targets_unique(signal_node_edge)
    # calculate size of node groups
    number_grn, number_ghost, number_signal = rk.node_total(
        nodes_grn, nodes_ghost, nodes_signal, nodes_unique)
    # additional information about network nodes
    nodeOrder_list, df_nodeLabels, signal_labels = rk.node_labels(
        nodes_grn, nodes_signal, nodes_ghost)
    # Total number of nodes in wiring diagram
    total_nodes = number_grn + number_ghost + number_signal
    # Total number of global states (size of state space)
    totalStates = 2 ** number_grn
    # Node order
    node_labels_filename = 'node_order_labels'
    rk.export_df(node_labels_filename, dir_out, motifName, df_nodeLabels)

    # # Use GRN data to fill interaction matrix
    interaction_matrix = rk.interaction_matrix(path_input, motifName, total_nodes, nodeOrder_list)

    # # Import ICs
    ics = rk.import_ICs(path_input, motifName, nodes_grn)

    """Loop through energy steps and produce figures"""
    for energy in energy_partition:
        # # loop through 0 to totalGlobalStates-1 in chronological/random order
        for globalState_0 in ics:
            # # Import mean "on" per time-step dataframe
            df_mean_filename = '%s/%s-timeseries-mean-%.0f-%s.csv' % (
                dir_out, motifName, float(energy)*100, globalState_0)
            df_mean = pd.read_csv(df_mean_filename)

            # # Import standard deviation per time-step dataframe
            df_std_filename = '%s/%s-timeseries-std-%.0f-%s.csv' % (
                dir_out, motifName, float(energy)*100, globalState_0)
            df_std = pd.read_csv(df_std_filename)

            """Plot figures"""
            fig_timeSteps = total_time

            # # Plot figure of time-evolution subplots
            fig_timeEvolution = rk.time_evolution_fig(
                globalState_0, nodes_grn, len(nodes_grn), fig_timeSteps,
                df_mean, df_std)  # , df_binary_count)

            # # Save time-evolution figure
            fig_timeEvolutionFilename = "%s/%s-timeseries-%.0f-%.0f-%s.pdf" % (
                dir_out, motifName, float(energy)*100, len(signal_range), globalState_0)
            fig_timeEvolution.savefig(fig_timeEvolutionFilename, format='pdf',
                                      transparent=False, bbox_inches='tight')
            plt.close(fig_timeEvolution)                                                                                                                                       
