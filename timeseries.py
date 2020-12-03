"""
Script to calculate % of time-steps each node is 'on' for network ICs
Outputs data as a dataframe
Energy Variability with Asynchronous updating
Personal mac: /Users/ryankerr/opt/anaconda3/bin/python
"""
# -*- coding: utf-8 - *-  # telling Python the source file you've saved is utf-8
"""Import modules"""
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
signal_range, sig_start, sig_end = rk.signal_range(15, 17)

"""
Make directory of python file the current working directory
"""
path_file = os.path.abspath(__file__)
path_parent = os.path.dirname(path_file)
os.chdir(path_parent)
# print(os.getcwd())

"""
Data accumulation
"""
for motifName in allMotifs:

    # # Create directory for script outputs
    path_out = rk.create_dir(update_rule, sig_start, sig_end, motifName,
                             len(signal_range), update_method)

    # # Input interaction matrix data using relative path of script
    path_in = '%s/grn-data' % os.path.dirname(path_parent)

    # # Calculate pre-simulation info and data
    # node-node and node-edge data
    node_node_data, node_edge_data = rk.regulation_data(path_in, motifName)
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
    rk.export_df(node_labels_filename, path_out, motifName, df_nodeLabels)

    # # Use GRN data to fill interaction matrix
    interaction_matrix = rk.interaction_matrix(path_in, motifName, total_nodes, nodeOrder_list)

    # # Output summary file of all motif data
    rk.summary_file(path_out, motifName, total_nodes, number_grn, nodes_grn, number_ghost,
                    nodes_ghost, number_signal, nodes_signal, totalStates, nodeOrder_list)

    # # Dataframe column headers
    # headers for time-evolution network behaviour dataframe
    df_raw_head = ['energy_simulation', 'initial_condition', 'time_step', 'signal_state']
    df_raw_head.extend(list(nodes_grn))
    # headers for mean node activation dataframe
    df_stats_head = list(nodes_grn)
    df_stats_head[:0] = ['time_step', 'signal_state']

    # # Import ICs
    ics_data_dir = '%s/initial-condition-files' % os.path.dirname(path_parent)
    ics = rk.import_ICs(ics_data_dir, motifName, nodes_grn)
    print(ics)

    # # Display information for considered motif
    print('Motif: %s' % str(motifName))
    print('Node order: %s' % str(nodes_grn))
    print('ICs: \n%s\n' % str(ics))
    print('Initial Interaction Matrix: \n%s' % str(interaction_matrix))
    print('Signal Interactions: \n%s\n' % str(signal_node_edge_targets))
    print('Unique Signal Interaction Pairs: \n%s\n' % str(signal_node_edge_targets))

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
        for energy_simulation in range(0, number_sims):
            if (energy_simulation+1) % 50000 == 0:
                # # display simulation information
                print('\n %s: Motif: %s, Update rule: %s, Signal range: [%s, %s], Energy = %.2f, Energy Simulation = %s'
                      % (dt.datetime.now().strftime('%H:%M:%S'), motifName, update_rule, sig_start, sig_end,
                         energy, energy_simulation+1))

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
                # print(globalState_extended)

                # # Add t=0 data to array
                array_raw_data[index_number] = rk.add_row(
                    energy_simulation, globalState_0_int, 0, 0, globalState_0)
                index_number += 1

                # # Loop through full time range (per simulation)
                # print('\nTime-step, Signal state, global state, global state extended')
                # print('0, 0, %s, %s' % (globalState_0, globalState_extended))
                for timestep in range(1, total_time):
                    # print('Time-step: %.0f. Pre-update global state: %s' %
                    #       (timestep, globalState_extended))

                    # # Modify signal node state if in pulse range
                    """CAN this be done better?"""
                    if timestep in signal_range:
                        globalState_extended = rk.node_state_change(globalState_extended,
                                                                    len(nodes_grn), '1')
                        # print('Time-step: %.0f. Pre-update global state: %s' %
                        #       (timestep, globalState_extended))
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

# #                     # """
# #                     # Print data for time-step
# #                     # """
# #                     # print('%.0f, %s, %s, %s' %
# #                     #       (timestep, signal_state, globalState, globalState_extended))
# #
                    # # Add data to array
                    array_raw_data[index_number] = rk.add_row(
                        energy_simulation, globalState_0_int, timestep, signal_state, globalState)
                    index_number += 1

        # # Create a Pandas DataFrame from the Numpy array of data & export as csv file
        # print(array_raw_data)
        df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:],
                                   index=range(0, index_length), columns=df_raw_head)
        full_filename = rk.export_df('timeseries-data-%.0f' % (float(energy)*100), path_out,
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
            rk.export_df(df_mean_filename, path_out, motifName, df_mean)

            # # Export standard deviation per time-step dataframe
            df_std = df_std.fillna(0)
            df_std_filename = 'timeseries-std-%.0f-%s' % (float(energy)*100, globalState_0)
            rk.export_df(df_std_filename, path_out, motifName, df_std)

            # # Calculate the cv (=standard deviation/mean) for each time-step and datframe column
            df_cv = df_std.div(df_mean)
            df_cv['time_step'] = df_std['time_step']

            # # Export cv per time-step dataframe
            # df_cv = df_cv.fillna(0)
            df_cv_filename = 'timeseries-cv-%.0f-%s' % (float(energy)*100, globalState_0)
            rk.export_df(df_cv_filename, path_out, motifName, df_cv)

"""
Produce figure(s)
"""
for motifName in allMotifs:
    print('Producing figures.')

    # # Create directory for script outputs
    path_out = rk.create_dir(update_rule, sig_start, sig_end, motifName,
                             len(signal_range), update_method)

    # # Input interaction matrix data using relative path of script
    path_in = '%s/grn-data' % os.path.dirname(path_parent)

    """Calculate pre-simulation info and data"""
    # # node-node and node-edge data
    node_node_data, node_edge_data = rk.regulation_data(path_in, motifName)
    # # unique nodes
    signal_node_edge, nodes_unique = rk.unique_nodes(node_node_data, node_edge_data)
    # # split unique nodes into groups
    nodes_grn, nodes_ghost, nodes_signal = rk.node_groups(nodes_unique)
    # # unique edges targeted by signal
    signal_node_edge_targets = rk.signal_targets_unique(signal_node_edge)
    # # calculate size of node groups
    number_grn, number_ghost, number_signal = rk.node_total(
        nodes_grn, nodes_ghost, nodes_signal, nodes_unique)
    # # additional information about network nodes
    nodeOrder_list, df_nodeLabels, signal_labels = rk.node_labels(
        nodes_grn, nodes_signal, nodes_ghost)
    # # Total number of nodes in wiring diagram
    total_nodes = number_grn + number_ghost + number_signal
    # # Total number of global states (size of state space)
    totalStates = 2 ** number_grn
    # # Node order
    node_labels_filename = 'node_order_labels'
    rk.export_df(node_labels_filename, path_out, motifName, df_nodeLabels)

    # # Use GRN data to fill interaction matrix
    interaction_matrix = rk.interaction_matrix(path_in, motifName, total_nodes, nodeOrder_list)

    # # Import ICs
    ics_data_dir = '%s/initial-condition-files' % os.path.dirname(path_parent)
    ics = rk.import_ICs(ics_data_dir, motifName, nodes_grn)

    """Loop through energy steps and produce figures"""
    for energy in energy_partition:
        # # loop through 0 to totalGlobalStates-1 in chronological/random order
        for globalState_0 in ics:
            # # Import mean "on" per time-step dataframe
            df_mean_filename = '%s/%s-timeseries-mean-%.0f-%s.csv' % (
                path_out, motifName, float(energy)*100, globalState_0)
            df_mean = pd.read_csv(df_mean_filename)

            # # Import standard deviation per time-step dataframe
            df_std_filename = '%s/%s-timeseries-std-%.0f-%s.csv' % (
                path_out, motifName, float(energy)*100, globalState_0)
            df_std = pd.read_csv(df_std_filename)

            """Plot figures"""
            fig_timeSteps = total_time

            # # Plot figure of time-evolution subplots
            fig_timeEvolution = rk.time_evolution_fig(
                globalState_0, nodes_grn, len(nodes_grn), fig_timeSteps,
                df_mean, df_std)  # , df_binary_count)

            # # Save time-evolution figure
            fig_timeEvolutionFilename = "%s/%s-timeseries-%.0f-%.0f-%s.pdf" % (
                path_out, motifName, float(energy)*100, len(signal_range), globalState_0)
            fig_timeEvolution.savefig(fig_timeEvolutionFilename, format='pdf',
                                      transparent=False, bbox_inches='tight')
            plt.close(fig_timeEvolution)                                                                                                                                       
