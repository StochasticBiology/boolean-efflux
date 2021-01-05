# -*- coding: utf-8 - *-  # telling Python the source file you've saved is utf-8
"""
Script calculates transition matrices for network architecture.
Outputs data and figures (heatmaps & state space diagram).
"""

# # Import modules
import numpy as np
import pandas as pd
import csv
import os
import datetime as dt
import matplotlib.pyplot as plt
import module_rk as rk

# # Model Inputs
# Regulatory motifs to loop through
motifName = input("Enter regulatory network name:")
# Signal state: 'active' or 'inactive'
signal_state = input("Enter signal state (active or inactive):")
# Energy availability levels to scan through - can use set levels or a linspace.
energy_partition = np.array([0.1, 0.5, 1])
# Number of simulations per energy level
number_sim = input("Enter total simulations:")  # 2*10**6
number_sims = int(number_sim)

# # Path to directory locations (input & output)
# Path to python file
path_file = os.path.abspath(__file__)
# Directory for scripts
path_scripts = os.path.dirname(path_file)
# Directory for script inputs: wiring diagram information & ICs
path_input = '%s/input-data' % os.path.dirname(path_scripts)
# Directory for script output files & directories
path_out = os.path.dirname(path_scripts)
# Create directory for script outputs
dir_out = rk.create_hm_dir(signal_state, motifName, path_out)

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

# # full set of ics
ics = [rk.get_bin(i, len(nodes_grn)) for i in range(totalStates)]

# # Output summary file of all motif data
rk.summary_file(dir_out, motifName, total_nodes, number_grn, nodes_grn, number_ghost,
                nodes_ghost, number_signal, nodes_signal, totalStates, nodeOrder_list,
                ics, interaction_matrix, signal_node_edge, df_nodeLabels)

# # Dataframe column headers
df_raw_head = ['energy', 'state(t)', 'state(t+1)']

# # create empty array, pre-allocate size and data type
index_length = totalStates*number_sims*energy_partition.shape[0]
array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.float64)
index_number = 0

# # Loop through each energy level
for energy in energy_partition:
    # # fix probability threshold
    prob_threshold = rk.energy_prob_function(energy)

    # # run numerous simulations per energy level
    for energy_sim in range(0, number_sims):
        if (energy_sim) % 100000 == 0:
            print('%s: Motif = %s, Signal: %s, Energy = %.2f, Probability Threshold = %.4f, Energy Simulation = %s' %
                  (dt.datetime.now().time(), motifName, signal_state, energy, prob_threshold, energy_sim))

        # # Reduce matrices & dataframes using probability-based condition
        sub_matrix = rk.sub_matrix(interaction_matrix, prob_threshold)
        # print('Sub-interaction matrix:\n%s' % str(sub_matrix))

        # """Shuffle "totalStates" order"""
        # randomNodeRange = list(range(totalStates))
        # rn.shuffle(randomNodeRange)

        # # Loop through all initial conditions
        for globalState_0_int in range(totalStates):  # randomNodeRange:

            # # Get binary form of "globalState_0_int" input; data class = string"""
            globalState_0 = rk.get_bin(globalState_0_int, len(nodes_grn))
            # print('Source State = [%.0f, %s]' % (globalState_0_int, globalState_0))

            # # modify state with signal & ghost nodes states on the end
            # print('Signal State = %.0f' % int(signal_state))
            if signal_state == 'active':
                globalState_signal_0 = rk.extend_state(globalState_0, len(nodes_signal), 1)
                globalState_extended_0 = rk.extend_state(
                    globalState_signal_0, len(nodes_ghost), 1)
                # # Update interaction matrix with signal regulation
                sub_matrix_signal = rk.signal_node_edge_regulation_matrix(
                    sub_matrix, signal_node_edge, signal_node_edge_targets, nodeOrder_list, globalState_extended_0)
            elif signal_state == 'inactive':
                globalState_signal_0 = rk.extend_state(globalState_0, len(nodes_signal), 0)
                globalState_extended_0 = rk.extend_state(
                    globalState_signal_0, len(nodes_ghost), 1)
                # Update interaction matrix with signal regulation
                sub_matrix_signal = sub_matrix.copy()
            else:
                sys.exit("\n\nError: Entered signal state is not known.")
            # print('Signal Extended State = ' + globalState_extended_0)
            # print('Interaction Matrix = \n' + str(sub_matrix_signal))

            # # Updating of global state - Update N nodes
            N = np.random.poisson(number_grn + number_signal)
            # print(N)
            if N == 0:
                globalState_extended_1 = globalState_extended_0[:]
                globalState_1 = globalState_extended_1[:len(nodes_grn)]
            else:
                globalState_1, globalState_extended_1 = \
                    rk.update_state(globalState_extended_0, number_grn, number_signal,
                                    sub_matrix_signal, total_nodes,  N)

            # # Integer form of global state
            globalState_1_int = int(globalState_1, 2)

            # # Add data to array
            transition = [energy, globalState_0_int, globalState_1_int]
            array_raw_data[index_number] = transition
            index_number += 1
# print(array_raw_data)

# # Create a Pandas DataFrame from the Numpy array of data & export as csv file
df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:], index=range(
    0, index_length), columns=df_raw_head, dtype=float)
full_filename = rk.export_df('hm-data', dir_out, motifName, df_raw_data)

# # Calulate transition count for dataframe at each energy level
for energy in energy_partition:
    sub_df = df_raw_data.loc[df_raw_data['energy'] == energy,
                             'state(t)':'state(t+1)'].reset_index(drop=True)

    dups = sub_df.groupby(sub_df.columns.tolist()).size(
    ).reset_index().rename(columns={0: 'count'})
    dups['count'] = dups['count'].div(number_sims)
    full_filename = rk.export_df('hm-count-%.0f' % (float(energy)*100),
                                 dir_out, motifName, dups)
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
print('Constructing figures.')

# # State Space Diagram
# Create blank state space diagram"
graphComment = "Asynchronous Energy Variability Boolean Network State Space Diagram for %s. Node label order = %s" % (
    motifName, nodeOrder_list[:-len(nodes_ghost)])
stateSpaceDiagram = rk.stateSpaceDiagram_attributes(graphComment)
# Add nodes to state space diagram
rk.stateSpaceDiagram_nodes(stateSpaceDiagram, totalStates, len(nodes_grn))
for energy in energy_partition:
    # Add edges to state space diagram
    df_fileName = "%s/%s-hm-count-%.0f.csv" % (dir_out, motifName, float(energy)*100)
    rk.stateSpaceDiagram_edges(stateSpaceDiagram, df_fileName, energy)

    # # Heatmap figure
    # Heatmap data
    data_hm = rk.hm_data(df_fileName, totalStates, len(nodes_grn))
    # Save dataframe
    df_fileName = "%s/%s-hm-%s-data-%.0f.csv" % (
        dir_out, motifName, signal_state, float(energy)*100)
    data_hm.to_csv(df_fileName, index=True, header=True)
    # Create heatmap and clustering figure
    fig_hm_dendrogram = plt.figure(figsize=(4, 3.2))
    rk.dendrogram_and_heatmap(fig_hm_dendrogram, data_hm, len(nodes_grn), number_sims,
                              cbarlabel="Transition Probability",
                              interpolation='nearest')
    # save heatmap figure
    dendrogramFilename = "%s/%s-hm-%s-%.0f.pdf" % (
        dir_out, motifName, signal_state, float(energy)*100)
    fig_hm_dendrogram.savefig(dendrogramFilename, format='pdf',
                              transparent=False, bbox_inches='tight')

# # save COMPLETE state space diagram file & figure
DOTFileName = "%s/%s-state-diag.gv" % (dir_out, motifName)
rk.save_stateSpaceDiagram(stateSpaceDiagram, DOTFileName)
