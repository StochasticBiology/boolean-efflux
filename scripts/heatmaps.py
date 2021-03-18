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
import sys
import datetime as dt
import matplotlib.pyplot as plt
import module_rk as rk
import argparse

# # Model Inputs
parser = argparse.ArgumentParser()
parser.add_argument("motif", type=str)  # Regulatory motifs to loop through
parser.add_argument("signal_status", type=str)
parser.add_argument("length_index", type=int)
args, unknown = parser.parse_known_args()
print(args)

print('Preamble')
# Energy availability levels to scan through - can use set levels or a linspace.
energy_levels = np.array([0.1, 0.5, 1])  # Energy availability levels to scan through
# Number of simulations per energy level
number_sims = 2*10**args.length_index
# signal features
if args.signal_status == "True":
    signal_state = 'active'
if args.signal_status == "False":
    signal_state = 'inactive'

# input and output locations
pathIn, pathOut = rk.pathOut(__file__)
# # Create directory for script outputs
dir_out = rk.create_hm_dir(signal_state, args.motif, pathOut)

# # Regulatory network info & boolean modelling prerequisites
# signal regulation & unique nodes
signal_node_edge, nodes_unique = rk.unique_nodes(pathIn, args.motif)
# split unique nodes into groups
nodes_grn, nodes_ghost, nodes_signal, total_nodes, number_ghost, number_signal, number_grn = \
    rk.node_groups(nodes_unique)
# unique edges targeted by signal
signal_node_edge_targets = rk.signal_targets_unique(signal_node_edge)
# additional information about network nodes
nodeOrder_list, df_nodeLabels, signal_labels = rk.node_labels(nodes_grn, nodes_signal, nodes_ghost)
# total number of global states (size of state space)
totalStates = 2 ** number_grn
# # fill interaction matrix
interactions = rk.interaction_matrix(pathIn, args.motif, total_nodes, nodeOrder_list)
# # set of ics to simulate
ics = [rk.get_bin(i, len(nodes_grn)) for i in range(totalStates)]

# # output summary file of all args.motif data
rk.summary_file(dir_out, args.motif, total_nodes, number_grn, nodes_grn, number_ghost,
                nodes_ghost, number_signal, nodes_signal, totalStates,
                ics, interactions, signal_node_edge, df_nodeLabels)

# # create empty array, pre-allocate size and data type
df_raw_head = ['energy', 'state(t)', 'state(t+1)']  #  dataframe column headers
index_length = totalStates*number_sims*energy_levels.shape[0]
array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.float64)
index_number = 0

print('Simulations.')
for energy in energy_levels:  # Loop through each energy level
    prob_threshold = rk.energy_prob_function(energy)  # fix probability threshold

    for energy_sim in range(0, number_sims):  # run numerous simulations per energy level
        if energy_sim % 100000 == 0:
            # # similation progress file
            progressFile = open('%s/simulation-progress.txt' % (pathOut), 'a')
            line = '%s [heatmaps]: %s, signal %s, energy level %.2f, energy simulation %s' \
                % (dt.datetime.now().strftime('%H:%M:%S'), args.motif, signal_state, energy, energy_sim)
            progressFile.write("%s\n" % line)
            # close progress file
            progressFile.close()

        # # Reduce matrices & dataframes using probability-based condition
        submatrix = rk.submatrix(interactions, prob_threshold)

        for globalState_0_int in range(totalStates):  # Loop through all initial conditions
            globalState_0 = rk.get_bin(globalState_0_int, len(nodes_grn))  # binary form

            # # modify state with signal & ghost nodes states on the end
            if signal_state == 'active':
                globalState_signal_0 = rk.extend_state(globalState_0, len(nodes_signal), 1)
                globalState_extended_0 = rk.extend_state(
                    globalState_signal_0, len(nodes_ghost), 1)

                # # Update interaction matrix with signal regulation
                submatrix_signal = rk.signal_node_edge_regulation_matrix(
                    submatrix, signal_node_edge, signal_node_edge_targets, nodeOrder_list, globalState_extended_0)

            elif signal_state == 'inactive':
                globalState_signal_0 = rk.extend_state(globalState_0, len(nodes_signal), 0)
                globalState_extended_0 = rk.extend_state(
                    globalState_signal_0, len(nodes_ghost), 1)
                # Update interaction matrix with signal regulation
                submatrix_signal = submatrix.copy()

            else:
                sys.exit("\n\nError: Entered signal state is not known.")

            # # Updating of global state
            # update N nodes
            N = np.random.poisson(number_grn + number_signal)
            if N == 0:
                globalState_extended_1 = globalState_extended_0[:]
                globalState_1 = globalState_extended_1[:len(nodes_grn)]
            else:
                globalState_1, globalState_extended_1 = \
                    rk.update_state(globalState_extended_0, number_grn, number_signal,
                                    submatrix_signal, total_nodes,  N)

            # # Integer form of succeeding global state
            globalState_1_int = int(globalState_1, 2)

            # # Add data to array
            transition = [energy, globalState_0_int, globalState_1_int]
            array_raw_data[index_number] = transition
            index_number += 1

# # Create a Pandas DataFrame from the Numpy array of data & export as csv file
df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:], index=range(
    0, index_length), columns=df_raw_head, dtype=float)
# full_filename = rk.export_df('hm-data', dir_out, args.motif, df_raw_data)

# # Calulate transition count for dataframe at each energy level
for energy in energy_levels:
    sub_df = df_raw_data.loc[df_raw_data['energy'] == energy,
                             'state(t)':'state(t+1)'].reset_index(drop=True)

    dups = sub_df.groupby(sub_df.columns.tolist()).size(
    ).reset_index().rename(columns={0: 'count'})
    dups['count'] = dups['count'].div(number_sims)
    full_filename = rk.export_df('hm-%s-count-energy=%.0f' % (signal_state, float(energy)*100),
                                 dir_out, args.motif, dups)
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
print('Constructing figures.')

# # Heatmaps
for energy in energy_levels:
    # Transition data
    df_fileName = "%s/%s-hm-%s-count-energy=%.0f.csv" % (
        dir_out, args.motif, signal_state, float(energy)*100)
    data_hm = rk.hm_data(df_fileName, totalStates, len(nodes_grn))
    # Save dataframe
    df_fileName = "%s/%s-hm-%s-data-energy=%.0f.csv" % (
        dir_out, args.motif, signal_state, float(energy)*100)
    data_hm.to_csv(df_fileName, index=True, header=True)

    # # Produce heatmap and clustering figure & save
    fig_hm = plt.figure(figsize=(4, 3.2))
    rk.dendrogram_and_heatmap(fig_hm, data_hm, len(nodes_grn), number_sims,
                              cbarlabel="Transition Probability",
                              interpolation='nearest')
    dendrogramFilename = "%s/%s-hm-%s-energy=%.0f.pdf" % (
        dir_out, args.motif, signal_state, float(energy)*100)
    fig_hm.savefig(dendrogramFilename, format='pdf', transparent=False, bbox_inches='tight')
