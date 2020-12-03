"""
Script to calculate transition matrices & state space diagram(s)
"""
# -*- coding: utf-8 - *-  # telling Python the source file you've saved is utf-8
"""Import modules"""
import numpy as np
import pandas as pd
import csv
import os
import datetime as dt
import matplotlib.pyplot as plt
import module_rk as rk

"""
Model Inputs
"""
# # Regulatory motifs to loop through
allMotifs = ["ecoli"]  # input('Enter regulatory network name:')
# # sync OR async
update_method = 'async'  # -- need to modify update_state to account for sync!!
# # 0 = blue, 1 = red
update_rule = 0
# # Signal state: 'active' or 'inactive'
signal_state = 'active'
# # Energy availability levels to scan through - can use set levels or a linspace.
energy_partition = np.array([0.1, 0.5, 1])
# # Number of simulations per energy level
number_sims = 2*10**2

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
    path_out = rk.create_hm_dir(signal_state, update_rule, motifName, update_method)

    """Input interaction matrix data using relative path of script"""
    path_in = '%s/grn-data' % os.path.dirname(path_parent)

    """Calculate pre-simulation info and data"""
    node_node_data, node_edge_data = rk.regulation_data(path_in, motifName)
    signal_node_edge, nodes_unique = rk.unique_nodes(node_node_data, node_edge_data)
    nodes_grn, nodes_ghost, nodes_signal = rk.node_groups(nodes_unique)
    signal_node_edge_targets = rk.signal_targets_unique(signal_node_edge)
    number_grn, number_ghost, number_signal = rk.node_total(
        nodes_grn, nodes_ghost, nodes_signal, nodes_unique)
    nodeOrder_list, df_nodeLabels, signal_labels = rk.node_labels(
        nodes_grn, nodes_signal, nodes_ghost)

    """Total number of nodes in wiring diagram"""
    total_nodes = number_grn + number_ghost + number_signal
    """Total number of global states (size of state space)"""
    totalStates = 2 ** number_grn
    """Node order"""
    node_labels_filename = 'node_order_labels'
    rk.export_df(node_labels_filename, path_out, motifName, df_nodeLabels)

    """Use GRN data to fill interaction matrix"""
    interaction_matrix = rk.interaction_matrix(path_in, motifName, total_nodes, nodeOrder_list)

    """Output summary file of all motif data"""
    rk.summary_file(path_out, motifName, total_nodes, number_grn, nodes_grn, number_ghost,
                    nodes_ghost, number_signal, nodes_signal, totalStates, nodeOrder_list)

    print([nodes_signal, nodes_ghost, nodes_grn, total_nodes])
    print(df_nodeLabels)
    print(nodeOrder_list)
    print(signal_labels)
    print(totalStates)
    print(signal_node_edge)
    print(signal_node_edge_targets)

    """Column headers for raw data dataframe"""
    df_raw_head = ['energy', 'state(t)', 'state(t+1)']

    """Create (zero) array for raw data, pre-allocate size and data type"""
    index_length = totalStates*number_sims*energy_partition.shape[0]
    array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.float64)
    index_number = 0

    """Loop through each energy level"""
    for energy in energy_partition:
        # # fix probability threshold
        prob_threshold = rk.energy_prob_function(energy)

        # # run numerous simulations per energy level
        for energy_simulation in range(0, number_sims):
            if (energy_simulation) % 100000 == 0:
                print('%s: Motif = %s, Update rule: %s, Signal: %s, Energy = %.2f, Probability Threshold = %.4f, Energy Simulation = %s' %
                      (dt.datetime.now().time(), motifName, update_rule, signal_state, energy, prob_threshold, energy_simulation))

            """Reduce matrices & dataframes using probability-based condition"""
            sub_matrix = rk.sub_matrix(interaction_matrix, prob_threshold)
            # print('Sub-interaction matrix:\n%s' % str(sub_matrix))

            # """Shuffle "totalStates" order"""
            # randomNodeRange = list(range(totalStates))
            # rn.shuffle(randomNodeRange)

            """Loop through all initial conditions in chronological/random order"""
            for globalState_0_int in range(totalStates):  # randomNodeRange:

                """Get binary form of "globalState_0_int" input; data class = string"""
                globalState_0 = rk.get_bin(globalState_0_int, len(nodes_grn))
                # print('Source State = [%.0f, %s]' % (globalState_0_int, globalState_0))

                """Modify state with signal & ghost nodes states on the end"""
                # print('Signal State = %.0f' % int(signal_state))
                if signal_state == 'active':
                    globalState_signal_0 = rk.extend_state(globalState_0, len(nodes_signal), 1)
                    globalState_extended_0 = rk.extend_state(
                        globalState_signal_0, len(nodes_ghost), 1)
                    """Update interaction matrix with signal regulation"""
                    sub_matrix_signal = rk.signal_node_edge_regulation_matrix(
                        sub_matrix, signal_node_edge, signal_node_edge_targets, nodeOrder_list, globalState_extended_0)
                elif signal_state == 'inactive':
                    globalState_signal_0 = rk.extend_state(globalState_0, len(nodes_signal), 0)
                    globalState_extended_0 = rk.extend_state(
                        globalState_signal_0, len(nodes_ghost), 1)
                    """Update interaction matrix with signal regulation"""
                    sub_matrix_signal = sub_matrix.copy()
                else:
                    sys.exit("\n\nError: Entered signal state is not known.")
                # print('Signal Extended State = ' + globalState_extended_0)
                # print('Interaction Matrix = \n' + str(sub_matrix_signal))

                """
                Asynchronous updating:
                Time-step definition: (i) Update X nodes (drawn from Poisson distribution with mean N)
                (ii) Choose random node, calc Boolean func sum and then update it using Boolean rule
                Note: Can choose the same node multiple times
                """

                """Updating of global state - Update N nodes"""
                if update_method == 'async':
                    N = np.random.poisson(number_grn + number_signal)
                elif update_method == 'sync':
                    N = number_grn + number_signal
                else:
                    sys.exit("Error: Update method not known.")
                # print(N)

                if N == 0:
                    globalState_extended_1 = globalState_extended_0[:]
                    globalState_1 = globalState_extended_1[:len(nodes_grn)]
                else:
                    globalState_1, globalState_extended_1 = \
                        rk.update_state(globalState_extended_0, update_method, number_grn, number_signal,
                                        sub_matrix_signal, total_nodes, update_rule,  N)

                globalState_1_int = int(globalState_1, 2)

                """Add data to array"""
                transition = [energy, globalState_0_int, globalState_1_int]
                array_raw_data[index_number] = transition
                index_number += 1
        # print(array_raw_data)

        # R = array_raw_data[(array_raw_data[:, 1] == 5) & (array_raw_data[:, 0] == 1)]

        """Create a Pandas DataFrame from the Numpy array of data & export as csv file"""
        # print(array_raw_data)
        df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:], index=range(
            0, index_length), columns=df_raw_head, dtype=float)
        full_filename = rk.export_df('hm-data', path_out, motifName, df_raw_data)

        """Calulate transition count for df at each energy level"""
        for energy in energy_partition:
            sub_df = df_raw_data.loc[df_raw_data['energy'] == energy,
                                     'state(t)':'state(t+1)'].reset_index(drop=True)

            dups = sub_df.groupby(sub_df.columns.tolist()).size(
            ).reset_index().rename(columns={0: 'count'})
            dups['count'] = dups['count'].div(number_sims)
            full_filename = rk.export_df('hm-count-%.0f' % (float(energy)*100),
                                         path_out, motifName, dups)
    # # ------------------------------------------------------------------------
    # # ------------------------------------------------------------------------
    print('Constructing figures.')
    """
    State Space Diagram
    """
    """Create blank state space diagram"""
    graphComment = "Asynchronous Energy Variability Boolean Network State Space Diagram for %s. Node label order = %s" % (
        motifName, nodeOrder_list[:-len(nodes_ghost)])
    stateSpaceDiagram = rk.stateSpaceDiagram_attributes(graphComment)

    """Add nodes"""
    rk.stateSpaceDiagram_nodes(stateSpaceDiagram, totalStates, len(nodes_grn))

    """Add edges"""
    for energy in energy_partition:
        df_fileName = "%s/%s-hm-count-%.0f.csv" % (path_out, motifName, float(energy)*100)
        rk.stateSpaceDiagram_edges(stateSpaceDiagram, df_fileName, energy)

        """
        Heatmaps
        """
        """Heatmap data"""
        data_hm = rk.hm_data(df_fileName, totalStates, len(nodes_grn))
        """Save dataframe"""
        df_fileName = "%s/%s-hm-%s-data-%.0f.csv" % (
            path_out, motifName, signal_state, float(energy)*100)
        data_hm.to_csv(df_fileName, index=True, header=True)

        """
        Clustered Heatmap and Row-Dendrogram figure
        """
        fig_hm_dendrogram = plt.figure(figsize=(4, 3.2))
        rk.dendrogram_and_heatmap(fig_hm_dendrogram, data_hm, len(nodes_grn), number_sims,
                                  cbarlabel="Transition Probability",
                                  interpolation='nearest')
        # # save heatmap figure
        dendrogramFilename = "%s/%s-hm-%s-%.0f.pdf" % (
            path_out, motifName, signal_state, float(energy)*100)
        fig_hm_dendrogram.savefig(dendrogramFilename, format='pdf',
                                  transparent=False, bbox_inches='tight')

    # # save COMPLETE state space diagram file & figure
    DOTFileName = "%s/%s-state-diag.gv" % (path_out, motifName)
    rk.save_stateSpaceDiagram(stateSpaceDiagram, DOTFileName)
