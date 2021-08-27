# -*- coding: utf-8 - *-  # telling Python the source file you've saved is utf-8
"""Calculate transition matrices for network architecture."""

# # Import modules
import csv
import os
import argparse
import sys
import random as rn
import datetime as dt
import numpy as np
import pandas as pd
import netInfo as ni
import matplotlib.pyplot as plt


def energy_prob_function(n):
    """Determine prob threshold using energy level"""
    return 1 * n  # (1+np.exp(8-16*n)) ** (-1)


def get_bin(x, n):  # convert integer to binary format
    """Get the binary representation of x"""
    return format(x, 'b').zfill(n)


def export_df(fileName, outputDirectory, motif, dataframe):  # export dataframe
    """Export dataframe to csv file format"""
    complete_filename = "%s/%s-%s.csv" % (outputDirectory, motif, fileName)
    dataframe.to_csv(complete_filename, index=False, header=True)
    return complete_filename


def add_row(energy_sim, globalState_0_int, time_step, signal_state, globalState_1):
    """Add array row data"""
    row = [energy_sim+1, globalState_0_int, time_step, signal_state]
    row.extend(list(map(int, globalState_1)))
    return row


def extend_state(globalState, extension_length, extension_state):  # append global state w. additional states
    """Create binary string to add to end of global state."""
    extension_list = ['%s' % extension_state for i in range(extension_length)]
    globalState_extension = ''.join(extension_list)
    globalState_extended = ''.join((globalState, globalState_extension))
    return globalState_extended


def SIM(interactionMatrix, data_dir, motifName, probThreshold, number_ghost, coupledInteractionsPair, coupledInteractionsInd, nodeOrder_list):
    """Sub-interaction matrix"""
    # # Import Data & make a copy of interaction matrix
    df_data = pd.read_csv('%s/%s-regulation-nodes.csv' % (data_dir, motifName), header=0)
    sub_matrix = interactionMatrix.copy()
    matrix_indices = [i for i in range(interactionMatrix.shape[0])]

    # # cycle through coupled interactions
    for item in coupledInteractionsPair:
        randNum = rn.uniform(0, 1)
        for i in np.array([0, 1]):
            node_start = nodeOrder_list.index(df_data.iloc[item[i], 0])
            node_end = nodeOrder_list.index(df_data.iloc[item[i], 1])
            sub_matrix[node_end][node_start] = sub_matrix[node_end][node_start] \
                if randNum <= probThreshold else 0
    # # temporary removal of couple interaction rows in dataframe
    df_temp = df_data.drop(df_data.index[coupledInteractionsInd], 0)

    # # cycle through non-coupled interactions
    for index, row in df_temp.iterrows():
        randNum = rn.uniform(0, 1)
        node_start = nodeOrder_list.index(row[0])
        node_end = nodeOrder_list.index(row[1])
        sub_matrix[node_end][node_start] = sub_matrix[node_end][node_start] \
            if randNum <= probThreshold else 0
    return sub_matrix


def SIM_signal(interaction_matrix, node_edge_data, signal_node_edge_unique, nodeOrder_list, globalState):
    """Modulate interactions based on signal state"""
    if len(signal_node_edge_unique.index) == 0:
        return interaction_matrix
    else:
        sub_interactionMatrix = interaction_matrix.copy()
        col_headers = node_edge_data.columns.values.tolist()
        index_regulation = col_headers.index('regulation')

        if len(node_edge_data.index) == len(signal_node_edge_unique.index):
            for row in range(node_edge_data.shape[0]):
                node_from_int = nodeOrder_list.index(node_edge_data.iloc[row, 1])
                node_to_int = nodeOrder_list.index(node_edge_data.iloc[row, 2])
                if node_edge_data.iloc[row, 3] < 0:
                    sub_interactionMatrix[node_to_int][node_from_int] = 0
                else:
                    sub_interactionMatrix[node_to_int][node_from_int] == sub_interactionMatrix[node_to_int][node_from_int]
        else:
            for item in range(signal_node_edge_unique.shape[0]):
                node_from = signal_node_edge_unique.iloc[item, 0]
                node_to = signal_node_edge_unique.iloc[item, 1]
                node_from_int = nodeOrder_list.index(node_from)
                node_to_int = nodeOrder_list.index(node_to)

                sub_df = node_edge_data.loc[(node_edge_data['target edge start'] == node_from)
                                            & (node_edge_data['target edge end'] == node_to), :]
                booleanSum = np.array([sub_df.iloc[df_row, index_regulation]*int(globalState[nodeOrder_list.index(sub_df.iloc[df_row, 0])])
                                       for df_row in range(sub_df.shape[0])]).sum()

                # # Modify matrix element value if condition met
                if np.sign(booleanSum) < 0:
                    sub_interactionMatrix[node_to_int][node_from_int] == 0
                else:
                    sub_interactionMatrix[node_to_int][node_from_int] == sub_interactionMatrix[node_to_int][node_from_int]
        return sub_interactionMatrix


def UpdateAsync(extendedState_0, matrix, totGroups, nodeOrder_list, coupledGenesIndexPair):
    """Boolean modelling asynchronous updating of nodes"""
    state_0 = extendedState_0[:totGroups[2]]

    # # Updating of global state - Update N nodes
    N = np.random.poisson(totGroups[2] + totGroups[1])
    if N == 0:
        extendedState_1 = extendedState_0[:]
        state_1 = state_0[:]
    else:
        state_0_list = list(extendedState_0)  # # convert to list
        for randomNode in range(0, N):
            random_node = rn.randint(0, (totGroups[2] + totGroups[1]-1))
            if random_node in [item for sublist in coupledGenesIndexPair for item in sublist]:
                for item in coupledGenesIndexPair:
                    getPair = item if random_node in item else None

                    # # Calculate Boolean function value
                    booleanSum = np.array([matrix[random_node][j]*int(state_0_list[j]) for j in range(totGroups[3])]).sum()

                    for node in getPair:
                        # # Threshold update function
                        # node remains on if a positive sum of inputs
                        if (booleanSum > 0):
                            state_0_list[node] = "1"
                        # entitiy remains in same state if overall input is zero
                        elif (booleanSum == 0):
                            state_0_list[node] = state_0_list[node]
                        # if the sum doesn't satisfy any criteria above then the entity is turned off
                        else:
                            state_0_list[node] = "0"
            else:
                # # Calculate Boolean function value
                booleanSum = np.array([matrix[random_node][j]*int(state_0_list[j]) for j in range(totGroups[3])]).sum()

                # # Threshold update function
                # node remains on if a positive sum of inputs
                if (booleanSum > 0):
                    state_0_list[random_node] = "1"
                # entitiy remains in same state if overall input is zero
                elif (booleanSum == 0):
                    state_0_list[random_node] = state_0_list[random_node]
                # if the sum doesn't satisfy any criteria above then the entity is turned off
                else:
                    state_0_list[random_node] = "0"
        extendedState_1 = "".join(state_0_list)
        state_1 = extendedState_1[:totGroups[2]]
    return state_1, extendedState_1


def UpdateSync(extendedState_0, matrix, totGroups, nodeOrder_list, coupledGenesIndexPair):
    """Boolean modelling synchronous updating of nodes"""
    # # convert to list
    state_0_list = list(extendedState_0)

    for node in range(totGroups[2] + totGroups[1]):
        if node in [item for sublist in coupledGenesIndexPair for item in sublist]:
            for item in coupledGenesIndexPair:
                getPair = item if node in item else None

                # # Calculate Boolean function value
                booleanSum = np.array([matrix[node][j]*int(state_0_list[j]) for j in range(totGroups[3])]).sum()
                # print("Boolean sum: %s" % booleanSum)

                for item in getPair:
                    # # Threshold update function
                    # node remains on if a positive sum of inputs
                    if (booleanSum > 0):
                        state_0_list[item] = "1"
                    # entitiy remains in same state if overall input is zero
                    elif (booleanSum == 0):
                        state_0_list[item] = state_0_list[item]
                    # if the sum doesn't satisfy any criteria above then the entity is turned off
                    else:
                        state_0_list[item] = "0"
        else:
            # # Calculate Boolean function value
            booleanSum = np.array([matrix[node][j]*int(state_0_list[j]) for j in range(totGroups[3])]).sum()
            # print("Boolean sum: %s" % booleanSum)

            # # Threshold update function
            # node remains on if a positive sum of inputs
            if (booleanSum > 0):
                state_0_list[node] = "1"
            # entitiy remains in same state if overall input is zero
            elif (booleanSum == 0):
                state_0_list[node] = state_0_list[node]
            # if the sum doesn't satisfy any criteria above then the entity is turned off
            else:
                state_0_list[node] = "0"

    extendedState_1 = "".join(state_0_list)
    state_1 = extendedState_1[:totGroups[2]]
    return state_1, extendedState_1


def hm_data(transitonsFile, ICs_total, numberNodes):
    """
    Produce a matrix to use for heatmaps and dendrograms.
    """
    # # data import
    df = pd.read_csv(transitonsFile)
    dfRows = df.shape[0]

    # # Empty array
    dataFile = [[0] * len(ICs_total) for rows in range(len(ICs_total))]

    # # Fill array with transition data from dataframe
    for index, hm_data_row in df.iterrows():
        hm_start_state = int(ICs_total.index(hm_data_row[0]))
        hm_end_state = int(ICs_total.index(hm_data_row[1]))
        hm_probability = hm_data_row[2]
        dataFile[hm_start_state][hm_end_state] = hm_probability

    # # Add column & index labels
    globalState_names = [get_bin(_, numberNodes) for _ in ICs_total]
    hm_data = pd.DataFrame(dataFile, index=globalState_names, columns=globalState_names)
    return hm_data


def dendrogram_and_heatmap(figure, data, numberNetworkNodes, length_index, cbarlabel="", cbar_kw={}, **kwargs):
    import matplotlib as mpl
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import dendrogram
    from scipy.cluster import hierarchy
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
    from matplotlib.colors import LogNorm
    import seaborn as sns
    import math
    """Heatmap figure with dendrogram"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    # # dendrogram
    # add dendrogram axis - values from the left to right are left, bottom, width, and height
    dendrogramAxisPositionLeft = -0.02 - (0.0145*numberNetworkNodes)
    axd = figure.add_axes([dendrogramAxisPositionLeft, 0.33, 0.06, 0.4])
    # make dendrogram multicoloured (1)
    hierarchy.set_link_color_palette(['k', 'g', 'r', 'c', 'm', 'y'])
    # calculate row clusters
    row_clusters = linkage(data.values, method='complete', metric='euclidean')
    # dendrogram / tree
    with plt.rc_context({'lines.linewidth': 0.3}):
        row_dendr = dendrogram(row_clusters, orientation='left', ax=axd, color_threshold=np.inf)
    # row clustered data
    data_rowclust = data.iloc[row_dendr['leaves'][::-1]]
    # dendrogram axis attributes
    axd.set_xticks([np.sqrt(2), 0])
    axd.set_xlabel('Euclidean distance\nbetween destination\nprobability sets', size=6)
    xticklabels = [r'$\sqrt{2}$', 0]
    axd.set_xticklabels(xticklabels, size=7)
    axd.set_yticklabels(list(data_rowclust.index)[::-1], size=7)
    axd.tick_params(axis='y', which='both', pad=-2)
    axd.tick_params(axis="x", width=0.4)
    plt.xlim(np.sqrt(2), 0)
    plt.grid(b=None, which='major', axis='x', ls='--', alpha=0.7)
    # remove axes spines from dendrogram
    for ax_spines in axd.spines.values():
        ax_spines.set_visible(False)

    # # Plot heatmap sub-figure
    axm = figure.add_axes([0.17, 0.1, 0.4, 0.76])  # x-pos, y-pos, width, height
    data_thresh = 1/10**length_index
    data_thresh_scientific = "{:.1e}".format(data_thresh)
    data[data < data_thresh] = np.nan

    v1 = float('1e-%.0f' % length_index)
    v2 = 1
    cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(v1)), 1+math.ceil(math.log10(v2)))]
    heat_map = sns.heatmap(data, norm=LogNorm(vmin=v1, vmax=v2), cmap=plt.cm.viridis, square=True, \
        linecolor='black', linewidths=0.15, cbar_kws={"shrink": .62, "ticks": cbar_ticks})

    cbar = axm.collections[0].colorbar
    cbar.ax.yaxis.set_ticks([], minor=True)
    cbar.ax.tick_params(labelsize=7, pad=0.3, width=0.8, length=2.2)
    cbar.set_label(cbarlabel, labelpad=8, rotation=-90, size=7)

    # # axis properties
    axesLabelLength = len(list(data.columns))
    axesTicksPositions = np.arange(0.5, axesLabelLength+0.5, 1)

    axm.xaxis.set_ticks_position('top')  # move x-ticks and labels to the top of heatmap
    axm.tick_params(axis="both", width=0.4, length=1.7, pad=0.8)
    axm.set_xlim(0, axesLabelLength)  # xlim(left, right)
    axm.set_xticks(axesTicksPositions)
    axm.set_xticklabels(list(data.columns), rotation=90, size=7)

    axm.set_ylim(axesLabelLength, 0)  # ylim(bottom,top)
    axm.set_yticks(axesTicksPositions)  # , minor=True)
    axm.set_yticklabels(list(data.index), size=7)

    # # axis labels & positioning
    axm.set_xlabel(r'Global state, $t = \tau+1$', labelpad=3, size=7)
    axm.xaxis.set_label_position('top')
    axm.set_ylabel(r'Global state, $t = \tau$', labelpad=2, size=7)

    # # 'zero' colourbar
    ax_zero_cbar = figure.add_axes([0.51, 0.18, 0.025, 0.025])
    zero_cbar_cmap = mpl.colors.ListedColormap([[1., 1., 1.]])
    cb2 = mpl.colorbar.ColorbarBase(ax_zero_cbar, cmap=zero_cbar_cmap, orientation='horizontal')
    ax_zero_cbar.axes.get_xaxis().set_visible(False)
    ax_zero_cbar.axes.get_yaxis().set_visible(False)
    ax_zero_cbar_label = figure.add_axes([0.54, 0.18, 0.03, 0.03])
    ax_zero_cbar_label.axes.get_xaxis().set_visible(False)
    ax_zero_cbar_label.axes.get_yaxis().set_visible(False)
    plt.text(0, 0, '$<10^{-%s}$' % length_index, fontsize=6)
    for ax_spines in ax_zero_cbar_label.spines.values():  # # remove axes spines
        ax_spines.set_visible(False)

#------------------------------------------------------------
# # Preamble work
#------------------------------------------------------------
# # Model Inputs
parser = argparse.ArgumentParser()
parser.add_argument("motif", type=str)  # Regulatory motifs to loop through
parser.add_argument("updating_method", type=str)
parser.add_argument("signal_status", type=str)
parser.add_argument("length_index", type=int)
args, unknown = parser.parse_known_args()
# print(args)

# print('\n[%s] Preamble.' % dt.datetime.now().strftime('%H:%M:%S'))
pathIn = '%s/input-data' % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Input data directory
pathOut = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Output location
energy_levels = np.array([0.1, 0.5, 1])  # can use set levels or a linspace
TOTAL_SIMS = 10**args.length_index  # Number of simulations per energy level

# signal features
if args.signal_status == "True":
    signal_state = 'active'
if args.signal_status == "False":
    signal_state = 'inactive'

# # Create directory for script outputs
motif_dir = '%s/outputs/%s/%s' % (pathOut, args.motif, args.updating_method)
dir_out = '%s/hms-signal-%s' % (motif_dir, signal_state)
for dirName in [motif_dir, dir_out]:
    try:
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    except OSError as err:
        print(err)

# # Network information
totGroups, groups, nodeOrder, interactions, ics, signal_node_edge, \
    nodes_unique, edge_targets = ni.netInf(pathIn, dir_out, args.motif)

# # Total number of global states (size of state space)
if args.motif == 'ecoli':
    allStates = [0,1,2,3,12,13,14,15]
    totalStates = len(allStates)
else:
    totalStates = 2 ** totGroups[2]
    allStates = [ic for ic in range(totalStates)]

# # Operon information
coupledInteractionsPair, coupledInteractionsInd, coupledGenesIndexPair = \
    ni.grouped_interactions(pathIn, args.motif, nodeOrder)

# # create empty array, pre-allocate size and data type
df_raw_head = ['energy', 'state(t)', 'state(t+1)']  #  dataframe column headers
index_length = totalStates*TOTAL_SIMS*energy_levels.shape[0]
array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.float64)
index_number = 0

#------------------------------------------------------------
# # Simulations
#------------------------------------------------------------
# print('[%s] Simulations.' % dt.datetime.now().strftime('%H:%M:%S'))
for energy in energy_levels:  # Loop through each energy level
    prob_threshold = energy_prob_function(energy)  # fix probability threshold

    for energy_sim in range(0, TOTAL_SIMS):  # run numerous simulations per energy level
        if energy_sim % 250000 == 0:
            # # similation progress file
            progressFile = open('%s/simulation-progress.txt' % (pathOut), 'a')
            line = '%s [heatmaps]: %s, signal %s, energy level %.2f, energy simulation %s' \
                % (dt.datetime.now().strftime('%H:%M:%S'), args.motif, signal_state, energy, energy_sim)
            progressFile.write("%s\n" % line)
            # close progress file
            progressFile.close()

        # # Reduce matrices & dataframes using probability-based condition
        submatrix = SIM(interactions, pathIn, args.motif, prob_threshold, totGroups[0], coupledInteractionsPair, coupledInteractionsInd, nodeOrder)

        for globalState_0_int in allStates:
            #  Loop through all initial conditions
            globalState_0 = get_bin(globalState_0_int, totGroups[2])  # binary form
            # # modify state with signal & ghost nodes states on the end
            if signal_state == 'active':
                globalState_signal_0 = extend_state(globalState_0, totGroups[1], 1)
                globalState_extended_0 = extend_state(globalState_signal_0, totGroups[0], 1)
                # # Update interaction matrix with signal regulation
                signalMatrix = SIM_signal(submatrix, signal_node_edge, edge_targets, \
                     nodeOrder, globalState_extended_0)
            elif signal_state == 'inactive':
                globalState_signal_0 = extend_state(globalState_0, totGroups[1], 0)
                globalState_extended_0 = extend_state(globalState_signal_0, totGroups[0], 1)
                # Update interaction matrix with signal regulation
                signalMatrix = submatrix.copy()
            else:
                sys.exit("\nError: Entered signal state is not known.")

            # # Updating of global state
            if args.updating_method == 'async':
                globalState_1, globalState_extended_1 = \
                    UpdateAsync(globalState_extended_0, signalMatrix, totGroups, nodeOrder, coupledGenesIndexPair)
            elif args.updating_method == 'sync':
                globalState_1, globalState_extended_1 = \
                    UpdateSync(globalState_extended_0, signalMatrix, totGroups, nodeOrder, coupledGenesIndexPair)
            else:
                sys.exit("Error: Updating method not known.")

            # # Integer form of succeeding global state
            globalState_1_int = int(globalState_1, 2)

            # # Add data to array
            transition = [energy, globalState_0_int, globalState_1_int]
            array_raw_data[index_number] = transition
            index_number += 1

# # Create a Pandas DataFrame from the Numpy array of data & export as csv file
df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:], index=range(
    0, index_length), columns=df_raw_head, dtype=float)
# full_filename = export_df('hm-data', dir_out, args.motif, df_raw_data)

# # Calulate transition count for dataframe at each energy level
for energy in energy_levels:
    sub_df = df_raw_data.loc[df_raw_data['energy'] == energy, 'state(t)':'state(t+1)'].reset_index(drop=True)
    dups = sub_df.groupby(sub_df.columns.tolist()).size().reset_index().rename(columns={0: 'count'})
    dups['count'] = dups['count'].div(TOTAL_SIMS)
    full_filename = export_df('hm-%s-count-energy=%.0f' % (signal_state, float(energy)*100),
                                 dir_out, args.motif, dups)

# #------------------------------------------------------------
# # # Produce figure(s)
# #------------------------------------------------------------
# # Import modules
# print('[%s] Figures.' % dt.datetime.now().strftime('%H:%M:%S'))
for energy in energy_levels:
    # Transition data
    df_fileName = "%s/%s-hm-%s-count-energy=%.0f.csv" % (
        dir_out, args.motif, signal_state, float(energy)*100)
    data_hm = hm_data(df_fileName, allStates, totGroups[2])

    # Save dataframe
    df_fileName = "%s/%s-hm-%s-data-energy=%.0f.csv" % (
        dir_out, args.motif, signal_state, float(energy)*100)
    data_hm.to_csv(df_fileName, index=True, header=True)

    # # Produce heatmap and clustering figure & save
    fig_hm = plt.figure(figsize=(4, 3.2))
    dendrogram_and_heatmap(fig_hm, data_hm, totGroups[2], args.length_index,
                              cbarlabel="Transition Probability",
                              interpolation='nearest')
    dendrogramFilename = "%s/%s-hm-%s-energy=%.0f.pdf" % (
        dir_out, args.motif, signal_state, float(energy)*100)
    fig_hm.savefig(dendrogramFilename, format='pdf', transparent=False, bbox_inches='tight')
