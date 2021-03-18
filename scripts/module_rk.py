"""
Self-defined functions for numerical simulation of Boolean network model
"""

# # Import modules
import numpy as np
import pandas as pd
import random as rn
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


def pathOut(file):
    """
    Path to directory locations (input & output)
    """
    path_scripts = os.path.dirname(os.path.abspath(file))  # Directory for scripts
    path_input = '%s/input-data' % os.path.dirname(path_scripts)  # Input data directory
    path_out = os.path.dirname(path_scripts)  # Output location
    return path_input, path_out


def unique_nodes(input_dir, motifName):
    """
    Signal regulation and unique components in architecture
    """

    # # Import regulatory interaction data
    node_node_file = '%s/%s-regulation-nodes.csv' % (input_dir, motifName)
    node_node_data = pd.read_csv(node_node_file, header=0)
    node_edge_file = '%s/%s-regulation-edges.csv' % (input_dir, motifName)
    node_edge_data = pd.read_csv(node_edge_file, header=0)
    # print(node_node_data)
    # print(node_edge_data)

    # # Node-node regulation dataframe column headers list
    col_head_nodes = node_node_data.columns.values.tolist()
    index_start_nodes = col_head_nodes.index('start node')
    index_end_nodes = col_head_nodes.index('end node')
    index_regulation_nodes = col_head_nodes.index('regulation')

    # # Node-edge regulation dataframe column headers list
    col_head_edges = node_edge_data.columns.values.tolist()
    index_regulator_edges = col_head_edges.index('regulator')
    index_start_edges = col_head_edges.index('target edge start')
    index_end_edges = col_head_edges.index('target edge end')

    # # Node-edge check
    signal_node_edge = node_edge_data[node_edge_data.iloc[:, index_regulator_edges]
                                      == 'signal'].reset_index(drop=True)
    if len(node_edge_data) != len(signal_node_edge):
        sys.exit("Error: Non-signal node-edge regulation detected.")

    # # Network original and ghost nodes
    start_unique_nodes = node_node_data[col_head_nodes[index_start_nodes]].unique()
    end_unique_nodes = node_node_data[col_head_nodes[index_end_nodes]].unique()
    start_unique_edges = node_edge_data[col_head_edges[index_start_edges]].unique()
    end_unique_edges = node_edge_data[col_head_edges[index_end_edges]].unique()

    nodes_unique = [item for item in start_unique_nodes]
    # print(nodes_unique)

    for node_item in end_unique_nodes:
        nodes_unique.append(node_item) if node_item not in nodes_unique \
            else nodes_unique
    # print(nodes_unique)

    for node_item in start_unique_edges:
        nodes_unique.append(node_item) if node_item not in nodes_unique \
            else nodes_unique
    # print(nodes_unique)

    for node_item in end_unique_edges:
        nodes_unique.append(node_item) if node_item not in nodes_unique \
            else nodes_unique
    # print(nodes_unique)

    return signal_node_edge, nodes_unique


def node_groups(nodes_unique):
    """
    Separate list into separte lists for original networks nodes, external signals & ghost nodes
    """
    # total
    nodes_total = len(nodes_unique)
    # ghost nodes
    nodes_ghost = [ss for ss in nodes_unique if "ghost" in ss]
    nodes_ghost.sort()
    number_ghost = len(nodes_ghost)
    # signal node
    nodes_signal = [ss for ss in nodes_unique if "signal" in ss]
    number_signal = len(nodes_signal)
    # original grn nodes
    nodes_grn = [ss for ss in nodes_unique if ((not 'signal' in ss) and (
        not 'ghost' in ss) and (not 'constitutive' in ss))]
    number_grn = len(nodes_grn)
    # print(nodes_ghost)
    # print(nodes_signal)
    # print(nodes_grn)

    # #Checkpoint: check calculated numbers of nodes are equivalent
    numberNodesCheck = nodes_total - number_ghost - number_signal
    if number_grn != numberNodesCheck:
        sys.exit("Error: Total nodes and sum of node sets do not match.")
    # print([number_grn, numberNodesCheck])

    return nodes_grn, nodes_ghost, nodes_signal, nodes_total, number_ghost, number_signal, number_grn


def signal_targets_unique(signal_node_edge):
    """
    Filter unique signal edge regulatory interactions
    """

    signal_node_edge_unique = signal_node_edge.drop_duplicates(
        ['target edge start', 'target edge end']).reset_index(drop=True)
    # print(signal_regulation_node_edge)

    signal_node_edge_targets = signal_node_edge_unique.drop(['regulator', 'regulation'], 1)

    return signal_node_edge_targets


def node_labels(nodes_grn, nodes_signal, nodes_ghost):
    """
    Node order for global states and assigned numeric labels
    """

    # # create empty dataframe for node labels
    df_nodeLabels = pd.DataFrame(index=[], columns=['nodeName', 'nodeLabel'])

    signal_labels = []
    # # label unique nodes to matrix col/row position
    for r in range(len(nodes_grn)):
        nodeSelected = nodes_grn[r]
        df_nodeLabels.loc[len(df_nodeLabels)] = [nodeSelected, r]
        # print('r = %s. Node = %s.' % (str(r), nodeSelected))
    if nodes_signal:
        for u in range(len(nodes_signal)):
            nodeSelected = nodes_signal[u]
            u2 = u + r + 1
            signal_labels.append(u2)
            df_nodeLabels.loc[len(df_nodeLabels)] = [nodeSelected, u2]
            # print('u2 = %s Node = %s.' % (str(u2), nodeSelected))
    else:
        u2 = r
        # print('u2 = %s' % str(u2))
    if nodes_ghost:
        for v in range(len(nodes_ghost)):
            nodeSelected = nodes_ghost[v]
            v2 = v + u2 + 1
            df_nodeLabels.loc[len(df_nodeLabels)] = [nodeSelected, v2]
            # print('v2 = %s Node = %s.' % (str(v2), nodeSelected))
    else:
        v2 = u2
        # print('v2 = %s' % str(v2))

    nodeOrder_list = df_nodeLabels['nodeName'].tolist()
    return nodeOrder_list, df_nodeLabels, signal_labels


def summary_file(path_out, motifName, nodes_total, number_grn, nodes_grn, number_ghost,
                 nodes_ghost, number_signal, nodes_signal, totalGlobalStates, ics,
                 interaction_matrix, signal_node_edge, df_nodeLabels):
    """
    Summary file of motif features
    """

    # # create/open summary file
    summary_filename = "%s/network-summary.txt" % (path_out)
    summary = open(summary_filename, "a+")
    summary.truncate(0)

    # # write info in text file
    summary.write("Motif: %s.\n\nTotal nodes = %s.\n\n" % (motifName, nodes_total))
    summary.write("Number of GRN nodes = %s.\nGRN nodes = %s.\n\nNumber of ghost nodes = %s.\nGhost nodes = %s.\n\n"
                  % (number_grn, nodes_grn, number_ghost, nodes_ghost))
    summary.write('Number of signal nodes = %s.\nSignal nodes = %s.\n\nTotal Global System States = %s.\n\n' %
                  (number_signal, nodes_signal, totalGlobalStates))
    summary.write('Node order and assigned labels: \n%s\n\n' % str(df_nodeLabels))
    summary.write('Initial conditions: %s.\n\n' % ics)
    summary.write('Initial Interaction Matrix: \n%s\n\n' % str(interaction_matrix))
    summary.write('Signal Interactions: \n%s\n\n' % str(signal_node_edge))

    # # close file
    summary.close()


def create_dir(signal_status, signal_start, signal_range, motifName, path_out):
    """
    Directories for simulation outputs
    """

    motif_dir = '%s/outputs/%s' % (path_out, motifName)

    if signal_status == False:
        out_dir = '%s/timeseries-no-signal' % (motif_dir)
    else:
        out_dir = '%s/timeseries-signalStart-%.0f-signalEnd-%.0f' % \
            (motif_dir, signal_start, signal_range[-1])

    create_directories = [motif_dir, out_dir]
    for dirName in create_directories:
        # print(dirName)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

    return out_dir


def create_hm_dir(signal_state, motifName, path_out):
    """
    Directories for simulation outputs
    """

    motif_dir = '%s/outputs/%s' % (path_out, motifName)
    out_dir = '%s/hms-signal-%s' % (motif_dir, signal_state)

    create_directories = [motif_dir, out_dir]
    for dirName in create_directories:
        if not os.path.exists(dirName):
            os.makedirs(dirName)

    return out_dir


def signal_range(signal_active, signal_start, signal_length):  # stress time range
    """
    List of time-steps for active signal
    """

    if signal_active == True:
        if signal_start < 0:
            sys.exit("Error in 'Signal Range'. Start point is before time-step 1.")
        if signal_length < 0:
            sys.exit("Error in 'Signal Length'. Value presented is negative.")

        signal_range = np.linspace(signal_start, signal_start +
                                   signal_length, signal_length, endpoint=False)

    elif signal_active == False:
        signal_range = np.array([])

    else:
        sys.exit("Error: Unknown 'signal' value.")

    return signal_range


def export_df(fileName, outputDirectory, motif, dataframe):  # export dataframe
    """
    Export dataframe to csv file format
    """

    complete_filename = "%s/%s-%s.csv" % (outputDirectory, motif, fileName)
    export_df = dataframe.to_csv(complete_filename, index=False, header=True)

    return complete_filename


def energy_prob_function(n):
    """
    determine prob threshold using energy level
    """
    return 1 * n  # (1+np.exp(8-16*n)) ** (-1)


def linear_energy_time_function(direction, energy_min, energy_max, timestep, totalTime):
    """
    Linearly increase or decrease energy at specified time-step
    """

    if timestep < 0:
        sys.exit("Error: Current timestep is before t=0.")
    if timestep > totalTime:
        sys.exit("Error: Current timestep is out of defined time range.")

    if direction == 'increase':
        energy = energy_min+(timestep*((energy_max-energy_min)/(totalTime-1)))
    elif direction == 'decrease':
        energy = energy_max - (timestep*((energy_max-energy_min)/(totalTime-1)))
    return energy


def discrete_energy_time_function(direction, matrix_low, matrix_high, low_energy, high_energy, switchpoint, timestep):
    """
    Increase or decrease energy at specified time-step
    """

    if timestep < 0:
        sys.exit("Error: Current timestep is before t=0.")
    # if timestep == switchpoint:
    #     print('Switchpoint [timestep = %.0f]' % timestep)

    if direction == 'increase':
        if timestep < switchpoint:
            energy = low_energy
            sub_matrix = matrix_low
        else:
            energy = high_energy
            sub_matrix = matrix_high
    elif direction == 'decrease':
        if timestep < switchpoint:
            energy = high_energy
            sub_matrix = matrix_high
        else:
            energy = low_energy
            sub_matrix = matrix_low
    else:
        sys.exit("Error: Direction of energy switch is unknown.")
    return energy, sub_matrix


def get_bin(x, n):  # convert integer to binary format
    """
    Get the binary representation of x
    """
    return format(x, 'b').zfill(n)


def extend_state(globalState, extension_length, extension_state):  # append global state w. additional states
    """
    Create binary string to add to end of global state.
    """

    extension_list = ['%s' % extension_state for i in range(extension_length)]
    # print(extension_list)
    globalState_extension = ''.join(extension_list)
    # print(globalState_extended)
    globalState_extended = ''.join((globalState, globalState_extension))
    # print(globalState_extended)
    return globalState_extended


def interaction_matrix(regulationDataDirectory, motifName, total_nodes, nodeOrder_list):
    """
    Produce interaction matrix from data
    """

    # #Zero array
    array_interactions = np.zeros(shape=(total_nodes, total_nodes), dtype=np.float64)
    # print(array_interactions)

    # # Import Data
    df_file = '%s/%s-regulation-nodes.csv' % (regulationDataDirectory, motifName)
    df_data = pd.read_csv(df_file, header=0)

    # # Populate the zero array using regulatory network data
    for index, row in df_data.iterrows():
        node_start = nodeOrder_list.index(row[0])
        node_end = nodeOrder_list.index(row[1])
        node_interaction = row[2]
        # print([node_start, node_end, node_interaction])
        array_interactions[node_end][node_start] = node_interaction
    # print('Interaction matrix is: %s' % interaction_matrix)
    return array_interactions


def import_ICs(ics_data_dir, motifName, nodes_grn):
    """
    ICs for simulations
    """

    ics_filename = '%s/%s-ICs.csv' % (ics_data_dir, motifName)
    ics_data = pd.read_csv(ics_filename,  # relative python path to subdirectory
                           # Parse the count "regulation" as an integer
                           # dtype={"start node": int, "end node": int, "regulation": int},
                           header=0,  # Skip the first 10 rows of the file
                           )

    # # Checkpoint: check calculated numbers of nodes are equivalent
    number_ics_nodes_check = ics_data.shape[1]
    if len(nodes_grn) != number_ics_nodes_check:
        sys.exit("\n\nERROR: Initial condition file contains incorrect number of columns.\n\n")

    # # Compile the set of ICs
    totalGlobalStates = 2 ** len(nodes_grn)
    if ics_data.empty:
        ics = [get_bin(i, len(nodes_grn)) for i in range(totalGlobalStates)]
    else:
        df_string = ics_data.to_string(header=False, index=False,
                                       index_names=False).split('\n')
        ics = [''.join(r.split()) for r in df_string]

    # # Randomly shuffle ICs order
    # ics = rn.sample(ics[:], len(ics))
    # print(ics)
    return ics


def submatrix(interactionMatrix, probThreshold):
    """
    Submatrix of interactions
    """

    sub_matrix = interactionMatrix.copy()
    matrix_indices = [i for i in range(interactionMatrix.shape[0])]

    # # Produce sub-matrix of node-node regulation
    for matrixRow in matrix_indices:
        for matrixCol in matrix_indices:
            # # (pseudo)random value from a uniform distribution
            randNum = rn.uniform(0, 1)
            # # set element value within sub-matrix
            sub_matrix[matrixRow][matrixCol] = sub_matrix[matrixRow][
                matrixCol] if randNum <= probThreshold else 0
    return sub_matrix


def add_row(energy_sim, globalState_0_int, time_step, signal_state, globalState_1):
    """
    add array row data
    """

    row = [energy_sim+1, globalState_0_int, time_step, signal_state]
    row.extend(list(map(int, globalState_1)))
    return row


def node_state_change(globalState_0, index_to_change, new_value):
    """
    flip state of node
    """
    globalState_0_list = list(globalState_0)
    globalState_0_list[index_to_change] = new_value
    globalState_0_new = ''.join(globalState_0_list)
    return globalState_0_new


def signal_node_edge_regulation_matrix(interaction_matrix, node_edge_data, signal_node_edge_unique, nodeOrder_list, globalState):
    """
    Modulate interactions based on signal state
    """
    if len(signal_node_edge_unique.index) != 0:
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
                # print(sub_df)

                booleanSum = np.array([sub_df.iloc[df_row, index_regulation]*int(globalState[nodeOrder_list.index(sub_df.iloc[df_row, 0])])
                                       for df_row in range(sub_df.shape[0])]).sum()
                # print('Sum of inputs: %.3f' % booleanSum)

                # # Modify matrix element value if condition met
                if np.sign(booleanSum) < 0:
                    sub_interactionMatrix[node_to_int][node_from_int] == 0
                else:
                    sub_interactionMatrix[node_to_int][node_from_int] == sub_interactionMatrix[node_to_int][node_from_int]
    else:
        sub_interactionMatrix = interaction_matrix.copy()

    return sub_interactionMatrix


def update_state(state_0, number_grn, number_signal, matrix, total_nodes, N):
    """
    Boolean modelling asynchronous updating of nodes
    """

    # # convert to list
    state_0_list = list(state_0)
    for randomNode in range(0, N):
        # print("Full state time = t: %s" % completeStartStateBinary)
        random_node = rn.randint(0, (number_grn + number_signal-1))
        # print("Node to be updated: %s" % random_node)

        # # Calculate Boolean function value
        booleanSum = np.array([matrix[random_node][j]*int(state_0[j])
                               for j in range(total_nodes)]).sum()  # calculates sum for node update
        # print("Boolean sum: %s" % booleanSum)

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
        state_1 = extendedState_1[:number_grn]

    return state_1, extendedState_1


def hm_data(transitonsFile, totalGlobalStates, numberNodes):
    """
    Produce a matrix to use for heatmaps and dendrograms.
    """
    # # data import
    df = pd.read_csv(transitonsFile)
    dfRows = df.shape[0]

    # # Empty array
    dataFile = [[0] * totalGlobalStates for rows in range(totalGlobalStates)]
    # print('\nBlank matrix is: ' + str(heatmapData))

    # # Fill array with transition data from dataframe
    for hm_data_row in range(dfRows):
        hm_start_state = int(df.iloc[hm_data_row]['state(t)'])
        hm_end_state = int(df.iloc[hm_data_row]['state(t+1)'])
        hm_probability = df.iloc[hm_data_row]['count']
        dataFile[hm_start_state][hm_end_state] = hm_probability

    # # Add column & index labels
    globalState_names = [get_bin(_, numberNodes) for _ in range(totalGlobalStates)]
    hm_data = pd.DataFrame(dataFile, index=globalState_names, columns=globalState_names)
    # print(hm_data)
    return hm_data


"""
Figure functions
(1) Heatmap plots of transition matrices
(2) Time-evolution figures
"""


def dendrogram_and_heatmap(figure, data, numberNetworkNodes, number_sims, cbarlabel="", cbar_kw={}, **kwargs):
    """
    Heatmap figure with dendrogram
    """
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
    axd.set_xlabel('Euclidean distance \nbetween destination \nprobability sets', size=4.5)
    xticklabels = [r'$\sqrt{2}$', 0]
    axd.set_xticklabels(xticklabels, size=7)
    axd.set_yticklabels(list(data_rowclust.index)[::-1], size=7)
    axd.tick_params(axis='y', which='both', pad=-2)
    # axd.set_ylabel('Global state, time t', size=18)
    axd.tick_params(axis="x", labelsize=5, width=0.4)
    # plt.xlim(dist_max, 0)  # ylim(left, right)
    plt.xlim(np.sqrt(2), 0)
    plt.grid(b=None, which='major', axis='x', ls='--', alpha=0.7)
    # remove axes spines from dendrogram
    for ax_spines in axd.spines.values():
        ax_spines.set_visible(False)

    # # Plot heatmap sub-figure
    mpl.rcParams['axes.linewidth'] = 0.4
    axm = figure.add_axes([0.17, 0.1, 0.3, 0.6])  # x-pos, y-pos, width, height
    cmap = plt.cm.viridis
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    data_thresh = 1/number_sims
    data_thresh_scientific = "{:.1e}".format(data_thresh)
    # print(data_thresh)
    data[data < data_thresh] = np.nan
    # Plot data in heatmap
    im = axm.imshow(data, cmap=cmap, vmin=0, vmax=1, **kwargs)
    # axis ticks
    axesLabelLength = len(list(data.columns))
    axesTicksPositions = np.arange(0, axesLabelLength, 1)
    axm.set_xticks(axesTicksPositions)
    axm.set_yticks(axesTicksPositions)  # , minor=True)

    axm.set_xticklabels(list(data.columns), rotation=90, size=6.8)
    axm.tick_params(axis="both", width=0.4, length=2, pad=1)
    axm.set_yticklabels(list(data.index), size=6.8)  # ([''] + list(df2.columns))
    # move x-ticks and labels to the top of heatmap
    axm.xaxis.set_ticks_position('top')
    # ax.minorticks_on()

    # # heatmap x and y limits
    axm.set_xlim(-0.5, axesLabelLength-0.5)  # xlim(left, right)
    axm.set_ylim(axesLabelLength-0.5, -0.5)  # ylim(bottom,top)

    # # axis labels & positioning
    axm.set_xlabel(r'Global state, $t = \tau+1$', labelpad=3, size=7)
    axm.xaxis.set_label_position('top')
    axm.set_ylabel(r'Global state, $t = \tau$', labelpad=2, size=7)

    # # Plot separate manual colour bar
    ax_cbar = figure.add_axes([0.5, 0.27, 0.015, 0.4])
    cbar_cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cbar_cmap, norm=norm, orientation='vertical')
    cb1.set_label(cbarlabel, rotation=-90, va="bottom", labelpad=-17, size=7)
    cb1.minorticks_on()
    start, end = ax_cbar.get_xlim()
    ax_cbar.yaxis.set_ticks(np.arange(start, end+0.2, 0.2))
    ax_cbar_labels = ['$%s$' % data_thresh_scientific, '0.2', '0.4', '0.6', '0.8', '1.0']
    ax_cbar.yaxis.set_ticklabels(ax_cbar_labels, size=6)
    ax_cbar.tick_params(axis="both", which='both', width=0.4, pad=1)

    ax_zero_cbar = figure.add_axes([0.5, 0.21, 0.025, 0.025])
    zero_cbar_cmap = mpl.colors.ListedColormap([[1., 1., 1.]])
    cb2 = mpl.colorbar.ColorbarBase(ax_zero_cbar, cmap=zero_cbar_cmap, orientation='horizontal')
    ax_zero_cbar.axes.get_xaxis().set_visible(False)
    ax_zero_cbar.axes.get_yaxis().set_visible(False)
    ax_zero_cbar_label = figure.add_axes([0.53, 0.212, 0.03, 0.03])
    ax_zero_cbar_label.axes.get_xaxis().set_visible(False)
    ax_zero_cbar_label.axes.get_yaxis().set_visible(False)
    plt.text(0, 0, '$%s$' % data_thresh_scientific, fontsize=6)

    # # remove axes spines
    for ax_spines in ax_zero_cbar_label.spines.values():
        ax_spines.set_visible(False)


def time_evolution_subplot(numberNetworkNodes):
    """
    Subplot layout (mxn)
    """
    x = math.sqrt(numberNetworkNodes)
    subplot_rows = math.ceil(x)
    subplot_cols = math.ceil(numberNetworkNodes/subplot_rows)
    return subplot_rows, subplot_cols


def time_evolution_fig(networkGlobalState_0, uniqueNodes, numberNodes, fig_timeSteps, df, df_error):
    """
    Time evolution figure
    """
    fig, axs = plt.subplots(2, 2, figsize=(3, 2.5), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.19, 'wspace': 0.08})
    # sharex='col',sharey = 'row', gridspec_kw = {'hspace': 0.2, 'wspace': 0.2}
    # constrained_layout=True
    ax = axs.ravel()

    # # set all text to arial font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.facecolor'] = '[1,1,1]'  # E5E5E5, [0.756,0.756,0.756,0.4]

    # fig.suptitle("Global state initial condition: %s" % str(networkGlobalState_0), fontsize=16)
    fig_subplotTitles = list(uniqueNodes)
    subplot_rows, subplot_cols = time_evolution_subplot(numberNodes)
    x_label_subplots = [((subplot_rows-1)*subplot_cols)+n for n in range(0, subplot_cols)]
    y_label_subplots = [(n*subplot_cols) for n in range(0, subplot_rows)]

    df_stress_sub = df[df.signal_state != 0]
    if not df_stress_sub.empty:
        stress_start = df_stress_sub.index[0]
    else:
        stress_start = 1

    for i in range(0, numberNodes):
        ax[i].fill_between(np.linspace(0, stress_start-1, stress_start-1), -0.1, 1.1,
                           facecolor='pink', alpha=0.5, zorder=0)

        ax[i].set_title('%s' % fig_subplotTitles[i], fontsize=8, style='italic', pad=1.5)

        ax[i].plot(df.iloc[:, 0], df.iloc[:, i+2], '-', color='#0023D1',
                   linewidth=1, zorder=2)  # ax[i].step / ax[i].plot
        # ax[i].step(df.iloc[:, 0], df.iloc[:, i+2], '-', color='#0023D1',
        #            linewidth=1.5, zorder=0, where='post')
        ax[i].fill_between(df.iloc[:, 0], df.iloc[:, i+2] - df_error.iloc[:, i+2],
                           df.iloc[:, i+2] + df_error.iloc[:, i+2],
                           color='#409AFF', alpha=0.25, linewidth=0.1, zorder=1)
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
