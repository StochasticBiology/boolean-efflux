# -*- coding: utf-8 - *-  # telling Python the source file you've saved is utf-8
"""Simulate network dynamics over a set of time-steps."""

# # Import modules
import datetime as dt
import csv
import os
import argparse
import sys
import random as rn
import numpy as np
import pandas as pd
import netInfo as ni


def Single_Cell(TOTAL_SIMS, number_SC):
    # # Randomly select X simulations for single-cell analysis
    if TOTAL_SIMS > 2500:
        rand_SC = rn.sample(range(0, TOTAL_SIMS), number_SC)
    else:
        rand_SC = [i for i in range(0, TOTAL_SIMS)]
    return sorted(rand_SC)

def energy_prob_function(n):
    # # Determine prob threshold using energy level
    return 1 * n  # (1+np.exp(8-16*n)) ** (-1)


def get_bin(x, n):  #
    # # Convert integer x to binary format
    return format(x, 'b').zfill(n)


def export_df(fileName, outputDirectory, motif, dataframe):
    # # Export dataframe in csv file format
    complete_filename = "%s/%s-%s.csv" % (outputDirectory, motif, fileName)
    dataframe.to_csv(complete_filename, index=False, header=True)
    return complete_filename


def add_row(energy_sim, GS0_int, time_step, signal_state, globalState_1):
    # # Add row to array
    row = [energy_sim+1, GS0_int, time_step, signal_state]
    row.extend(list(map(int, globalState_1)))
    return row


def extend_state(globalState, extension_length, extension_state):
    # # Add binary string to end of global state
    extension_list = ['%s' % extension_state for i in range(extension_length)]
    # print(extension_list)
    globalState_extension = ''.join(extension_list)
    # print(GS_extended)
    GS_extended = ''.join((globalState, globalState_extension))
    # print(GS_extended)
    return GS_extended


def node_state_change(GS0, index_to_change, new_value):
    # # Flip state of node
    GS0_list = list(GS0)
    GS0_list[index_to_change] = new_value
    GS0_new = ''.join(GS0_list)
    return GS0_new


def SIM(interactionMatrix, data_dir, motifName, probThreshold, number_ghost, coupledInteractionsPair, coupledInteractionsInd, nodeOrder_list):
    # # Sub-interaction matrix

    # Import Data & make a copy of interaction matrix
    df_data = pd.read_csv('%s/%s-regulation-nodes.csv' % (data_dir, motifName), header=0)
    sub_matrix = interactionMatrix.copy()
    matrix_indices = [i for i in range(interactionMatrix.shape[0])]

    # cycle through coupled interactions
    for item in coupledInteractionsPair:
        randNum = rn.uniform(0, 1)
        for i in np.array([0, 1]):
            node_start = nodeOrder_list.index(df_data.iloc[item[i], 0])
            node_end = nodeOrder_list.index(df_data.iloc[item[i], 1])
            sub_matrix[node_end][node_start] = sub_matrix[node_end][node_start] \
                if randNum <= probThreshold else 0

    # temporary removal of couple interaction rows in dataframe
    df_temp = df_data.drop(df_data.index[coupledInteractionsInd], 0)

    # cycle through non-coupled interactions
    for index, row in df_temp.iterrows():
        randNum = rn.uniform(0, 1)
        node_start = nodeOrder_list.index(row[0])
        node_end = nodeOrder_list.index(row[1])
        sub_matrix[node_end][node_start] = sub_matrix[node_end][node_start] \
            if randNum <= probThreshold else 0

    return sub_matrix


def SIM_signal(interaction_matrix, node_edge_data, signal_node_edge_unique, nodeOrder_list, globalState):
    # # Modulate interactions based on signal state

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

                # Modify matrix element value if condition met
                if np.sign(booleanSum) < 0:
                    sub_interactionMatrix[node_to_int][node_from_int] == 0
                else:
                    sub_interactionMatrix[node_to_int][node_from_int] == sub_interactionMatrix[node_to_int][node_from_int]
        return sub_interactionMatrix


def UpdateAsync(extendedState_0, matrix, totGroups, nodeOrder_list, timestep, signal_range, coupledGenesIndexPair):
    # # Boolean modelling asynchronous updating of nodes
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

                    # # Calculate Boolean function value
                    booleanSum = np.array([matrix[random_node][j]*int(state_0_list[j]) for j in range(totGroups[3])]).sum()
                    # print("Boolean sum: %s" % booleanSum)

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
                # # Calculate Boolean function value
                booleanSum = np.array([matrix[random_node][j]*int(state_0_list[j]) for j in range(totGroups[3])]).sum()
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
        state_1 = extendedState_1[:totGroups[2]]

    if timestep not in signal_range:  # list(map(lambda x: x+1, signal_range[:-1]))
        signal_state = extendedState_1[totGroups[2]:totGroups[2]+1]
    else:
        signal_state = '1'

    return state_1, extendedState_1, signal_state


def UpdateSync(extendedState_0, matrix, totGroups, nodeOrder_list, timestep, signal_range, coupledGenesIndexPair):
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

    if timestep not in signal_range:  # list(map(lambda x: x+1, signal_range[:-1]))
        signal_state = extendedState_1[totGroups[2]:totGroups[2]+1]
    else:
        signal_state = '1'

    return state_1, extendedState_1, signal_state

def timeFigOther(stress_start, GS0, uniqueNodes, fig_timeSteps, motif, dir_out, energy_levels, tick_stepsize):
    """Time evolution figure"""
    uniqueNodes.remove('acrAB')

    fig, axs = plt.subplots(len(uniqueNodes), 1, figsize=(2, 2.8), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.5, 'wspace': 0.0})
    ax = axs.ravel()

    # # set all text to arial font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.facecolor'] = '[1,1,1]'

    fig_subplotTitles = list(uniqueNodes)
    colors = ["#FF0000", "#1D9300", "#0904A4"]
    colorsStd = ["#FF0000", "#34E806", "#0F07F3"]
    for i in range(0, len(uniqueNodes)):
        clr = 0
        for energyLevel in energy_levels:
            # # Import mean "on" per time-step dataframe
            df_mean = pd.read_csv('%s/%s-activation-%s-energy=%.0f-mean.csv' % (
                dir_out, motif, GS0, float(energyLevel)*100))
            # # Import standard deviation per time-step dataframe
            df_std = pd.read_csv('%s/%s-activation-%s-energy=%.0f-std.csv' % (
                dir_out, motif, GS0, float(energyLevel)*100))

            if fig_subplotTitles[i] != 'signal':
                j = df_mean.columns.get_loc(fig_subplotTitles[i])
            else:
                j = df_mean.columns.get_loc("signal_state")

            ax[i].plot(df_mean.iloc[:, 0], df_mean.iloc[:, j], '-', color=colors[clr],
                       linewidth=0.7, zorder=1+clr)
            transparency = [0.15, 0.18, 0.15]
            ax[i].fill_between(df_mean.iloc[:, 0], df_mean.iloc[:, j] - df_std.iloc[:, j],
                               df_mean.iloc[:, j] + df_std.iloc[:, j],
                               color=colorsStd[clr], alpha=transparency[clr], linewidth=0.5, zorder=1+clr)
            clr+=1
        right_xlim = 250
        ax[i].set_xlim(-2, right_xlim)
        ax[i].set_ylim(-0.02, 1.02)
        ax[i].xaxis.set_major_locator(MultipleLocator(tick_stepsize))
        ax[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax[i].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i].set_yticks(np.linspace(0, 1, 5))  # , minor=True)
        ax[i].tick_params(axis="both", labelsize=5.5, length=3, pad=1, width=0.5)
        ax[i].tick_params(which='minor', length=1.5, width=0.4)

        if fig_subplotTitles[i] != 'signal':
            j = df_mean.columns.get_loc(fig_subplotTitles[i])
            ax[i].set_ylabel("Mean\nActivation", size=6.5, labelpad=2)
            ax[i].set_title('%s' % fig_subplotTitles[i], fontsize=7, style='italic', pad=1.5)
        else:
            j = df_mean.columns.get_loc("signal_state")
            ax[i].set_ylabel("Mean\nPresence", size=6.5, labelpad=2)
            ax[i].set_title('Stressor', fontsize=7, pad=1.5)

        if stress_start != 0:
            ax[i].fill_between(np.linspace(0, stress_start-1, stress_start-1), -0.1, 1.1,
                           facecolor='#171717', alpha=0.15, zorder=0) #facecolor='pink'

        if i not in np.arange(0, len(uniqueNodes)-1):
            ax[i].set_xlabel("Time-step", size=6.5, labelpad=2)

        ax[i].grid(True, which='both', linestyle='--', linewidth='0.1')
    return fig


def timeFigEfflux(stress_start, GS0, uniqueNodes, fig_timeSteps, motif, dir_out, energy_levels, tick_stepsize):
    """Time evolution figure"""
    fig, axs = plt.subplots(len(energy_levels), 1, figsize=(2, 2.8), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0.5, 'wspace': 0.0})
    ax = axs.ravel()

    # # set all text to arial font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.facecolor'] = '[1,1,1]'

    fig_subplotTitles = ['acrAB' for i in range(len(energy_levels))]
    i = 0
    colors = ["#FF0000", "#1D9300", "#0904A4"]
    colorsStd = ["#FF0000", "#34E806", "#0F07F3"]
    for energyLevel in energy_levels:
        # # Import mean "on" per time-step dataframe
        df_mean = pd.read_csv('%s/%s-activation-%s-energy=%.0f-mean.csv' % (
            dir_out, motif, GS0, float(energyLevel)*100))
        # # Import standard deviation per time-step dataframe
        df_std = pd.read_csv('%s/%s-activation-%s-energy=%.0f-std.csv' % (
            dir_out, motif, GS0, float(energyLevel)*100))

        j = df_mean.columns.get_loc(fig_subplotTitles[i])
        ax[i].plot(df_mean.iloc[:, 0], df_mean.iloc[:, j], '-', color=colors[i],
                   linewidth=0.7, zorder=2)
        ax[i].fill_between(df_mean.iloc[:, 0], df_mean.iloc[:, j] - df_std.iloc[:, j],
                           df_mean.iloc[:, j] + df_std.iloc[:, j],
                           color=colorsStd[i], alpha=0.15, linewidth=0.5, zorder=1)

        right_xlim = 250
        ax[i].set_xlim(-2, right_xlim)
        ax[i].set_ylim(-0.02, 1.02)
        ax[i].xaxis.set_major_locator(MultipleLocator(tick_stepsize))
        ax[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax[i].xaxis.set_minor_locator(MultipleLocator(5))
        ax[i].set_yticks(np.linspace(0, 1, 5))  # , minor=True)
        ax[i].tick_params(axis="both", labelsize=5.5, length=3, pad=1, width=0.5)
        ax[i].tick_params(which='minor', length=1.5, width=0.4)

        ax[i].set_ylabel("Mean\nActivation", size=6.5, labelpad=2)
        ax[i].set_title('%s' % fig_subplotTitles[i], fontsize=7, style='italic', pad=1.5)

        if stress_start != 0:
            ax[i].fill_between(np.linspace(0, stress_start-1, stress_start-1), -0.1, 1.1,
                           facecolor='#171717', alpha=0.2, zorder=0) #facecolor='#0023D1'

        if i not in np.arange(0, len(uniqueNodes)-1):
            ax[i].set_xlabel("Time-step", size=6.5, labelpad=2)

        ax[i].grid(True, which='both', linestyle='--', linewidth='0.1')
        i+=1
    return fig


def singleCell_fig(df_SC):
    timeStart = 80
    timeEnd = 200

    # # Plot figure of signal time-evolution
    fig, ax = plt.subplots(figsize=(3, 2.5))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.facecolor'] = '[1,1,1]'

    # Create an array with the colors you want to use
    colors = ["#cacaca", "#a3a3a3"]
    # Set your custom color palette
    customPalette = sns.color_palette(colors)
    heat_map = sns.heatmap(df_SC, cbar=False, linecolor='black', linewidths=0.05, cmap=customPalette)
    # heat_map.invert_yaxis()

    for i in range(df_SC.shape[1]+1):
        ax.axhline(i, color='white', lw=0.5)

    # make frame visible
    for _, spine in heat_map.spines.items():
        spine.set_visible(True)

    # # sns.palplot(sns.mpl_palette("Set3", 11))
    xaxis_spacing = 10
    ax.set_xlabel('Time-step', size=8)
    ax.set_xticks([xaxis_spacing*i for i in range(int(timeStart/xaxis_spacing), int(timeEnd/xaxis_spacing)+1)])
    ax.set_xticklabels([xaxis_spacing*i for i in range(int(timeStart/xaxis_spacing),int(timeEnd/xaxis_spacing)+1)], rotation=60, fontsize=6.5)
    ax.set_xlim(timeStart, timeEnd)
    ax.set_ylabel('Simulation', size=8)
    ax.set_yticks([i+0.5 for i in range(0,int(df_SC.shape[0]))])#, rotation=60)
    ax.set_yticklabels(list(df_SC.index), rotation=20, fontsize=6.5)
    ax.set_ylim(0, df_SC.shape[0])
    ax.tick_params(axis='both', which='major', pad=1)
    return fig

def timePulseFig(energy_levels, df_PTD, gene, ic):

    # # general qualities
    colors = ["#FF0000", "#1D9300", "#0904A4"]
    recs = []
    for i in range(0,len(energy_levels)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=colors[i]))

    leftlim_x = 25
    rightlim_x = 225
    x_spacing = 25
    axis_labelsize = 12

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['axes.facecolor'] = '[1,1,1]'

    # # mean pulse length figure
    fig_mu, axes_mu = plt.subplots(figsize=(5, 2.5))
    axes_mu.set_ylabel("Mean Pulse Length", size=axis_labelsize, labelpad=2)
    lim_y = 150
    axes_mu.set_ylim(10**0, lim_y)
    axes_mu.set_yscale('log')
    axes_mu.legend(recs, ['Energy = %.2f' % i for i in energy_levels], loc='upper right', fontsize=6)
    axes_mu.set_xlim(leftlim_x, rightlim_x)
    axes_mu.grid(True, which='both', linestyle='--', linewidth='0.1')
    axes_mu.set_xlabel("Time-step of Pulse Start", size=axis_labelsize, labelpad=2)
    axes_mu.set_xticks(np.linspace(leftlim_x, rightlim_x, int((rightlim_x-leftlim_x)/x_spacing)+1), minor=False)
    axes_mu.set_facecolor("#F6F6F6")
    axes_mu.tick_params(axis="both", labelsize=11, length=2.5, pad=1.5)

    # # cv of pulse length figure
    fig_cv, axes_cv = plt.subplots(figsize=(5, 2.5))
    axes_cv.set_ylabel("CV of Pulse Length", size=axis_labelsize, labelpad=2)
    lim_y = 2
    y_spacing = 0.5
    axes_cv.set_ylim(0, lim_y)
    axes_cv.set_yticks(np.linspace(0, lim_y, int(lim_y/y_spacing)+1), minor=False)
    axes_cv.legend(recs, ['Energy = %.2f' % i for i in energy_levels], loc='upper right', fontsize=6)
    axes_cv.set_xlim(leftlim_x, rightlim_x)
    axes_cv.grid(True, which='both', linestyle='--', linewidth='0.1')
    axes_cv.set_xlabel("Time-step of Pulse Start", size=axis_labelsize, labelpad=2)
    axes_cv.set_xticks(np.linspace(leftlim_x, rightlim_x, int((rightlim_x-leftlim_x)/x_spacing)+1), minor=False)
    axes_cv.set_facecolor("#F6F6F6")
    axes_cv.tick_params(axis="both", labelsize=11, length=2.5, pad=1.5)

    clr = 0
    for energy in energy_levels:
        df_sub = df_PTD[df_PTD.energy_level == energy].reset_index(drop=True)
        axes_mu.scatter(df_sub.iloc[:, 3], df_sub.iloc[:, 4], color=colors[clr], alpha=0.5, s=1)
        axes_cv.scatter(df_sub.iloc[:, 3], df_sub.iloc[:, 6], color=colors[clr], alpha=0.5, s=1)
        clr+=1
    return fig_mu, fig_cv

#------------------------------------------------------------
# # Preamble work
#------------------------------------------------------------
# # Model Inputs
parser = argparse.ArgumentParser()
parser.add_argument("motif", type=str)
parser.add_argument("updating_method", type=str)
parser.add_argument("signal_0", type=int)
parser.add_argument("signal_length", type=int)
parser.add_argument("length_index", type=int)
args, unknown = parser.parse_known_args()
# print(args)

# print('\n[%s] Preamble.' % dt.datetime.now().strftime('%H:%M:%S'))
pathIn = '%s/input-data' % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Input data directory
pathOut = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Output location

energy_levels = np.array([0.1, 0.5, 1])  # can use set levels or a linspace
TOTAL_TIME = 300

if args.signal_0 < 0:
    sys.exit("Error in 'Signal Range'. Value presented is negative.")
if args.signal_length < 0:
    sys.exit("Error in 'Signal Length'. Value presented is negative.")

motif_dir = '%s/outputs/%s/%s' % (pathOut, args.motif, args.updating_method)
if (args.signal_0 == 0) and (args.signal_length == 0):  # timesteps signal is active
    signal = False
    signal_range = np.array([])
    dir_out = '%s/timeseries-no-signal' % (motif_dir)
elif (args.signal_0 != 0) and (args.signal_length == 0):
    sys.exit("Error. Check entered values for signal.")
else:
    signal = True
    signal_range = np.linspace(args.signal_0, args.signal_0 +
                               args.signal_length, args.signal_length, endpoint=False)
    dir_out = '%s/timeseries-signal-%.0fto%.0f' % (motif_dir, args.signal_0, signal_range[-1])
for dirName in [motif_dir, dir_out]:
    try:
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    except OSError as err:
        print(err)
signal_range = list(map(lambda x: x-1, signal_range))

# # Network infmormation for framework inputs
totGroups, groups, nodeOrder, interactions, ics, signal_node_edge, \
    nodes_unique, edge_targets = ni.netInf(pathIn, dir_out, args.motif)

# # Total number of global states (size of state space)
totalStates = 2 ** totGroups[2]
# # Operon information
coupledInteractionsPair, coupledInteractionsInd, coupledGenesIndexPair = \
    ni.grouped_interactions(pathIn, args.motif, nodeOrder)

# # Dataframe column headers
df_raw_head = ['energy_sim', 'initial_condition', 'time_step', 'signal_state']
df_raw_head.extend(list(groups[2]))
df_stats_head = list(groups[2])
df_stats_head[:0] = ['time_step', 'signal_state']

# # Single-simulation pulse data
arr_Pulse = np.zeros(shape=(len(ics)*len(energy_levels)*2500*totGroups[2]*TOTAL_TIME, 6), dtype=np.float64)
ind_pulse = 0
#------------------------------------------------------------
# # Simulations
#------------------------------------------------------------
# print('[%s] Simulations.' % dt.datetime.now().strftime('%H:%M:%S'))
# # Loop through each energy level
for energy in energy_levels:
    # # fix probability threshold
    prob_threshold = energy_prob_function(energy)

    # # Set total simulations & number of 'single-cells'
    if args.updating_method == "sync":
        TOTAL_SIMS = 1 if prob_threshold == 1 or prob_threshold == 0 else 10**args.length_index
    else:
        TOTAL_SIMS = 1 if prob_threshold == 0 else 10**args.length_index
    number_SC = 2500 if TOTAL_SIMS >= 2500 else TOTAL_SIMS
    rand_SCs = Single_Cell(TOTAL_SIMS, number_SC)

    # # create empty array, pre-allocate size and data type
    index_length = (TOTAL_TIME+1)*TOTAL_SIMS*len(ics)
    array_raw_data = np.zeros(shape=(index_length, len(df_raw_head)), dtype=np.int64)
    index_number = 0

    # # create empty array, pre-allocate size and data type
    SC_index_length = (TOTAL_TIME+1)*number_SC*len(ics)
    SC_raw_data = np.zeros(shape=(SC_index_length, len(df_raw_head)), dtype=np.int64)
    SC_index_number = 0

    # # run numerous simulations per energy level
    for energy_sim in range(0, TOTAL_SIMS):
        if energy_sim % 5000 == 0:
            # # Similation progress file
            progressFile = open('%s/simulation-progress.txt' % (pathOut), 'a')
            line = '%s: %s, signal length %.0f, energy level %.2f, energy simulation %s' \
                % (dt.datetime.now().strftime('%H:%M:%S'), args.motif, len(signal_range),
                   energy, energy_sim)
            progressFile.write("%s\n" % line)
            progressFile.close()

        # # Loop through all initial conditions
        for GS0 in ics:
            # # Loop through full time range (per simulation)
            for timestep in range(0, TOTAL_TIME):
                submatrix = SIM(interactions, pathIn, args.motif, prob_threshold, totGroups[0], coupledInteractionsPair, coupledInteractionsInd, nodeOrder)
                if timestep == 0:
                    # # integer conversion
                    GS0_int = int(GS0, 2)
                    # print([GS0, GS0_int])

                    # # modify state with signal & ghost nodes states on the end
                    GS_signal = extend_state(GS0, totGroups[1], 0)
                    GS_extended = extend_state(GS_signal, totGroups[0], 1)

                    # # Add t=0 data to arraya
                    array_raw_data[index_number] = add_row(energy_sim, GS0_int, 0, 0, GS0)
                    index_number += 1
                    if energy_sim in rand_SCs:
                        SC_raw_data[SC_index_number] = add_row(energy_sim, GS0_int, 0, 0, GS0)
                        SC_index_number += 1

                # # Modify signal node state if in pulse range
                GS_extended = node_state_change(GS_extended, totGroups[2], '1') \
                    if timestep in signal_range else GS_extended[:]
                signal_state = GS_extended[totGroups[2]:totGroups[2]+1]

                # # Update interaction matrix with signal regulation
                if signal_state == '1':
                    signalMatrix = SIM_signal(
                        submatrix, signal_node_edge, edge_targets, nodeOrder, GS_extended)
                else:
                    signalMatrix = submatrix.copy()
                # print(signalMatrix)

                if args.updating_method == 'async':
                    globalState, GS_extended, signal_state = \
                        UpdateAsync(GS_extended, signalMatrix, totGroups, nodeOrder, timestep, signal_range, coupledGenesIndexPair)
                elif args.updating_method == 'sync':
                    globalState, GS_extended, signal_state = \
                        UpdateSync(GS_extended, signalMatrix, totGroups, nodeOrder, timestep, signal_range, coupledGenesIndexPair)
                else:
                    sys.exit("Error: Updating method not known.")

                # # Add data to array
                array_raw_data[index_number] = add_row(energy_sim, GS0_int, timestep+1, signal_state, globalState)
                index_number += 1

                if energy_sim in rand_SCs:
                    SC_raw_data[SC_index_number] = add_row(energy_sim, GS0_int, timestep+1, signal_state, globalState)
                    SC_index_number += 1

    # # Create a Pandas DataFrame from the Numpy array of data & export as csv file
    df_raw_data = pd.DataFrame(data=array_raw_data[0:, 0:], index=range(0, index_length), columns=df_raw_head)
    # export_df('timeseries-data-energy=%.0f' % (float(energy)*100), dir_out, args.motif, df_raw_data)

    # # Create a Pandas DataFrame from the Numpy array of single-cell data
    SC_raw_data[~np.all(SC_raw_data == 0, axis=1)]
    df_SC_data = pd.DataFrame(data=SC_raw_data[0:, 0:], index=range(0, len(SC_raw_data)), columns=df_raw_head)
    # export_df('timeseries-data-energy=%.0f-SC' % (float(energy)*100), dir_out, args.motif, df_SC_data)

    # # Calculate mean "on" and standard deviation for each time-step
    for GS0 in ics:#[ics[1]]:#
        # # integer conversion
        GS0_int = int(GS0, 2)
        df_sub_raw1 = df_raw_data[df_raw_data.initial_condition == GS0_int].reset_index(drop=True)  # .astype(float)

        # # initiate dataframes for mean, std & cv
        df_mean = pd.DataFrame(index=[], columns=df_stats_head)
        df_std = pd.DataFrame(index=[], columns=df_stats_head)
        df_cv = pd.DataFrame(index=[], columns=df_stats_head)

        for time_iter in range(0, TOTAL_TIME+1):
            df_sub_raw = df_sub_raw1[df_sub_raw1.time_step ==time_iter].astype(float).reset_index(drop=True)
            stat_summary = df_sub_raw.describe()  # exclude=[np.object]
            # print(stat_summary)
            mean = stat_summary.iloc[1, 2:len(df_raw_head)].tolist()
            df_mean.loc[len(df_mean)] = mean
            # print(mean)
            std = stat_summary.iloc[2, 3:len(df_raw_head)].tolist()
            std.insert(0, time_iter)
            df_std.loc[len(df_std)] = std
            # print(std)

        # # Export mean "on" per time-step dataframe
        df_mean = df_mean.fillna(0)
        export_df('activation-%s-energy=%.0f-mean' % (GS0, float(energy)*100), dir_out, args.motif, df_mean)

        # # Export standard deviation per time-step dataframe
        df_std = df_std.fillna(0)
        export_df('activation-%s-energy=%.0f-std' % (GS0, float(energy)*100), dir_out, args.motif, df_std)

        # # Calculate the cv (=standard deviation/mean) for each time-step and datframe column
        df_cv = df_std.div(df_mean)
        df_cv['time_step'] = df_std['time_step']

        # # Export cv per time-step dataframe
        # df_cv = df_cv.fillna(0)
        export_df('activation-%s-energy=%.0f-cv' % (GS0, float(energy)*100), dir_out, args.motif, df_cv)

        # # Single-cell "barcode" data for IC
        SC_data = df_SC_data[df_SC_data.initial_condition == int(GS0, 2)].reset_index(drop=True)

        for gene in nodeOrder[:totGroups[2]]:
            gene_index = nodeOrder.index(gene)
            df = SC_data.copy()

            maxCells = 20 if TOTAL_SIMS >= 20 else TOTAL_SIMS
            cols = list(df.columns)
            m = [df.columns.get_loc('energy_sim'), df.columns.get_loc('time_step'), df.columns.get_loc(gene)]
            totCols = len(df.columns)
            removeCols = [i for i in range(totCols) if i not in m]  # remove all except time, sim and selected gene

            df.drop(df.columns[tuple([removeCols])], axis=1, inplace=True)
            df_pivot = df.pivot(index='energy_sim', columns='time_step', values=gene)
            pivotRows = df_pivot.index.tolist()
            randRows_ind = [pivotRows.index(i) for i in sorted(rn.sample(pivotRows, maxCells))]
            dfPivot_sub = df_pivot.iloc[randRows_ind, :]
            export_df('SC-%s-%s-energy=%.0f' % (GS0, gene, float(energy)*100), dir_out, args.motif, dfPivot_sub)

            """Pulse statistics on simulated single-cell data"""
            pulseList = []
            for index, row in df_pivot.iterrows():
                # row positions where element is a '1'
                position1 = np.where(row == 1)
                item = position1[0]

                # count pulse lengths
                count = 1
                consec_list = []
                consec_items = []
                timestepPulseInfo_list = []
                for element in range(len(item)-1):
                    consec_items.append(item[element])
                    if item[element]+1 == item[element+1]:
                        count += 1
                    else:
                        consec_list.append(count)
                        timestepPulseInfo_list.append([consec_items[0], count])
                        arr_Pulse[ind_pulse] = [energy, int(GS0, 2), gene_index, index, consec_items[0], count]
                        ind_pulse += 1
                        count = 1
                        consec_items = []

                    if item[element] == item[-2]:
                        if count > 1:
                            consec_list.append(count)
                            timestepPulseInfo_list.append([consec_items[0], count])
                            arr_Pulse[ind_pulse] = [energy, int(GS0, 2), gene_index, index, consec_items[0], count]
                            ind_pulse += 1
                        else:
                            consec_list.append(1)
                            timestepPulseInfo_list.append([item[element+1], 1])
                            arr_Pulse[ind_pulse] = [energy, int(GS0, 2), gene_index, index, item[element+1], 1]
                            ind_pulse += 1
arr_Pulse[~np.all(arr_Pulse == 0, axis=1)]
# print(arr_Pulse)

# # Create a Pandas DataFrame from the Numpy array of data & export as csv file
df_hd = ['energy_level', 'ic', 'gene_index', 'cell', 'pulse_start', 'pulse_length']
df = pd.DataFrame(data=arr_Pulse[0:, 0:], index=range(0, arr_Pulse.shape[0]), columns=df_hd)
# export_df('pulse-data', dir_out, args.motif, df)

for GS0 in ics:#[ics[1]]:#
    for gene in nodeOrder[:totGroups[2]]:
        gene_index = nodeOrder.index(gene)

        # Modify data
        df_sub = df[(df.ic == int(GS0, 2)) & (df.gene_index == gene_index)].reset_index(drop=True)
        clr = 0
        arr_PulseTimeData = np.zeros(shape=(TOTAL_TIME*len(energy_levels), 6), dtype=np.float64)
        PTD_index = 0
        for energy in energy_levels:
            sub_df_sub = df_sub[df_sub.energy_level == energy].reset_index(drop=True)
            for timePoint in range(TOTAL_TIME):
                df_stats = sub_df_sub[sub_df_sub.pulse_start == timePoint].reset_index(drop=True)
                stat_summary = df_stats.describe()
                mu = stat_summary.iloc[1, 5]
                sigma = stat_summary.iloc[2, 5]
                arr_PulseTimeData[PTD_index] = [gene_index, energy, int(GS0, 2), timePoint, mu, sigma]
                PTD_index += 1

        # # Create a Pandas DataFrame from the Numpy array of data & export as csv file
        df_PTD_Head = ['gene_index', 'energy_level', 'ic', 'time_step', 'mean', 'std']
        df_PTD = pd.DataFrame(data=arr_PulseTimeData[0:, 0:], index=range(0, arr_PulseTimeData.shape[0]), columns=df_PTD_Head)
        df_PTD["cv"] = df_PTD["std"] / df_PTD["mean"]
        export_df('SC-pulse-stats-%s-%s' % (gene, GS0), dir_out, args.motif, df_PTD)

# #------------------------------------------------------------
# # # Produce figure(s)
# #------------------------------------------------------------
# print('[%s] Figures.' % dt.datetime.now().strftime('%H:%M:%S'))
# # Import modules
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
for GS0 in ics:#[ics[1]]:#
    """Timeseries figures"""
    fig_timeSteps = TOTAL_TIME
    tick_stepsize = 25

    # # network minus efflux
    fig_timeOther = timeFigOther(args.signal_0, GS0, (groups[2]+groups[1]), fig_timeSteps, args.motif, dir_out, energy_levels, tick_stepsize)
    fn_timeOther = "%s/%s-activation-%s-other.pdf" % (dir_out, GS0, args.motif)
    fig_timeOther.savefig(fn_timeOther, format='pdf', bbox_inches='tight', transparent=False)
    plt.close(fig_timeOther)

    # # efflux
    fig_timeEfflux = timeFigEfflux(args.signal_0, GS0, (groups[2]+groups[1]), fig_timeSteps, args.motif, dir_out, energy_levels, tick_stepsize)
    fn_timeEfflux = "%s/%s-activation-%s-acrAB.pdf" % (dir_out, GS0, args.motif)
    fig_timeEfflux.savefig(fn_timeEfflux, format='pdf', bbox_inches='tight', transparent=False)
    plt.close(fig_timeEfflux)

    for energy in energy_levels:
        for gene in nodeOrder[:totGroups[2]]:
            gene_index = nodeOrder.index(gene)
            df_SC = pd.read_csv('%s/%s-SC-%s-%s-energy=%.0f.csv' % (dir_out, args.motif, GS0, gene, float(energy)*100))

            # # Single-Cell "barcode" Plots
            fig_SC = singleCell_fig(df_SC)
            fig_SCFilename = "%s/%s-SC-%s-%s-energy=%.0f.pdf" % (
                dir_out, args.motif, GS0, gene, float(energy)*100)
            fig_SC.savefig(fig_SCFilename, format='pdf', bbox_inches='tight', transparent=False)
            plt.close(fig_SC)

            # # start time-step & mean or cv of pulse length figure
            df_PTD = pd.read_csv('%s/%s-SC-pulse-stats-%s-%s.csv' % (dir_out, args.motif, gene, GS0))
            figPulse_mean, figPulse_cv = timePulseFig(energy_levels, df_PTD, gene, GS0)
            figPulse_mean.savefig('%s/%s-SC-%s-pulseStats-%s-mean.pdf' % (dir_out, args.motif, GS0, gene), format='pdf', bbox_inches='tight', transparent=False)
            figPulse_cv.savefig('%s/%s-SC-%s-pulseStats-%s-cv.pdf' % (dir_out, args.motif, GS0, gene), format='pdf', bbox_inches='tight', transparent=False)
            plt.close(figPulse_mean)
            plt.close(figPulse_cv)
