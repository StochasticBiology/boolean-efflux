# # Import modules
import sys
import numpy as np
import pandas as pd

def netInf(input_dir, output_dir, name):
    # # Import regulatory interaction data
    data_node = pd.read_csv('%s/%s-regulation-nodes.csv' % (input_dir, name), header=0, index_col=None)
    data_edge = pd.read_csv('%s/%s-regulation-edges.csv' % (input_dir, name), header=0, index_col=None)
    # print(data_node)
    # print(data_edge)

    # # Node-edge regulation dataframe column headers list
    col_edge = data_edge.columns.values.tolist()

    # # Node-edge check
    df_edge = data_edge[data_edge.iloc[:, col_edge.index('regulator')] == 'signal'].reset_index(drop=True)
    if len(data_edge) != len(df_edge):
        sys.exit("Error: Non-signal node-edge regulation detected.")

    data1, first_positions1 = np.unique(data_node[['start node', 'end node']].values, return_index=True)
    data2, first_positions2 = np.unique(data_edge[['target edge start', 'target edge end']].values, return_index=True)

    # # Unique elements of network
    nodes = [t for t in [*data2[np.argsort(first_positions2)], *data1[np.argsort(first_positions1)]]]
    nodes_unique = list(dict.fromkeys(nodes))
    totNodes = len(nodes_unique)

    # # Split unique nodes into groups
    ghostNodes = sorted(filter(lambda x: x if "ghost" in x else None, nodes_unique))
    totGhost = len(ghostNodes)

    signalNodes = sorted(filter(lambda x: x if "signal" in x else None, nodes_unique))
    if not signalNodes:
        signalNodes = ['signal']
        totNodes = len(nodes_unique)+1
    else:
        totNodes = len(nodes_unique)
    totSignal = len(signalNodes)

    primaryNodes = list(filter(lambda x: x if ((not 'signal' in x) and (not 'ghost' in x) \
        and (not 'constitutive' in x)) else None, nodes_unique))
    totPrimary = len(primaryNodes)

    totGroups = [totGhost, totSignal, totPrimary, totNodes]
    groups = [ghostNodes, signalNodes, primaryNodes]

    """Checkpoint: check calculated numbers of nodes are equivalent"""
    if totNodes != totPrimary + totGhost + totSignal:
        sys.exit("Error: Total nodes and sum of node sets do not match.")

    # # Filter unique signal edge regulatory interactions --- NOT FINISHED!
    edgeUnique = df_edge.drop_duplicates(['target edge start', 'target edge end']).reset_index(drop=True)
    edge_targets = edgeUnique.drop(['regulator', 'regulation'], 1)

    # # Node order for global states and assigned numeric labels
    df_labels = pd.DataFrame(index=[], columns=['nodeName', 'nodeLabel'])

    signal_labels = []
    # # label unique nodes to matrix col/row position
    r = 0
    for node in [*primaryNodes, *signalNodes, *ghostNodes]:
        df_labels.loc[len(df_labels)] = [node, r]
        r += 1
    nodeOrder = df_labels['nodeName'].tolist()

    """Produce interaction matrix from data"""
    # # Populate zero array using regulatory network data
    array_interactions = np.zeros(shape=(totNodes, totNodes), dtype=np.float64)
    for index, row in data_node.iterrows():
        node_start = nodeOrder.index(row[0])
        node_end = nodeOrder.index(row[1])
        node_interaction = row[2]
        array_interactions[node_end][node_start] = node_interaction
    # print('Interaction matrix is: %s' % array_interactions)

    # # IC data
    ics_data = pd.read_csv('%s/%s-ICs.csv' % (input_dir, name),  # relative python path to subdirectory
                           # Parse the count "regulation" as an integer
                           # dtype={"start node": int, "end node": int, "regulation": int},
                           header=0,  # Skip the first 10 rows of the file
                           )

    """Checkpoint: check calculated numbers of nodes are equivalent"""
    if len(primaryNodes) != ics_data.shape[1]:
        sys.exit("\nERROR: Initial condition file contains incorrect number of columns.\n")

    # # ICs for network
    totalStates = 2 ** len(primaryNodes)
    if ics_data.empty:
        ics = [get_bin(i, len(primaryNodes)) for i in range(totalStates)]
    else:
        df_string = ics_data.to_string(header=False, index=False, index_names=False).split('\n')
        ics = [''.join(r.split()) for r in df_string]

    # # Randomly shuffle ICs order
    # ics = rn.sample(ics[:], len(ics))

    # # Summary file of motif features
    summary_filename = "%s/network-summary.txt" % (output_dir)
    summary = open(summary_filename, "a+")
    summary.truncate(0)
    summary.write("Motif: %s.\n\nTotal nodes = %s.\n\n" % (name, totNodes))
    summary.write("Number of GRN nodes = %s.\nGRN nodes = %s.\n\nNumber of ghost nodes = %s.\nGhost nodes = %s.\n\n"
                  % (totPrimary, primaryNodes, totGhost, ghostNodes))
    summary.write('Number of signal nodes = %s.\nSignal nodes = %s.\n\nTotal Global System States = %s.\n\n' %
                  (totSignal, signalNodes, totalStates))
    summary.write('Node order and assigned labels: \n%s\n\n' % str(df_labels))
    summary.write('Initial conditions: %s.\n\n' % ics)
    summary.write('Initial Interaction Matrix: \n%s\n\n' % str(array_interactions))
    summary.write('Signal Interactions: \n%s\n\n' % str(data_edge))
    summary.close()

    return totGroups, groups, nodeOrder, array_interactions, ics, df_edge, nodes_unique, edge_targets

def grouped_interactions(input_dir, name, nodeOrder):
    # # Import regulatory interaction data
    node_node_file = '%s/%s-regulation-nodes.csv' % (input_dir, name)
    node_node_data = pd.read_csv(node_node_file, header=0)

    # # Node-node regulation dataframe column headers list
    col_head_nodes = node_node_data.columns.values.tolist()
    # indexGroupedInteractions = col_head_nodes.index('grouped regulation')

    coupledInteractionsPair = []
    coupledInteractionsInd = []
    coupledGenesIndexPair = []
    # coupledGenesIndexInd = []
    for index, row in node_node_data.iterrows():
        # print([index, node_node_data.iloc[index, indexGroupedInteractions]])

        pair1_index = index
        pair1_group = node_node_data.iloc[index, 3]
        # print([pair1_index, pair1_group])
        # print(np.isnan(pair1_group))
        if not np.isnan(pair1_group):
            pair2_index = int(pair1_group)
            pair2_group = int(node_node_data.iloc[pair2_index, 3])
            # print([pair2_index, pair2_group])
            if (pair1_index == pair2_group) and (pair1_group == pair2_index):
                pair = [pair1_index, pair2_index]
                coupledGenes = [node_node_data.iloc[pair1_index,0], node_node_data.iloc[pair1_index,1]] + \
                    [node_node_data.iloc[pair2_index,0], node_node_data.iloc[pair2_index,1]]
                uniq = sorted(list(dict.fromkeys(filter(lambda x: x if "ghost" not in x else None, coupledGenes))))
                genes_index = [nodeOrder.index(gene) for gene in uniq]

                if sorted(genes_index) not in coupledGenesIndexPair:
                    coupledGenesIndexPair.append(sorted(genes_index))
                    # coupledGenesIndexInd.append(genes_index[0])
                    # coupledGenesIndexInd.append(genes_index[1])

                if sorted(pair) not in coupledInteractionsPair:
                    coupledInteractionsPair.append(sorted(pair))
                    coupledInteractionsInd.append(pair1_index)
                    coupledInteractionsInd.append(pair2_index)

    coupledInteractionsInd = list(dict.fromkeys(sorted(coupledInteractionsInd)))
    # coupledGenesIndexInd = list(dict.fromkeys(sorted(coupledGenesIndexInd)))

    return coupledInteractionsPair, coupledInteractionsInd, coupledGenesIndexPair#, coupledGenesIndexInd
