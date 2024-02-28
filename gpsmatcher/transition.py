import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from gpsmatcher.utils import load_param, print_step, save_param


def node2edge_graph(graph_nx, show_print=True):
    """
    Convert a node-weighted graph to an edge-weighted directed graph.

    Parameters
    ----------
    graph_nx : networkx.DiGraph
        The input graph with node weights.

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    new_graph : networkx.DiGraph
        Converted graph with edge weights.
    """
    edges = list(graph_nx.edges())
    edges2weight =nx.get_edge_attributes(graph_nx, 'weight')
    edges2id =nx.get_edge_attributes(graph_nx, 'edge_id')
    new_edges = []
    for from_node, to_node in edges:
        id_edge = edges2id[(from_node, to_node)]
        for i in graph_nx.in_edges(from_node):
            id_row = edges2id[i]
            weight = (edges2weight[(from_node, to_node)] + edges2weight[i]) / 2
            new_edges.append((id_row, id_edge, weight))
    new_graph = nx.DiGraph()
    new_graph.add_weighted_edges_from(new_edges)
    
    if show_print:
        print("Node to edge graph conversion done")
    return(new_graph)

def edge_graph_trans(new_graph, max_weight, show_print=True):
    """
    Compute edge-to-edge transitions in a weighted directed graph.

    Parameters
    ----------
    new_graph : networkx.DiGraph
        The input graph with edge weights.

    max_weight : float
        Maximum weight for considering transitions.

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    all_transition : pd.DataFrame
        DataFrame containing edge-to-edge transitions with columns 'ori_edge', 'dest_edge', 'weight'.
    """
    all_transition = []
    for id_node, ori_node in enumerate(list(new_graph.nodes)):
        one_edge_trans = nx.single_source_dijkstra_path_length(new_graph, ori_node, cutoff=max_weight)
        all_transition.append((ori_node, list(one_edge_trans.keys()), list(one_edge_trans.values())))
    all_transition = pd.DataFrame(all_transition, columns=['ori_edge', 'dest_edge', 'weight'])
    
    if show_print:
        print("Edge to edge transition computation done")
    return(all_transition)

def transition_matrix(graph_nx, max_weight, beta = 1/500, save=True, folder_name="mm_input", show_print=True):
    """
    Compute a transition matrix for a graph.

    Parameters
    ----------
    graph_nx : networkx.DiGraph
        The input graph.

    max_weight : float
        Maximum weight for considering transitions.

    beta : float, optional
        Scaling factor for the exponential term. Default is 1/500.

    save : bool, optional
        Whether to save the computed transition matrix. Default is True.

    folder_name : str, optional
        Folder name for saving/loading. Default is "mm_input".

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    transition_matrix : scipy.sparse.csr_matrix
        Computed transition matrix.
    """
    if show_print:
        print_step("Start process transition")
        
    if save and (os.path.isfile(folder_name + '/transition_matrix.pickle')):
        transition_matrix = load_param(folder_name, "transition_matrix", show_print=show_print)
        return(transition_matrix)
    else:
        new_graph = node2edge_graph(graph_nx, show_print=show_print)
        all_transition = edge_graph_trans(new_graph, max_weight, show_print=show_print)
        all_transition = all_transition.explode(['dest_edge', 'weight']).reset_index(drop=True).astype(np.int64)
        int_max_edge = max(all_transition[["ori_edge", "dest_edge"]].max()) + 1
        row = all_transition['ori_edge'].values
        col = all_transition['dest_edge'].values
        data = np.exp(- beta * all_transition['weight'].values)
        transition_matrix = csr_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1))
        transition_matrix = (transition_matrix * 100).astype(np.int8)
        if save:
            save_param(folder_name, "transition_matrix", transition_matrix)
        return(transition_matrix)