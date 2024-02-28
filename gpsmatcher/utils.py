import os
import pickle
import networkx as nx
from itertools import chain


def remove_unneccesarry_att(G, atts_to_keep = ['source', 'target', 'weight', 'geometry']):
    """
    Remove unnecessary attributes from edges in a graph.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    atts_to_keep : list, optional
        List of attributes to keep. Default is ['source', 'target', 'weight', 'geometry'].

    Returns
    -------
    G : networkx.Graph
        The graph with only the specified attributes.
    """
    atts_name = list(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True))))
    atts_to_remove = [i for i in atts_name if i not in ['source', 'target', 'weight', 'geometry']]
    for n1, n2, d in G.edges(data=True):
        for att in atts_to_remove:
            d.pop(att, None)
    return(G)

def print_step(msg):
    """
    Print a step message with dashes above and below.

    Parameters
    ----------
    msg : str
        Message to print.
    """
    print("-"*len(msg))
    print(msg)
    print("-"*len(msg))
    return(None)

def load_param(folder_name, param_name, show_print=True):
    """
    Load a parameter from a pickle file.

    Parameters
    ----------
    folder_name : str
        Folder name containing the pickle file.

    param_name : str
        Name of the parameter.

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    param : object
        Loaded parameter.
    """
    with open('{}/{}.pickle'.format(folder_name, param_name), 'rb') as handle:
        param = pickle.load(handle)
    if show_print:
        print("Load existing {} done".format(param_name))
    return(param)

def save_param(folder_name, param_name, param):
    """
    Save a parameter to a pickle file.

    Parameters
    ----------
    folder_name : str
        Folder name for saving the pickle file.

    param_name : str
        Name of the parameter.

    param : object
        Parameter to save.
    """
    isExist = os.path.exists(folder_name)
    if not isExist:
        os.makedirs(folder_name)
    with open('{}/{}.pickle'.format(folder_name, param_name), 'wb') as handle:
        pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return(None)
