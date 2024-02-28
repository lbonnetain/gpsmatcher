import numpy as np
import networkx as nx
import pandas as pd
import pygeohash_fast
import pygeohash as pgh
from utils import load_param, save_param
import os

def process_dic_cand_edges(G, radius = 150, save=True, folder_name="mm_input", show_print=True):
    """
    Process a dictionary of candidate edges for geohashes in a graph: load if dic_cand_edges has been already computed else dic_cand_edges is computed.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    radius : float, optional
        Radius for computing candidate edges. Default is 150 meters.

    save : bool, optional
        Whether to save the processed candidate edges. Default is True.

    folder_name : str, optional
        Folder name for saving/loading. Default is "mm_input".

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    dic_cand_edges : dict
        Dictionary mapping geohashes to candidate edges.
    """
    
    if save and (os.path.isfile(folder_name + '/cand_edges.pickle')):
        dic_cand_edges = load_param(folder_name, "cand_edges", show_print=show_print)

    else:
        dic_cand_edges = get_dic_cand_edges(G, radius = radius, show_print=show_print)

        if save and (~os.path.isfile(folder_name + '/cand_edges.pickle')):
            save_param(folder_name, "cand_edges", dic_cand_edges)
    return(dic_cand_edges)


def get_all_neigbors(hash_central, dir_geo, dic_neighbor, nb_steps = 19):
    """
    Get all neighbors in a specified direction for a given geohash.

    Parameters
    ----------
    hash_central : str
        Central geohash.

    dir_geo : str
        Direction of neighbors ("top", "bottom", "left", "right").

    dic_neighbor : dict
        Dictionary to store computed neighbors.

    nb_steps : int, optional
        Number of steps in each direction. Default is 19.

    Returns
    -------
    all_neigbors_dir : list
        List of all geohashes in the specified direction.
    """
    all_keys = dic_neighbor.keys()
    all_neigbors_dir = [hash_central]
    i = 0
    for i in range(nb_steps):
        if (hash_central, dir_geo) in all_keys:
            neigbhor_dir = dic_neighbor[(hash_central, dir_geo)]
        else:
            neigbhor_dir = pgh.get_adjacent(hash_central, dir_geo)
            dic_neighbor[(hash_central, dir_geo)] = neigbhor_dir
            
            all_neigbors_dir.append(neigbhor_dir)
        hash_central = neigbhor_dir
    return(all_neigbors_dir, dic_neighbor)

def compute_neighbors(hash_central, nb_steps_lon, nb_steps_lat, dic_neighbor):
    """
    Compute neighbors for a central geohash in both horizontal and vertical directions.

    Parameters
    ----------
    hash_central : str
        Central geohash.

    nb_steps_lon : int
        Number of steps in the horizontal direction.

    nb_steps_lat : int
        Number of steps in the vertical direction.

    dic_neighbor : dict
        Dictionary to store computed neighbors.

    Returns
    -------
    all_geohashes : list
        List of all computed geohashes.
    """
    
    all_neigbors_dir_north, dic_neighbor = get_all_neigbors(hash_central, "top",dic_neighbor, nb_steps=nb_steps_lat)
    all_neigbors_dir_south, dic_neighbor = get_all_neigbors(hash_central, "bottom", dic_neighbor, nb_steps=nb_steps_lat)
    vertical_geoshes = all_neigbors_dir_south + all_neigbors_dir_north
    all_geohashes = vertical_geoshes.copy()
    for hash_v in vertical_geoshes:
        all_neigbors_dir_west, dic_neighbor = get_all_neigbors(hash_v, "left", dic_neighbor, nb_steps=nb_steps_lon)
        all_neigbors_dir_est, dic_neighbor  = get_all_neigbors(hash_v, "right", dic_neighbor, nb_steps=nb_steps_lon)
        all_geohashes.extend(all_neigbors_dir_west + all_neigbors_dir_est)
    
    return(all_geohashes, dic_neighbor)

def get_all_emission_tiles(geohashes, edge, nb_steps_lon, nb_steps_lat, dic_neighbor, dic_cand_edges): 
    """
    Get all geohashes neighbors for a list of central geohashes (geohashes of all edges).

    Parameters
    ----------
    geohashes : list
        List of central geohashes.

    edge : int
        Edge ID associated with the geohashes.

    nb_steps_lon : int
        Number of steps in the horizontal direction.

    nb_steps_lat : int
        Number of steps in the vertical direction.

    dic_neighbor : dict
        Dictionary to store computed neighbors.

    dic_cand_edges : dict
        Dictionary mapping geohashes to candidate edges.

    Returns
    -------
    all_geohashes_tiles : list
        List of all geohashes for a list of central geohashes (geohashes of all edges).
    """
    all_geohashes_tiles = []
    dic_neighbor = {}
    for i in geohashes:
        sub_geohashes, dic_neighbor = compute_neighbors(i, nb_steps_lon, nb_steps_lat ,dic_neighbor)
        all_geohashes_tiles.extend(sub_geohashes)
    all_geohashes_tiles = list(set(all_geohashes_tiles))
    
    keys_geohash = dic_cand_edges.keys()
    for i in all_geohashes_tiles:
        if i not in keys_geohash:
            dic_cand_edges[i] = [edge]
        else:
            dic_cand_edges[i] += [edge]
    return(all_geohashes_tiles)

def get_number_tiles(radius):
    """
    Get the number of steps in the horizontal and vertical directions for geohashes within a radius.

    Parameters
    ----------
    radius : float
        Radius for computing emission tiles.

    Returns
    -------
    nb_lat_steps : int
        Number of steps in the vertical direction.

    nb_lon_steps : int
        Number of steps in the horizontal direction.
    """
    width, height = 38.2, 19.1
    nb_lon_steps = int((radius // width) + 1)
    nb_lat_steps = int((radius // height) + 1)
    return(nb_lat_steps, nb_lon_steps)

def get_dic_cand_edges(G, radius = 150, show_print=True):
    """
    Get a dictionary of candidate edges for geohashes in a graph.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    radius : float, optional
        Radius for computing candidate edges. Default is 150 meters.

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    dic_cand_edges : dict
        Dictionary mapping geohashes to candidate edges.
    """
    edges2id =nx.get_edge_attributes(G, 'edge_id')
    edge2geohash = nx.get_edge_attributes(G, 'geohashes')
    edge_geohash = pd.DataFrame(edge2geohash.items(),columns=['edge', 'geohash'])
    edge_geohash['edge_id'] = edge_geohash['edge'].map(edges2id)
    dic_cand_edges = {}
    dic_neighbor = {}
    nb_lat_steps, nb_lon_steps = get_number_tiles(radius)
    edge_geohash['geohash_bis'] = edge_geohash.apply(lambda row : get_all_emission_tiles(row['geohash'], row['edge_id'], nb_lon_steps, nb_lat_steps, dic_neighbor, dic_cand_edges), axis=1)
    if show_print:
        print("Get candidate edges for all geohashes done")
    return(dic_cand_edges)
