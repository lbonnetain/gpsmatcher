import itertools

import networkx as nx
import numba
import numpy as np
import pandas as pd
import pygeohash_fast
from joblib import Parallel, delayed

from gpsmatcher.emission import emission_matrix
from gpsmatcher.gps import process_gps
from gpsmatcher.graph import process_graph
from gpsmatcher.precomputation_emission import process_dic_cand_edges
from gpsmatcher.transition import transition_matrix


def get_cand_edge_gps(gps, cand_edge):
    """
    Filter candidate edges based on GPS data.

    Parameters
    ----------
    gps : pandas.DataFrame
        GPS data.

    cand_edge : dict
        Dictionary mapping geohashes to candidate edges.

    Returns
    -------
    cand_edge : dict
        Updated dictionary with filtered candidate edges.
    """
    all_geohashes = set(pygeohash_fast.encode_many(gps['lon'].values, gps['lat'].values, 8))
    cand_edge = {k: cand_edge[k] for k in cand_edge.keys() & all_geohashes}
    return(cand_edge)

def format_result(gps, gps_mm):
    """
    Format map-matching results.

    Parameters
    ----------
    gps : pandas.DataFrame
        GPS data.

    gps_mm : pandas.DataFrame
        Map-matched GPS data.

    Returns
    -------
    gps : pandas.DataFrame
        Updated GPS data with edge information.

    gps_mm : pandas.DataFrame
        Formatted map-matching results.
    """
    gps.reset_index(drop=True, inplace=True)
    gps_mm['shortest_path_nodes'] = gps_mm.apply(lambda row: row['map_match'][1], axis=1)
    gps_mm['edges'] = gps_mm.apply(lambda row: row['map_match'][0], axis=1)
    gps = gps[['lon', 'lat', 'ID_trip']]
    gps['edge'] = gps_mm['edges'].explode().reset_index(drop=True)
    gps_mm = gps_mm["shortest_path_nodes"].reset_index()
    return(gps, gps_mm)

def mm_precomputation(G, beta=1/500, radius=150, save=True, folder_name="mm_input", show_print=True):
    """
    Perform map-matching precomputation.

    Parameters
    ----------
    G : networkx.DiGraph
        The road network graph.

    beta : float, optional
        Parameter for transition matrix computation.

    radius : int, optional
        Radius for candidate edges computation.

    save : bool, optional
        Whether to save the precomputed data.

    folder_name : str, optional
        Folder name for saving data.

    show_print : bool, optional
        Whether to print progress messages.

    Returns
    -------
    G_mm : networkx.DiGraph
        Processed road network graph for map-matching.

    trans : scipy.sparse.csr_matrix
        Transition matrix for map-matching.

    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.

    id2edges : dict
        Dictionary mapping edge_id to edge tuples.
    """
    G_mm, id2edges = process_graph(G, max_length=500, save=save, folder_name=folder_name, show_print=show_print)
    trans = transition_matrix(G_mm, 120, beta=beta, save=save, folder_name=folder_name, show_print=show_print)
    dic_candidates = process_dic_cand_edges(G_mm, radius = radius, save=True, folder_name=folder_name, show_print=show_print)
    return(G_mm, trans, dic_candidates, id2edges)

def mm_gps(gps, G_mm, trans, dic_candidates, id2edges, alpha=0.1, radius=150):
    """
    Given the precomputation, map-match gps data (without multiprocessing)

    Parameters
    ----------
    gps : pandas.DataFrame
        GPS data.

    G_mm : networkx.DiGraph
        Processed road network graph for map-matching.

    trans : scipy.sparse.csr_matrix
        Transition matrix for map-matching.

    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.

    id2edges : dict
        Dictionary mapping edge_id to edge.

    alpha : float, optional
        Parameter for emission matrix computation.

    radius : int, optional
        Radius for candidate edges computation.

    Returns
    -------
    gps : pandas.DataFrame
        Updated GPS data with edge information.

    gps_mm : pandas.DataFrame
        Map-matched GPS data. Each ID_trip with most likely path in the graph.
    """
    gps, gps_mm, dic_geohash, dic_candidates = process_gps(gps, dic_candidates)
    emit = emission_matrix(gps, G_mm, dic_geohash, dic_candidates, alpha = alpha, radius = radius)
    gps_mm['map_match'] = gps_mm.apply(lambda row: one_traj_mm(row['traj'], G_mm, trans, emit, id2edges, row['sub_edges'], row['first_emission'], row['last_emission']), axis=1)
    gps, gps_mm = format_result(gps, gps_mm)
    return(G_mm, gps, gps_mm)

def mm_gps_parrallel(gps, G_mm, trans, dic_candidates, id2edges, alpha=0.1, radius=150, nb_cores=4):
    """
    Given the precomputation, map-match gps data (with multiprocessing)

    Parameters
    ----------
    gps : pandas.DataFrame
        GPS data.

    G_mm : networkx.DiGraph
        Processed road network graph for map-matching.

    trans : scipy.sparse.csr_matrix
        Transition matrix for map-matching.

    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.

    id2edges : dict
        Dictionary mapping edge_id to edge.

    alpha : float, optional
        Parameter for emission matrix computation.

    radius : int, optional
        Radius for candidate edges computation.

    nb_cores : int, optional
        Number of CPU cores to use.

    Returns
    -------
    gps : pandas.DataFrame
        Updated GPS data with edge information.

    gps_mm : pandas.DataFrame
        Map-matched GPS data. Each ID_trip with most likely path in the graph.
    """
    dic_candidates = get_cand_edge_gps(gps, dic_candidates)
    all_chunks_gps =  np.array_split(gps, nb_cores)
    result = Parallel(n_jobs=nb_cores, verbose=20)(delayed(mm_gps)(chunk_gps, G_mm, trans, dic_candidates, id2edges, alpha=alpha, radius=radius) for chunk_gps in all_chunks_gps)
    gps = pd.concat([i_result[0] for i_result in result])
    gps_mm = pd.concat([i_result[1] for i_result in result])
    return(G_mm, gps, gps_mm)

def big_gps_file_mm(G, gps, radius = 150, alpha = 0.1, beta=1/500, nb_rows_chunk = 100000, save=True, folder_name="mm_input", show_print=True, parrallel = False, nb_cores=3):
    """
    Perform map-matching on large GPS data where the data is splitted into chunks to process.

    Parameters
    ----------
    G : networkx.DiGraph
        The road network graph.

    gps : pandas.DataFrame
        GPS data with three columns lon, lat, ID_trip.

    save : bool, optional
        Whether to save the precomputation or load if already computed.

    radius : int, optional
        Radius for candidate edges computation.

    alpha : float, optional
        Parameter for emission matrix computation.

    beta : float, optional
        Parameter for transition matrix computation.

    folder_name : str, optional
        Folder name for saving data.

    show_print : bool, optional
        Whether to print progress messages.

    parallel : bool, optional
        Whether to use parallel processing.

    nb_cores : int, optional
        Number of CPU cores to use.

    Returns
    -------
    gps : pandas.DataFrame
        Updated GPS data with edge information.

    gps_mm : pandas.DataFrame
        Map-matched GPS data. Each ID_trip with most likely path in the graph.
    """
    G_mm, trans, dic_candidates, id2edges = mm_precomputation(G, beta=beta, radius=radius, save=save, folder_name=folder_name, show_print=show_print)
    
    nb_chunks = len(gps) // nb_rows_chunk
    all_chunks_gps =  np.array_split(gps, nb_chunks)
    all_edges  = []
    all_gps_mm  = []
    for idx, chunk_gps in enumerate(all_chunks_gps):
        print("{}/{} chunk done".format(idx+1, nb_chunks))
        if parrallel:
            chunk_gps, chunk_gps_mm = mm_gps_parrallel(chunk_gps, G_mm, trans, dic_candidates, id2edges, alpha=alpha, radius=radius, nb_cores=nb_cores)
        else:
            chunk_gps, chunk_gps_mm = mm_gps(chunk_gps, G_mm, trans, dic_candidates, id2edges, alpha=alpha, radius=radius)
        all_edges.extend(chunk_gps['edge'].values)
        all_gps_mm.append(chunk_gps_mm)
    
    all_gps_mm = pd.concat(all_gps_mm)
    gps['edge'] = all_edges
    return(G_mm, gps, all_gps_mm)

def gps_file_mm(G, gps, save=True, radius = 150, alpha = 0.1, beta=1/500, folder_name="mm_input", show_print=True, parrallel = False, nb_cores=3):
    """
    Perform map-matching on GPS data.

    Parameters
    ----------
    G : networkx.DiGraph
        The road network graph.

    gps : pandas.DataFrame
        GPS data with three columns lon, lat, ID_trip.

    save : bool, optional
        Whether to save the precomputation or load if already computed.

    radius : int, optional
        Radius for candidate edges computation.

    alpha : float, optional
        Parameter for emission matrix computation.

    beta : float, optional
        Parameter for transition matrix computation.

    folder_name : str, optional
        Folder name for saving data.

    show_print : bool, optional
        Whether to print progress messages.

    parallel : bool, optional
        Whether to use parallel processing.

    nb_cores : int, optional
        Number of CPU cores to use.

    Returns
    -------
    gps : pandas.DataFrame
        Updated GPS data with edge information.

    gps_mm : pandas.DataFrame
        Map-matched GPS data. Each ID_trip with most likely path in the graph
    """
    G_mm, trans, dic_candidates, id2edges = mm_precomputation(G, beta=beta, radius=radius, save=save, folder_name=folder_name, show_print=show_print)
    if parrallel:
        G_mm, gps, gps_mm = mm_gps_parrallel(gps, G_mm, trans, dic_candidates, id2edges, alpha=alpha, radius=radius, nb_cores=nb_cores)
    else:
        G_mm, gps, gps_mm = mm_gps(gps, G_mm, trans, dic_candidates, id2edges, alpha=alpha, radius=radius)
    return(G_mm, gps, gps_mm)

def one_traj_mm(GPS_traj, graph, transition_matrix, emission_matrix, id2edges, sub_edges, start, end):
    """
    Perform map-matching on a single GPS trajectory.

    Parameters
    ----------
    GPS_traj : numpy.ndarray
        Array containing GPS observations.

    graph : networkx.DiGraph
        The road network graph.

    transition_matrix : scipy.sparse.csr_matrix
        Transition matrix for the graph.

    emission_matrix : scipy.sparse.csr_matrix
        Emission matrix for the graph.

    id2edges : dict
        Dictionary mapping edge_id to edge of the graph.

    sub_edges : list
        List of edge IDs candidates considered for map-matching.

    start : int
        Start index in emission_matrix.

    end : int
        End index in emission_matrix).

    Returns
    -------
    edges : list
        List of edges matched to the GPS trajectory.

    path : list
        List of nodes representing the matched path on the road network graph.
    """
    try:
        emit_p = (emission_matrix[start:end+1, : ][:, sub_edges].toarray())/10
        trans_p = (transition_matrix[sub_edges, :][:, sub_edges].toarray())/100
        start_p = np.ones(len(sub_edges))/len(sub_edges)
        obs = np.array([i for i in range(len(GPS_traj))])
        states = np.array([i for i in range(len(sub_edges))])
        (V,prob, state,path) = fast_viterbi(obs, states, start_p, trans_p, emit_p)
        if prob != 0:
            edges = [id2edges[sub_edges[int(i)]] for i in state]
            edges_without_redundancy = [i[0] for i in itertools.groupby(edges)]
            path = [nx.shortest_path(graph, edges_without_redundancy[i][1], edges_without_redundancy[i+1][0]) for i in range(len(edges_without_redundancy) - 1)]
            path = [edges_without_redundancy[0][0]] + [item for sublist in path for item in sublist] + [edges_without_redundancy[-1][1]]
            return(edges, path)
        else:
            return([np.nan]*len(GPS_traj),[])
    except:
        return(["Problem"], ["Problem"])

@numba.jit(nopython=True)
def fast_viterbi(obs, states, start_p, trans_p, emit_p):
    """
    Implement the Viterbi algorithm for map-matching.

    Parameters
    ----------
    obs : numpy.ndarray
        Array containing observation indices.

    states : numpy.ndarray
        Array containing state indices.

    start_p : numpy.ndarray
        Initial state probabilities.

    trans_p : numpy.ndarray
        Transition probabilities.

    emit_p : numpy.ndarray
        Emission probabilities.

    Returns
    -------
    V : numpy.ndarray
        Viterbi matrix.

    prob : float
        Probability of the most likely path.

    seq_nodes : numpy.ndarray
        Sequence of node indices representing the most likely path.

    path : numpy.ndarray
        most likely path matrix.
    """
    nb_obs = obs.shape[0]
    nb_states = states.shape[0]

    V = np.zeros((nb_obs, nb_states))
    path = np.zeros((1,nb_states))

    # Initialize base cases (t == 0)
    for idx, y in enumerate(states):
        V[0, y] = start_p[y] * emit_p[ obs[0],y]
        path[0, idx] = y

    for itime in range(1, nb_obs):   
        states_cand = np.where(emit_p[ obs[itime],:] != 0)[0]
        newpath = np.zeros((itime + 1 ,states_cand.shape[0]))
        states_bis =np.where(V[itime-1, :] != 0)[0]
        for idx, y in enumerate(states_cand):
            prob = 0
            for y0 in states_bis:
                if V[itime-1, y0] * trans_p[y0,y] * emit_p[ obs[itime],y] > prob:
                    prob = V[itime-1, y0] * trans_p[y0,y] * emit_p[ obs[itime],y]
                    state = y0
            V[itime, y] = prob
            if prob>0: #check this if problem
                row = np.where(path[-1,:] == state)[0][0]
                newpath[:itime, idx] = path[:, row]
                newpath[-1, idx] = y
        # Don't need to remember the old paths
        path = newpath

    idx_max = np.argmax(V[itime,:])
    (prob, state) = V[itime, idx_max], idx_max
    if prob !=0:
        row = np.where(path[-1,:] == state)[0][0]
        seq_nodes = path[:, row]
    else:
        path = np.zeros((1,nb_states))
        seq_nodes = np.zeros((1,nb_states))[:,0]
    return (V,prob, seq_nodes,path)