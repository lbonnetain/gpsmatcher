import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pygeohash_fast
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import snap, split

from gpsmatcher.utils import (load_param, print_step, remove_unneccesarry_att,
                              save_param)


def compute_edge_length(G, show_print=True):
    """
    Compute and assign lengths to edges in a networkx graph based on their geometries.

    Parameters
    ----------
    G : networkx.DiGraph
        The graph with edges containing 'geometry' attributes.

    show_print : bool, optional
        Whether to print a message indicating the completion of the computation. Default is True.

    Returns
    -------
    G : networkx.DiGraph
        The input graph with added 'length' attributes for each edge.
    """
    
    edges2geom = nx.get_edge_attributes(G, 'geometry')
    geom_edges = pd.DataFrame(edges2geom.items(),columns=['edge', 'geometry'])
    geom_edges['points'] = geom_edges.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
    geom_points = geom_edges.explode('points')
    geom_points[['lon', 'lat']] = pd.DataFrame(geom_points['points'].tolist(), index=geom_points.index)
    geom_points['dist'] = haversine(geom_points["lat"], geom_points["lon"], geom_points["lat"].shift(-1), geom_points["lon"].shift(-1))
    geom_points['dist'] = 1000 * geom_points['dist'] * (geom_points['edge'] == geom_points['edge'].shift(-1))
    edges2length = geom_points.groupby('edge')['dist'].sum().to_dict()
    nx.set_edge_attributes(G, edges2length, 'length')
    
    if show_print:
        print("Compute edge legnth done")
    return(G)


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Calculate the great circle distance between two points on the Earth.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Latitude and longitude coordinates of two points.

    to_radians : bool, optional
        Whether the input coordinates are in radians. Default is True.

    earth_radius : float, optional
        Earth's radius in kilometers. Default is 6371.

    Returns
    -------
    distance : float
        The great circle distance between the two points.
    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def geom2length(geom_edge):
    """
    Calculate the length of a LineString geometry.

    Parameters
    ----------
    geom_edge : shapely.geometry.LineString
        LineString geometry.

    Returns
    -------
    length : float
        Length of the LineString in meters.
    """
    coords = np.array(geom_edge.coords)
    length = 1000 *sum(haversine(coords[:-1, 1], coords[:-1, 0], coords[1:, 1], coords[1:, 0]))
    return(length)

def interpolate_graph(G, max_length = 500, show_print=True):
    """
    Interpolate long edges in a graph by adding new nodes and edges.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    max_length : float, optional
        Maximum length for an edge. If an edge's length exceeds this value, it will be interpolated.
        Default is 500 meters.

    show_print : bool, optional
        Whether to print a message indicating the completion of the interpolation. Default is True.

    Returns
    -------
    G : networkx.DiGraph
        The graph with interpolated edges.
    """
    edges2length =nx.get_edge_attributes(G, 'length')
    long_edges = [key for key, value in edges2length.items() if value >= max_length]
    start_node_id = max(G.nodes) + 1
    for edge in long_edges:
        atts_edges = G.edges[edge]
        atts_nodes = {key: value for (key, value) in G.nodes[edge[0]].items()}
        nb_new_nodes = int(atts_edges['length'] // max_length)
        line = atts_edges['geometry']
        new_nodes = MultiPoint([Point(line.coords[0])] + [line.interpolate(i/(nb_new_nodes + 1), normalized=True) for i in range(1,nb_new_nodes+1)] + [Point(line.coords[-1])])
        int_nodes = new_nodes.geoms[1:-1].geoms
        for j in range(len(int_nodes)):
            atts_nodes['x'] = int_nodes[j].x
            atts_nodes['y'] = int_nodes[j].y
            atts_nodes['edge'] = edge
            G.add_node(start_node_id + j, **atts_nodes)

        new_edges = split(snap(line, new_nodes, 1), new_nodes).geoms
        parent_edge_length, parent_edge_weight = atts_edges['length'], atts_edges['weight']
        
        for i in range(len(new_edges)):
            atts_edges['geometry'] = new_edges[i]
            atts_edges['length'] = geom2length(atts_edges['geometry'])
            atts_edges['weight'] = parent_edge_weight * atts_edges['length'] / parent_edge_length
            atts_edges['parent_edges'] = edge
            if i == 0:
                from_node = edge[0]
                to_node = start_node_id
            elif i == len(new_edges) - 1:
                from_node = start_node_id
                to_node = edge[1]
            else:
                from_node = start_node_id
                start_node_id += 1
                to_node = start_node_id
            G.add_edge(from_node, to_node, **atts_edges)
        start_node_id += 1
        G.remove_edge(edge[0], edge[1])
        
    if show_print:
        print("interpolate graph - interpolation distance: {} m - done".format(max_length))
    return(G)

def get_listgeohahses_edge(G, show_print=True):
    """
    Get geohashes associated with each edge in a graph.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    G : networkx.Graph
        The graph with added 'geohashes' attribute for each edge.
    """
    edges2interpolation = interpolation_edge_geom(G, dist_max = 5)
    geom_interpolation = pd.DataFrame(edges2interpolation.items(),columns=['edge', 'geometry'])
    geom_interpolation = gpd.GeoDataFrame(geom_interpolation, geometry=geom_interpolation['geometry'])
    geom_interpolation = geom_interpolation.explode('geometry', index_parts=True)
    geom_interpolation["lon"] = geom_interpolation["geometry"].x
    geom_interpolation["lat"] = geom_interpolation["geometry"].y
    geom_interpolation['geohash']   = pygeohash_fast.encode_many(geom_interpolation['lon'].values, geom_interpolation['lat'].values, 8)
    edge_geohash = geom_interpolation.groupby('edge')['geohash'].apply(set).apply(list).to_dict()
    nx.set_edge_attributes(G, edge_geohash, 'geohashes')
    
    if show_print:
        print("Get geoashes of edges done")
    return(G)

def edge_subsampling(edge_geom, edge_length, dist_max = 5):
    """
    Subsample points along an edge geometry to achieve a specified maximum distance between points.

    Parameters
    ----------
    edge_geom : shapely.geometry.LineString
        LineString geometry representing an edge.

    edge_length : float
        Length of the edge.

    dist_max : float, optional
        Maximum distance between subsampled points. Default is 5 meters.

    Returns
    -------
    geom_edge : shapely.geometry.MultiPoint
        Subsampled points along the edge geometry.
    """
    geom_edge, len_edge = edge_geom, round(edge_length) 
    if len_edge > dist_max:
        geom_edge = MultiPoint([geom_edge.interpolate(i/len_edge, normalized=True) for i in range(0,len_edge+1,dist_max)])
    elif len_edge ==0:
        geom_edge = MultiPoint(list(geom_edge.coords))
    else:
        geom_edge = geom_edge.boundary
    return(geom_edge)

def interpolation_edge_geom(G, dist_max = 5):
    """
    Interpolate edge geometries in a graph by subsampling points to achieve a specified maximum distance.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    dist_max : float, optional
        Maximum distance between subsampled points. Default is 5 meters.

    Returns
    -------
    edge2newinterpolatedgeom : dict
        Dictionary mapping edges to their interpolated geometries.
    """
    interpolated_geom = [edge_subsampling(G.edges[i]['geometry'],G.edges[i]['length'], dist_max = dist_max) for i in G.edges]
    keys = list(G.edges)
    edge2newinterpolatedgeom = dict(zip(keys, interpolated_geom))
    return(edge2newinterpolatedgeom)

def fill_missing_geometry(G, show_print=True):
    """
    Fill missing 'geometry' attributes in edges of a graph by creating a LineString between nodes when missing.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    show_print : bool, optional
        Whether to print a message indicating the completion of the operation. Default is True.

    Returns
    -------
    G : networkx.Graph
        The graph with filled 'geometry' attributes when missing.
    """
    for u, v in G.edges:
        atts = G.edges[u,v].keys()
        if 'geometry' not in atts:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            edge_geom = LineString((Point((u_node['x'], u_node['y'])), Point((v_node['x'], v_node['y']))))
            G.edges[u,v]['geometry'] = edge_geom
            
    if show_print:
        print("Fill missing geometry done")
    return(G)



def process_graph(G, max_length = 500, save=True, folder_name="mm_input", show_print=True):
    """
    Process a graph by removing unnecessary attributes, filling missing geometries, computing edge lengths,
    interpolating long edges, assigning geohashes to edges, and optionally saving the processed graph.

    Parameters
    ----------
    G : networkx.DiGraph
        The input graph.

    max_length : float, optional
        Maximum length for an edge. If an edge's length exceeds this value, it will be interpolated.
        Default is 500 meters.

    save : bool, optional
        Whether to save the processed graph. Default is True.

    folder_name : str, optional
        Folder name for saving/loading. Default is "mm_input".

    show_print : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    G : networkx.DiGraph
        The processed graph.

    id2edges : dict
        Dictionary mapping orignal edge to a new edge_id.
    """
    if show_print:
        print_step("Start the process graph")
        
    if save and (os.path.isfile(folder_name + '/graph.pickle')):
        G = load_param(folder_name, "graph", show_print=show_print)
        edges = nx.to_pandas_edgelist(G)
        edges['edge'] = list(zip(edges["source"], edges["target"]))
        id2edges = edges.set_index("edge_id")['edge'].to_dict()
        edges2id = edges.set_index("edge")['edge_id'].to_dict()
    
    else:
        G = remove_unneccesarry_att(G)
        G = fill_missing_geometry(G, show_print=show_print)
        G = compute_edge_length(G, show_print=show_print)
        G = interpolate_graph(G, max_length = max_length, show_print=show_print)
        
        edges = nx.to_pandas_edgelist(G)
        edges['edge'] = list(zip(edges["source"], edges["target"]))
        edges['edge_id'] = range(len(edges))
        edges2id = edges.set_index("edge")['edge_id'].to_dict()
        id2edges = edges.set_index("edge_id")['edge'].to_dict()
        nx.set_edge_attributes(G, edges2id, 'edge_id')
        G = get_listgeohahses_edge(G, show_print=show_print)

        if save:
            save_param(folder_name, "graph", G)
    return(G, id2edges, edges2id)