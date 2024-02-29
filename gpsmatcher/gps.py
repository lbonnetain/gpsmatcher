import os

os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeohash_fast


def process_gps(gps, cand_edge):
    """
    Process GPS data by encoding geohashes, mapping edges, and creating a GeoDataFrame.

    Parameters
    ----------
    gps : pandas.DataFrame
        DataFrame containing GPS data with columns 'lon', 'lat'.

    cand_edge : dict
        Dictionary mapping geohashes to candidate edges.

    Returns
    -------
    gps : gpd.GeoDataFrame
        GeoDataFrame with additional columns 'id', 'geohash_int', and 'geometry'.
    
    gps_mm : pandas.DataFrame
        DataFrame containing aggregated GPS data per trip, with columns 'traj', 'first_emission', 'last_emission', 'sub_edges'.
    
    dic_geohash : dict
        Dictionary mapping geohashes to integer indices.
    
    dic_candidates : dict
        Dictionary mapping geohashes to candidate edges.
    """
    gps.reset_index(drop=True, inplace=True)
    gps['id'] = range(len(gps))
    gps["id"] = gps["id"].astype(np.uint32)
    gps['geohash']   = pygeohash_fast.encode_many(gps['lon'].values, gps['lat'].values, 8)
    all_geohashes = set(gps['geohash'] )
    dic_geohash = dict(zip(all_geohashes, range(len(all_geohashes))))
    dic_candidates = {k: cand_edge[k] for k in cand_edge.keys() & all_geohashes}

    gps['edge'] = gps['geohash'].map(dic_candidates)
    gps['geohash_int'] = gps['geohash'].map(dic_geohash)
    gps.set_index('geohash_int', inplace=True)
    gps = gpd.GeoDataFrame(gps, geometry=gpd.points_from_xy(gps['lon'], gps['lat']), crs=4326)
    gps.to_crs(3035, inplace=True)
    gps['points'] = list(zip(gps['lon'], gps['lat']))
    
    gps_mm = gps.groupby('ID_trip').agg({'points' : lambda row: np.array(list(row), dtype=float),  
                                            "id":['first', 'last'], 
                                            "edge":  sum})
    gps_mm.columns = ['traj', 'first_emission', 'last_emission', 'sub_edges']
    gps_mm = gps_mm[gps_mm['sub_edges']!=0]
    gps_mm['sub_edges'] = gps_mm['sub_edges'].apply(lambda row: list(set(row)))
    gps.drop(columns=['geohash', 'points', "edge"], inplace=True)
    return(gps, gps_mm, dic_geohash, dic_candidates)