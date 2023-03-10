import numpy as np
import csv
import networkx as ntx
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
###################
# Utils for link prediction 
###################



def nodes_featuring(df_song, DATA_PATH, read=True, path='edge_list_600k.pkl'):
    """
    This one is surprisingly super fast. Thanks chat gpt
    """
    if read:
        nodes_df = pd.read_pickle(DATA_PATH + path)
    else:
        # create a list of dictionaries for the edge list
        edges = []
        for _, x in df_song.iterrows():
            # generate all pairs of artists
            artists = sorted(x.artists)
            pairs = list(itertools.combinations(artists, 2))
            # append the pairs to the edges list
            for pair in pairs:
                edges.append({'artist_1': pair[0], 'artist_2': pair[1], 'number': 1, 'track_ids': [x.track_id]})
        # create the edge list dataframe
        prep_df = pd.DataFrame(edges)
        # compute the number of collaborations between each pair of artists
        nodes_df = pd.crosstab(index=[prep_df['artist_1'], prep_df['artist_2']], 
                               values=prep_df['number'], aggfunc='sum',
                               columns=['num_feats']).reset_index()
        nodes_df['track_ids'] = pd.crosstab(index=[prep_df['artist_1'], prep_df['artist_2']], 
                                            values=prep_df['track_ids'], aggfunc='sum',
                                            columns=['track_ids']).reset_index()['track_ids']
        # save the edge list dataframe to a pickle file
        nodes_df.to_pickle(DATA_PATH + path)

    return nodes_df


def read_spotify_600(DATA_PATH, read=True, 
spotify_path='tracks_600k.csv', 
artists_path='artists_600k.csv',
pkl_spotify_path='tracks_600k_processed.pkl', 
pkl_artists_path='artists_600k_processed.pkl',
):
    """
    I did a function because it takes 1min12sec sec to run. Boring
    """

    if read:
        spotify_600 = pd.read_pickle(DATA_PATH+pkl_spotify_path)
        artists_600 = pd.read_pickle(DATA_PATH+pkl_artists_path)
        
    else :
        spotify_600 = pd.read_csv(DATA_PATH+spotify_path)
        artists_600 = pd.read_csv(DATA_PATH+artists_path)
        
        spotify_600 = spotify_600.rename(columns={'id':'track_id', 'popularity':'track_popularity'})

        spotify_600['artists'] = spotify_600.artists.apply(lambda x: eval(x))
        spotify_600['id_artists'] = spotify_600.id_artists.apply(lambda x: eval(x))
        spotify_600['artist_id'] = spotify_600.id_artists.apply(lambda x: x[0])
        spotify_600['num_artists'] = spotify_600.artists.apply(lambda x: len(x))
        spotify_600['release_date'] = pd.to_datetime(spotify_600.release_date)


        artists_600 = artists_600.rename(columns={'id':'artist_id', 'popularity':'artist_popularity'})
        artists_600['genres'] = artists_600.genres.apply(lambda x: eval(x))

        spotify_600.to_pickle(DATA_PATH+pkl_spotify_path)
        artists_600.to_pickle(DATA_PATH+pkl_artists_path)

    return spotify_600, artists_600



def artists_features_creation(artists_600, spotify_600, DATA_PATH, read=True, 
pkl_features_artist_path='features_artists_600k.pkl', 
):
    """
    I did a function because it takes 1min12sec sec to run. Boring
    """

    if read:
        artists_600_features = pd.read_pickle(DATA_PATH+pkl_features_artist_path)
        
    else:
        #we select all the artists that are in the spotify tracks
        artists_600_features = artists_600[artists_600.artist_id.isin(spotify_600.id_artists.explode().unique())].copy()
        #features: number of genres
        artists_600_features['num_genres'] = artists_600_features.genres.apply(lambda x: len(x))
        #set artist_id as index
        artists_600_features = artists_600_features.set_index('artist_id')
        #the features we want from spotify track
        feature_cols = ['track_popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key',
       'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
       'liveness', 'valence', 'tempo', 'time_signature']
       
        ##### SOLO TRACKS
        #features for solo tracks of artists
        solo_features = spotify_600[spotify_600.num_artists == 1].groupby('artist_id').agg({
            a:'mean' for a in feature_cols
        })

        ##### FEATS INITIATED
        #features for featured tracks of artists: mean for track initiated by the artist
        initiatied_feat_features = spotify_600[spotify_600.num_artists > 1].groupby('artist_id').agg({
            a:'mean' for a in feature_cols
        })

        ##### FEATS MEMBER OF
        #features for featured tracks of artists: mean for tracks where the artist is just part of it
        cop_spot_600 = spotify_600[spotify_600.num_artists > 1].copy()
        cop_spot_600.artists = cop_spot_600.artists.apply(lambda x: x[1:])

        # split artist column into multiple rows
        cop_spot_600 = cop_spot_600.explode('id_artists')

        # group by artist and compute mean values
        member_feat_features = cop_spot_600.groupby('id_artists').agg({
            a:'mean' for a in feature_cols
        })
        #renaming for the join
        
        ###### MERGES
        artists_600_features = artists_600_features.join(solo_features.add_prefix('solo_'), 
        how='left'
        )

        artists_600_features = artists_600_features.join(initiatied_feat_features.add_prefix('initiated_'), 
        how='left',)

        artists_600_features = artists_600_features.join(member_feat_features.add_prefix('member_'), 
        how='left')
        
        ##### SAVE
        artists_600_features.to_pickle(DATA_PATH+pkl_features_artist_path)

    return artists_600_features