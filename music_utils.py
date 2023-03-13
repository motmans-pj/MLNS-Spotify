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
    :param df_song: a dataframe of songs that have more than one artist, and are younger than a certain year
    :param DATA_PATH: where to find or save data
    :param read: boolean variable that states whether file already saved locally
    :param path: where to find or save the edge_list

    :return: edge list with number of featurings and songs in which they collaborated
    """
    if read:
        nodes_df = pd.read_pickle(DATA_PATH + path)
    else:
        # create a list of dictionaries for the edge list
        edges = []
        for _, x in df_song.iterrows():
            # generate all pairs of artists
            artists = sorted(x.id_artists)
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
    :param DATA_PATH: where to find or save data
    :param spotify_path: original csv file with tracks
    :param artists_path: original csv file with artists
    :param pkl_spotify_path: saved pickle for tracks
    :param pkl_artists_path: saved pickle for artists

    :return: processed spotify and artists dataframes
    """

    if read:
        spotify_600 = pd.read_pickle(DATA_PATH + pkl_spotify_path)
        artists_600 = pd.read_pickle(DATA_PATH + pkl_artists_path)

    else:
        spotify_600 = pd.read_csv(DATA_PATH + spotify_path)
        artists_600 = pd.read_csv(DATA_PATH + artists_path)

        spotify_600 = spotify_600.rename(columns={'id': 'track_id', 'popularity': 'track_popularity'})

        spotify_600['artists'] = spotify_600.artists.apply(lambda x: eval(x))
        spotify_600['id_artists'] = spotify_600.id_artists.apply(lambda x: eval(x))
        spotify_600['artist_id'] = spotify_600.id_artists.apply(lambda x: x[0])
        spotify_600['num_artists'] = spotify_600.artists.apply(lambda x: len(x))
        spotify_600['release_date'] = pd.to_datetime(spotify_600.release_date)

        artists_600 = artists_600.rename(columns={'id': 'artist_id', 'popularity': 'artist_popularity'})
        artists_600['genres'] = artists_600.genres.apply(lambda x: eval(x))

        spotify_600.to_pickle(DATA_PATH + pkl_spotify_path)
        artists_600.to_pickle(DATA_PATH + pkl_artists_path)

    return spotify_600, artists_600


def artists_features_creation(artists_600, spotify_600, DATA_PATH, read=True,
                              pkl_features_artist_path='features_artists_600k.pkl',
                              ):
    """
    :param artists_600: the dataframe with artists
    :param spotify_600: the dataframe with tracks
    :param DATA_PATH: where to save or read the data
    :param read: has data already been saved to pickle? then just read
    :param pkl_features_artist_path: where to save or read artists features data

    :return: dataframe with for each artist the features related to tracks in which they were involved
    """

    if read:
        artists_600_features = pd.read_pickle(DATA_PATH + pkl_features_artist_path)

    else:
        # we select all the artists that are in the spotify tracks
        artists_600_features = artists_600[artists_600.artist_id.isin(spotify_600.id_artists.explode().unique())].copy()
        # features: number of genres
        artists_600_features['num_genres'] = artists_600_features.genres.apply(lambda x: len(x))
        # set artist_id as index
        artists_600_features = artists_600_features.set_index('artist_id')
        # the features we want from spotify track
        feature_cols = ['track_popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key',
                        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo', 'time_signature']

        ##### SOLO TRACKS
        # features for solo tracks of artists
        solo_features = spotify_600[spotify_600.num_artists == 1].groupby('artist_id').agg({
            a: 'mean' for a in feature_cols
        })

        ##### FEATS INITIATED
        # features for featured tracks of artists: mean for track initiated by the artist
        initiatied_feat_features = spotify_600[spotify_600.num_artists > 1].groupby('artist_id').agg({
            a: 'mean' for a in feature_cols
        })

        ##### FEATS MEMBER OF
        # features for featured tracks of artists: mean for tracks where the artist is just part of it
        cop_spot_600 = spotify_600[spotify_600.num_artists > 1].copy()
        cop_spot_600.artists = cop_spot_600.artists.apply(lambda x: x[1:])

        # split artist column into multiple rows
        cop_spot_600 = cop_spot_600.explode('id_artists')

        # group by artist and compute mean values
        member_feat_features = cop_spot_600.groupby('id_artists').agg({
            a: 'mean' for a in feature_cols
        })
        # renaming for the join

        ###### MERGES
        artists_600_features = artists_600_features.join(solo_features.add_prefix('solo_'),
                                                         how='left'
                                                         )

        artists_600_features = artists_600_features.join(initiatied_feat_features.add_prefix('initiated_'),
                                                         how='left', )

        artists_600_features = artists_600_features.join(member_feat_features.add_prefix('member_'),
                                                         how='left')

        ##### SAVE
        artists_600_features.to_pickle(DATA_PATH + pkl_features_artist_path)

    return artists_600_features


# Yearly featurings functions
def get_largest_cc_by_year(spotify_600, DATA_PATH, read, yearly_dir_name="yearly_600k"):
    """
    :param spotify_600: song dataframe
    :param DATA_PATH: where to find or save data
    :param read: read or create data
    :param yearly_dir_name: directory to save

    :return:  returns two dictionaries, one with a dataframe for every year and one for the
    largest_cc for every year.
    """
    # Create year column
    df_600 = spotify_600.copy()
    df_600["year"] = [date.year for date in df_600.release_date]
    # Get range of years
    min_year, max_year = min(df_600.year), max(df_600.year)
    songs_by_year = {}
    nodes_yearly = {}
    largest_cc_year = {}
    for year in range(min_year, max_year + 1):
        # Songs for a given year
        songs_by_year[year] = df_600[df_600.year == year].copy()
        # Featurings for that year
        featurings_year = songs_by_year[year][songs_by_year[year].num_artists > 1]
        # Check that we have songs in that year
        if len(featurings_year) == 0:
            continue
        # Create nodes for that year and the largest cc
        nodes_yearly[year] = nodes_featuring(featurings_year, DATA_PATH=DATA_PATH, read=read,
                                             path=f'/{yearly_dir_name}/edge_list{year}.pkl')
        edge_list = [tuple(l[:2]) for l in nodes_yearly[year].values.tolist()]
        G = ntx.from_edgelist(edge_list)
        largest_cc_year[year] = G.subgraph(max(ntx.connected_components(G), key=len))
    return songs_by_year, largest_cc_year


def draw_graph(G, with_labels=False, node_size=20, fig_size=(15, 15)):
    """
    Function that draws a graph.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ntx.draw_networkx(G, with_labels=with_labels, node_size=node_size, ax=ax)
