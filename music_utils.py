import numpy as np
import csv
import networkx as ntx
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from music_utils import *
from tqdm.auto import tqdm  # import tqdm for progress bar
tqdm.pandas()
from gensim.models import Word2Vec
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

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

        # solo_features = spotify_600[spotify_600.num_artists == 1].groupby('artist_id').agg({
        #    a: 'mean' for a in feature_cols
        # })

        ##### FEATS INITIATED
        # features for featured tracks of artists: mean for track initiated by the artist
        # initiatied_feat_features = spotify_600[spotify_600.num_artists > 1].groupby('artist_id').agg({
        #    a: 'mean' for a in feature_cols
        # })

        ##### FEATS MEMBER OF
        # features for featured tracks of artists: mean for tracks where the artist is just part of it
        # cop_spot_600 = spotify_600[spotify_600.num_artists > 1].copy()
        # cop_spot_600.artists = cop_spot_600.artists.apply(lambda x: x[1:])
        #
        # split artist column into multiple rows
        # cop_spot_600 = cop_spot_600.explode('id_artists')

        # group by artist and compute mean values
        # member_feat_features = cop_spot_600.groupby('id_artists').agg({
        #    a: 'mean' for a in feature_cols
        # })
        # renaming for the join

        ####################
        # Average features

        avg_features = spotify_600.explode('id_artists').groupby('id_artists').agg({
            a: 'mean' for a in feature_cols
        })

        ###### MERGES
        # artists_600_features = artists_600_features.join(solo_features.add_prefix('solo_'),
        #                                                how='left'
        #                                                )

        # artists_600_features = artists_600_features.join(initiatied_feat_features.add_prefix('initiated_'),
        #                                                 how='left', )

        # artists_600_features = artists_600_features.join(member_feat_features.add_prefix('member_'),
        #                                                how='left')

        artists_600_features = artists_600_features.join(avg_features.add_prefix('avg_'),
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


def draw_graph(G, with_labels=False, node_size=20, fig_size=(15, 15), colors = None):
    """
    Function that draws a graph.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ntx.draw_networkx(G, with_labels=with_labels, node_size=node_size, ax=ax, node_colors = colors)


def explosive_df(df, column: str):
    """
    Function to go from a dataframe obtained with the explode function to the df from before.
    """
    def implode(data_frame=()):
        inner_df = data_frame if len(data_frame) > 0 else df

        def reduce_columns():
            acc = {}
            for col in inner_df.columns:
                if col == column:
                    acc[col] = lambda x: x.dropna().tolist()
                    continue
                acc[col] = lambda x: x.iloc[0]

            return acc

        return inner_df.groupby(inner_df.index, as_index=False).agg(
            reduce_columns()
        ).reset_index(drop=True)

    return df.explode(column), implode


def change_genres(genre):
    """
    This function takes in a genre, and will create 'overarching' genres.
    If a genre is french rap, it will become rap
    It is not perfect, because dance pop will become dance in this case.
    Input: a single genre (exploded artists table based on genre)
    Return: the changed genre
    """
    if 'dance' in genre:
        genre = 'dance'
    elif 'classical' in genre:
        genre = 'classical'
    elif 'rap' in genre or 'hip hop' in genre:
        genre = 'rap'
    elif 'latin' in genre:
        genre = 'latin'
    elif 'rock' in genre:
        genre = 'rock'
    elif 'pop' in genre:
        genre = 'pop'
    return genre


def retrieve_popular_genres(genres):
    """
    :param genres: a dataframe of genres sorted by number of artists
    :return: a dataframe with the six most popular genres
    """
    genres.loc[:, 'genre'] = genres.genre.apply(change_genres)
    genres = genres.groupby('genre').agg({
        'number_of_artists': 'sum'
    }).reset_index()
    popular_genres = genres.sort_values('number_of_artists', ascending=False).iloc[:6, :]
    return popular_genres


def get_graph_features_node_classification(genres, artists_600, artists_600_features, nodes_600):
    """
    :param genres: a dataframe of genres sorted by number of artists
    :param artists_600: dataframe of artists from which we will filter only artists in a popular genre
    :param artists_600_features: after filtering we select the features
    :param nodes_600: df containing edge lists and number of song featurings
    :return: Graph and Feature Matrix
    """

    popular_genres = retrieve_popular_genres(genres)

    # select only artists with at least 1 genre
    artists_600['num_genres'] = artists_600.genres.apply(lambda x: len(x))
    artists_with_genres = artists_600[artists_600.num_genres >= 1].copy().explode('genres')
    artists_with_genres['genres'] = artists_with_genres.genres.apply(change_genres)
    artists_with_genres['popular'] = artists_with_genres['genres'].apply(lambda x: x in list(popular_genres.genre))
    artists_popular_genres = artists_with_genres[artists_with_genres.popular == True]
    artists_popular_genres['idx'] = artists_popular_genres.index
    exploded, implode = explosive_df(artists_popular_genres, 'genres')
    artists_node_classification = implode(exploded).set_index('idx')
    artists_node_classification.drop(columns=['popular', 'num_genres'], inplace=True)

    # then we select as genre only the least popular genre that artist belongs to.
    # this is done 1) to balance classes
    # 2) because it is usually more characteristic to an artist
    # many artists might have done 'pop' once, but if they usually do classical
    # they should be considered as such.
    for row in artists_node_classification.itertuples():
        for popular_genre in list(popular_genres.genre)[::-1]:
            # Assign the most popular genre that the artist has
            if popular_genre in row.genres:
                artists_node_classification.at[row.Index, 'genres'] = popular_genre
                break

    # On this dataframe of artists, we merge the features matrix
    data = artists_node_classification.merge(artists_600_features, how='left', on='artist_id').set_index('artist_id')
    data.dropna(subset='name_y', inplace=True)

    # Then we select only the edges for which the source and target are both in the table of artists that belong
    # to a popular genre. From that, we create an edgelist (edge_info is a dataframe)
    edge_info = nodes_600[(nodes_600.artist_1.isin(data.index.unique())) &
                          (nodes_600.artist_2.isin(data.index.unique()))]

    edgelist = [tuple(l[:2]) for l in edge_info.values.tolist()]
    print(edgelist[:5])
    # Create the graph
    G = ntx.from_edgelist(edgelist)
    # Select the largest connected component
    S = [G.subgraph(c).copy() for c in ntx.connected_components(G)]
    graph = S[0]

    ## Then we create the feature matrix
    # Make sure the index is not lost when we create the feature matrix
    data['artist_id'] = data.index
    node_info = pd.DataFrame([list(data.loc[idx]) for idx in list(graph.nodes)],
                             columns=data.columns).set_index('artist_id')
    # Drop duplicate columns
    node_info.drop(['followers_y', 'genres_y', 'name_y', 'artist_popularity_y'], axis=1, inplace=True)

    return graph, node_info


class DeepWalk():
    '''
    Using this class: after initializing, run the generate_walks method to generate the random walks.
    To generate the embeddings, run the skipgram method.
    '''

    def __init__(self, G: ntx.Graph, window: int, dims: int, num_walks: int, walk_length: int):
        '''
        G: the graph of which the node embeddings will be created
        window: the window size
        dims: the embedding size
        num_walks: The number of random walks performed for each individual node
        walk_length: The random walk length
        '''

        self.G = G
        self.window = window
        self.dims = dims
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.walks = None

    def generate_walks(self):
        '''
        Generating the walks
        '''

        # store all vertices
        V = list(self.G.nodes)
        self.walks = []
        for _ in range(self.num_walks):
            # each iteration here makes a pass over the data to sample one walk for each node
            # the random order of the vertices speeds up the convergence of SGD
            random.shuffle(V)
            for v in V:
                self.walks.append(self.RandomWalk(v))
        return self.walks

    def RandomWalk(self, v):
        # v is the root
        walk = [v]
        current = v
        for _ in range(self.walk_length):
            # neighbors of the current node
            neighbors = [node for node in self.G.neighbors(current)]
            # choose a neighbor to go to
            current = np.random.choice(neighbors)
            walk.append(current)

        return walk

    def skipgram(self):
        '''
        Calling the Word2Vec model that we will implement ourselves
        '''

        model = Word2Vec(sentences=self.walks, vector_size=self.dims, window=self.window, workers=4)

        # return the embeddings (word vectors)
        return model.wv


def visualize(graph, node2embedding, y, id_to_number, labeled=False):
    nodelist = list(graph.nodes())
    y = list(y)

    # Embedding vectors
    x = np.asarray([node2embedding[id_to_number[node]] for node in nodelist])
    x_embedded = TSNE(n_components=2).fit_transform(x)
    print(f'TSNE succesfully applied: x now of shape: {x_embedded.shape}')
    df = pd.DataFrame({"Axis 1": x_embedded[:, 0], "Axis 2": x_embedded[:, 1]})

    if not labeled:
        # plt.figure(figsize=(10,10))
        ax = sns.scatterplot(
            x="Axis 1", y="Axis 2",
            # palette=sns.color_palette(["mediumseagreen", "navy"]),
            data=df,
            alpha=0.8
        )
    else:
        # plt.figure(figsize=(10,10))
        ax = sns.scatterplot(
            x="Axis 1", y="Axis 2",
            # palette=sns.color_palette(["navy", "mediumseagreen"]),
            data=df,
            alpha=0.8,
            hue=y
        )

    plt.figure(figsize=(20, 20))


def predict(embeddings, y, model='lr'):
    """
    :param G: the networkx graph
    :param embeddings: embeddings created by deepwalk
    :param return_f1_score: return f1 scores (both binary and macro f1), select this when you want to keep track of scores
    :param return_train_test: return the train and test labels for later use (keep same test set over all benchmarks)
    :param model: the model to use
    This function runs an ML model on the embeddings generated by deepwalk
    """
    plt.set_cmap('RdYlGn')

    y = list(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, shuffle=True, stratify=y)

    sample_weights = compute_sample_weight('balanced', y_train)

    # Train logistic regression
    if model == 'lr':
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_train, y_train)

    elif model == 'xgb':
        # need to have done the label encoding
        model = XGBClassifier(max_depth=10)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    elif model == 'rf':
        model = RandomForestClassifier(max_depth=10, class_weight='balanced')
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate predictions

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f'Macro test F1: {macro_f1}')
    accuracy = sum(y_test == y_pred) / len(y_pred)
    print(f'Accuracy: {accuracy}')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    return macro_f1