import numpy as np
import numpy.testing as npt
import os
import re
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import gluonnlp as nlp
import networkx as nx
import hetergraph
import dgl

READ_DATASET_PATH = os.path.join("data_set")
GENRES_ML_100K =\
    ['unknown', 'Action', 'Adventure', 'Animation',
     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
     'Thriller', 'War', 'Western']
GENRES_ML_1M = GENRES_ML_100K[1:]
GENRES_ML_10M = GENRES_ML_100K + ['IMAX']

_word_embedding = nlp.embedding.GloVe('glove.840B.300d')
_tokenizer = nlp.data.transforms.SpacyTokenizer()

class MovieLens(object):
    def __init__(self, name):
        self.name = name

        print("Starting processing {} ...".format(self.name))
        self._load_raw_user_info()
        self._load_raw_movie_info()
        if self.name == 'ml-100k':
            self.train_rating_info = self._load_raw_rates(os.path.join(READ_DATASET_PATH, self.name, 'u1.base'), '\t')
            self.test_rating_info = self._load_raw_rates(os.path.join(READ_DATASET_PATH, self.name, 'u1.test'), '\t')
            self.all_rating_info = pd.concat([self.train_rating_info, self.test_rating_info])
        elif self.name == 'ml-1m' or self.name == 'ml-10m':
            self.all_rating_info = self._load_raw_rates(os.path.join(READ_DATASET_PATH, self.name, 'ratings.dat'), '::')
            num_test = int(np.ceil(self.all_rating_info.shape[0] * 0.1))
            shuffled_idx = np.random.permutation(self.all_rating_info.shape[0])
            self.test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
            self.train_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: ]]
        else:
            raise NotImplementedError
        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tTest rating pairs : {}".format(self.test_rating_info.shape[0]))
        print("Filter user and movie info unseen in rating pairs ...")
        self.user_info = self._drop_unseen_nodes(orign_info=self.user_info,
                                            cmp_col_name="id",
                                            reserved_ids_set=set(self.all_rating_info["user_id"].values),
                                            label="user")
        self.movie_info = self._drop_unseen_nodes(orign_info=self.movie_info,
                                             cmp_col_name="id",
                                             reserved_ids_set=set(self.all_rating_info["movie_id"].values),
                                             label="movie")

        # Map user/movie to the global id
        print("  -----------------")
        print("Generating user id map and movie id map ...")
        global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        print('Total user number = {}, movie number = {}'.format(len(global_user_id_map),
                                                                 len(global_movie_id_map)))

        ### Generate features
        self._process_user_fea()
        self._process_movie_fea()
        print("user_features: shape ({},{})".format(self.user_features.shape[0], self.user_features.shape[1]))
        print("movie_features: shape ({},{})".format(self.movie_features.shape[0], self.movie_features.shape[1]))


        user_movie_ratings_coo = sp.coo_matrix(
            (self.all_rating_info["rating"].values.astype(np.float32),
             (np.array([global_user_id_map[ele] for ele in self.all_rating_info["user_id"]], dtype=np.int64),
              np.array([global_movie_id_map[ele] for ele in self.all_rating_info["movie_id"]], dtype=np.int64))),
            shape=(len(global_user_id_map), len(global_movie_id_map)),
            dtype = np.float32)
        #movie_user_ratings_coo = user_movie_ratings_coo.transpose()
        self.all_graph = dgl.DGLBipartiteGraph(
            metagraph=nx.MultiGraph([('user', 'movie', 'rating')]),
            number_of_nodes_by_type={'user': len(global_user_id_map),
                                     'movie': len(global_movie_id_map)},
            edge_connections_by_type={('user', 'movie', 'rating'): user_movie_ratings_coo},
            node_frame={"user": self.user_features, "movie": self.movie_features},
            readonly=True)

        # self.all_graph = hetergraph.DGLBipartiteGraph(
        #     metagraph=nx.MultiGraph([('user', 'movie', 'rating'),
        #                              ('movie', 'user', 'rating')]),
        #     number_of_nodes_by_type={'user': len(global_user_id_map),
        #                              'movie': len(global_movie_id_map)},
        #     edge_connections_by_type={('user', 'movie', 'rating'): user_movie_ratings_coo,
        #                               ('movie', 'user', 'rating'): movie_user_ratings_coo},
        #     node_frame={"user": self.user_features, "movie": self.movie_features})





        user_movie_train_ratings_coo = sp.coo_matrix(
            (self.train_rating_info["rating"].values.astype(np.float32),
             (np.array([global_user_id_map[ele] for ele in self.train_rating_info["user_id"]], dtype=np.int64),
              np.array([global_movie_id_map[ele] for ele in self.train_rating_info["movie_id"]], dtype=np.int64))),
            shape=(len(global_user_id_map), len(global_movie_id_map)),
            dtype=np.float32)
        #movie_user_train_ratings_coo = user_movie_train_ratings_coo.transpose()
        self.train_graph = dgl.DGLBipartiteGraph(
            metagraph=nx.MultiGraph([('user', 'movie', 'rating')]),
            number_of_nodes_by_type={'user': len(global_user_id_map),
                                     'movie': len(global_movie_id_map)},
            edge_connections_by_type={('user', 'movie', 'rating'): user_movie_train_ratings_coo},
            node_frame={"user": self.user_features, "movie": self.movie_features},
            readonly=True)

        # self.train_graph = hetergraph.DGLBipartiteGraph(
        #     metagraph=nx.MultiGraph([('user', 'movie', 'rating'),
        #                              ('movie', 'user', 'rating')]),
        #     number_of_nodes_by_type={'user': len(global_user_id_map),
        #                              'movie': len(global_movie_id_map)},
        #     edge_connections_by_type={('user', 'movie', 'rating'): user_movie_train_ratings_coo,
        #                               ('movie', 'user', 'rating'): movie_user_train_ratings_coo},
        #     node_frame={"user": self.user_features, "movie": self.movie_features})


    ### check whether the user/items in info all appear in the rating
    def _drop_unseen_nodes(self, orign_info, cmp_col_name, reserved_ids_set, label):
        print("  -----------------")
        print("{}: {}(reserved) v.s. {}(from info)".format(label, len(reserved_ids_set),
                                                             len(set(orign_info[cmp_col_name].values))))
        if reserved_ids_set != set(orign_info[cmp_col_name].values):
            pd_rating_ids = pd.DataFrame(list(reserved_ids_set), columns=["id_graph"])
            print("\torign_info: ({}, {})".format(orign_info.shape[0], orign_info.shape[1]))
            data_info = orign_info.merge(pd_rating_ids, left_on=cmp_col_name, right_on='id_graph', how='outer')
            data_info = data_info.dropna(subset=[cmp_col_name, 'id_graph'])
            data_info = data_info.drop(columns=["id_graph"])
            data_info = data_info.reset_index(drop=True)
            print("\tAfter dropping, data shape: ({}, {})".format(data_info.shape[0], data_info.shape[1]))
            return data_info
        else:
            orign_info = orign_info.reset_index(drop=True)
            return orign_info


    def _load_raw_rates(self, file_path, sep):
        """In MovieLens, the rates have the following format

        ml-100k
        user id \t movie id \t rating \t timestamp

        ml-1m/10m
        UserID::MovieID::Rating::Timestamp

        timestamp is unix timestamp and can be converted by pd.to_datetime(X, unit='s')

        Parameters
        ----------
        file_path : str

        Returns
        -------
        rating_info : pd.DataFrame
        """
        rating_info = pd.read_csv(
            file_path, sep=sep, header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id' : np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python')
        return rating_info


    def _load_raw_user_info(self):
        """In MovieLens, the user attributes file have the following formats:

        ml-100k:
        user id | age | gender | occupation | zip code

        ml-1m:
        UserID::Gender::Age::Occupation::Zip-code

        For ml-10m, there is no user information. We read the user id from the rating file.

        Parameters
        ----------
        name : str

        Returns
        -------
        user_info : pd.DataFrame
        """
        if self.name == 'ml-100k':
            self.user_info = pd.read_csv(os.path.join(READ_DATASET_PATH, self.name, 'u.user'), sep='|', header=None,
                                    names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
        elif self.name == 'ml-1m':
            self.user_info = pd.read_csv(os.path.join(READ_DATASET_PATH, self.name, 'users.dat'), sep='::', header=None,
                                    names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')
        elif self.name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(READ_DATASET_PATH, self.name, 'ratings.dat'), sep='::', header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                dtype={'user_id': np.int32, 'movie_id': np.int32, 'ratings': np.float32,
                       'timestamp': np.int64}, engine='python')
            self.user_info = pd.DataFrame(np.unique(rating_info['user_id'].values.astype(np.int32)),
                                     columns=['id'])
        else:
            raise NotImplementedError


    def _process_user_fea(self):
        """

        Parameters
        ----------
        user_info : pd.DataFrame
        name : str
        For ml-100k and ml-1m, the column name is ['id', 'gender', 'age', 'occupation', 'zip_code'].
            We take the age, gender, and the one-hot encoding of the occupation as the user features.
        For ml-10m, there is no user feature and we set the feature to be a single zero.

        Returns
        -------
        user_features : np.ndarray

        """
        if self.name == 'ml-100k' or self.name == 'ml-1m':
            ages = self.user_info['age'].values.astype(np.float32)
            gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
            all_occupations = set(self.user_info['occupation'])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)),
                                          dtype=np.float32)
            occupation_one_hot[np.arange(self.user_info.shape[0]),
                               np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
            self.user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                            gender.reshape((self.user_info.shape[0], 1)),
                                            occupation_one_hot], axis=1)
        elif self.name == 'ml-10m':
            self.user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)
        else:
            raise NotImplementedError


    def _load_raw_movie_info(self):
        """In MovieLens, the movie attributes may have the following formats:

        In ml_100k:

        movie id | movie title | release date | video release date | IMDb URL | [genres]

        In ml_1m, ml_10m:

        MovieID::Title (Release Year)::Genres

        Also, Genres are separated by |, e.g., Adventure|Animation|Children|Comedy|Fantasy

        Parameters
        ----------
        name : str

        Returns
        -------
        movie_info : pd.DataFrame
            For ml-100k, the column name is ['id', 'title', 'release_date', 'video_release_date', 'url'] + [GENRES (19)]]
            For ml-1m and ml-10m, the column name is ['id', 'title'] + [GENRES (18/20)]]
        """
        if self.name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self.name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self.name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        if self.name == 'ml-100k':
            file_path = os.path.join(READ_DATASET_PATH, self.name, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self.name == 'ml-1m' or self.name == 'ml-10m':
            file_path = os.path.join(READ_DATASET_PATH, self.name, 'movies.dat')
            movie_info = pd.read_csv(file_path, sep='::', header=None,
                                     names=['id', 'title', 'genres'], engine='python')
            genre_map = {ele: i for i, ele in enumerate(GENRES)}
            genre_map['Children\'s'] = genre_map['Children']
            genre_map['Childrens'] = genre_map['Children']
            movie_genres = np.zeros(shape=(self.movie_info.shape[0], len(GENRES)), dtype=np.float32)
            for i, genres in enumerate(self.movie_info['genres']):
                for ele in genres.split('|'):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        print('genres not found, filled with unknown: {}'.format(genres))
                        movie_genres[i, genre_map['unknown']] = 1.0
            for idx, genre_name in enumerate(GENRES):
                assert idx == genre_map[genre_name]
                movie_info[genre_name] = movie_genres[:, idx]
            self.movie_info = movie_info.drop(columns=["genres"])
        else:
            raise NotImplementedError

    def _process_movie_fea(self):
        """

        Parameters
        ----------
        movie_info : pd.DataFrame
        name :  str

        Returns
        -------
        movie_features : np.ndarray
            Generate movie features by concatenating embedding and the year

        """
        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)
        p = re.compile(r'(.+)\s*\((\d+)\)')
        for i, title in enumerate(self.movie_info['title']):
            match_res = p.match(title)
            if match_res is None:
                print('{} cannot be matched, index={}, name={}'.format(title, i, self.name))
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            title_embedding[i, :] =_word_embedding[_tokenizer(title_context)].asnumpy().mean(axis=0)
            release_years[i] = float(year)
            self.movie_features = np.concatenate((title_embedding, (release_years - 1950.0) / 100.0), axis=1)


if __name__ == '__main__':
    MovieLens("ml-100k")
    # MovieLens("ml-1m")
    # MovieLens("ml-10m")