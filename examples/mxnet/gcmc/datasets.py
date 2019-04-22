import os
import io
import numpy as np
from mxnet.gluon.utils import check_sha1
from graph import HeterGraph
from zipfile import ZipFile
import warnings
import gluonnlp as nlp
import logging
import pandas as pd
import re
from mxgraph.utils import logging_config
try:
    import requests
except ImportError:
    class requests_failed_to_import(object):
        pass
    requests = requests_failed_to_import


_PROCESSED_PATH = os.path.join("datasets")
_RAW_PATH = os.path.join("datasets", "raw")
if not os.path.exists(_RAW_PATH):
    os.makedirs(_RAW_PATH)

MOVIELENS = ['ml-100k', 'ml-1m', 'ml-10m']
_word_embedding = nlp.embedding.GloVe('glove.840B.300d')
_tokenizer = nlp.data.transforms.SpacyTokenizer()

class LoadData(object):
    def __init__(self, name, val_ratio=0.1, test_ratio=0.1,
                 force_download=False, seed=1024):
        """

        Parameters
        ----------
        name : str the dataset name


        use_input_test_set : bool (for transductive)
        test_ratio : decimal (for transductive) if use_input_test_set=True, then this value has no usage
        val_ratio : decimal (for transductive and inductive)

        inductive_key : str (for inductive)
        inductive_node_frac : int (for inductive)
        inductive_edge_frac : int (for inductive)

        force_download : bool
        seed : int
        """
        self._name = name
        if name in ['ml-1m', 'ml-10m']:
            self._test_ratio = test_ratio
        self._val_ratio = val_ratio
        if name in MOVIELENS:
            self._data_file = {'ml-100k': ['ml-100k.zip',
                                           'http://files.grouplens.org/datasets/movielens/ml-100k.zip'],
                               'ml-1m': ['ml-1m.zip',
                                         'http://files.grouplens.org/datasets/movielens/ml-1m.zip'],
                               'ml-10m': ['ml-10m.zip',
                                          'http://files.grouplens.org/datasets/movielens/ml-10m.zip']}
        else:
            raise NotImplementedError
        self._get_data(force_download)
        ### for test and valid set
        self._rng = np.random.RandomState(seed=seed)
        self._graph = self._preprocess_data()

    @property
    def name_user(self):
        return 'user'
    @property
    def name_item(self):
        if self._name in MOVIELENS:
            return 'movie'
        else:
            raise NotImplementedError

    @property
    def graph(self):
        """

        Returns
        -------
        ret : HeterGraph
            The inner heterogenous graph
        """
        return self._graph

    @property
    def valid_data(self):
        """

        Returns
        -------
        node_pair_ids : np.ndarray
            Shape (2, TOTAL_NUM)
            First row --> user_id
            Second row --> item_id
        ratings : np.ndarray
            Shape (TOTAL_NUM,)
        """
        return self._valid_data
    @property
    def test_data(self):
        """

        Returns
        -------
        node_pair_ids : np.ndarray
            Shape (2, TOTAL_NUM)
            First row --> user_id
            Second row --> item_id
        ratings : np.ndarray
            Shape (TOTAL_NUM,)
        """
        return self._test_data


    def _get_data(self, force_download=False):
        file_name, url = self._data_file[self._name]
        path = os.path.join(_RAW_PATH, file_name)

        if not os.path.exists(path) or force_download:
            print("\n\n\n=====================> Download dataset")
            self.download(url, path=path)
            with ZipFile(path, 'r') as zf:
                zf.extractall(path=os.path.join(_RAW_PATH, file_name))
            if self._name == 'ml-10m':
                os.rename(os.path.join(_RAW_PATH, 'ml-10M100K'), os.path.join(_RAW_PATH, self._name))

    def download(self, url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True, proxy_dict=None):
        """Download an given URL

        Parameters
        ----------
        url : str
            URL to download
        path : str, optional
            Destination path to store downloaded file. By default stores to the
            current directory with same name as in url.
        overwrite : bool, optional
            Whether to overwrite destination file if already exists.
        sha1_hash : str, optional
            Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
            but doesn't match.
        retries : integer, default 5
            The number of times to attempt the download in case of failure or non 200 return codes
        verify_ssl : bool, default True
            Verify SSL certificates.

        Returns
        -------
        str
            The file path of the downloaded file.
        """
        if path is None:
            fname = url.split('/')[-1]
            # Empty filenames are invalid
            assert fname, 'Can\'t construct file-name from this URL. ' \
                          'Please set the `path` option manually.'
        else:
            path = os.path.expanduser(path)
            if os.path.isdir(path):
                fname = os.path.join(path, url.split('/')[-1])
            else:
                fname = path
        assert retries >= 0, "Number of retries should be at least 0"

        if not verify_ssl:
            warnings.warn(
                'Unverified HTTPS request is being made (verify_ssl=False). '
                'Adding certificate verification is strongly advised.')

        if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
            dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            while retries + 1 > 0:
                # Disable pyling too broad Exception
                # pylint: disable=W0703
                try:
                    print('Downloading %s from %s...' % (fname, url))
                    try:
                        r = requests.get(url, stream=True, verify=verify_ssl)
                    except:
                        r = requests.get(url, stream=True, verify=verify_ssl, proxies=proxy_dict)
                    if r.status_code != 200:
                        raise RuntimeError("Failed downloading url %s" % url)
                    with open(fname, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                    if sha1_hash and not check_sha1(fname, sha1_hash):
                        raise UserWarning('File {} is downloaded but the content hash does not match.' \
                                          ' The repo may be outdated or download may be incomplete. ' \
                                          'If the "repo_url" is overridden, consider switching to ' \
                                          'the default repo.'.format(fname))
                    break
                except Exception as e:
                    retries -= 1
                    if retries <= 0:
                        raise e
                    else:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

        return fname


    def _preprocess_data(self):
        save_dir = os.path.join(_PROCESSED_PATH, self._name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.isfile(os.path.join(save_dir, "data_log.log")):
            os.remove(os.path.join(save_dir, "data_log.log"))
        logging_config(folder=save_dir, name="data_log", no_console=False)
        logging.info("Starting processing {} ...".format(self._name))


        if self._name == 'ml-100k':
            train_rating_info = self.load_raw_rates(os.path.join(_RAW_PATH, self._name, 'u1.base'), '\t')
            num_train_val = train_rating_info.shape[0]
            num_valid = int(np.ceil(num_train_val * self._val_ratio))
            shuffled_idx = self._rng.permutation(num_train_val)
            valid_rating_info = train_rating_info.iloc[shuffled_idx[:num_valid]]
            test_rating_info = self.load_raw_rates(os.path.join(_RAW_PATH, self._name, 'u1.test'), '\t')
            self.all_rating_info = pd.concat([train_rating_info, test_rating_info])
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            self.all_rating_info = self.load_raw_rates(os.path.join(_RAW_PATH, self._name, 'ratings.dat'), '::')
            num_all = self.all_rating_info.shape[0]
            num_test = int(np.ceil( num_all * self._test_ratio))
            num_valid = int(np.ceil( (num_all-num_test) * self._val_ratio))
            shuffled_idx = np.random.permutation(num_all)
            test_rating_info = self.all_rating_info.iloc[shuffled_idx[: num_test]]
            valid_rating_info = self.all_rating_info.iloc[shuffled_idx[num_test: num_test+num_valid]]
        else:
            raise NotImplementedError

        self.process_users()
        self.process_items()

        logging.info("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        logging.info("\tValid rating pairs : {}".format(valid_rating_info.shape[0]))
        logging.info("\tTest rating pairs : {}".format(test_rating_info.shape[0]))

        # Map user/movie to the global id
        logging.info("  -----------------")

        logging.info("Generating user id map and movie id map ...")
        global_user_id_map = {ele: i for i, ele in enumerate(self.user_info['id'])}
        global_movie_id_map = {ele: i for i, ele in enumerate(self.movie_info['id'])}
        logging.info('Total user number = {}, movie number = {}'.format(len(global_user_id_map),
                                                                        len(global_movie_id_map)))
        all_ratings_CSRMat = _gen_csr_rating(self.all_rating_info, global_user_id_map, global_movie_id_map,
                                             user_col_name="user_id", item_col_name="movie_id",
                                             rating_col_name="rating")
        uniq_ratings = np.unique(all_ratings_CSRMat.values)
        all_ratings_CSRMat.multi_link = uniq_ratings

        ### generate the rating indices pairs
        test_node_pairs = np.stack([np.array([global_user_id_map[ele] for ele in test_rating_info['user_id']],
                                             dtype=np.int32),
                                    np.array([global_movie_id_map[ele] for ele in test_rating_info['movie_id']],
                                             dtype=np.int32)])
        test_values = test_rating_info['rating'].values.astype(np.float32)
        self._test_data = (test_node_pairs,
                           test_values)
        valid_node_pairs = np.stack([np.array([global_user_id_map[ele] for ele in valid_rating_info['user_id']],
                                              dtype=np.int32),
                                     np.array([global_movie_id_map[ele] for ele in valid_rating_info['movie_id']],
                                              dtype=np.int32)])
        valid_values = valid_rating_info['rating'].values.astype(np.float32)
        self._valid_data = (valid_node_pairs,
                            valid_values)

        # build Graph
        G = HeterGraph(features={'user': self.user_features,
                                 'movie': self.movie_features},
                       csr_mat_dict={('user', 'movie'): all_ratings_CSRMat},
                       multi_link={('user', 'movie'): uniq_ratings})
        G.summary()
        G.save(os.path.join(save_dir, 'graph'))
        self.user_info.to_csv(os.path.join(save_dir, 'user_info.pd'), sep="\t", header=True)
        self.movie_info.to_csv(os.path.join(save_dir, 'movie_info.pd'), sep="\t", header=True)

        return G

    def load_raw_rates(self, file_path, sep):
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
        rating_info = pd.read_csv(file_path, sep=sep, header=None,
                                  names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                  dtype={'user_id': np.int32, 'movie_id': np.int32,
                                         'ratings': np.float32, 'timestamp': np.int64}, engine='python')
        return rating_info

    def process_users(self):
        """In MovieLens, the user attributes file have the following formats:

        ml-100k:
        user id | age | gender | occupation | zip code

        ml-1m:
        UserID::Gender::Age::Occupation::Zip-code

        ml-10m:
         there is no user information. We read the user id from the rating file.

        Features:
        ml-100k & ml-1m: the column name is ['id', 'gender', 'age', 'occupation', 'zip_code'].
            We take the age, gender, and the one-hot encoding of the occupation as the user features.

        ml-10m: there is no user feature and we set the feature to be a single zero.

        Parameters
        ----------

        Returns
        -------
        self.user_info : pd.DataFrame
        self.user_features : np.ndarray
        """

        if self._name == 'ml-100k' or self._name == 'ml-1m':
            if self._name == 'ml-100k':
                user_info = pd.read_csv(os.path.join(_RAW_PATH, self._name, 'u.user'), sep='|', header=None,
                                        names=['id', 'age', 'gender', 'occupation', 'zip_code'], engine='python')
            elif self._name == 'ml-1m':
                user_info = pd.read_csv(os.path.join(_RAW_PATH, self._name, 'users.dat'), sep='::', header=None,
                                        names=['id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')

            # user_info = user_info[user_info['id'].map(lambda x: x == set(self.all_rating_info["user_id"].values))]
            rating_users = pd.DataFrame(list(set(self.all_rating_info["user_id"].values)), columns=["id_graph"])
            user_info = user_info.merge(rating_users, left_on=["id"], right_on='id_graph', how='outer')
            user_info = user_info.dropna(subset=["id", 'id_graph'])
            user_info = user_info.drop(columns=["id_graph"])
            self.user_info = user_info.reset_index(drop=True)
            print(self.user_info)

            ages = self.user_info['age'].values.astype(np.float32)
            gender = (self.user_info['gender'] == 'F').values.astype(np.float32)
            all_occupations = set(self.user_info['occupation'])
            occupation_map = {ele: i for i, ele in enumerate(all_occupations)}
            occupation_one_hot = np.zeros(shape=(self.user_info.shape[0], len(all_occupations)), dtype=np.float32)
            occupation_one_hot[np.arange(self.user_info.shape[0]),
                               np.array([occupation_map[ele] for ele in self.user_info['occupation']])] = 1
            self.user_features = np.concatenate([ages.reshape((self.user_info.shape[0], 1)) / 50.0,
                                                 gender.reshape((self.user_info.shape[0], 1)),
                                                 occupation_one_hot], axis=1)

        elif self._name == 'ml-10m':
            rating_info = pd.read_csv(
                os.path.join(_RAW_PATH, self._name, 'ratings.dat'), sep='::', header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                dtype={'user_id': np.int32, 'movie_id': np.int32, 'ratings': np.float32,
                       'timestamp': np.int64}, engine='python')
            self.user_info = pd.DataFrame(np.unique(rating_info['user_id'].values.astype(np.int32)),
                                     columns=['id'])
            self.user_features = np.zeros(shape=(self.user_info.shape[0], 1), dtype=np.float32)

        else:
            raise NotImplementedError
        logging.info("user_features: shape ({},{})".format(self.user_features.shape[0], self.user_features.shape[1]))





    def process_items(self):
        """In MovieLens, the movie attributes may have the following formats:

        In ml_100k:
            movie id | movie title | release date | video release date | IMDb URL | [genres]

        In ml_1m, ml_10m:
            MovieID::Title (Release Year)::Genres
            Also, Genres are separated by |, e.g., Adventure|Animation|Children|Comedy|Fantasy

        Parameters
        ----------

        Returns
        -------
        self.movie_info : pd.DataFrame
            For ml-100k, the column name is ['id', 'title', 'release_date', 'video_release_date', 'url'] + [GENRES (19)]]
            For ml-1m and ml-10m, the column name is ['id', 'title'] + [GENRES (18/20)]]

        self.movie_features : np.ndarray
            Generate movie features by concatenating embedding and the year

        """

        GENRES_ML_100K = \
            ['unknown', 'Action', 'Adventure', 'Animation',
             'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
             'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
             'Thriller', 'War', 'Western']
        GENRES_ML_1M = GENRES_ML_100K[1:]
        GENRES_ML_10M = GENRES_ML_100K + ['IMAX']
        if self._name == 'ml-100k':
            GENRES = GENRES_ML_100K
        elif self._name == 'ml-1m':
            GENRES = GENRES_ML_1M
        elif self._name == 'ml-10m':
            GENRES = GENRES_ML_10M
        else:
            raise NotImplementedError

        if self._name == 'ml-100k':
            file_path = os.path.join(_RAW_PATH, self._name, 'u.item')
            self.movie_info = pd.read_csv(file_path, sep='|', header=None,
                                          names=['id', 'title', 'release_date', 'video_release_date', 'url'] + GENRES,
                                          engine='python')
        elif self._name == 'ml-1m' or self._name == 'ml-10m':
            file_path = os.path.join(_RAW_PATH, self._name, 'movies.dat')
            movie_info = pd.read_csv(file_path, sep='::', header=None,
                                     names=['id', 'title', 'genres'], engine='python')
            genre_map = {ele: i for i, ele in enumerate(GENRES)}
            genre_map['Children\'s'] = genre_map['Children']
            genre_map['Childrens'] = genre_map['Children']
            movie_genres = np.zeros(shape=(movie_info.shape[0], len(GENRES)), dtype=np.float32)
            for i, genres in enumerate(movie_info['genres']):
                for ele in genres.split('|'):
                    if ele in genre_map:
                        movie_genres[i, genre_map[ele]] = 1.0
                    else:
                        logging.info('genres not found, filled with unknown: {}'.format(genres))
                        movie_genres[i, genre_map['unknown']] = 1.0
            for idx, genre_name in enumerate(GENRES):
                assert idx == genre_map[genre_name]
                movie_info[genre_name] = movie_genres[:, idx]
            self.movie_info = movie_info.drop(columns=["genres"])
        else:
            raise NotImplementedError

        ### processing features
        title_embedding = np.zeros(shape=(self.movie_info.shape[0], 300), dtype=np.float32)
        release_years = np.zeros(shape=(self.movie_info.shape[0], 1), dtype=np.float32)
        p = re.compile(r'(.+)\s*\((\d+)\)')
        for i, title in enumerate(self.movie_info['title']):
            match_res = p.match(title)
            if match_res is None:
                logging.info('{} cannot be matched, index={}, name={}'.format(title, i, self._name))
                title_context, year = title, 1950
            else:
                title_context, year = match_res.groups()
            # We use average of glove
            title_embedding[i, :] = _word_embedding[_tokenizer(title_context)].asnumpy().mean(axis=0)
            release_years[i] = float(year)
        self.movie_features = np.concatenate((title_embedding,
                                              (release_years - 1950.0) / 100.0,
                                              self.movie_info[GENRES].values.astype(np.float32)),
                                             axis=1)
        logging.info("movie_features: shape ({},{})".format(self.movie_features.shape[0], self.movie_features.shape[1]))



    def __repr__(self):
        stream = io.StringIO()
        if self._name in MOVIELENS:
            print('Dataset Name={}'.format(self._name),
                  file=stream)
        print(self.graph, file=stream)
        print('#Val/Test edges: {}/{}'.format(self.valid_data[1].size, self.test_data[1].size),
              file=stream)
        print('------------------------------------------------------------------------------',
              file=stream)
        return stream.getvalue()

if __name__ == '__main__':
    LoadData("ml-100k")
    LoadData("ml-1m")
    LoadData("ml-10m")
