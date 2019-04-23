import io
import numpy as np
from graph import HeterGraph, CSRMat


class HeterIterator(object):
    def __init__(self, all_graph, name_user, name_item,
                 test_node_pairs=None, valid_node_pairs=None,
                 seed=100):
        """

        Parameters
        ----------
        all_graph : HeterGraph
        name_user : str
        name_item : str
        valid_node_pairs : np.ndarray or None
            The node pairs for validation. It should only be set in the transductive setting.
        test_node_pairs : np.ndarray or None
            The node pairs for validation. It should only be set in the transductive setting.
        seed : int or None
            The seed of the random number generator
        """
        self._rng = np.random.RandomState(seed=seed)
        self._all_graph = all_graph
        self._is_inductive = False
        self._name_user = name_user
        self._name_item = name_item

        ### Generate graphs
        ### test_graph is for testing data to aggregate neighbors
        ### val_graph is for validation data to aggregate neighbors
        ### train_graph is for training data to aggregate neighbors, require to remove batch pairs first
        self._test_graph = all_graph.remove_edges_by_id(name_user, name_item, test_node_pairs)
        self._val_graph = self._test_graph.remove_edges_by_id(name_user, name_item, valid_node_pairs)
        self._train_graph = self._val_graph

        self._test_node_pairs = test_node_pairs
        self._valid_node_pairs = valid_node_pairs
        self._train_node_pairs = self._train_graph[name_user, name_item].node_pair_ids
        self._train_ratings = self._train_graph[name_user, name_item].values
        self._valid_ratings = self._all_graph.fetch_edges_by_id(src_key=name_user,
                                                                dst_key=name_item,
                                                                node_pair_ids=self._valid_node_pairs)
        self._test_ratings = self._all_graph.fetch_edges_by_id(src_key=name_user,
                                                               dst_key=name_item,
                                                               node_pair_ids=self._test_node_pairs)

    @property
    def possible_rating_values(self):
        return self.all_graph[self._name_user, self._name_item].multi_link
    @property
    def all_graph(self):
        return self._all_graph
    @property
    def test_graph(self):
        return self._test_graph
    @property
    def val_graph(self):
        return self._val_graph
    @property
    def train_graph(self):
        return self._train_graph

    def rating_sampler(self, batch_size, segment='train', sequential=None):
        """ Return the sampler for ratings

        Parameters
        ----------
        batch_size : int, -1 means the whole data samples
        segment : str
        sequential : bool or None
            Whether to sample in a sequential manner. If it's set to None, it will be
            automatically determined based on the sampling segment.

        Returns
        -------
        node_pairs : np.ndarray
            Shape (2, #Edges)
        ratings : np.ndarray
            Shape (#Edges,)
        """
        if segment == 'train':
            sequential = False if sequential is None else sequential
            node_pairs, ratings = self._train_node_pairs, self._train_ratings
        elif segment == 'valid':
            sequential = True if sequential is None else sequential
            node_pairs, ratings = self._valid_node_pairs, self._valid_ratings
        elif segment == 'test':
            sequential = True if sequential is None else sequential
            node_pairs, ratings = self._test_node_pairs, self._test_ratings
        else:
            raise NotImplementedError('segment must be in {}, received {}'.format(['train', 'valid', 'test'], segment))
        if batch_size < 0:
            batch_size = node_pairs.shape[1]
        else:
            batch_size = min(batch_size, node_pairs.shape[1])
        if sequential:
            for start in range(0, node_pairs.shape[1], batch_size):
                end = min(start + batch_size, node_pairs.shape[1])
                yield node_pairs[:, start:end], ratings[start:end]
        else:
            while True:
                if batch_size == node_pairs.shape[1]:
                    yield node_pairs, ratings
                else:
                    sel = self._rng.choice(node_pairs.shape[1], batch_size, replace=False)
                    yield node_pairs[:, sel], ratings[sel]

    def __repr__(self):
        stream = io.StringIO()
        print('All Graph=', file=stream)
        print(self.all_graph, file=stream)
        print('Test Graph=', file=stream)
        print(self.test_graph, file=stream)
        print('Val Graph=', file=stream)
        print(self.val_graph, file=stream)
        print('Train Graph=', file=stream)
        print(self.train_graph, file=stream)
        return stream.getvalue()
