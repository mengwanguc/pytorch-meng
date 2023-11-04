r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""

import io
from PIL import Image

class _BaseDatasetFetcher(object):
    def __init__(self, worker_id, dataset, auto_collation, collate_fn, drop_last):
        self.worker_id = worker_id
        if dataset.async_loader:
            print("AAAA")
            self.user_state = dataset.async_loader.get_worker_context(worker_id)
        elif dataset.ladcache:
            print("BBBB")
            self.user_state = dataset.ladcache.get_user_state(worker_id)
        else:
            self.user_state = None
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, worker_id, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(worker_id, dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, worker_id, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(worker_id, dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if self.dataset.load_indices:
                data = self.dataset.load_indices(self.user_state, self.dataset, possibly_batched_index)
            else:
                data = [self.dataset[index] for index in possibly_batched_index]

        else:
            # Async loader must be run with auto collation.
            assert(False)
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
