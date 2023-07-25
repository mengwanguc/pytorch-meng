r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""


class _BaseDatasetFetcher(object):
    def __init__(self, worker_id, dataset, auto_collation, collate_fn, drop_last):
        self.worker_id = worker_id
        self.async_worker = dataset.async_loader.get_worker_context(id=worker_id)
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
            # Request images to be loaded.
            for index in possibly_batched_index:
                print("Requesting {} from async loader", self.dataset.samples[index])
                self.async_worker.request(self.dataset.samples[index])

            # Get loaded images.
            data = [self.async_worker.wait_get() for _ in possibly_batched_index]
        else:
            # Async loader must be run with auto collation.
            assert(False)
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
