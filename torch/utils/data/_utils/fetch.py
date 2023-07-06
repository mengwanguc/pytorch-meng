r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""

import asyncio
import time

class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
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
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.max_threads = dataset._n_loader_threads
        print("max threads {}".format(self.max_threads))

    def load_single_data_blocking(self, index):
        print("begin {}".format(index))
        data = self.dataset[index]
        time.sleep(0.5)
        print("end   {}".format(index))
        return data

    async def load_single_data(self, index, semaphore):
        async with semaphore:
            return await asyncio.to_thread(self.load_single_data_blocking, index)

    async def load_many_data(self, indices):
        semaphore = asyncio.Semaphore(self.max_threads)
        return await asyncio.gather(*(self.load_single_data(index, semaphore) for index in indices))

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = asyncio.run(self.load_many_data(possibly_batched_index))
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
