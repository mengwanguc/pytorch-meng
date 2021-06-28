r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import concurrent.futures


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

def same(x):
    return x

class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def getitem(self, idx):
        return self.dataset[idx]
#        return idx

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            print("self.auto_collation")
            if False:
                data = [self.dataset[idx] for idx in possibly_batched_index]
#            print(len(data))
#            data = []
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    results = executor.map(self.getitem, possibly_batched_index)
                    data = []
                    for result in results:
                        data.append(result)  
#                print(data)
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
