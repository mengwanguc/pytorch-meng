r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""


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

    def fetch(self, possibly_batched_index):
        if hasattr(self.dataset, 'is_mytar') and self.dataset.is_mytar == True:
#            print("	using mytar loader....!!!")
            # print("possibly_batched_index: {}".format(possibly_batched_index))
            if self.auto_collation:
                data = mytar_load_data(self.dataset, possibly_batched_index)
            else:
                data = self.dataset[possibly_batched_index]
 #           print("data size {}".format(len(data)))
            return self.collate_fn(data)
        
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)

def mytar_load_data(dataset, possibly_batched_index):
    data = []
    for idx in possibly_batched_index:
        imgs, targets = dataset[idx]
        for i in range(len(imgs)):
            per_image = imgs[i], targets[i]
            data.append(per_image)
    return data