r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import time
import asyncio

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
        print("_IterableDatasetFetcher")
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

load_count = 0
load_total = 0

class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    def fetch(self, possibly_batched_index):
#        print("_MapDatasetFetcher")
#        print("self.dataset type:{}".format(type(self.dataset).__name__))

        if hasattr(self.dataset, 'is_zip') and self.dataset.is_zip == True:
#            print("	using zip loader....!!!")
            if self.auto_collation:
                data = []
                for idx in possibly_batched_index:
                    temp_data, class_index = self.dataset[idx]
                    for per_image_data in temp_data:
                        per_image = per_image_data, class_index
                        data.append(per_image)
#                        print('\n\n\n')
#                        print(per_image)
            else:
                data = self.dataset[possibly_batched_index]
 #           print("data size {}".format(len(data)))
            return self.collate_fn(data)

        if hasattr(self.dataset, 'is_tar') and self.dataset.is_tar == True:
#            print("	using tar loader....!!!")
            if self.auto_collation:
                data = []
                for idx in possibly_batched_index:
                    temp_data, class_index = self.dataset[idx]
                    for per_image_data in temp_data:
                        per_image = per_image_data, class_index
                        data.append(per_image)
#                        print('\n\n\n')
#                        print(per_image)
            else:
                data = self.dataset[possibly_batched_index]
 #           print("data size {}".format(len(data)))
            return self.collate_fn(data)


        if hasattr(self.dataset, 'is_meng') and self.dataset.is_meng == True:
            print("	using MengImageFolder....!!!")
            if self.auto_collation:
                print("    auto_collation")
                data = []
                for idx in possibly_batched_index:
                    temp_data, class_index = self.dataset[idx]
                    for per_image_data in temp_data:
                        per_image = per_image_data, class_index
                        data.append(per_image)
#                        print('\n\n\n')
#                        print(per_image)
            else:
                print("    not auto_collation!!!!!")
                data = self.dataset[possibly_batched_index]
#            print("data size {}".format(len(data)))
            return self.collate_fn(data)
            

        if self.auto_collation:

            end = time.time()
            data = []
#            data = [(self.dataset[idx]) for idx in possibly_batched_index]
            for idx in range(len(possibly_batched_index)):
                sample, target = self.dataset[idx]
                sample = self.dataset.transform(sample)
                data.append((sample, target))
            if self.dataset.transform is not None:
                print("dataset.transform is NOT None")
            else:
                print("dataset.transform is None")
            data_time = time.time() - end
            print("one batch load time: {}".format(data_time))
            global load_total
            global load_count
            load_total += data_time
            load_count += 1
            print("average per batch load time: {}, load_total: {}, load_count: {}".format(
                           load_total / load_count, load_total, load_count))
        else:
            data = self.dataset[possibly_batched_index]
#        print("data size {}".format(len(data)))
        return self.collate_fn(data)


def get_data(dataset, idx):
    return dataset[idx]

async def async_load_data(dataset, possibly_batched_index):
    #data = [dataset[idx] async for idx in possibly_batched_index]
    loop = asyncio.get_event_loop()
    futures = [asyncio.ensure_future(loop.run_in_executor(None, get_data, dataset, idx)) for idx in possibly_batched_index]
    await asyncio.gather(*futures)
