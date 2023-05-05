from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, asdict
from functools import cached_property
from random import Random
from typing import Tuple, List, Any

from torch.utils.data import Dataset


@dataclass
class Sample:
    sid: Any = None
    time: float = -1
    label: bool = False


@dataclass
class _Sample(Sample):
    feature_index: int = -1


merged_data = namedtuple('merged_data', ['accindex', 'array'])


class BaseSurvivalDataset(ABC, Dataset):
    @abstractmethod
    def info(self, index: int) -> Sample:
        """
        :param index: index of the item
        :return: should be a instance of Sample and sorted by id
        :Raises: IndexError if index is out of range
        """
        raise NotImplementedError

    @abstractmethod
    def feature(self, index: int):
        """
        :param index: index of the item
        :return: should return the feature as the input of models
        """
        raise NotImplementedError

    @cached_property
    def merged_data2(self):
        data = defaultdict(list)
        index = 0
        while True:
            try:
                sample = self.info(index)
                assert isinstance(sample, Sample)
                data[sample.sid].append((sample, index))
                index += 1
            except IndexError:
                break
        # sort each set of samples by time
        for k, v in data.items():
            v.sort(key=lambda x: x[0].time)
        # make the data into a list of samples
        accumulated_index = []
        flat_data = []
        for v in data.values():
            accumulated_index.append(len(flat_data))
            flat_data.extend(v)
        accumulated_index.append(len(flat_data))
        return merged_data(accumulated_index, flat_data)


class SurvivalFuncEstimateDataset(BaseSurvivalDataset, ABC):
    """
    this class is the Base dataset class for survival function
    estimation.
    """

    @cached_property
    def merged_data(self) -> List[List[_Sample]]:
        data = defaultdict(list)
        index = 0
        while True:
            try:
                sample = self.info(index)
                assert isinstance(sample, Sample)
                s = _Sample(feature_index=index, **asdict(sample))
                data[sample.sid].append(s)
                index += 1
            except IndexError:
                break
        # sort each set of samples by time
        for k, v in data.items():
            v.sort(key=lambda x: x.time)
        return list(filter(lambda x: not x[0].label, data.values()))

    def __len__(self):
        return len(self.merged_data)

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        infos = self.merged_data[item]
        assert not infos[0].label
        t1 = infos[0].time
        t2 = infos[-1].time
        for item in infos:
            if item.label:
                t2 = min(t2, item.time)
            else:
                t1 = max(t1, item.time)
        return self.feature(infos[0].feature_index), t1 - infos[0].time, t2 - infos[0].time, t2 > t1, infos[0].sid


class PairedRandomSampleDataset(BaseSurvivalDataset, ABC):
    def __init__(self, testing=False, sample_seed=None) -> None:
        super().__init__()
        self._testing = testing
        self._rnd = Random(sample_seed)

    def __len__(self):
        if self._testing:
            return len(self.merged_data2.array) // 2
        else:
            return len(self.merged_data2.accindex) - 1

    def _item_train(self, index):
        assert 0 <= index < len(self)
        records = self.merged_data2.array[
                  self.merged_data2.accindex[index]:
                  self.merged_data2.accindex[index + 1]]
        # randomly sample tow records with replacement
        records = self._rnd.choices(records, k=2)
        return records

    def _item_test(self, index):
        """
        generate test samples. test samples are iteration of the whole dataset
        if the size of the dataset is odd, the last sample is duplicated
        """
        assert 0 <= index < len(self)
        sample1 = index * 2
        sample2 = index * 2 + 1
        return (self.merged_data2.array[sample1],
                self.merged_data2.array[sample2])

    def _handle_paired_sample(self, sample_pair: Tuple[Tuple[Sample, int], Tuple[Sample, int]]):
        """
        generate the trainable data for the model given a pair of samples
        """
        sample1, index1 = sample_pair[0]
        sample2, index2 = sample_pair[1]
        feat1 = self.feature(index1)
        feat2 = self.feature(index2)
        label1 = sample1.label
        label2 = sample2.label
        dt = sample2.time - sample1.time
        return feat1, feat2, label1, label2, dt

    def __getitem__(self, index):
        if self._testing:
            item = self._item_test(index)
        else:
            item = self._item_train(index)
        return self._handle_paired_sample(item)
