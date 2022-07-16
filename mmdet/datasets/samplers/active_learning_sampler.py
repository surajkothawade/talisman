from __future__ import division
import math
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

class ActiveLearningSampler(Sampler):

    def __init__(self, dataset, indicesFile, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.samples_per_gpu = samples_per_gpu
        self.indices = np.loadtxt(indicesFile,dtype=int)
        self.dataset = dataset
        #print('Selected Indices, Dataset Length->',self.indices, len(self.dataset))
        self.num_samples = len(self.indices)

    def __iter__(self):
        indices = []
        print("invoking Active Learning Sampler.iter() method...")
        np.random.shuffle(self.indices)
        indices = self.indices
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        #assert len(indices) == self.num_samples
        print('iterable indices length:',len(indices))
        return iter(indices)

    def __len__(self):
        return self.num_samples