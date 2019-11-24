import os
import time, random

import numpy as np
import pandas as pd

import json

# import ujson as json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# set the random seeds for reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MySet(Dataset):
    def __init__(self):
        super(MySet, self).__init__()
        self.content = open('./json/file.json').readlines()

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):

        rec = json.loads(self.content[idx])

        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1
        return rec




class MyTrainSet(Dataset):
    def __init__(self):
        super(MyTrainSet, self).__init__()
        self.content = open('./json/train.json').readlines()

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):

        rec = json.loads(self.content[idx])

        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1
        return rec


class MyTestSet(Dataset):
    def __init__(self):
        super(MyTestSet, self).__init__()
        self.content = open('./json/test.json').readlines()

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):

        rec = json.loads(self.content[idx])

        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1
        return rec


def collate_fn(recs):

    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))


    def to_tensor_dict(recs):
        values = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        masks = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))
        deltas = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        forwards = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        evals = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        eval_masks = torch.FloatTensor(
            list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))


        # print('values:{}'.format(values.size()))
        # print('!!')
        # print('masks:{}'.format(masks.size()))
        # print('deltas:{}'.format(deltas.size()))
        # print('forwards:{}'.format(forwards.size()))
        # print('evals:{}'.format(evals.size()))
        # print('eval_masks:{}'.format(eval_masks.size()))

        return {
            'values': values.permute(0,2,1),
            'forwards': forwards.permute(0,2,1),
            'masks': masks.permute(0,2,1),
            'deltas': deltas.permute(0,2,1),
            'evals': evals.permute(0,2,1),
            'eval_masks': eval_masks.permute(0,2,1)
        }

    ret_dict = {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward)
    }

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'],
                                                 recs)))

    return ret_dict


def get_loader(batch_size=100, shuffle=True):
    data_set = MySet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 1, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter


def get_train_loader(batch_size=100, shuffle=True):
    data_set = MyTrainSet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 1, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
    


def get_test_loader(batch_size=100, shuffle=True):
    data_set = MyTestSet()
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 1, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter

if __name__ == '__main__':

    test_data = MySet()
    # print(test_data.__getitem__(0))

    for key, value in test_data.__getitem__(0).items():
        # print(key, '->', value, '\n')
        print(key, '\n')

    print(test_data.__getitem__(1))


