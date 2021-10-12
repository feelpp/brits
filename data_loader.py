import os
import time

import ujson as json
import csv
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Seems to be unused in the rest of the code.
# class MySet(Dataset):
#     def __init__(self):
#         super(MySet, self).__init__()
#         self.content = open('./json/json').readlines()

#         indices = np.arange(len(self.content))
#         val_indices = np.random.choice(indices, len(self.content) // 5)

#         self.val_indices = set(val_indices.tolist())

#     def __len__(self):
#         return len(self.content)

#     def __getitem__(self, idx):
#         rec = json.loads(self.content[idx])
#         if idx in self.val_indices:
#             rec['is_train'] = 0
#         else:
#             rec['is_train'] = 1
#         return rec

# Original
# class MyTrainSet(Dataset):
#     def __init__(self):
#         super(MyTrainSet, self).__init__()
#         self.content = open('./json/EMS/USA/USA_nitrate_2train1012.json').readlines()

#         indices = np.arange(len(self.content))

#         val_indices = np.random.choice(indices, len(self.content) // 5)

#         self.val_indices = set(val_indices.tolist())

#     def __len__(self):
#         return len(self.content)

#     def __getitem__(self, idx):

#         rec = json.loads(self.content[idx])

#         # if idx in self.val_indices:
#         #     rec['is_train'] = 0
#         # else:
#         #     rec['is_train'] = 1
#         return rec

# Modif
class MyTrainSet(Dataset):
    def __init__(self, input_path=None):
        super(MyTrainSet, self).__init__()
        # self.content = open('./csv/ibat/preprocess/train_raw_results_demo.csv').readlines()
        # self.path = './csv/ibat/preprocess/train_raw_results_demo.csv'
        self.path = input_path
        self.content = pd.read_csv(self.path, header=0) # , chunksize=self.chunksize)

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):

        # rec = csv.reader (self.content[idx])
        rec = self.content.iloc[[idx]]

        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1
        return rec

# my_train_set = MyTrainSet()
# print(my_train_set[3])
# print(len(my_train_set))
# input("waiting")

# Original
# class MyTestSet(Dataset):
#     def __init__(self):
#         super(MyTestSet, self).__init__()
#         self.content = open('./json/EMS/USA/USA_nitrate_2test1012.json').readlines()

#         indices = np.arange(len(self.content))

#         val_indices = np.random.choice(indices, len(self.content) // 5)

#         self.val_indices = set(val_indices.tolist())

#     def __len__(self):
#         return len(self.content)

#     def __getitem__(self, idx):

#         rec = json.loads(self.content[idx])

#         # if idx in self.val_indices:
#         #     rec['is_train'] = 0
#         # else:
#         #     rec['is_train'] = 1
#         return rec


# Modif
class MyTestSet(Dataset):
    def __init__(self, input_path=None):
        super(MyTestSet, self).__init__()
        # self.content = open('./json/EMS/USA/USA_nitrate_2test1012.json').readlines()
        # self.path = './csv/ibat/preprocess/test_raw_results_demo.csv'
        self.path = input_path
        self.content = pd.read_csv(self.path, header=0) # , chunksize=self.chunksize)

        indices = np.arange(len(self.content))

        val_indices = np.random.choice(indices, len(self.content) // 5)

        self.val_indices = set(val_indices.tolist())

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):

        # rec = json.loads(self.content[idx])
        rec = self.content.iloc[[idx]]

        # if idx in self.val_indices:
        #     rec['is_train'] = 0
        # else:
        #     rec['is_train'] = 1
        return rec

# my_test_set = MyTestSet()
# print(my_test_set[4])
# print(len(my_test_set))
# input("waiting")


def collate_fn(recs):
    forward = list(map(lambda x: x['forward'], recs))
    backward = list(map(lambda x: x['backward'], recs))

    def to_tensor_dict(recs):
        # values = torch.FloatTensor(list(map(lambda r: r['values'], recs)))
        # masks = torch.FloatTensor(list(map(lambda r: r['masks'], recs)))
        # deltas = torch.FloatTensor(list(map(lambda r: r['deltas'], recs)))

        # evals = torch.FloatTensor(list(map(lambda r: r['evals'], recs)))
        # eval_masks = torch.FloatTensor(
        #     list(map(lambda r: r['eval_masks'], recs)))
        # forwards = torch.FloatTensor(list(map(lambda r: r['forwards'], recs)))

        values = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['values'], r)), recs)))
        masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['masks'], r)), recs)))

        deltas = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['deltas'], r)), recs)))
        forwards = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['forwards'], r)), recs)))

        evals = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['evals'], r)), recs)))
        eval_masks = torch.FloatTensor(list(map(lambda r: list(map(lambda x: x['eval_masks'], r)), recs)))

        # print('values:{}'.format(values.size()))
        # print('!!')
        # print('masks:{}'.format(masks.size()))
        # print('deltas:{}'.format(deltas.size()))
        # print('forwards:{}'.format(forwards.size()))
        # print('evals:{}'.format(evals.size()))
        # print('eval_masks:{}'.format(eval_masks.size()))

        return {
            'values': values.permute(0, 2, 1),
            'forwards': forwards.permute(0, 2, 1),
            'masks': masks.permute(0, 2, 1),
            'deltas': deltas.permute(0, 2, 1),
            'evals': evals.permute(0, 2, 1),
            'eval_masks': eval_masks.permute(0, 2, 1)
        }

    ret_dict = {
        'forward': to_tensor_dict(forward),
        'backward': to_tensor_dict(backward)
    }

    ret_dict['labels'] = torch.FloatTensor(list(map(lambda x: x['label'], recs)))
    ret_dict['is_train'] = torch.FloatTensor(list(map(lambda x: x['is_train'], recs)))

    # print('values:{}'.format(ret_dict['forward']['values'].size()))
    # print('!!')
    # print('masks:{}'.format(masks.size()))
    # print('deltas:{}'.format(deltas.size()))
    # print('forwards:{}'.format(forwards.size()))
    # print('evals:{}'.format(evals.size()))
    # print('eval_masks:{}'.format(eval_masks.size()))

    return ret_dict

# For now, get_loader is not used in the rest of the code
# def get_loader(batch_size=64, shuffle=False):
#     data_set = MySet()
#     data_iter = DataLoader(dataset=data_set,
#                            batch_size=batch_size,
#                            num_workers=1,
#                            shuffle=shuffle,
#                            pin_memory=True,
#                            collate_fn=collate_fn)

#     return data_iter


def get_train_loader(batch_size=100, shuffle=False):
    data_set = MyTrainSet()
    data_iter = DataLoader(dataset=data_set,
                           batch_size=batch_size,
                           num_workers=1,
                           shuffle=shuffle,
                           pin_memory=True,
                           collate_fn=collate_fn)

    return data_iter


def get_test_loader(batch_size=100, shuffle=False):
    data_set = MyTestSet()
    data_iter = DataLoader(dataset=data_set,
                           batch_size=batch_size,
                           num_workers=1,
                           shuffle=shuffle,
                           pin_memory=True,
                           collate_fn=collate_fn)

    return data_iter
