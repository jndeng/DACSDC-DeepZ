#!/usr/bin/python
# coding: utf-8
# python 2.7.12

from __future__ import print_function
import os
import shutil
import random


"""
Revise the paths based on 4 meta files.
"""


if __name__ == '__main__':
    # set paths
    TRAIN_DATA_ROOT = os.path.abspath(os.path.pardir)

    train_set_cls_meta_divided = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'divided_dataset', 'train_dataset_cls.txt')
    train_set_meta_divided = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'divided_dataset', 'train_dataset.txt')
    valid_set_meta_divided = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'divided_dataset', 'valid_dataset.txt')
    valid_set_label_meta_divided = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'divided_dataset', 'valid_dataset_label.txt')

    train_set_cls_meta = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'train_dataset_cls.txt')
    train_set_meta = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'train_dataset.txt')
    valid_set_meta = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'valid_dataset.txt')
    valid_set_label_meta = os.path.join(
        TRAIN_DATA_ROOT, 'dataset', 'valid_dataset_label.txt')

    # revise train_set_cls
    with open(train_set_cls_meta_divided, 'r') as f:
        train_set_cls = [line.strip() for line in f if line]
    train_set_cls_new = []
    for train_cls in train_set_cls:
        cls = train_cls.split('_')[-1].split('.')[0]
        train_set_cls_new.append(
            os.path.join(TRAIN_DATA_ROOT, 'train_dataset', cls, train_cls)
        )
    with open(train_set_cls_meta, 'w') as f:
        f.write('\n'.join(train_set_cls_new))

    # revise train_set
    with open(train_set_meta_divided, 'r') as f:
        train_set = [line.strip() for line in f if line]
    train_set_new = []
    for train_path in train_set:
        cls = train_path.split('_')[0]
        train_set_new.append(
            os.path.join(TRAIN_DATA_ROOT, 'train_dataset', cls, train_path)
        )
    with open(train_set_meta, 'w') as f:
        f.write('\n'.join(train_set_new))

    # revise valid_set
    with open(valid_set_meta_divided, 'r') as f:
        valid_set = [line.strip() for line in f if line]
    valid_set_new = []
    for valid_path in valid_set:
        cls = valid_path.split('_')[0]
        valid_set_new.append(
            os.path.join(TRAIN_DATA_ROOT, 'train_dataset', cls, valid_path)
        )
    with open(valid_set_meta, 'w') as f:
        f.write('\n'.join(valid_set_new))

    # copy
    shutil.copyfile(valid_set_label_meta_divided, valid_set_label_meta)

    # revise train_cls
    train_cls_dict = {}
    for train_path in train_set_new:
        cls = train_path.split(os.path.sep)[-1].split('_')[0]
        if cls in train_cls_dict:
            train_cls_dict[cls].append(train_path)
        else:
            train_cls_dict[cls] = [train_path]

    for cls, path_lst in train_cls_dict.items():
        random.shuffle(path_lst)
        cls_dir = os.path.join(TRAIN_DATA_ROOT, 'train_dataset', cls)
        with open(os.path.join(cls_dir, 'train_%s.txt' % cls), 'w') as f:
            f.write('\n'.join(path_lst))