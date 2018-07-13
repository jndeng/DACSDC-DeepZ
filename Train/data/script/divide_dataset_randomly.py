#!/usr/bin/python
# coding: utf-8
# python 2.7.12

from __future__ import print_function
import os
import glob
import random


def divide_cls(image_path_lst, train_set_cls, train_set_lst, valid_set_lst):
    ratio = 0.8  # train:valid = 8:2
    image_path_size = len(image_path_lst)
    train_set = image_path_lst[:int(ratio * image_path_size)]
    valid_set = image_path_lst[int(ratio * image_path_size):]
    
    with open(train_set_cls, 'w') as f:
        f.write('\n'.join(train_set))
    train_set_lst += train_set
    valid_set_lst += valid_set

    # check
    for valid_path in valid_set:
        assert valid_path not in train_set


def shuffle_data(data):
    random.shuffle(data)
    return data


def load_data(data_dir, cls):
    return glob.glob(os.path.join(data_dir, cls, '*.jpg'))


if __name__ == '__main__':
    # set paths
    TRAIN_DATA_ROOT = os.path.abspath(os.path.pardir)
    train_set_cls_meta = os.path.join(TRAIN_DATA_ROOT, 'dataset', 'train_dataset_cls.txt')
    train_set_meta = os.path.join(TRAIN_DATA_ROOT, 'dataset', 'train_dataset.txt')
    valid_set_meta = os.path.join(TRAIN_DATA_ROOT, 'dataset', 'valid_dataset.txt')
    valid_set_label_meta = os.path.join(TRAIN_DATA_ROOT, 'dataset', 'valid_dataset_label.txt')
    if not os.path.exists(os.path.join(TRAIN_DATA_ROOT, 'dataset')):
      os.mkdir(os.path.join(TRAIN_DATA_ROOT, 'dataset'))
    # load the list of class
    data_dir = os.path.join(TRAIN_DATA_ROOT, 'train_dataset')
    cls_lst = sorted(os.listdir(data_dir))
    # for each class
    train_set_cls_lst, train_set_lst, valid_set_lst = [], [], []
    for cls in cls_lst:
        print('processing class %s ...' % cls)
        # initialize
        train_set_cls = os.path.join(data_dir, cls, 'train_%s.txt' % cls)
        # add path
        train_set_cls_lst.append(train_set_cls)
        # load and shuffle
        image_path_lst = shuffle_data(load_data(data_dir, cls))
        # divide
        divide_cls(image_path_lst, train_set_cls, train_set_lst, valid_set_lst)

    # write dataset into meta files
    with open(train_set_cls_meta, 'w') as f:
        f.write('\n'.join(train_set_cls_lst))
    # shuffle
    train_set_lst = shuffle_data(train_set_lst)
    with open(train_set_meta, 'w') as f:
        f.write('\n'.join(train_set_lst))
    with open(valid_set_meta, 'w') as f:
        f.write('\n'.join(valid_set_lst))
        
    # generate the corresponding label file
    label_dict = {}
    for cls in cls_lst:
        label_file = os.path.join(TRAIN_DATA_ROOT, 'train_dataset', cls, 'label_all.txt')
        with open(label_file, 'r') as f:
            for line in f:
                lst = line.strip().split()
                frame, box = lst[0], ' '.join(lst[1:])
                label_dict[frame] = box
    label_lst = []
    for frame_path in valid_set_lst:
        frame = frame_path.split(os.path.sep)[-1].split('.')[0]
        label_lst.append(label_dict[frame])
    with open(valid_set_label_meta, 'w') as f:
        f.write('\n'.join(label_lst))