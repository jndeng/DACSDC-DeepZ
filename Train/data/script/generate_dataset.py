#!/usr/bin/python
# coding: utf-8
# python 2.7.12

from __future__ import print_function
import os
import glob
import shutil
import xml.dom.minidom


def generate_cls(src_path, des_path, cls, idx):
    # get name list of all frames of class cls
    # the order of images does not matter, so we won't do sorting here
    frame_list = [
        frame.split(os.path.sep)[-1].split('.')[0]
        for frame in glob.glob(src_path + '/*.xml')
    ]

    # this file contains all frame and the corresponding label in class cls
    label_all_file = 'label_all.txt'
    with open(os.path.join(des_path, label_all_file), 'w') as f: pass

    # create a new label file for each frame
    cnt = 0
    for frame in frame_list:
        print('processing frame %06d of class "%s" ...' % (cnt, cls))
        # get source data
        image_src = os.path.join(src_path, frame + '.jpg')
        label_src = os.path.join(src_path, frame + '.xml')
        # parse xml file to get information of label
        tr = xml.dom.minidom.parse(label_src)
        width = float(tr.getElementsByTagName('width')[0].firstChild.data)
        height = float(tr.getElementsByTagName('height')[0].firstChild.data)
        xmin = tr.getElementsByTagName('xmin')
        if xmin:
            xmin = float(xmin[0].firstChild.data)
        else:
            continue
        ymin = tr.getElementsByTagName('ymin')
        if ymin:
            ymin = float(ymin[0].firstChild.data)
        else:
            continue
        xmax = tr.getElementsByTagName('xmax')
        if xmax:
            xmax = float(xmax[0].firstChild.data)
        else:
            continue
        ymax = tr.getElementsByTagName('ymax')
        if ymax:
            ymax = float(ymax[0].firstChild.data)
        else:
            continue
        # check
        assert xmin <= xmax
        assert ymin <= ymax

        # 1. save the txt file for training, which is formatted as follow:
        # <id> cx cy w h
        # Note: cx, w are relative to width, cy, h are relative to height
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        w  = (xmax - xmin) / width
        h  = (ymax - ymin) / height
        frame_label_file = '%s_%06d.txt' % (cls, cnt)  # re-ordered
        with open(os.path.join(des_path, frame_label_file), 'w') as f:
            f.write('%d %f %f %f %f\n' % (idx, cx, cy, w, h))

        # 2. save the txt file for validation, which is formatted as follow:
        # <frame_name> left top right bottom
        frame_name = '%s_%06d' % (cls, cnt)
        with open(os.path.join(des_path, label_all_file), 'a') as f:
            f.write('%s %d %d %d %d\n' %
                (frame_name, int(xmin), int(ymin), int(xmax), int(ymax)))

        # 3. copy image
        image_des = os.path.join(des_path, '%s_%06d.jpg' % (cls, cnt))
        shutil.copyfile(image_src, image_des)

        # counter +1
        cnt += 1


if __name__ == '__main__':
    TRAIN_DATA_ROOT = os.path.abspath(os.path.pardir)
    src_dir = os.path.join(TRAIN_DATA_ROOT, 'raw_dataset')
    des_dir = os.path.join(TRAIN_DATA_ROOT, 'train_dataset')
    if not os.path.exists(des_dir):
      os.mkdir(des_dir)
    # retrieve all classes (98 classes totally)
    cls_lst = sorted(os.listdir(src_dir))
    #assert 98 == len(cls_lst)
    print(len(cls_lst))
    # for each class
    for idx, cls in enumerate(cls_lst):
        # set paths
        src_path = os.path.join(src_dir, cls)
        des_path = os.path.join(des_dir, cls)
        if not os.path.exists(os.path.join(des_dir, cls)):
            os.mkdir(des_path)
        # generate
        generate_cls(src_path, des_path, cls, idx)