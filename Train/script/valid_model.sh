#!/bin/bash


# change it to your $TRAIN_ROOT
train_root=/home/djn/projects/DAC-Contest/DACSDC-DeepZ/Train

# set the path of validation cfg file
va_cfg_path=${train_root}/cfg/valid.cfg
# set the path of initial weights
weights_path=${train_root}/model/yolo_tiny_dacsdc_best.weights

# set the root path of datax
data_root=${train_root}/data
# set the path of validation images
valid_path=${data_root}/dataset/valid_dataset.txt
# set the path of validation labels
valid_gt_path=${data_root}/dataset/valid_dataset_label.txt


# start training
# use parameter -i to choose the GPU device for validation
${train_root}/darknet detector valid \
	-i 0 \
	-va_cfg ${va_cfg_path} \
	-weights ${weights_path} \
	-va_dir ${valid_path} \
	-va_gt_dir ${valid_gt_path}
