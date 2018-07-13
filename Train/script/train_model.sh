#!/bin/bash


# change it to your $TRAIN_ROOT
train_root=/home/djn/projects/DAC-Contest/DACSDC-DeepZ/Train

# set the path of training cfg file
tr_cfg_path=${train_root}/cfg/train.cfg
# set the path of validation cfg file
va_cfg_path=${train_root}/cfg/valid.cfg
# set the path of initial weights
weights_path=${train_root}/model/yolov3_tiny_COCO.conv.weights

# set the root path of data
data_root=${train_root}/data
# set the path of training data
train_path=${data_root}/dataset/train_dataset.txt
# set the path of validation images
valid_path=${data_root}/dataset/valid_dataset.txt
# set the path of validation labels
valid_gt_path=${data_root}/dataset/valid_dataset_label.txt

# set the name of the model
model_name=yolo_tiny_dacsdc
# set the path to store trained models
model_path=${train_root}/model
# set the path of the logging file of validation
log_path=${train_root}/log
# set the path of the logging file of training
log_file=${train_root}/log/${model_name}.out


# start training
# use -i parameter to choose the GPU device
# use -avg_loss parameter to set the value of initial loss
nohup \
	${train_root}/darknet detector train \
	-i 0 \
	-tr_cfg ${tr_cfg_path} \
	-va_cfg ${va_cfg_path} \
	-weights ${weights_path} \
	-tr_dir ${train_path} \
	-va_dir ${valid_path} \
	-va_gt_dir ${valid_gt_path} \
	-model_name ${model_name} \
	-model_dir ${model_path} \
	-log_dir ${log_path} \
	-avg_loss -1 > ${log_file} 2>&1 &
