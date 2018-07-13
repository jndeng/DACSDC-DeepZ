#!/usr/bin/env python2.7
# coding=utf-8


# standard libraries
from __future__ import print_function
import os
import math
import time
# third-party libraries
import numpy as np
# local packages
import procfunc


# set root path of the detector
detector_root = '..'


if __name__ == "__main__":
    DAC = os.path.join(detector_root, 'data')
    [imgDir, resultDir, timeDir, xmlDir, myXmlDir, allTimeFile] = procfunc.setupDir(DAC)
    # load all the images paths
    [allImageName, imageNum] = procfunc.getImageNames(imgDir)
    img_path_lst = [os.path.join(imgDir, imgName) for imgName in allImageName]

    ##############################################
    ############### detection part ###############
    ##############################################
    # initialize
    cfg_path = os.path.join(detector_root, 'cfg', 'valid.cfg')
    model_path = os.path.join(detector_root, 'model', 'yolo_tiny_dacsdc_best.weights')
    res_bboxes = np.zeros((imageNum, 4), dtype=int)

    # inference
    time_start = time.time()
    net = procfunc.loadModel(cfg_path, model_path)  # load trained model
    procfunc.detectionAndTrackingInParallel(net, img_path_lst, imageNum, res_bboxes)
    #procfunc.detectionAndTrackingSerially(net, img_path_lst, imageNum, res_bboxes)
    time_end = time.time()
    resultRunTime = time_end - time_start

    # write results (write time to allTimeFile and detection results to xml)
    procfunc.storeResultsToXML(res_bboxes, allImageName, myXmlDir)
    procfunc.write(imageNum, resultRunTime, allTimeFile)
