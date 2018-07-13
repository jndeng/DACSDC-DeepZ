#!/usr/bin/env python2.7
# coding=utf-8

# standard libraries
from __future__ import print_function
import os
import glob
from ctypes import *
# import numpy.ctypeslib as npct
import xml.dom.minidom
# third-party libraries
import numpy as np


# image size: width=640, height=360, channel=3
imageSize = (360, 640, 3)


# load shared library
detector_root = '..'
shared_lib_path = os.path.join(detector_root, 'libdarknet.so')
libdarknet = cdll.LoadLibrary(shared_lib_path)

load_model = libdarknet.load_model
load_model.argtypes = (c_char_p, c_char_p)
load_model.restype = c_void_p

detect_all_parallel = libdarknet.detect_all_parallel
detect_all_parallel.argtypes = (c_void_p, c_void_p, c_int)
detect_all_parallel.restype = POINTER(c_int)

detect_all_batch = libdarknet.detect_all_batch
detect_all_batch.argtypes = (c_void_p, c_void_p, c_int)
detect_all_batch.restype = POINTER(c_int)


##must be called to creat default directory
def setupDir(homeFolder):
    imgDir = homeFolder + '/images'
    resultDir = homeFolder + '/result'
    timeDir = resultDir + '/time'
    xmlDir = resultDir + '/xml'
    myXmlDir = xmlDir
    allTimeFile = timeDir + '/time.txt'
    if os.path.isdir(homeFolder):
        pass
    else:
        os.mkdir(homeFolder)

    if os.path.isdir(imgDir):
        pass
    else:
        os.mkdir(imgDir)

    if os.path.isdir(resultDir):
        pass
    else:
        os.mkdir(resultDir)

    if os.path.isdir(timeDir):
        pass
    else:
        os.mkdir(timeDir)

    if os.path.isdir(xmlDir):
        pass
    else:
        os.mkdir(xmlDir)

    if os.path.isdir(myXmlDir):
        pass
    else:
        os.mkdir(myXmlDir)
    ##create timefile file
    ftime = open(allTimeFile, 'a+')
    ftime.close()

    return [imgDir, resultDir, timeDir, xmlDir, myXmlDir, allTimeFile]


## get image name list
def getImageNames(imgDir):
    nameset1 = []
    nameset2 = []
    namefiles = os.listdir(imgDir)
    for f in namefiles:
        if 'jpg' in f:
            imgname = f.split('.')[0]
            nameset1.append(imgname)
    nameset1.sort()
    for f in nameset1:
        f = f + ".jpg"
        nameset2.append(f)
    imageNum = len(nameset2)
    return [nameset2, imageNum]


def readImagesBatch(imgDir, allImageName, imageNum, iter, batchNumDiskToDram):
    start = iter * batchNumDiskToDram
    end = start + batchNumDiskToDram
    if end > imageNum:
        end = imageNum
    batchImageData = np.zeros((end - start, imageSize[0], imageSize[1], imageSize[2]))
    for i in range(start, end):
        imgName = imgDir + '/' + allImageName[i]
        img = cv2.imread(imgName, 1)
        batchImageData[i - start, :, :] = img[:, :]
    return batchImageData


# Load pre-train model
def loadModel(cfg_path, model_path):
    return load_model(cfg_path, model_path)


# Inference
def detectionAndTrackingInParallel(net, img_path_lst, tot_image, res_bboxes):
    """
    loading & inference in parallel
    """
    assert len(img_path_lst) == tot_image
    img_dir = (c_char_p * tot_image)(*img_path_lst[:tot_image])
    pd_bboxes = detect_all_parallel(net, img_dir, tot_image)
    for i in xrange(tot_image):
        res_bboxes[i][...] = pd_bboxes[i*4: (i + 1)*4]


def detectionAndTrackingSerially(net, img_path_lst, tot_image, res_bboxes):
    """
    serially loading & inference
    """
    assert len(img_path_lst) == tot_image
    img_dir = (c_char_p * tot_image)(*img_path_lst[:tot_image])
    pd_bboxes = detect_all_batch(net, img_dir, tot_image)
    for i in xrange(tot_image):
        res_bboxes[i][...] = pd_bboxes[i*4: (i + 1)*4]


## store the results about detection accuracy to XML files
def storeResultsToXML(resultRectangle, allImageName, myXmlDir):
    # remove all .xml file first
    for file in glob.glob(os.path.join(myXmlDir, '*.xml')):
        if os.path.isfile(file):
            os.remove(file)
    # store
    for i in range(len(allImageName)):
        doc = xml.dom.minidom.Document()
        root = doc.createElement('annotation')

        doc.appendChild(root)
        nameE = doc.createElement('filename')
        nameT = doc.createTextNode(allImageName[i])
        nameE.appendChild(nameT)
        root.appendChild(nameE)

        sizeE = doc.createElement('size')
        nodeWidth = doc.createElement('width')
        nodeWidth.appendChild(doc.createTextNode("640"))
        nodelength = doc.createElement('length')
        nodelength.appendChild(doc.createTextNode("360"))
        sizeE.appendChild(nodeWidth)
        sizeE.appendChild(nodelength)
        root.appendChild(sizeE)

        object = doc.createElement('object')
        nodeName = doc.createElement('name')
        nodeName.appendChild(doc.createTextNode("NotCare"))
        nodebndbox = doc.createElement('bndbox')
        nodebndbox_xmin = doc.createElement('xmin')
        nodebndbox_xmin.appendChild(doc.createTextNode(str(resultRectangle[i, 0])))
        nodebndbox_ymin = doc.createElement('ymin')
        nodebndbox_ymin.appendChild(doc.createTextNode(str(resultRectangle[i, 1])))
        nodebndbox_xmax = doc.createElement('xmax')
        nodebndbox_xmax.appendChild(doc.createTextNode(str(resultRectangle[i, 2])))
        nodebndbox_ymax = doc.createElement('ymax')
        nodebndbox_ymax.appendChild(doc.createTextNode(str(resultRectangle[i, 3])))
        nodebndbox.appendChild(nodebndbox_xmin)
        nodebndbox.appendChild(nodebndbox_xmax)
        nodebndbox.appendChild(nodebndbox_ymin)
        nodebndbox.appendChild(nodebndbox_ymax)

        # nodebndbox.appendChild(doc.createTextNode("360"))
        object.appendChild(nodeName)
        object.appendChild(nodebndbox)
        root.appendChild(object)

        fileName = allImageName[i].replace('jpg', 'xml')
        fp = open(myXmlDir + "/" + fileName, 'w')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
    return


##write time result to alltime.txt
def write(imageNum, runTime, allTimeFile):
    FPS = imageNum / runTime
    ftime = open(allTimeFile, 'a+')
    ftime.write(
        "\n" + "Frames per second:" + str((FPS)) + ", imgNum: "
        + str(imageNum) + ", runtime: " + str(runTime) + '\n')
    ftime.close()
    return