#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import numpy as np
import SimpleITK as sitk
import time
import sys, os, cv2
from scipy import ndimage
from shutil import rmtree


def worldToVoxelCoodrd(worldCoord, origin, outputSpacing):
    return np.absolute(worldCoord - origin) / outputSpacing


def save3DImage(image, path):
    ''' just save image
    '''
    if os.path.exists(path):
        rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)
    _max = np.max(image)
    if _max > 1:
        for i, im in enumerate(image):
            cv2.imwrite(os.path.join(path, '{}.png'.format(i)), im)
    else:
        for i, im in enumerate(image):
            cv2.imwrite(os.path.join(path, '{}.png'.format(i)), image * 255)


def createPatchCoordinates(centraCoor, size):
    topMostPoint = np.array([centraCoor[0] - size / 2, centraCoor[1] - size /2, centraCoor[2] - size / 2])
    pointOne = np.array(centraCoor[0] + size / 2, centraCoor[1] - size / 2, centraCoor[2] - size / 2)
    pointTwo = np.array(centraCoor[0] - size / 2, centraCoor[1] + size / 2, centraCoor[2] - size / 2)
    pointThree = np.array(centraCoor[0] - size / 2, centraCoor[1] - size / 2, centraCoor[2] + size / 2)

    coordinates = np.vstack((topMostPoint, pointOne, pointTwo, pointThree))
    return coordinates


def translate(coordinates, translation):
    translation = np.vstack((translation, translation, translation, translation))
    coordinates = coordinates + translation
    return coordinates


def zoom_diy(coordinates, zoomCenter, scale):
    for i in range(coordinates.shape[0]):
        coordinates[i, :] = (coordinates[i, :] - zoomCenter) * scale + zoomCenter
    return coordinates


def rotate(coordinates, rotateCenter, rotateAngle):
    rotation = sitk.Euler3DTransform()
    rotation.SetCenter(rotateCenter)
    rotation.SetRotation(rotateAngle[0], rotateAngle[1], rotateAngle[2])
    coordinates = np.array(coordinates, np.int32)
    for i in range(coordinates.shape[0]):
        point = np.array(coordinates, np.float)
        point = tuple(point[i, :])
        rotatePoint = rotation.TransformPoint(point)
        coordinates[i, :] = np.array(rotatePoint)

    return coordinates


def pointBoardcastToMatrix(coordinates, patchSize):
    topMostPoint, pointOne, pointTwo, pointThree = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    point1VectorZ = np.linspace()







































