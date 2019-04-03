# Preprocess.py

import cv2
import numpy as np
import math
import Main

# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 5 # Default was 9

# NEW PREPROCESS
###################################################################################################
def preprocess(imgOriginal):

    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    imgBlurred = cv2.GaussianBlur(imgGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    _, threshold = cv2.threshold(imgBlurred, 175, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)

    dilation = cv2.dilate(threshold, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    mean_c = cv2.adaptiveThreshold(erosion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)

    imgThresh = cv2.adaptiveThreshold(erosion, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                      ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    # cv2.imshow("Image Blurred", imgBlurred)
    # cv2.imshow("Binary threshold", threshold)
    # cv2.imshow("Dilation", dilation)
    # cv2.imshow("Erosion", erosion)
    #
    # cv2.imshow("mean_c", mean_c)
    # cv2.imshow("imgThresh", imgThresh)

    # cv2.waitKey(0)

    return imgGrayscale, imgThresh
###################################################################################################

# ORIGINAL PREPROCESS
# ###################################################################################################
# def preprocess(imgOriginal):
#     imgGrayscale = extractValue(imgOriginal)
#
#     if Main.scottShowSteps == True:
#         cv2.imshow("imgGrayscale", imgGrayscale)
#         cv2.waitKey(0)
#
#     imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
#
#     height, width = imgGrayscale.shape
#
#     # Just makes a black empty image the size of the image
#     imgBlurred = np.zeros((height, width, 1), np.uint8)
#
#     imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
#
#     if Main.scottShowSteps == True:
#         cv2.imshow("imgBlurred", imgBlurred)
#         cv2.waitKey(0)
#
#     # cv2.imwrite("imgBlurred.jpg", imgBlurred)
#
#     imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
#
#     return imgGrayscale, imgThresh
# # end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):

    SHOW_STEPS = False

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    # structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    if SHOW_STEPS == True:
        cv2.imshow("0a structuringElement", structuringElement)
        cv2.waitKey(0)

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)

    if SHOW_STEPS == True:
        cv2.imshow("0b imgTopHat", imgTopHat)
        cv2.waitKey(0)

    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    if SHOW_STEPS == True:
        cv2.imshow("0c imgGrayscalePlusTopHat", imgBlackHat)
        cv2.waitKey(0)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)

    if SHOW_STEPS == True:
        cv2.imshow("0d imgGrayscalePlusTopHat", imgGrayscalePlusTopHat)
        cv2.waitKey(0)

    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    if SHOW_STEPS == True:
        cv2.imshow("0e imgGrayscalePlusTopHatMinusBlackHat", imgGrayscalePlusTopHatMinusBlackHat)
        cv2.waitKey(0)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function