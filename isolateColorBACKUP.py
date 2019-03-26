import cv2
import numpy as np
import matplotlib.pyplot as plt

HBUFFER = 50
SVBUFFER = 70
# SVBUFFER = HBUFFER * 255 / 180

RGB_UNIQUE_COLOR = [62, 96, 117] # RGBFIELD = (234, 234, 226)

raceImage = cv2.imread('/Users/Scott/Desktop/DATA/SORT/CodingProgrammingPython/ocr-text-extraction/TestRaceImages/SRL_8136.jpg')
raceImage = cv2.GaussianBlur(raceImage,(5,5),0)
raceImage = cv2.cvtColor(raceImage, cv2.COLOR_BGR2RGB)
raceImage = cv2.cvtColor(raceImage, cv2.COLOR_BGR2HSV)
# plt.imshow(raceImage)
# plt.show()

def ConvertRGB_HSV(rgb):
    rgbToConvert = np.uint8([[rgb]])
    hsvRaw = cv2.cvtColor(rgbToConvert, cv2.COLOR_BGR2HSV)

    return hsvRaw[0][0]

hsvColor = ConvertRGB_HSV(RGB_UNIQUE_COLOR)

def hsvLimits(hsv):
    hsvHigh = []
    hsvLow = []
    for i in range(len(hsv)):
        if i == 0:
            hsvHigh.append(hsv[i] + HBUFFER if hsv[i] + HBUFFER <= 180 else 180)
            hsvLow.append(hsv[i] - HBUFFER if hsv[i] - HBUFFER >= 0 else 0)
        if i > 0:
            hsvHigh.append(hsv[i] + SVBUFFER if hsv[i] + SVBUFFER <= 255 else 255)
            hsvLow.append(hsv[i] - SVBUFFER if hsv[i] - SVBUFFER >= 0 else 0)
    return hsvHigh, hsvLow

hsvHigh, hsvLow = hsvLimits(hsvColor)

hsvLow = np.array(hsvLow)
hsvHigh = np.array(hsvHigh)

mask = cv2.inRange(raceImage, hsvLow, hsvHigh)
result = cv2.bitwise_and(raceImage, raceImage, mask=mask)

cv2.imwrite( "./ColorIsolated.jpg", result )

# # plt.subplot(1, 2, 1)
# plt.imshow(mask, cmap="gray")
# # plt.subplot(1, 2, 2)
# plt.imshow(result)
# plt.show()