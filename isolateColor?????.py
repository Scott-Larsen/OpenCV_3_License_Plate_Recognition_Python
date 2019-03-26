import cv2
import numpy as np
import matplotlib.pyplot as plt


SHOW_STEPS = 4 # 0 for off; 1 for all steps; higher numbers for later in the program
HBUFFER = 70
SVBUFFER = HBUFFER
# 50 and 70 for Number Color
# SVBUFFER = HBUFFER * 255 / 180

RGB_NUMBER_COLOR = [62, 96, 117]
RGB_BIB_BACKGROUND_COLOR = [234, 234, 226]

raceImage = cv2.imread('/Users/Scott/Desktop/DATA/SORT/CodingProgrammingPython/ocr-text-extraction/TestRaceImages/SRL_8136.jpg')

if SHOW_STEPS != 0 and SHOW_STEPS <= 1:
    cv2.imshow("Img", raceImage)
    cv2.waitKey(0)

raceImage = cv2.GaussianBlur(raceImage,(7,7),0)

if SHOW_STEPS != 0 and SHOW_STEPS <= 2:
    cv2.imshow("Gaussian", raceImage)
    cv2.waitKey(0)

raceImage = cv2.cvtColor(raceImage, cv2.COLOR_BGR2RGB)
raceImage = cv2.cvtColor(raceImage, cv2.COLOR_BGR2HSV)

if SHOW_STEPS != 0 and SHOW_STEPS <= 3:
    cv2.imshow("HSV", raceImage)
    cv2.waitKey(0)

def Mask_from_RGB(rgb):
    #Convert an RGB color into HSV
    def ConvertRGB_HSV(inputRGB):
        rgbToConvert = np.uint8([[inputRGB]])
        hsvRaw = cv2.cvtColor(rgbToConvert, cv2.COLOR_BGR2HSV)
        return hsvRaw[0][0]

    # hsvColor = ConvertRGB_HSV(rgb)

    #Set margins on either side of the selected colors
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

    # hsvColor = ConvertRGB_HSV(rgb)
    #
    # if SHOW_STEPS >= 2:
    #     print(f"hsvColor is {hsvColor}")

    hsvHighColor, hsvLowColor = hsvLimits(ConvertRGB_HSV(rgb))

    hsvLow = np.array(hsvLowColor)
    hsvHigh = np.array(hsvHighColor)

    # mask = cv2.inRange(raceImage, hsvLow, hsvHigh)
    return cv2.inRange(raceImage, hsvLow, hsvHigh)

mask1 = Mask_from_RGB(RGB_NUMBER_COLOR)
mask2 = Mask_from_RGB(RGB_BIB_BACKGROUND_COLOR)

# cv2.imshow("Mask", mask1)
# cv2.waitKey(0)

# combinedMask = mask1 + mask2
#
# cv2.imshow("Mask", combinedMask)
# cv2.waitKey(0)

# print(raceImage.shape[0:2])
# dimensions = extractValue(imgOriginal)
height, width = raceImage.shape[0:2]
# print(height, width)
# #
blankImage = np.zeros((height, width, 1), np.uint8)

cv2.imshow("Mask", blankImage)
cv2.waitKey(0)

# result = cv2.bitwise_and(raceImage, raceImage, mask=mask)

# cv2.imwrite( "./ColorIsolated.jpg", combinedMask )

# cv2.imwrite( "./ColorIsolated.jpg", result )
# cv2.imshow("Isolated Color", result)
# cv2.waitKey(0)

# # plt.subplot(1, 2, 1)
# plt.imshow(mask, cmap="gray")
# # plt.subplot(1, 2, 2)
# plt.imshow(result)
# plt.show()