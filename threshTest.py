import cv2
import numpy as np

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 5 # Default was 9

img = cv2.imread("/Users/Scott/Desktop/DATA/SORT/CodingProgrammingPython/License_Plate_Recognition_Python/OpenCV_3_KNN_Character_Recognition_Python/test5.jpg")

# height, width = img.shape

imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
# imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
imgBlurred = cv2.GaussianBlur(imgGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

# _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
_, threshold = cv2.threshold(imgBlurred, 175, 255, cv2.THRESH_BINARY)



kernel = np.ones((7,7),np.uint8)
# closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

dilation = cv2.dilate(threshold,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 1)
# dilation2 = cv2.dilate(erosion,kernel,iterations = 1)

mean_c = cv2.adaptiveThreshold(erosion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)

imgThresh = cv2.adaptiveThreshold(erosion, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                  ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

# for i in range(-100, 100):
#     gauss = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, i)

# cv2.imshow("Img", img)
cv2.imshow("Image Blurred", imgBlurred)
cv2.imshow("Binary threshold", threshold)
# cv2.imshow("Closing", closing)
cv2.imshow("Dilation", dilation)
cv2.imshow("Erosion", erosion)
# cv2.imshow("Dilation2", dilation2)

cv2.imshow("mean_c", mean_c)
cv2.imshow("imgThresh", imgThresh)
# cv2.imshow("Gauss", gauss)

    # name = "Gauss/Gaus" + str(i) + ".jpg"
    # cv2.imwrite(name, gauss)

cv2.waitKey(0)
    # cv2.destroyAllWindows()