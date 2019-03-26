try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

import cv2
import numpy as np
import os

testImage = '/Users/Scott/Desktop/DATA/SORT/CodingProgrammingPython/License_Plate_Recognition_Python/OpenCV_3_KNN_Character_Recognition_Python/test1.jpg'
# print(pytesseract.image_to_string(Image.open('/Users/Scott/Desktop/DATA/SORT/CodingProgrammingPython/License_Plate_Recognition_Python/ColorIsolated.jpeg')))
print(pytesseract.image_to_boxes(Image.open(testImage)))