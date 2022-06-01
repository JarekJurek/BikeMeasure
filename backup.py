import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\1.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\2.png'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx1.png'
path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx2.png'

img = cv2.imread(path)
img = cv2.resize(img, (1000, 700))

imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img2 = np.copy(imgHSV) * 0

# wysoka saturacja 150 - 255, szkieletyzacja,

img2 = cv2.inRange(imgHSV, (0, 200, 0), (17, 255, 255))

# cv2.imshow('line_image', imgHSV[:,:,2])

cv2.imshow('2', img2)

kernel = 11

imgGauss = cv2.GaussianBlur(img2, (kernel, kernel), 0)

imgCanny = cv2.Canny(imgGauss, 50, 150)

rho = 1  # Distance resolution of the accumulator in pixels.
theta = np.pi / 180  # Angle resolution of the accumulator in radians.
threshold = 15  # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
minLineLength = 90  # od wielkosci zdjecia
maxLineGap = 90  # Maximum allowed gap between points on the same line to link them.

imgLines = np.copy(imgHSV) * 0

lines = cv2.HoughLinesP(img2, rho, theta, threshold, np.array([]), minLineLength, maxLineGap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(imgLines, (x1, y1), (x2, y2), (255, 0, 0), 4)

imgFinal = cv2.addWeighted(img, 0.8, imgLines, 1, 0)

cv2.imshow('imgFinal ', imgFinal)
cv2.waitKey(0)
