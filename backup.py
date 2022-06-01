import cv2
import numpy as np

# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\1.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\2.png'
path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx2.png'
img = cv2.imread(path)
img = cv2.resize(img, (1000, 700))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = 11
imgGauss = cv2.GaussianBlur(imgGray, (kernel, kernel), 0)
imgCanny = cv2.Canny(imgGauss, 50, 150)

# imgDilate = cv2.dilate(imgCanny, np.ones((3, 3), np.uint8), iterations=1)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
minLineLength = 170  # minimum number of pixels making up a line
maxLineGap = 20  # maximum gap in pixels between connectable line segments
imgLines = np.copy(img) * 0  # creating a blank to draw lines on

lines = cv2.HoughLinesP(imgCanny, rho, theta, threshold, np.array([]),
                        minLineLength, maxLineGap)

# cv2.imshow('imgDilate', imgDilate)
cv2.imshow('imgCanny ', imgCanny)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(imgLines, (x1, y1), (x2, y2), (255, 0, 0), 5)

# cv2.imshow('line_image', imgLines)

cv2.waitKey(0)
