import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import math

# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\4.jpg'
path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx2.png'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx3.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx4.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx5.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx6.jpg'

img = cv2.imread(path)
wymiarX = 1000
wymiarY = 700
img = cv2.resize(img, (wymiarX, wymiarY))
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

y, x, _ = plt.hist(imgHSV[:, :, 0].flatten(), 50)

it = 0
for piece in y:  # wyzerowanie wartości pomijalnych
    if y[it] < 0.02 * np.max(y):  # jeżeli ileś procent masymalnej wartości
        y[it] = 0
    it += 1

maksy = []
maximums = argrelextrema(y, np.greater)  # szukanie maksimów lokalnych spośród tego co zostało
for maximum in maximums:
    maksy = maximum

index = 0
for maks in maksy:
    imgRange = cv2.inRange(imgHSV, ((x[maksy[index]] - (0.4 * x[maksy[index]])), 50, 0),  # maska z przedziałem
                           ((x[maksy[index]] + (0.4 * x[maksy[index]])), 255, 255))
    maska = np.ones((3, 3), np.uint8)

    # imgRange = cv2.erode(imgRange, maska, iterations=1)
    imgRange = cv2.dilate(imgRange, maska, iterations=3)
    imgCanny = cv2.Canny(imgRange, 50, 250)  # na razie nie używany
    cv2.imshow("Canny", imgCanny)

    iloscBieli = np.sum(imgRange == 255)
    stosunek = int((iloscBieli / (wymiarX * wymiarY)) * 100)
    if stosunek > 6 or stosunek < 2:
        index += 1
        continue
    cv2.imshow('Range', imgRange)

    imgLines = np.copy(imgHSV) * 0
    lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, 100)  # 100 było 210

    if lines is not None:

        # averaging lines
        lines2 = [[0] * 2] * len(lines)
        for line in lines:
            for i in range(len(lines)):
                lines2[i] = [lines[i][0][0], lines[i][0][1]]  # nowa zwykła lista dla prostrzego działania
        lines2.sort()
        for it in range(len(lines2) - 1):
            limitRho = 40
            limitTheta = 3
            if abs(lines2[it + 1][0] - lines2[it][0]) < limitRho and abs(
                    lines2[it + 1][1] - lines2[it][1]) < limitTheta:
                lines2[it] = [0, 0]
        while True:
            try:
                lines2.remove([0, 0])  # usuwanie oflagowanch (wyzerowanych) pól
            except ValueError:
                break

        for i in range(len(lines2)):
            rho = lines2[i][0]
            theta = lines2[i][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(imgLines, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)

        imgFinal = cv2.addWeighted(img, 0.4, imgLines, 1, 0)

        cv2.imshow('imgFinal ', imgFinal)
    else:
        print("No lines detected")
    index += 1
    cv2.waitKey(0)
