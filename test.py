import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import math

# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\4.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx2.png'
path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx3.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx4.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx5.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx6.jpg'

img = cv2.imread(path)
wymiarX = 1000
wymiarY = 700
img = cv2.resize(img, (wymiarX, wymiarY))
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#######################################################################################################################

y, x, _ = plt.hist(imgHSV[:, :, 0].flatten(), 50)

maxy = np.max(y)

it = 0
for piece in y:  # wyzerowanie wartości pomijalnych
    if y[it] < 0.02 * maxy:  # ileś procent masymalnej wartości
        y[it] = 0
    it += 1
it = 0

maksy = []
maximums = argrelextrema(y, np.greater)
for maximum in maximums:
    maksy = maximum

index = 0
for maks in maksy:
    imgRange = cv2.inRange(imgHSV, ((x[maksy[index]] - (0.4 * x[maksy[index]])), 50, 0),  # maska z przedziałem od
                           ((x[maksy[index]] + (0.4 * x[maksy[index]])), 255, 255))
    imgCanny = cv2.Canny(imgRange, 50, 250)
    cv2.imshow("Canny", imgCanny)

    iloscBieli = np.sum(imgRange == 255)
    stosunek = int((iloscBieli / (wymiarX * wymiarY)) * 100)
    print(stosunek)
    if stosunek > 6 or stosunek < 2:
        index += 1
        continue
    cv2.imshow('Range', imgRange)

    imgLines = np.copy(imgHSV) * 0
    lines = cv2.HoughLines(imgRange, 1, np.pi / 180, 210, None, 0, 0)  # 100 było 210
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(imgLines, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)

        imgFinal = cv2.addWeighted(img, 0.8, imgLines, 1, 0)

        cv2.imshow('imgFinal ', imgFinal)
    else:
        print("No lines detected")
    index += 1
    cv2.waitKey(0)
