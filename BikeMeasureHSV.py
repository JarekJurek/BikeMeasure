import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import math

# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx2.png'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx6.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\olx4.jpg'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\Untitled2.png'
path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\Untitled3.png'
# path = r'C:\Users\grzeg\Documents\Studia\Semestr 6\Widzenie Maszynowe\Projekt\BikeMeasure\data\Untitled4.png'


def intersection(rho1, theta1, rho2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    be = np.array([[rho1], [rho2]])
    iks, ygrek = np.linalg.solve(A, be)  # rozwiązanie równania macieżowego liniowego (linear matrix equation)
    iks, ygrek = int(np.round(iks)), int(np.round(ygrek))
    return [[iks, ygrek]]


def fillPipeLength(x1, y1, x2, y2):
    length = int(
        np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)) * (2.5 * dlugoscGRurki) / wymiarX)  # razy długość piksela w cm
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    return [length, x, y]


wymiarX = 1000
wymiarY = 700
dlugoscGRurki = 55
imgOG = cv2.imread(path)
img = cv2.resize(imgOG, (wymiarX, wymiarY))
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

y, x, _ = plt.hist(imgHSV[:, :, 0].flatten(), 50)  # jakie zakresy x i y i która warstwa - tutaj H; flatten lista list

it = 0
for piece in y:  # wyzerowanie wartości pomijalnych
    if y[it] < 0.02 * np.max(y):  # jeżeli ileś procent masymalnej wartości
        y[it] = 0
    it += 1

maksy = [] # szukanie maksimów lokalnych spośród tego co zostało
maximums = argrelextrema(y, np.greater)  # np.greater to porównanie dwóch sąsiednich wartości
for maximum in maximums:
    maksy = maximum

index = 0
for maks in maksy:
    imgRange = cv2.inRange(imgHSV, ((x[maksy[index]] - (0.4 * x[maksy[index]])), 50, 0),  # maska z przedziałami H,S,V
                           ((x[maksy[index]] + (0.4 * x[maksy[index]])), 255, 255))
    iloscBieli = np.sum(imgRange == 255)
    stosunek = int((iloscBieli / (wymiarX * wymiarY)) * 100)
    if stosunek > 6 or stosunek < 2:  # odrzucanie peaku histogramu ze względu na ilości danego koloru
        index += 1
        continue

    imgLines = np.copy(imgHSV) * 0
    lines = cv2.HoughLines(imgRange, 1, np.pi / 180, 210)  # krawędzie, rho, theta threshold

    if len(lines) > 1:
        # averaging lines
        lines2 = [[0] * 2] * len(lines)
        for line in lines:
            for i in range(len(lines)):
                lines2[i] = [lines[i][0][0], lines[i][0][1]]  # nowa zwykła lista dla prostrzego działania

        limitRho = 380
        limitTheta = 0.5
        for i in range(len(lines2) - 1):
            for j in range(i + 1, (len(lines2))):
                if abs(lines2[i][1] - lines2[j][1]) < limitTheta and abs(
                        lines2[i][0] - lines2[j][0]) < limitRho:
                    lines2[j] = [0, 0]
        while True:
            try:
                lines2.remove([0, 0])  # usuwanie "oflagowanch" (wyzerowanych) pól
            except ValueError:
                break

        # fiding intersections
        intersections = []
        for i in range(len(lines2) - 1):
            rho1 = lines2[i][0]
            theta1 = lines2[i][1]
            for j in range(i + 1, (len(lines2))):
                rho2 = lines2[j][0]
                theta2 = lines2[j][1]
                intersections.append(intersection(rho1, theta1, rho2, theta2))

        # averaging intersections
        intersections.sort()
        for it in range(len(intersections) - 1):
            limit = 40
            if abs(intersections[it + 1][0][0] - intersections[it][0][0]) < limit and abs(
                    intersections[it + 1][0][1] - intersections[it][0][1]) < limit:
                intersections[it] = [[0, 0]]
            if intersections[it][0][0] > 0.78 * wymiarX or intersections[it][0][1] > 0.7 * wymiarY:
                intersections[it] = [[0, 0]]
            if intersections[it + 1][0][0] > 0.78 * wymiarX or intersections[it + 1][0][1] > 0.7 * wymiarY:
                intersections[it + 1] = [[0, 0]]
            if intersections[it][0][0] < 0.1 * wymiarX or intersections[it][0][1] < 0.1 * wymiarY:
                intersections[it] = [[0, 0]]
            if intersections[it + 1][0][0] < 0.1 * wymiarX or intersections[it + 1][0][1] < 0.1 * wymiarY:
                intersections[it + 1] = [[0, 0]]

        while True:
            try:
                intersections.remove([[0, 0]])  # usuwanie oflagowanch (wyzerowanych) pól
            except ValueError:
                break

        for i in range(len(lines2)):
            rho = lines2[i][0]
            theta = lines2[i][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = [int(x0 + 1000 * (-b)), int(y0 + 1000 * a)]
            pt2 = [int(x0 - 1000 * (-b)), int(y0 - 1000 * a)]
            cv2.line(imgLines, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)  # color, thickness style

        imgFinal = cv2.addWeighted(img, 0.4, imgLines, 1, 0)
        for inter in intersections:
            imgFinal = cv2.circle(imgFinal, (inter[0][0], inter[0][1]), radius=10, color=(0, 0, 255), thickness=-1)

        pipeLength = []
        global grzesX0, grzesY0, grzesXk, grzesYk
        for it in range(len(intersections) - 1):
            x1 = intersections[it][0][0]
            y1 = intersections[it][0][1]
            x2 = intersections[it + 1][0][0]
            y2 = intersections[it + 1][0][1]
            if it == 0:
                grzesX0 = x1
                grzesY0 = y1
            if it == len(intersections) - 2:
                grzesXk = x2
                grzesYk = y2
            pipeLength.append(fillPipeLength(x1, y1, x2, y2))
        pipeLength.append(fillPipeLength(grzesX0, grzesY0, grzesXk, grzesYk))

        font = cv2.FONT_HERSHEY_SIMPLEX
        for tmp in range(0, len(pipeLength)):
            cv2.putText(imgFinal, (str(pipeLength[tmp][0]) + "cm"), (pipeLength[tmp][1], pipeLength[tmp][2]), font, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('imgFinal', imgFinal)
    else:
        print("No lines detected")
    index += 1
    cv2.waitKey(0)
