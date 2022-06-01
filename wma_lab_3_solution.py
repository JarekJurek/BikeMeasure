import cv2
import numpy as np

wideo = cv2.VideoCapture('./data/wm_lab_3/1.webm')
ret, klatka_prev = wideo.read()
x, y, w, h = cv2.selectROI(klatka_prev)
rysowanie = klatka_prev.copy()

dobre_kontury = []

while True:
    ret, klatka = wideo.read()
    if not ret:
        break
    roznica = cv2.absdiff(klatka, klatka_prev)
    roznica = cv2.cvtColor(roznica, cv2.COLOR_BGR2GRAY)
    _, prog = cv2.threshold(roznica, 20, 255, cv2.THRESH_BINARY)
    img_morph = cv2.morphologyEx(prog, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    kontury, _ = cv2.findContours(img_morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for kontur in kontury:
        S = cv2.contourArea(kontur)
        L = cv2.arcLength(kontur, True)
        M = L/(2*np.sqrt(np.pi*S)) - 1
        if S > 50 and M<0.5:
            dobre_kontury.append(kontur)
            break
    cv2.imshow('Klatka', klatka)
    cv2.imshow('Roznica', roznica)
    cv2.imshow('Progowanie', prog)
    cv2.imshow('Otwarcie', img_morph)
    cv2.waitKey(100)
    klatka_prev = klatka.copy()

srodki = []
for kontur in dobre_kontury:
    M = cv2.moments(kontur)
    x0 = int(M['m10']/M['m00'])
    y0 = int(M['m01']/M['m00'])
    srodki.append((x0, y0))
for srodek in srodki:
    cv2.circle(rysowanie, srodek, 3, (0, 255, 0), 2)
for i in range(len(srodki)-1):
    if srodki[i+1][1] > srodki[i][1] + 15:
        hit_point = srodki[i]
        break

cv2.circle(rysowanie, hit_point, 3, (0, 0, 255), 2)
if hit_point[0] >= x and hit_point[0] <= x+w and hit_point[1] >= y and hit_point[1] <= y+h:
    print('Trafione!')
else:
    print('Pudło')
cv2.imshow('Rysowanie', rysowanie)
cv2.waitKey()
