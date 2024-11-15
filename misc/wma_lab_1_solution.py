'''# 0. Import biblioteki'''

# Import biblioteki opencv i numpy
import cv2
import numpy as np

"""# 1. Wczytanie i wyświetlenie obrazu"""

# Wczytaj obraz z pliku
img_rel_path = cv2.imread('./data/wm_lab_1/gandalf.png')
# img_absolute_path = cv2.imread('/home/piotr/Pobrane/WMA/data/wm_lab_1/gandalf.png')
img_grayscale = cv2.imread('./data/wm_lab_1/gandalf.png', cv2.IMREAD_GRAYSCALE)
img_grayscale_2 = cv2.cvtColor(img_rel_path, cv2.COLOR_BGR2GRAY)
# Wyświetl obraz
cv2.imshow("Img", img_rel_path)
# cv2.imshow("Img absolute path", img_absolute_path)
cv2.imshow("Img grayscale", img_grayscale)
cv2.imshow("Again image grayscale", img_grayscale_2)
cv2.waitKey()
cv2.destroyAllWindows()

"""# 2. Zmiana kontrastu"""

# Dokonanie zmiany kontrastu
img_contr = cv2.add(cv2.multiply(img_grayscale, 2), 15)
# Wyświetlenie obrazu
cv2.imshow("In", img_grayscale)
cv2.imshow("Out", img_contr)
cv2.waitKey()
cv2.destroyAllWindows()

"""# 3. Progowanie"""

# Dokonaj progowania
_, img_thresh = cv2.threshold(img_grayscale, 50, 255, cv2.THRESH_BINARY)
# Wyświetlenie obrazu
cv2.imshow("In", img_grayscale)
cv2.imshow("Out", img_thresh)
cv2.waitKey()
cv2.destroyAllWindows()

"""# 4. Operacje morfologiczne"""

# Dylatacja
img_dil = cv2.dilate(img_thresh, np.ones((3,3), np.uint8))
img_dil_5 = cv2.dilate(img_thresh, np.ones((3, 3), np.uint8), iterations=5)
# Erozja
img_ero = cv2.erode(img_thresh, np.ones((3, 3), np.uint8))
# Wyświetlenie obrazu
cv2.imshow("In", img_thresh)
cv2.imshow("Dilate 1 iter", img_dil)
cv2.imshow("Dilate 5 iter", img_dil_5)
cv2.imshow("Erode", img_ero)
cv2.imshow("Difference", img_dil-img_ero)
# Stworzenie elementu strukturalnego i wyświetlenie go
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
print(kernel)
cv2.waitKey()
cv2.destroyAllWindows()

"""# 5. Filtrowanie"""

# Filtr uśredniający
img_blur = cv2.blur(img_rel_path, (35, 35))
# Filtr gaussowski
img_gauss = cv2.GaussianBlur(img_rel_path, (35, 35), 35)
# Filtr medianowy
img_median = cv2.medianBlur(img_rel_path, 35)
# Wyświetlenie obrazu
cv2.imshow("In", img_rel_path)
cv2.imshow("Blur", img_blur)
cv2.imshow("Gaussian Blur", img_gauss)
cv2.imshow("Median blur", img_median)
cv2.waitKey()
cv2.destroyAllWindows()

"""# 6. Maskowanie"""

# Tworzenie maski
mask = np.zeros(img_rel_path.shape[:2], np.uint8)
# Ustaw okrąg (610, 390), 100
mask = cv2.circle(mask, (610, 390), 100, 255, -1)
# Skopiuj zdjęcie i przemaskuj
img_copy = img_rel_path.copy()
img_copy[mask!=255] = 0
# Wyświetlanie
cv2.imshow("Mask", mask)
cv2.imshow("Masked img", img_copy)
cv2.waitKey()
cv2.destroyAllWindows()

"""# 7. Iteracja po pikselach"""

# Przekreśl obraz
img_copy_2 = img_rel_path.copy()
shape = img_copy_2.shape
kier = shape[1]/shape[0]
for x in range(shape[0]):
  for y in range(shape[1]):
    if abs(x*kier-y)<10:
      img_copy_2[x, y] = [0, 255, 00]
# Wyświetlanie
cv2.imshow("Diagonal", img_copy_2)
cv2.waitKey()
cv2.destroyAllWindows()
