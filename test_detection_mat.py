import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# Initialise le lecteur de vidéo
cap = cv2.VideoCapture('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

# Lit la première frame
ret, frame = cap.read()

# Convertit la frame en niveaux de gris
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Applique un seuillage pour isoler les graduations
thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

# Détecte les contours des graduations
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]


#Draw countours not approximated
drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
CountersImg = cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)

# Initialisation donnée rectangle
#x_min, y_max, w_max, h_max = cv2.boundingRect(contours[0])


l_cnt_max = 0
x_min = 0
y_max = 0
w_max = 0
h_max = 0


# Boucle sur chaque graduation
for c in contours:
    l_cnt = cv2.arcLength(c, True)

    if l_cnt > l_cnt_max :
        l_cnt_max = l_cnt
        # Calcule les statistiques des contours
        x, y, w, h = cv2.boundingRect(c)


# Dessine un rectangle autour de la graduation
cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 0, 255), 2)

w = int(2.5* w)
h = int(4 * h)

y = int(0.1 * y)
x = int(0.8 * x)

#creer une image coupe
crop_img = CountersImg[y:y+h, x:x+w ]

# Affiche la frame avec les rectangles dessinés
#cv2.imshow("Graduations", CountersImg)

# Affiche la frame avec les rectangles dessinés
cv2.imshow("Crop", crop_img)


# Affiche la frame avec le seuillage
#cv2.imshow("Seuillage", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Libère la mémoire occupée par le lecteur de vidéo
cap.release()


#detection du texte

d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
print(d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('detection', thresh)
cv2.waitKey(0)
