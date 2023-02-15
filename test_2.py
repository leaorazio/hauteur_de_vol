import cv2
import numpy as np

# Initialise le lecteur de vidéo
cap = cv2.VideoCapture('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

# Lit la première frame
ret, frame = cap.read()

# Convertit la frame en niveaux de gris
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Applique un seuillage pour isoler les graduations
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Détecte les contours des graduations
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Boucle sur chaque graduation
for c in contours:
    # Calcule les statistiques des contours
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / float(h)
    extent = cv2.contourArea(c) / float(w * h)

    # Vérifie si le contour correspond à une graduation
    if aspect_ratio >= 0.1 and extent >= 0.1:
        # Dessine un rectangle autour de la graduation
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Affiche la frame avec les rectangles dessinés
cv2.imshow("Graduations", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Libère la mémoire occupée par le lecteur de vidéo
cap.release()
