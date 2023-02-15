import cv2
import numpy as np
import pytesseract

# Initialise le lecteur de vidéo
cap = cv2.VideoCapture('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

# Select the time where the image is extracted
cap.set(cv2.CAP_PROP_POS_MSEC, 32500)

#read the frame at this time
ret, img = cap.read()

#using pytesseract
h, w, c = img.shape
boxes = pytesseract.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow('img', img)