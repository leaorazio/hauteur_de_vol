import cv2
import numpy as np
import pytesseract

# Initialise le lecteur de vidéo
cap = cv2.VideoCapture('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

# Select the time where the image is extracted
cap.set(cv2.CAP_PROP_POS_MSEC, 32500)

#read the frame at this time
ret, frame = cap.read()


# Using cv2.split() to split channels of coloured frame
gray,g,r = cv2.split(frame)

# Threshold the grayscale frame to make it binary
#_, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 40)

# Find contours in the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Draw countours not approximated
drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
CountersImg = cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)


# Show the frame
#cv2.imshow("contours",CountersImg)

# Loop through the contours
straight_contours = []

longest_contour = contours[0]


for iteration, cnt in enumerate(contours):

        l_cnt = cv2.arcLength(cnt, True)

        # Approximate the contour with a polygonal curve
        epsilon = 0.1 * l_cnt
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if l_cnt > 500:

            CountersImg = cv2.drawContours(drawing, approx, -1, (0, 0, 255), 3)

            straight_contours += [approx]

        # Find the longest contour
        if cv2.arcLength(longest_contour, True) < cv2.arcLength(approx, True):
            longest_contour = approx


        #draw frame around countuor on the image
        #x, y, w, h = cv2.boundingRect(cnt)
        #frame = cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

CountersImg = cv2.drawContours(drawing,straight_contours, -1, (0, 0, 255), 3)


#draw frame around longest contour on the image
x, y, w, h = cv2.boundingRect(longest_contour)
print(x, y, w, h)
CountersImg = cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 255, 255), 3)

# Display the length on the video
cv2.putText(CountersImg, "Length: {:.2f} mm".format(cv2.arcLength(longest_contour, True)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Lit l'image
img = cv2.imread("image_ext.jpg")


# Tracez le rectangle sur l'image
#result = cv2.rectangle(result, (x, y), (x+w, y+h), (127), 2)

# Afficher le frame
#cv2.imshow("Contours", frame)

# Show the frame
#cv2.imshow("image_extraite", img)

# Show the frame
#cv2.imshow("gray", gray)


# Show the frame
#cv2.imshow("b", b)

# Show the frame
cv2.imshow("tresh img", thresh)


# Show the frame
#cv2.imshow("mask", mask)

# Show the frame
#cv2.imshow("result", result)

# Show the frame
cv2.imshow("contours",CountersImg)


# Sauvegarde the extract image
cv2.imwrite("contours.jpg",CountersImg)

# Wait for the user to close the image
cv2.waitKey(0)

# Ferme la fenêtre d'affichage
cv2.destroyAllWindows()


# Libère la mémoire occupée par le lecteur de vidéo
cap.release()






