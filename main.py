import argparse
import time
import cv2
from datetime import datetime, time
import numpy as np
import time as time2
import os, sys
import matplotlib.pyplot as plt



def lecturevideo(file):

    #creation de la capture video
    cap = cv2.VideoCapture(file)

    #verification de l'ouverture de la camera
    if (cap.isOpened() == False):
        print("Error opening video  file")

    return cap





def mesure_hauteur(cap):

    #create empty list for length
    list_h = []

    #time step
    time_step = 0

    # create empty list for time
    list_t = []

    #choose time step for showwing frames of video
    time_step_video = 1

    # Read frames from the video
    while True:

        # Read a frame
        ret, frame = cap.read()

        # Check if a frame is read
        if not ret:
            break

        # Using cv2.split() to split channels of coloured frame
        gray, g, r = cv2.split(frame)

        # Threshold the grayscale frame to make it binary
        # _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 40)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through the contours
        straight_contours = []

        for cnt in contours:
            # Approximate the contour with a polygonal curve
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            straight_contours += [approx]

        # Find the longest contour
        longest_contour = max(straight_contours, key=cv2.contourArea)

        # Calibrate the image to obtain the conversion factor from pixels to real-world units
        pixels_per_metric = 1

        # Calculate the length of the object in real-world units
        mast_length = round(cv2.arcLength(longest_contour, True) * pixels_per_metric)

        # Display the length on the video
        cv2.putText(result, "Length: {:.2f} mm".format(mast_length), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Calcul the minimal rectangle around the longest contour
        x, y, w, h = cv2.boundingRect(longest_contour)
        result = cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the filtered frame
        cv2.imshow("Result", result)


        #dessin du contour
        drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        CountersImg = cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)

        # draw the contour on the image
        cv2.drawContours(CountersImg, longest_contour, -1, (0, 0, 255), 2)


        # Show the contour frame
        cv2.imshow("contours", CountersImg)




        # Break the loop if the user presses 'q' key
        if cv2.waitKey(time_step_video) & 0xFF == ord('q'):
            break

        time_step += 1

        if mast_length < 1000:
            # add lenght on the list
            list_h += [mast_length]

            #add time to list
            list_t += [time_step]

    # Release the VideoCapture object
    cap.release()


    return list_t, list_h



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    cap = lecturevideo('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

    list_t, list_h = mesure_hauteur(cap)
    plt.plot(np.array(list_t), np.array(list_h))
    plt.show()


    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
