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

# function to check whether the list is empty or not
def is_list_empty(list):
    # checking the length
    if len(list) == 0:
        # returning true as length is 0
        return True
    # returning false as length is greater than 0
    return False

import numpy as np

#this function plot a graph

def graph(x,y,title, x_label = 'time (s)', y_label = 'height (cm)'):

    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()



def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def qualibrage(cap):

    # Read a frame
    ret, frame = cap.read()


    # Convertit la frame en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applique un seuillage pour isoler les graduations
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

    # Detecte les contours des graduations
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Draw countours not approximated
    drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    CountersImg = cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)

    # Initialisation donnee rectangle
    # x_min, y_max, w_max, h_max = cv2.boundingRect(contours[0])

    l_cnt_max = 0

    # Boucle sur chaque graduation
    for c in contours:
        l_cnt = cv2.arcLength(c, True)

        if l_cnt > l_cnt_max:
            l_cnt_max = l_cnt
            # Calcule les statistiques des contours
            x, y, w, h = cv2.boundingRect(c)

    # Dessine un rectangle autour de la graduation
    #cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 0, 255), 2)

    w = int(2.5 * w)
    h = int(4 * h)

    y = int(0.1 * y)
    x = int(0.8 * x)

    # creer une image coupe
    crop_img = CountersImg[y:y + h, x:x + w]


    # Affiche la frame avec les rectangles dessine
    cv2.imshow("Crop", crop_img)

    # cv2.imshow("Seuillage", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x,y,w,h


def mesure_hauteur(cap,x0,y0, w0, h0):

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
        gray, g, r = cv2.split(frame[y0:y0+h0, x0:x0+w0,])

        # Threshold the grayscale frame to make it binary
        thresh = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 40)

        # Find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw countours not approximated
        drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        CountersImg = cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)

        if not is_list_empty(contours):

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

                # draw frame around countuor on the image
                # x, y, w, h = cv2.boundingRect(cnt)
                # frame = cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

            CountersImg = cv2.drawContours(drawing, straight_contours, -1, (0, 0, 255), 3)

            # Calibrate the image to obtain the conversion factor from pixels to real-world units
            pixels_per_metric = 35/1000

            # Calculate the length of the object in real-world units
            mast_length = round(cv2.arcLength(longest_contour, True) * pixels_per_metric)

            # draw frame around countour on the image
            x, y, w, h = cv2.boundingRect(longest_contour)
            CountersImg = cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # add lenght on the list
            list_h += [mast_length]

            # add time to list
            list_t += [time_step]


            # Display the length on the video
            cv2.putText(CountersImg, "Length: {:.2f} cm".format(mast_length), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Calcul the minimal rectangle around the longest contour
            #x, y, w, h = cv2.boundingRect(longest_contour)
            #result = cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the filtered frame
            cv2.imshow("Result", thresh)


            # Show the contour frame
            cv2.imshow("contours", CountersImg)


        # Break the loop if the user presses 'q' key
        if cv2.waitKey(time_step_video) & 0xFF == ord('q'):
            print(time_step)
            break

        #implement time step
        time_step += 1


    # Release the VideoCapture object
    cap.release()


    return list_t, list_h




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    cap = lecturevideo('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

    x,y,w,h = qualibrage(cap)

    list_t, list_h = mesure_hauteur(cap, x,y,w,h)

    for e in  list_t : e = e * 60/len(list_t)
    graph(list_t, list_h, 'Hauteur de vol')

    window = 80

    list_h_lisee = moving_average(list_h, window)

    graph(list_t, list_h_lisee, 'Hauteur de vol lisee')



    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
