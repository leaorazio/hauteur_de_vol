import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import newton
from functools import partial


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


#this function plot a graph
def graph(x,*lists, title=None, x_axis_title=None, y_axis_title=None, colors=None, labels=None):
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'gray']
    if labels is None:
        labels = [f'Liste {i + 1}' for i in range(len(lists))]
    for i in range(len(lists)):
        plt.plot(x,lists[i], color=colors[i], label=labels[i])
    plt.legend()
    if title is not None:
        plt.suptitle(title)
    if x_axis_title is not None:
        plt.xlabel(x_axis_title)

    if y_axis_title is not None:
        plt.ylabel(y_axis_title)
    plt.show()


def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def get_mouse_points(event, x, y, flags, param):
    'Fonction permettant de remplir une liste des positions ou l\ont a cliqué sur l\'image.'
    global draw, l_points
    if event == cv2.EVENT_LBUTTONUP:
        l_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONUP:
        draw = False

#différentes fonctions polynomiales permettant d'approximer le résultat
def func1(x, a, b, c):
    return a*x**2+b*x+c

def func2(x, a, b, c):
    return a*x**3+b*x+c

def func3(x, a, b, c):
    return a*x**3+b*x**2+c

def func4(x, a, b, c):
    return a*np.exp(b*x)+c

#fonction permettant de calculer l'inverse
def f_inv(f,y):
    x0 = 0
    g_with_fixed_y = lambda x: f(x) - y
    x = newton(g_with_fixed_y, x0)
    return x


def qualibrage(cap):
    'Fonction permettant à l\'utilisateur de qualibrer les données. \n  L\'utilisateur peut donner la position de toutes les graduations, ainsi que vérifier la réduction du cadrage de l\'image.'

    ret_save, frame_save = cap.read()

    # Boucle pour lire chaque image de la vidéo
    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:
            # Afficher l'image actuelle dans une fenêtre OpenCV
            cv2.imshow('Video', frame)

            # Attendre l'entrée de l'utilisateur
            cv2.waitKey(1)

            # Si l'utilisateur appuie sur la touche "s", enregistrer l'image actuelle
            if cv2.waitKey(1) & 0xFF == ord('s'):
                ret_save, frame_save = ret, frame
                cv2.imwrite('image.jpg', frame)
                print("Image enregistrée avec succès!")
                break

            # Si l'utilisateur appuie sur la touche "q", quitter la boucle
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Convertit la frame en niveaux de gris
    gray = cv2.cvtColor(frame_save, cv2.COLOR_BGR2GRAY)

    # Applique un seuillage pour isoler les graduations
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

    # Detecte les contours des graduations
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Draw countours not approximated
    drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    CountersImg = cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)

    l_cnt_max = 0

    # Boucle sur chaque graduation
    for c in contours:
        l_cnt = cv2.arcLength(c, True)

        if l_cnt > l_cnt_max:
            l_cnt_max = l_cnt
            # Calcule les statistiques des contours
            x, y, w, h = cv2.boundingRect(c)



    # Dessine un rectangle autour de la graduation

    w = int(2.5 * w)
    h = int(4 * h)

    y = int(0.1 * y)
    x = int(0.8 * x)

    # creer une image coupe
    crop_img = frame_save[y:y + h, x:x + w]

    #variables global

    global draw, l_points

    draw = True
    l_points = []

    # detection des graduations

    cv2.namedWindow("Image pour établir échelle")
    cv2.setMouseCallback("Image pour établir échelle", get_mouse_points)

    while True:
        #donne instruction à utilisateur
        cv2.putText(crop_img, "Placer le premier point sur l'image, gradutaion 70.", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if draw:
            i = 0
            for point in l_points:
                i += 1
                # Display the coordonne of the point
                cv2.putText(crop_img, "Point: {:.2f} ".format(i), point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                #draw the point on the image
                cv2.circle(crop_img, point, 3, (0, 0, 255))
        cv2.imshow("Image pour établir échelle", crop_img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if not draw and len(l_points) > 0:
            draw = True



    # fonction assosicant N(bit) à la distance réelle

    l_distances = [] #distances en pixel
    l_distance_real = [] #distances en réel

    distance_real = 70  #initialisation avec la plus grande variable

    x1 = l_points[0][0] #ordonnée x base pour chaque distance
    y1 = l_points[0][1] #ordonnée y pour chaque image

    #boucle pour calculer les distances entre les points
    for i in range(len(l_points) - 1):
        # recupération des points sur le mat
        x2 = l_points[i + 1][0]
        y2 = l_points[i + 1][1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # calcul de la distance eucliedienne entre chaque point
        l_distances.append(distance)
        l_distance_real.append(distance_real)
        distance_real -= 5  # ajout de la distance entre chaque graduation réelle

    l_distances = np.array(l_distances) #transfomation en tableau
    l_distance_real = np.array(l_distance_real)

    # comparatif des différentes focntions
    l_fonctions = [func1, func2, func3] #liste des fonctions disponible
    l_result = [["f_1", "f_2", "f_3"], []] #liste pour stocker résultats et titre  de graph

    #tracer différentes fonctions parametree pour correspondre à la courbe mesuree

    for i in range(len(l_fonctions)):
        f = l_fonctions[i]
        name = l_result[0][i]

        # Fit the polynomial function to the data points
        params, params_covariance = curve_fit(f, l_distance_real, l_distances)

        # Get the coefficients of the polynomial function
        a, b, c = params

        # store fonctions
        l_result[1].append((a, b, c))

        result = f(l_distance_real, a, b, c)
        plt.plot(l_distance_real, result, '-', label=name)

    # Plot the original data points and the best-fit line
    plt.plot(l_distance_real, l_distances, 'o-', label="Initial data")

    plt.legend()
    plt.show()

    return x,y,w,h, l_fonctions[0], l_result[1][0]

#fonction retourne les résultats et les paramètres

def mesure_hauteur(cap,x0,y0, w0, h0, f, param):
    'Cette fonction permet de mesurer au cours de la vidéo la hauteur de vol. \n Elle affiche la vidéo et encadre la ligne de bord d\'attaque du mat relié à l hauteur de vol'

    #create empty list for length
    list_h = []
    list_h_f = []

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
        cv2.drawContours(drawing, contours, -1, (255, 255, 0), 3)

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

            frame = cv2.drawContours(drawing, straight_contours, -1, (0, 0, 255), 3)

            # Calibrate the image to obtain the conversion factor from pixels to real-world units
            pixels_per_metric = 75/1000

            #longueur qui nous interesse

            y = cv2.arcLength(longest_contour, True)

            # Calculate the length of the object in real-world units
            mast_length = round(y*pixels_per_metric)

            # fonction sans paramètre pour appliquer inverse

            f_param = partial(f, param[0], param[1], param[2])

            #mast_length_fonction = round(f_inv(f_param,cv2.arcLength(longest_contour, True)))

            mast_length_fonction = f_inv(f_param,y)
            mast_length_fonction = round(mast_length_fonction)

            # draw frame around countour on the image
            x, y, w, h = cv2.boundingRect(longest_contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # add lenght on the list
            list_h += [mast_length]
            list_h_f += [mast_length_fonction]

            # add time to list
            list_t += [time_step]


            # Display the length on the video
            cv2.putText(CountersImg, "Length: {:.2f} cm".format(mast_length), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Calcul the minimal rectangle around the longest contour
            #x, y, w, h = cv2.boundingRect(longest_contour)
            #result = cv2.rectangle(CountersImg, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the filtered frame
            #cv2.imshow("Result", thresh)


            # Show the contour frame
            cv2.imshow("Frame", frame)


        # Break the loop if the user presses 'q' key
        if cv2.waitKey(time_step_video) & 0xFF == ord('q'):
            print(time_step)
            break

        #implement time step
        time_step += 1


    # Release the VideoCapture object
    cap.release()


    return list_t, list_h, list_h_f





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    cap = lecturevideo('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

    x,y,w,h, f, param  = qualibrage(cap)

    list_t, list_h, list_h_f = mesure_hauteur(cap, x,y,w,h, f, param)

    print(list_h)
    print(list_h_f)
    for e in  list_t : e = e * 60/len(list_t)


    #graph(list_t, list_h, list_h_f, title = 'Hauteur de vol')

    window = 80

    list_h_lisee = moving_average(list_h, window)
    list_h_f_lisee = moving_average(list_h_f, window)

    #graph(list_t, list_h_lisee, list_h_f_lisee, title = 'Hauteur de vol lisee')

    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
