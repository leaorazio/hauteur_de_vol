import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import pytesseract
from pytesseract import Output
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit



def get_mouse_points(event, x, y, flags, param):
    global l_points, stop
    if event == cv2.EVENT_LBUTTONUP:
        l_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONUP:
        stop = False

def func1(x, a, b, c):
    return a*x**2+b*x+c

def func2(x, a, b, c):
    return a*x**3+b*x+c

def func3(x, a, b, c):
    return a*x**3+b*x**2+c

def func4(x, a, b, c):
    return a*np.exp(b*x)+c


# Initialise le lecteur de vidéo
cap = cv2.VideoCapture('/media/lea/92DA-552D/nextcloud_zaclys/Documents/Documents/ENSTA/S5/AS/ENSTA.mp4')

# Get the frame rate of the video
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

#selecte randomly the frame
frame_number = random.randint(1, frame_rate)

# Set the position of the video to the specified time
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Lit la première frame
ret, frame = cap.read()

# Enregistrer la nouvelle image rognée
cv2.imwrite("img.jpg", frame)

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
#cv2.imshow("Crop", crop_img)


# Affiche la frame avec le seuillage
#cv2.imshow("Seuillage", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Libère la mémoire occupée par le lecteur de vidéo
cap.release()


#detection des graduations


l_points = []
stop = False

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)

while True:
    if stop:
        for point in l_points:
            cv2.circle(frame, point,2,(0,0,255))
    cv2.imshow("image", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    if not stop and len(l_points) > 0:
        stop = True

#fonction assosicant N(bit) à la distance réelle

l_distances = []
l_distance_real = []

distance_real = 70

x1 = l_points[0][0]
y1 = l_points[0][1]

for i in range(len(l_points)-1):

    #recupération des points sur le mat
    x2 = l_points[i+1][0]
    y2 = l_points[i+1][1]
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) #calcule de la distance eucliedienne entre chaque point
    l_distances.append(distance)
    l_distance_real.append(distance_real)
    distance_real -= 5 #ajout de la distance entre chaque graduation réelle

l_distances = np.array(l_distances)
l_distance_real = np.array(l_distance_real)

#comparatif des différentes focntions
l_fonctions = [func1, func2, func3]
l_result = [["f_1", "f_2", "f_3"],[]]


cov_save = 1000

for i in range(len(l_fonctions)):
    f = l_fonctions[i]
    name = l_result[0][i]

    # Fit the polynomial function to the data points
    params, params_covariance = curve_fit(f, l_distance_real, l_distances)

    # Get the coefficients of the polynomial function
    a, b, c = params

    #store fonctions
    l_result[1].append((a,b,c))

    result = f(l_distance_real,a,b,c)
    plt.plot(l_distance_real, result, '-', label=name)




# Plot the original data points and the best-fit line
plt.plot(l_distance_real, l_distances,'o-', label = "Initial data")

plt.legend()
plt.show()

