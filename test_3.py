import cv2

points = []

def get_mouse_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))

img = cv2.imread("contours.jpg")
cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)

while True:
    cv2.imshow("image", img)
    print(points)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print("Points:", points)