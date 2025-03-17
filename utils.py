import cv2

def preprocesar_imagen(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    return blurred

def aplicar_deteccion_circulos(frame):
    circles = cv2.HoughCircles(frame,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=30,
                               param1=100,
                               param2=20,
                               minRadius=15,
                               maxRadius=40)
    return circles
