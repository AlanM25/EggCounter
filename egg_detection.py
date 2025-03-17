import cv2
import numpy as np

def detectar_tapa(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_carton = np.array([0, 0, 30])
    upper_carton = np.array([180, 70, 170])

    mask = cv2.inRange(hsv, lower_carton, upper_carton)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50000: 
            if area > max_area:
                max_area = area
                best_contour = cnt

    if best_contour is not None:
        cv2.drawContours(frame, [best_contour], -1, (255, 255, 0), 2)

    return frame, best_contour


def contar_huevos(frame, contorno_tapa):
    output = frame.copy()
    total = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
        param1=80, param2=32, minRadius=18, maxRadius=35
    )

    if circles is not None and contorno_tapa is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            if cv2.pointPolygonTest(contorno_tapa, (int(x), int(y)), False) >= 0:
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, 255, -1)
                mean_intensity = cv2.mean(gray, mask=mask)[0]

                if mean_intensity > 140: 
                    cv2.circle(output, (x, y), r, (0, 0, 255), 2)
                    total += 1

    return output, total
