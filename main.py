import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def contar_huevos():
    # Ocultar la ventana principal de Tkinter
    Tk().withdraw()

    # Abrir cuadro de diálogo para seleccionar el archivo
    video_path = askopenfilename(title="Seleccionar un video", filetypes=[("Archivos de video", ".mp4;.avi;*.mov")])

    if not video_path:
        print("No se seleccionó ningún video.")
        return

    # Cargar el video
    captura = cv2.VideoCapture(video_path)

    if not captura.isOpened():
        print("No se pudo abrir el video.")
        return

    conteos = []

    while True:
        # Leer un frame del video
        ret, frame = captura.read()
        if not ret:
            break

        # Convertir a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar un desenfoque para reducir ruido
        gris = cv2.GaussianBlur(gris, (7, 7), 1.5)

        # Detectar círculos usando la Transformada de Hough
        circulos = cv2.HoughCircles(gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                    param1=50, param2=30, minRadius=10, maxRadius=50)

        huevos_detectados = 0
        if circulos is not None:
            circulos = np.uint16(np.around(circulos))

            for (x, y, r) in circulos[0, :]:
                # Obtener el color promedio dentro del círculo
                mascara = np.zeros_like(gris)
                cv2.circle(mascara, (x, y), r, 255, -1)
                color_promedio = cv2.mean(gris, mask=mascara)[0]

                # Filtrar círculos que tengan color gris-blanco (brillo medio)
                if 150 < color_promedio < 230:
                    huevos_detectados += 1
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Dibujar el círculo
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Dibujar el centro

        conteos.append(huevos_detectados)

        # Mostrar conteo en pantalla
        cv2.putText(frame, f"Huevos detectados: {huevos_detectados}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Conteo de Huevos', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Mostrar el conteo promedio de huevos
    if conteos:
        promedio_huevos = round(sum(conteos) / len(conteos))
        print(f"Numero estimado de huevos: {promedio_huevos}")
    else:
        print("No se detectaron huevos.")

    # Liberar recursos
    captura.release()
    cv2.destroyAllWindows()


# Ejecutar el programa
contar_huevos()