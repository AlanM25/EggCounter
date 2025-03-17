import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from egg_detection import detectar_tapa, contar_huevos

class EggCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proyecto - Contador de Huevos")
        self.setGeometry(100, 100, 900, 700)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.contador_label = QLabel("Huevos detectados: 0", self)
        self.btn_seleccionar = QPushButton("Seleccionar Video", self)
        self.btn_seleccionar.clicked.connect(self.abrir_video)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.contador_label)
        layout.addWidget(self.btn_seleccionar)
        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.leer_frame)
        self.contorno_tapa = None

    def abrir_video(self):
        ruta_video, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Videos (*.mp4 *.avi *.mov)")
        if ruta_video:
            self.cap = cv2.VideoCapture(ruta_video)
            if not self.cap.isOpened():
                print("[ERROR] No se pudo abrir el video")
                return

            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] No se pudo leer el primer frame del video")
                return

            frame, self.contorno_tapa = detectar_tapa(frame)
            self.mostrar_frame(frame)
            self.timer.start(30)

    def leer_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return

        frame_procesado, total_huevos = contar_huevos(frame, self.contorno_tapa)
        self.mostrar_frame(frame_procesado)
        self.contador_label.setText(f"Huevos detectados: {total_huevos}")

    def mostrar_frame(self, frame):
        
        max_width = 750
        max_height = 550
        h, w, _ = frame.shape
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = EggCounterApp()
    ventana.show()
    sys.exit(app.exec())
