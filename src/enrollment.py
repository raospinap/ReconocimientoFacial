import cv2
import mediapipe as mp

class EnrollmentManager:
    """
    Gestiona la detección facial multirrostro en tiempo real (RF-03).
    """
    def __init__(self):
        # Inicialización estándar de MediaPipe Solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # RF-03: Detección con umbral de confianza del 70%[cite: 1, 2]
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,           # 0: Optimizado para cámaras frontales[cite: 1]
            min_detection_confidence=0.7  # Requerimiento de precisión mínima[cite: 1, 2]
        )

    def test_camera(self):
        """RF-02: Captura de video en vivo desde hardware local[cite: 1, 2]."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] No se pudo acceder a la cámara.")
            return

        print("Presiona 'q' para salir de la prueba de cámara.")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # MediaPipe procesa imágenes en RGB[cite: 1]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)

            # RF-09: Visualización de indicadores sobre el rostro[cite: 1, 2]
            if results.detections:
                for detection in results.detections:
                    self.mp_drawing.draw_detection(image, detection)

            cv2.imshow('Fase 2: Detección Multirrostro SOTA', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    enrollment = EnrollmentManager()
    enrollment.test_camera()