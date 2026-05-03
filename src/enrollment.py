import cv2
import numpy as np
from deepface import DeepFace
from mediapipe.python.solutions import face_detection as mp_face_detection
import logging

# Desactivar logs de TensorFlow para mantener la terminal limpia
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class EnrollmentManager:
    """
    Gestiona la detección (MediaPipe) y extracción de vectores (ArcFace).
    """
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7  # Umbral de precisión mínima
        )
        # Modelo SOTA definido en la arquitectura técnica[cite: 1]
        self.model_name = "ArcFace"

    def extract_embedding(self, frame):
        """
        Transforma el rostro detectado en un vector de 512 dimensiones (RF-04)[cite: 1].
        """
        try:
            # Representación biométrica usando ArcFace[cite: 1]
            embeddings = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend="mediapipe",
                align=True
            )
            return embeddings[0]["embedding"]
        except Exception as e:
            # En caso de que el rostro no sea lo suficientemente claro[cite: 1]
            return None

    def enroll_student(self, nombre, codigo):
        """
        Captura un fotograma, extrae el vector y lo prepara para persistencia[cite: 1].
        """
        cap = cv2.VideoCapture(0)
        print(f"Iniciando enrolamiento para: {nombre}. Mira a la cámara...")
        
        vector_final = None
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # Feedback visual para el usuario[cite: 1, 2]
            display_frame = frame.copy()
            cv2.putText(display_frame, "Presiona 'SPACE' para capturar", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Enrolamiento - Registro de Usuario", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Capturar al presionar Espacio
                print("Procesando biometría...")
                vector_final = self.extract_embedding(frame)
                if vector_final:
                    print(f"[OK] Vector de {len(vector_final)} dimensiones generado.")
                    break
                else:
                    print("[ERROR] Rostro no detectado o mala calidad. Intenta de nuevo.")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return vector_final