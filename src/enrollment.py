import cv2
import numpy as np
from deepface import DeepFace
from mediapipe.python.solutions import face_detection as mp_face_detection
import logging
import time

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
        self.model_name = "ArcFace"

    def extract_embedding(self, frame):
        """
        Transforma el rostro detectado en un vector de 512 dimensiones.
        """
        try:
            # Representación biométrica usando ArcFace
            embeddings = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend="mediapipe",
                align=True
            )
            return embeddings[0]["embedding"]
        except Exception as e:
            return None

    def enroll_student(self, nombre, codigo):
        """
        Flujo de captura automática en tres etapas.
        """
        cap = cv2.VideoCapture(0)
        
        # 1. Definición de las etapas de captura
        stages = [
            {"label": "MIRA AL FRENTE", "color": (0, 255, 0)},
            {"label": "GIRA LEVEMENTE A LA DERECHA", "color": (255, 255, 0)},
            {"label": "GIRA LEVEMENTE A LA IZQUIERDA", "color": (255, 255, 0)}
        ]
        
        current_stage = 0
        captured_vectors = []
        stable_start_time = None
        required_stable_duration = 1.5 
        
        while cap.isOpened() and current_stage < len(stages):
            success, frame = cap.read()
            if not success: break

            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # 2. Nueva Geometría: Óvalo más alargado y vertical
            oval_w, oval_h = int(w * 0.20), int(h * 0.45) 

            results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_aligned = False

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    fx, fy, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                    f_center_x, f_center_y = fx + (fw // 2), fy + (fh // 2)

                    dist_to_center = np.sqrt((f_center_x - center_x)**2 + (f_center_y - center_y)**2)
                    
                    # Validación de alineación técnica[cite: 13]
                    if dist_to_center < 35 and (fw > w * 0.25): 
                        face_aligned = True
                        if stable_start_time is None:
                            stable_start_time = time.time()
                    else:
                        stable_start_time = None

            # 3. Feedback Visual[cite: 13]
            display_frame = frame.copy()
            stage_info = stages[current_stage]
            draw_color = stage_info["color"] if face_aligned else (0, 0, 255)
            
            cv2.ellipse(display_frame, (center_x, center_y), (oval_w, oval_h), 0, 0, 360, draw_color, 2)
            cv2.putText(display_frame, stage_info["label"], (center_x - 120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

            if face_aligned and stable_start_time:
                elapsed = time.time() - stable_start_time
                progress = int((elapsed / required_stable_duration) * 100)
                cv2.putText(display_frame, f"CAPTURANDO: {progress}%", (center_x - 80, center_y + oval_h + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 4. Disparo automático tras estabilidad[cite: 13]
                if elapsed >= required_stable_duration:
                    print(f"Captura {current_stage + 1} exitosa.")
                    vector = self.extract_embedding(frame)
                    if vector:
                        captured_vectors.append(vector)
                        current_stage += 1
                    stable_start_time = None 

            cv2.imshow("Multi-Enrolamiento Biometrico", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        return captured_vectors # Retorna una lista de 3 vectores[cite: 13]