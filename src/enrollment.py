import cv2
import numpy as np
from deepface import DeepFace
from mediapipe.python.solutions import face_detection as mp_face_detection
import logging
import time

logging.getLogger("tensorflow").setLevel(logging.ERROR)

class EnrollmentManager:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7 
        )
        self.model_name = "ArcFace"

    def _check_face_collision(self, current_vec, registry):
        """
        Compara el vector recién capturado contra todos los usuarios registrados.
        """
        for uid, data in registry.items():
            # Recuperación segura sin operador 'or' para evitar ValueError en NumPy
            stored_vectors = data.get("vector")
            if stored_vectors is None:
                stored_vectors = data.get(b"vector")
            
            if stored_vectors is None: 
                continue
            
            # Normalización: Si es matriz (3, 512) iteramos filas; si es vector (512,) lo envolvemos en lista
            search_list = stored_vectors if (isinstance(stored_vectors, np.ndarray) and stored_vectors.ndim > 1) else [stored_vectors]
            
            for vec in search_list:
                stored_emb = np.array(vec)
                # Cálculo de Similitud Coseno
                similarity = np.dot(current_vec, stored_emb) / (
                    np.linalg.norm(current_vec) * np.linalg.norm(stored_emb)
                )
                
                # Si la similitud supera el 70%, es el mismo rostro
                if similarity > 0.70:
                    name = data.get("nombre")
                    if name is None:
                        name = data.get(b"nombre", "Usuario Registrado")
                    return True, name
        return False, None

    def extract_embedding(self, frame):
        try:
            embeddings = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend="mediapipe",
                align=True
            )
            return embeddings[0]["embedding"]
        except:
            return None

    def enroll_student(self, nombre, codigo, registry):
        """
        Requiere el parámetro 'registry' para validar colisiones de rostro.
        """
        cap = cv2.VideoCapture(0)
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
            oval_w, oval_h = int(w * 0.20), int(h * 0.45) 

            results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_aligned = False

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    fx, fy, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                    f_center_x, f_center_y = fx + (fw // 2), fy + (fh // 2)
                    dist_to_center = np.sqrt((f_center_x - center_x)**2 + (f_center_y - center_y)**2)
                    
                    if dist_to_center < 35 and (fw > w * 0.25): 
                        face_aligned = True
                        if stable_start_time is None:
                            stable_start_time = time.time()
                    else:
                        stable_start_time = None

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
                
                if elapsed >= required_stable_duration:
                    vector = self.extract_embedding(frame)
                    if vector:
                        # VALIDACIÓN DE COLISIÓN (Solo en la primera captura)
                        if current_stage == 0:
                            is_duplicate, name = self._check_face_collision(np.array(vector), registry)
                            if is_duplicate:
                                print(f"\n[ALARMA] El rostro ya pertenece a: {name}")
                                cap.release()
                                cv2.destroyAllWindows()
                                return None # Cancela el registro

                        captured_vectors.append(vector)
                        current_stage += 1
                    stable_start_time = None 

            cv2.imshow("Multi-Enrolamiento Biometrico", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        return captured_vectors