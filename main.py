# --- PUENTE DE COMPATIBILIDAD ---
import sys
try:
    import keras
    sys.modules['tensorflow.keras'] = keras
except ImportError:
    pass
# --------------------------------

import cv2
import numpy as np
from deepface import DeepFace
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager
import time


def start_identification():
    # 1. Inicialización de gestores y carga de perfiles cifrados
    security = SecurityManager()
    persistence = PersistenceManager(security)
    
    # Carga el registro (diccionario de vectores y nombres)
    registry = persistence.load_profiles()
    
    if not registry:
        print("[ERROR] Registro vacío o no encontrado.")
        return

    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    frame_count = 0
    process_every_n_frames = 5 
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # Cálculo de FPS para telemetría
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time

        # Procesamiento diferido para mantener la fluidez
        if frame_count % process_every_n_frames == 0:
            try:
                # Extracción con MediaPipe y alineación forzada para paridad técnica[cite: 14]
                last_results = DeepFace.represent(
                    img_path=frame,
                    model_name="ArcFace",
                    detector_backend="mediapipe", 
                    enforce_detection=False,
                    align=True
                )
            except:
                last_results = []

        # 2. Análisis de rostros detectados
        for res in last_results:
            face_area = res["facial_area"]

            # FILTRO DE ÁREA: Ignora rostros que ocupen menos del 20% del ancho del frame[cite: 14]
            # Esto evita identificar accidentalmente cuadros o retratos en el fondo.
            frame_width = frame.shape[1]
            if face_area['w'] < (frame_width * 0.20):
                continue 
            
            current_embedding = np.array(res["embedding"])
            match_name = "Desconocido"
            best_dist = 0.0

            # 3. Comparación contra la base de datos (Registry)
            for user_id, data in registry.items():
                # RECUPERACIÓN SEGURA: Evita el error de ambigüedad de NumPy[cite: 14]
                stored_vectors = data.get("vector")
                if stored_vectors is None:
                    stored_vectors = data.get(b"vector")
                
                user_name = data.get("nombre")
                if user_name is None:
                    user_name = data.get(b"nombre", "Sin Nombre")
                
                if stored_vectors is None: 
                    continue 

                # --- CORRECCIÓN DE DIMENSIONES PARA MULTI-VECTOR ---
                # Si el dato recuperado es un array de NumPy
                if isinstance(stored_vectors, np.ndarray):
                    # Si tiene 2 dimensiones (matriz de 3x512), lo usamos directamente para iterar filas
                    # Si tiene 1 dimensión (vector de 512), lo metemos en una lista para el bucle
                    search_list = stored_vectors if stored_vectors.ndim > 1 else [stored_vectors]
                else:
                    # Si es una lista de Python, la usamos tal cual
                    search_list = stored_vectors if isinstance(stored_vectors, list) else [stored_vectors]

                current_best_match = 0.0
                for vec in search_list:
                    stored_embedding = np.array(vec)
                    
                    # Cálculo de Similitud Coseno (Ahora las formas coinciden: 512 y 512)
                    dist = np.dot(current_embedding, stored_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
                    )
                    
                    if dist > current_best_match:
                        current_best_match = dist
                # ---------------------------------------------------

                # Umbral de confianza ajustado a 0.60 para mayor flexibilidad[cite: 14]
                if current_best_match > 0.60 and current_best_match > best_dist:
                    best_dist = current_best_match
                    match_name = user_name

            # 4. Renderizado de resultados en pantalla[cite: 14]
            x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
            color = (0, 255, 0) if match_name != "Desconocido" else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{match_name} ({best_dist:.2f})"
            cv2.putText(frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Telemetría de FPS[cite: 14]
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Asistencia - Telemetria", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_identification()