#.\gui_app.py

import os
import cv2
import numpy as np
import time
import datetime
import unicodedata
import multiprocessing
from deepface import DeepFace
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager

def _clean_text(text):
    """Elimina tildes y eñes"""
    if not text: return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).replace('ñ', 'n').replace('Ñ', 'N').replace('á', 'a')

def ai_camera_worker(mode, name, code, active_class_id, active_class_name, allowed_students, result_queue, stop_event):
    """
    Motor de IA con manejo explícito de None para NumPy y color de texto corregido.
    """
    security = SecurityManager()
    persistence = PersistenceManager(security)
    
    cap = cv2.VideoCapture(0)
    # === OPTIMIZACIÓN DE RESOLUCIÓN Y FPS ===
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # =========================================
    
    # Calibración inicial
    for _ in range(5): cap.read()

    enroll_vectors = []
    enroll_stage = 0
    stages_text = ["Mire al frente", "Giro leve Izquierda", "Giro leve Derecha"]
    win_title = "REGISTRO BIOMETRICO" if mode == "enroll" else "CONTROL DE ASISTENCIA"
    last_capture_time = 0
    is_processing = False

    # Carga de perfiles
    all_profiles = persistence.load_profiles()
    # Memoria de sesión para evitar registros duplicados
    already_marked = persistence.get_already_marked_today(active_class_id)
    session_cooldown = {} 
    COOLDOWN_SEC = 8  
    frame_counter = 0  
    skip_frames = 5    

    # WARM-UP
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        DeepFace.represent(dummy_frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=False)
    except: pass
    
    status_bar_msg = "ESPERANDO ROSTRO..."
    status_bar_color = (255, 255, 255)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        key = cv2.waitKey(1) & 0xFF
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        light_ok = brightness > 90 
        sharp_ok = sharpness > 55

        face_ready = False
        try:
            faces = DeepFace.extract_faces(frame, detector_backend="mediapipe", enforce_detection=True)
            if faces and faces[0]["facial_area"]['w'] > (w * 0.10):
                face_ready = True
        except Exception as e:
            # Solo loguea si es un error real de librería, no por falta de rostro en frame
            if "Face could not be detected" not in str(e):
                print(f"[ERROR CRÍTICO DETECCIÓN] {e}")
            
        if not light_ok:
            status_color = (0, 165, 255)
            cv2.putText(display_frame, "ADVERTENCIA: POCA LUZ", (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        elif not sharp_ok:
            status_color = (0, 165, 255)
            cv2.putText(display_frame, "ADVERTENCIA: ENFOQUE POBRE", (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            status_color = (0, 255, 0) if face_ready else (0, 0, 255)

        cv2.ellipse(display_frame, (w//2, h//2), (int(w*0.22), int(h*0.35)), 0, 0, 360, status_color, 2)

        # Lógica de Enrolamiento
        if mode == "enroll":
            if enroll_stage < 3:
                txt_stage = f"ETAPA {enroll_stage+1}/3: {stages_text[enroll_stage]}"
                cv2.putText(display_frame, txt_stage, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                if light_ok and sharp_ok and face_ready:
                    cv2.putText(display_frame, "[PRESIONE ESPACIO PARA CAPTURAR]", (w//2 - 180, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if key == ord(' ') and not is_processing:
                        is_processing = True
                        try:
                            res = DeepFace.represent(frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=True)
                            emb = np.array(res[0]["embedding"])
                            
                            duplicate = False
                            if enroll_stage == 0:
                                for uid, data in all_profiles.items():
                                    stored = data.get("vector")
                                    if stored is None: stored = data.get(b"vector")
                                    
                                    if stored is not None:
                                        search_list = np.array(stored)
                                        if search_list.ndim == 1: search_list = [search_list]
                                        for v in search_list:
                                            v_arr = np.array(v)
                                            dist = np.dot(emb, v_arr) / (np.linalg.norm(emb) * np.linalg.norm(v_arr))
                                            print(f"[DEBUG] Comparando con {uid}: Similitud = {dist:.4f}")
                                            if dist > 0.80: # Umbral anti-duplicados
                                                print(f"!!! POSIBLE DUPLICADO DETECTADO: {dist:.4f}")
                                                duplicate = True; break
                                    if duplicate: break
                            
                            if duplicate:
                                result_queue.put("DUPLICATE")
                                break 
                            else:
                                enroll_vectors.append(emb)
                                enroll_stage += 1
                        except Exception as e:
                            print(f"Error técnico en represent: {e}")
                        is_processing = False
                else:
                    msg = "ACERQUESE O MEJORE LUZ" if not face_ready else "ESTABILICE LA CAMARA"
                    cv2.putText(display_frame, msg, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                current_registry = persistence.load_profiles()
                final_vector = np.mean(enroll_vectors, axis=0).astype(np.float32)
                final_vector = final_vector / np.linalg.norm(final_vector)
                
                current_registry[str(code)] = {
                    "nombre": name,
                    "vector": final_vector,
                    "fecha_registro": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S")
                }
                if persistence.save_profiles(current_registry):
                    result_queue.put("SUCCESS")
                    enroll_vectors = [] 
                    enroll_stage = 0
                break

        # Lógica de Asistencia
        elif mode == "attendance":
            cv2.putText(display_frame, f"CLASE: {active_class_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.rectangle(display_frame, (0, h - 50), (w, h), (0, 0, 0), -1)

            try:
                faces = DeepFace.extract_faces(frame, detector_backend="mediapipe", enforce_detection=False)
                if not faces:
                    status_bar_msg, status_bar_color = "CAMARA VACIA", (200, 200, 200)

                for face_data in faces:
                    facial_area = face_data["facial_area"]
                    x, y, w_f, h_f = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    cv2.rectangle(display_frame, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)

                    if w_f > (w * 0.10) and frame_counter % skip_frames == 0:
                        res = DeepFace.represent(frame[y:y+h_f, x:x+w_f], model_name="ArcFace", detector_backend="skip", enforce_detection=False)
                        emb = np.array(res[0]["embedding"])
                        best_dist, match_id, match_name = 0.0, None, None
                        
                        for uid, data in all_profiles.items():
                            stored = data.get("vector") if data.get("vector") is not None else data.get(b"vector")
                            if stored is not None:
                                v_arr = np.array(stored)
                                vectors_to_check = v_arr if v_arr.ndim > 1 else [v_arr]
                                for v_vec in vectors_to_check:
                                    v_vec_arr = np.array(v_vec)
                                    dist = np.dot(emb, v_vec_arr) / (np.linalg.norm(emb) * np.linalg.norm(v_vec_arr))
                                    
                                    if dist > 0.85:
                                        print(f"[DEBUG] Posible coincidencia: {data.get('nombre')} (ID: {uid}) - Dist: {dist:.4f}")
                                    
                                    if dist > 0.75 and dist > best_dist: # Umbral de asistencia
                                        best_dist, match_id, match_name = dist, uid, data.get("nombre")
                        
                        if match_id:
                            clean_name = _clean_text(match_name)
                            confianza_vip = round(float(best_dist), 2)
                            now = time.time()
                            last_log = session_cooldown.get(match_id, 0)

                            if str(match_id) in already_marked or (now - last_log < COOLDOWN_SEC):
                                status_bar_msg = f"{clean_name} (Ya registrado - Cooldown)"
                                status_bar_color = (255, 255, 0)
                            elif str(match_id) in allowed_students:
                                success = persistence.log_attendance({
                                    "codigo": str(match_id), 
                                    "nombre": match_name, 
                                    "confianza": confianza_vip,
                                    "clase_id": active_class_id,  
                                    "clase_nombre": active_class_name
                                })
                                if success:
                                    already_marked.add(str(match_id))
                                    session_cooldown[match_id] = now
                                    status_bar_msg = f"REGISTRADO: {clean_name} ({confianza_vip:.2f})"
                                    status_bar_color = (0, 255, 0)
                                    print(f"[ASISTENCIA] {clean_name} - Confianza: {confianza_vip}")
                                else:
                                    status_bar_msg = f"Error al guardar: {clean_name}"
                                    status_bar_color = (0, 0, 255)
                            else:
                                status_bar_msg = f"{clean_name}: NO MATRICULADO"
                                status_bar_color = (0, 0, 255)
                        else:
                            status_bar_msg = "ROSTRO NO REGISTRADO"
                            status_bar_color = (0, 0, 255)
                
                cv2.putText(display_frame, status_bar_msg, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_bar_color, 2)
            except: pass

        cv2.imshow(win_title, display_frame)
        if key == 27 or cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
            result_queue.put("CANCELLED")
            break
        frame_counter += 1
    cap.release()
    cv2.destroyAllWindows()