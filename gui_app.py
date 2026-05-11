import os
# # Silencia logs de TensorFlow (0 = todos, 1 = sin INFO, 2 = sin WARNING, 3 = ERROR solamente)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# Deshabilita advertencias de deprecación internas
tf.get_logger().setLevel('ERROR')

import customtkinter as ctk
import json
import unicodedata
import datetime
import cv2
import numpy as np
import time
import multiprocessing 
from PIL import Image
from deepface import DeepFace 
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager
from tkinter import messagebox

def _clean_text(text):
    """Elimina tildes y eñes para compatibilidad con OpenCV."""
    if not text: return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).replace('ñ', 'n').replace('Ñ', 'N').replace('á', 'a')

# ==========================================================
# PROCESO INDEPENDIENTE: MOTOR DE IA (SIDE-CAR)
# ==========================================================
def ai_camera_worker(mode, name, code, active_class_id, active_class_name, allowed_students, result_queue, stop_event):
    """
    Motor de IA con manejo explícito de None para NumPy y color de texto corregido.
    """
    security = SecurityManager()
    persistence = PersistenceManager(security)
    cap = cv2.VideoCapture(0)
    
    # Calibración inicial
    for _ in range(5): cap.read()

    enroll_vectors = []
    enroll_stage = 0
    stages_text = ["Mire al frente", "Gire a la izquierda", "Gire a la derecha"]
    win_title = "REGISTRO BIOMETRICO" if mode == "enroll" else "CONTROL DE ASISTENCIA"
    last_capture_time = 0
    is_processing = False
    
    # Carga de perfiles
    all_profiles = persistence.load_profiles()
    # Memoria de sesión para evitar registros duplicados en el mismo encendido de cámara
    already_marked = persistence.get_already_marked_today(active_class_id)
    frame_counter = 0  # Contador de cuadros
    skip_frames = 5    # PUNTO DULCE: Procesar cada 5 cuadros

    # WARM-UP: Forzamos la carga de modelos en RAM antes de abrir la ventana
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        DeepFace.represent(dummy_frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=False)
    except: pass
    # Alrededor de la línea 60, antes del while
    status_bar_msg = "ESPERANDO ROSTRO..."
    status_bar_color = (255, 255, 255) # Blanco inicial
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Evento de teclado único
        key = cv2.waitKey(1) & 0xFF
        
        # Análisis de Calidad
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        light_ok = brightness > 65 
        sharp_ok = sharpness > 35

        face_ready = False
        try:
            faces = DeepFace.extract_faces(frame, detector_backend="mediapipe", enforce_detection=True)
            if faces and faces[0]["facial_area"]['w'] > (w * 0.20):
                face_ready = True
        except: pass

        # UI y Colores
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
                # REQUERIMIENTO: Texto en color NEGRO (0, 0, 0) para contraste
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
                                    # CORRECCIÓN: Verificación explícita de None para evitar error de ambigüedad
                                    stored = data.get("vector")
                                    if stored is None:
                                        stored = data.get(b"vector")
                                    
                                    if stored is not None:
                                        search_list = np.array(stored)
                                        if search_list.ndim == 1: search_list = [search_list]
                                        for v in search_list:
                                            v_arr = np.array(v)
                                            dist = np.dot(emb, v_arr) / (np.linalg.norm(emb) * np.linalg.norm(v_arr))
                                            if dist > 0.70:
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
                all_profiles[code] = {
                    "nombre": name,
                    "vector": np.array(enroll_vectors, dtype=np.float32),
                    "fecha_registro": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S")
                }
                persistence.save_profiles(all_profiles)
                result_queue.put("SUCCESS")
                break

        # Lógica de Asistencia
        elif mode == "attendance":
            # REQUERIMIENTO: Nombre de materia en NEGRO (esquina superior)
            cv2.putText(display_frame, f"CLASE: {active_class_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Dibujar fondo para la barra de estado inferior (rectángulo negro opaco)
            cv2.rectangle(display_frame, (0, h - 50), (w, h), (0, 0, 0), -1)

            try:
                faces = DeepFace.extract_faces(frame, detector_backend="mediapipe", enforce_detection=False)
                
                # Si no hay rostros, actualizamos el mensaje (opcional)
                if not faces:
                    status_bar_msg, status_bar_color = "CAMARA VACIA", (200, 200, 200)

                for face_data in faces:
                    facial_area = face_data["facial_area"]
                    x, y, w_f, h_f = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Dibujar cuadro SIEMPRE para feedback visual (muy bajo consumo)
                    cv2.rectangle(display_frame, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)

                    # RECONOCIMIENTO PESADO: Solo en el "Punto Dulce"
                    if w_f > (w * 0.15) and frame_counter % skip_frames == 0:
                        res = DeepFace.represent(frame[y:y+h_f, x:x+w_f], model_name="ArcFace", detector_backend="skip", enforce_detection=False)
                        emb = np.array(res[0]["embedding"])
                        best_dist, match_id, match_name = 0.0, None, None
                        
                        for uid, data in all_profiles.items():
                            stored = data.get("vector") if data.get("vector") is not None else data.get(b"vector")
                            if stored is not None:
                                v_arr = np.array(stored)
                                if v_arr.ndim > 1: v_arr = v_arr[0] # Usar primer vector del enrolamiento
                                dist = np.dot(emb, v_arr) / (np.linalg.norm(emb) * np.linalg.norm(v_arr))
                                if dist > 0.82 and dist > best_dist:
                                    best_dist, match_id, match_name = dist, uid, data.get("nombre")

                        if match_id:
                            clean_name = _clean_text(match_name)
                            confianza_vip = round(float(best_dist), 2)
                            if str(match_id) in already_marked:
                                status_bar_msg = f"{clean_name} (Registrado)"
                                status_bar_color = (255, 255, 0) # Cian
                            elif str(match_id) in allowed_students:
                                success = persistence.log_attendance({
                                    "codigo": str(match_id), 
                                    "nombre": match_name, 
                                    "confianza": confianza_vip,
                                    "clase_id": active_class_id,  
                                    "clase_nombre": active_class_name
                                })
                                if success:
                                    status_bar_msg = f"REGISTRADO: {clean_name}"
                                    status_bar_color = (0, 255, 0) # Verde
                                    already_marked.add(str(match_id))
                                    print(f"[ASISTENCIA] {clean_name} - Confianza: {confianza_vip}")
                            else:
                                status_bar_msg = f"{clean_name}: NO MATRICULADO"
                                status_bar_color = (0, 0, 255) # Rojo
                        else:
                            status_bar_msg = "ROSTRO NO REGISTRADO"
                            status_bar_color = (0, 0, 255)
                
                # Renderizar el mensaje en la barra fija inferior
                cv2.putText(display_frame, status_bar_msg, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_bar_color, 2)
                
            except: pass

        cv2.imshow(win_title, display_frame)
        # Detecta Cierre por Tecla ESC o por el botón [X] del mouse
        if key == 27 or cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
            result_queue.put("CANCELLED")
            break
        frame_counter += 1
    cap.release()
    cv2.destroyAllWindows()

# ==========================================================
# GUI ADMINISTRATIVA
# ==========================================================
class ReconApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema Biométrico UT v1.0")
        self.geometry("1100x700")
        
        self.sidebar_buttons = []
        
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.worker_process = None
        
        self.security = SecurityManager()
        self.persistence = PersistenceManager(self.security)
        
        self.active_session_file = "data/attendance/active_session.json"
        self.classes_path = "data/meta/classes.json"
        self.current_session = self._load_active_session()
        
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        

    def _load_active_session(self):
        if os.path.exists(self.active_session_file):
            with open(self.active_session_file, "r", encoding='utf-8') as f:
                return json.load(f)
        return None

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._create_sidebar()
        self.main_view = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.show_dashboard()

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="RECON-FACIAL", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        
        # Guardamos cada botón en la lista self.sidebar_buttons
        btn_dash = ctk.CTkButton(self.sidebar, text="Dashboard", command=self.show_dashboard)
        btn_dash.pack(pady=5, padx=20); self.sidebar_buttons.append(btn_dash)
        
        btn_clas = ctk.CTkButton(self.sidebar, text="Gestión de Clases", command=self.show_classes)
        btn_clas.pack(pady=5, padx=20); self.sidebar_buttons.append(btn_clas)
        
        btn_enro = ctk.CTkButton(self.sidebar, text="Registrar Estudiante", command=self.show_enrollment)
        btn_enro.pack(pady=5, padx=20); self.sidebar_buttons.append(btn_enro)
        
        btn_assi = ctk.CTkButton(self.sidebar, text="Tomar Asistencia", command=self.show_live_attendance)
        btn_assi.pack(pady=5, padx=20); self.sidebar_buttons.append(btn_assi)
        
        btn_repo = ctk.CTkButton(self.sidebar, text="Reportes", command=self.show_reports)
        btn_repo.pack(pady=5, padx=20); self.sidebar_buttons.append(btn_repo)
        
        ctk.CTkButton(self.sidebar, text="Salir", fg_color="#e74c3c", command=self._on_closing).pack(pady=5, padx=20)

    def show_dashboard(self):
        self._clear_view()
        status = f"Sesión Activa: {self.current_session['clase']}" if self.current_session else "No hay clase activa"
        ctk.CTkLabel(self.main_view, text=f"PANEL DE CONTROL\n\n{status}", font=ctk.CTkFont(size=24)).pack(expand=True)

    def show_classes(self):
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="ADMINISTRACIÓN DE ASIGNATURAS", font=ctk.CTkFont(size=22)).pack(pady=20)
        f = ctk.CTkFrame(self.main_view); f.pack(fill="x", padx=40, pady=10)
        self.new_class_entry = ctk.CTkEntry(f, placeholder_text="Nombre de la materia"); self.new_class_entry.pack(side="left", padx=20, expand=True, fill="x")
        ctk.CTkButton(f, text="Añadir", command=self._create_new_class_logic).pack(side="right", padx=20)
        s_frame = ctk.CTkFrame(self.main_view); s_frame.pack(fill="both", expand=True, padx=40, pady=20)
        if self.current_session:
            ctk.CTkButton(s_frame, text="FINALIZAR SESIÓN", fg_color="#e74c3c", command=self._close_class_logic).pack(pady=10)
        else:
            if os.path.exists(self.classes_path):
                with open(self.classes_path, "r", encoding='utf-8') as f:
                    content = json.load(f)
                    classes_dict = content.get("classes", {})
                
                for cid, cinfo in classes_dict.items():
                    if cinfo.get("status") == "active":
                        row = ctk.CTkFrame(s_frame, fg_color="transparent")
                        row.pack(fill="x", pady=2)
                        
                        # Nombre y ID
                        btn_text = f"{cinfo['name']} ({cid})"
                        ctk.CTkButton(row, text=btn_text, width=220, anchor="w",
                                     command=lambda c=cid, n=cinfo['name']: self._activate_class_logic(c, n)).pack(side="left", padx=5)
                        
                        # Botón para gestionar quién entra a esta clase
                        ctk.CTkButton(row, text="Matricular", fg_color="#3498db", width=80,
                                     command=lambda c=cid: self._manage_enrollment_view(c)).pack(side="left", padx=5)
                        
                        # 3. BOTÓN DE RENOMBRAR
                        ctk.CTkButton(row, text="Renombrar", fg_color="#063B57", width=70,
                                     command=lambda c=cid, n=cinfo['name']: self._rename_class_logic(c, n)).pack(side="left", padx=5)
                        
                        # Botón de eliminación lógica
                        ctk.CTkButton(row, text="Eliminar", fg_color="#c0392b", width=80,
                                     command=lambda c=cid: self._delete_class_logic(c)).pack(side="right", padx=5)

    def show_enrollment(self):
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="REGISTRO DE ESTUDIANTE", font=ctk.CTkFont(size=22)).pack(pady=10)
        self.ent_name = ctk.CTkEntry(self.main_view, placeholder_text="Nombre Completo", width=400); self.ent_name.pack(pady=5)
        self.ent_code = ctk.CTkEntry(self.main_view, placeholder_text="Código", width=400); self.ent_code.pack(pady=5)
        ctk.CTkButton(self.main_view, text="ABRIR CÁMARA DE REGISTRO", command=self._launch_enroll_worker).pack(pady=20)
        self.enroll_status = ctk.CTkLabel(self.main_view, text="")
        self.enroll_status.pack(pady=10)

    def _launch_enroll_worker(self):
        n, c = self.ent_name.get().strip(), self.ent_code.get().strip()
        if not n or not c: return
        self.enroll_status.configure(text="Iniciando cámara...", text_color="#3498db")
        self.stop_event.clear()
        self._set_sidebar_state("disabled")
        self.worker_process = multiprocessing.Process(
            target=ai_camera_worker, 
            args=("enroll", n, c, None, None, self.result_queue, self.stop_event)
        )
        self.worker_process.start()
        self._listen_for_result()

    def _listen_for_result(self):
        try:
            res = self.result_queue.get_nowait()
            
            # Si el mensaje es de registro exitoso, NO desbloqueamos aún 
            # porque la cámara sigue abierta esperando más gente.
            if res.startswith("REG:"):
                nombre = res.split(":")[1]
                if hasattr(self, 'lbl_last_reg'):
                    self.lbl_last_reg.configure(text=f"✅ REGISTRO EXITOSO: {nombre}", text_color="#2ecc71")
            
            # Si el proceso se cierra (SUCCESS final, DUPLICATE o CANCELLED)
            elif res in ["SUCCESS", "DUPLICATE", "CANCELLED"]:
                if res == "SUCCESS":
                    self.enroll_status.configure(text="✅ PROCESO COMPLETADO", text_color="#2ecc71")
                elif res == "CANCELLED":
                    if hasattr(self, 'enroll_status'):
                        self.enroll_status.configure(text="⚠️ CÁMARA CERRADA", text_color="#f1c40f")
                
                self._set_sidebar_state("normal") # DESBLOQUEAMOS EL MENÚ
                
        except:
            # Si el proceso sigue vivo pero no hay mensajes finales, seguimos escuchando
            if self.worker_process and self.worker_process.is_alive():
                self.after(500, self._listen_for_result)
            else:
                # Si el proceso murió inesperadamente, desbloqueamos por seguridad
                self._set_sidebar_state("normal")

    def show_live_attendance(self):
        self._clear_view()
        if not self.current_session: return
        
        # Cargamos los estudiantes permitidos navegando por la nueva estructura de IDs
        allowed = []
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                # Obtenemos el ID de la sesión actual (ej: CLS-001)
                cid = self.current_session.get("class_id")
                # Acceso correcto según ISO 25012: classes -> ID -> students
                allowed = data.get("classes", {}).get(cid, {}).get("students", [])

        # Título de la materia y feedback visual para el usuario
        ctk.CTkLabel(self.main_view, text=f"ASISTENCIA: {self.current_session['clase']}", 
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        
        self.lbl_last_reg = ctk.CTkLabel(self.main_view, text="Esperando registros...", 
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_last_reg.pack(pady=20)

        self._set_sidebar_state("disabled")
        self.stop_event.clear()
        
        self.worker_process = multiprocessing.Process(
            target=ai_camera_worker, 
            args=(
                "attendance", 
                None, 
                None, 
                self.current_session['class_id'],
                self.current_session['clase'],   
                allowed, 
                self.result_queue, 
                self.stop_event
            )
        )
        self.worker_process.start()
        self._listen_for_result()

    def _set_sidebar_state(self, state="normal"):
        """Habilita o deshabilita los botones del menú lateral."""
        for btn in self.sidebar_buttons:
            btn.configure(state=state)
    
    def _clear_view(self):
        for w in self.main_view.winfo_children(): w.destroy()

    def _activate_class_logic(self, cid, name):
        self.current_session = {
            "class_id": cid, 
            "clase": name, 
            "session_id": f"{cid}_{int(time.time())}"
        }
        with open(self.active_session_file, "w", encoding='utf-8') as f:
            json.dump(self.current_session, f)
        
        # Registro en el log administrativo
        self.persistence.log_admin_action(
            event_type="SESSION_START",
            target_id=cid,
            description=f"Sesión de asistencia abierta para: {name}"
        )
        self.show_dashboard()

    def _close_class_logic(self):
        if self.current_session:
            cid = self.current_session.get("class_id")
            name = self.current_session.get("clase")
            
            # Registro en el log administrativo antes de borrar la sesión
            self.persistence.log_admin_action(
                event_type="SESSION_END",
                target_id=cid,
                description=f"Sesión de asistencia cerrada para: {name}"
            )
            
        if os.path.exists(self.active_session_file): 
            os.remove(self.active_session_file)
        self.current_session = None
        self.show_dashboard()

    def _create_new_class_logic(self):
        name = self.new_class_entry.get().strip()
        if not name: return

        # Estructura inicial si el archivo no existe
        data = {"classes": {}}
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r", encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if "classes" not in data: data = {"classes": {}}
                except: data = {"classes": {}}

        # Generación de ID Automático (CLS-001, CLS-002...)
        existing_ids = [int(k.split('-')[1]) for k in data["classes"].keys() if k.startswith("CLS-")]
        next_id_num = max(existing_ids + [0]) + 1
        new_id = f"CLS-{next_id_num:03d}"

        # Inserción con metadatos para cumplimiento ISO 25012
        data["classes"][new_id] = {
            "name": name,
            "students": [],
            "status": "active", # active | hidden
            "created_at": datetime.datetime.now().isoformat()
        }

        with open(self.classes_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            # Registro en el log administrativo
            self.persistence.log_admin_action(
                event_type="CREATE_CLASS",
                target_id=new_id,
                description=f"Creación de materia: {name}"
            )
        
        self.new_class_entry.delete(0, 'end')
        self.show_classes()

    def show_reports(self): self._clear_view(); ctk.CTkLabel(self.main_view, text="REPORTES").pack(pady=20)

    def _on_closing(self):
        self.stop_event.set()
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate()
        self.destroy()
        
    def _delete_class_logic(self, cid):
        """Marca una clase como 'hidden' con confirmación previa."""
        if not os.path.exists(self.classes_path): return
        
        # Cuadro de diálogo de confirmación
        if not messagebox.askyesno("Confirmar Eliminación", "¿Está seguro de eliminar esta materia?"):
            return

        with open(self.classes_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        if cid in data.get("classes", {}):
            old_name = data["classes"][cid].get("name", "Desconocida")
            data["classes"][cid]["status"] = "hidden"
            data["classes"][cid]["deactivated_at"] = datetime.datetime.now().isoformat()
            
            with open(self.classes_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            # Log administrativo inmediatamente después del dump
            self.persistence.log_admin_action(
                event_type="DELETE_CLASS",
                target_id=cid,
                description=f"Materia '{old_name}' marcada como oculta (Soft Delete)"
            )
        
        if self.current_session and self.current_session.get("class_id") == cid:
            self._close_class_logic()
        else:
            self.show_classes()
            
    def _manage_enrollment_view(self, cid):
        self._clear_view()
        
        with open(self.classes_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        cinfo = data["classes"][cid]
        
        ctk.CTkLabel(self.main_view, text=f"MATRÍCULA: {cinfo['name']}", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Panel superior: Añadir estudiante por código
        add_frame = ctk.CTkFrame(self.main_view)
        add_frame.pack(fill="x", padx=40, pady=10)
        
        self.enroll_code_entry = ctk.CTkEntry(add_frame, placeholder_text="Ingrese código del estudiante")
        self.enroll_code_entry.pack(side="left", padx=20, pady=10, expand=True, fill="x")
        
        ctk.CTkButton(add_frame, text="Inscribir", command=lambda: self._add_student_to_class(cid)).pack(side="right", padx=20)
        
        # Panel inferior: Lista de matriculados actualmente
        ctk.CTkLabel(self.main_view, text="Estudiantes Inscritos:").pack(pady=5)
        scroll_frame = ctk.CTkScrollableFrame(self.main_view, height=300)
        scroll_frame.pack(fill="both", expand=True, padx=40, pady=10)
        
        profiles = self.persistence.load_profiles()
        
        for student_code in cinfo["students"]:
            s_row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            s_row.pack(fill="x", pady=2)
            
            # Buscamos el nombre en los perfiles biométricos para mostrarlo
            s_name = profiles.get(student_code, {}).get("nombre", "Código no registrado")
            ctk.CTkLabel(s_row, text=f"{student_code} - {s_name}", anchor="w").pack(side="left", padx=10)
            
            ctk.CTkButton(s_row, text="Retirar", fg_color="#e67e22", width=60, height=20,
                         command=lambda sc=student_code: self._remove_student_from_class(cid, sc)).pack(side="right", padx=10)
        
        ctk.CTkButton(self.main_view, text="Volver", command=self.show_classes).pack(pady=10)

    def _add_student_to_class(self, cid):
        code = self.enroll_code_entry.get().strip()
        if not code: return
        
        # Verificar si el estudiante existe en la base biométrica
        profiles = self.persistence.load_profiles()
        if code not in profiles:
            # Aquí podrías poner un mensaje de error en la UI
            print(f"Error: Estudiante {code} no existe en el sistema biométrico.")
            return

        with open(self.classes_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            
        if code not in data["classes"][cid]["students"]:
            data["classes"][cid]["students"].append(code)
            with open(self.classes_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            # Registro en el log administrativo
            self.persistence.log_admin_action(
                event_type="ENROLL_STUDENT",
                target_id=cid,
                description=f"Estudiante {code} matriculado en la clase"
            )
                
        self._manage_enrollment_view(cid)

    def _remove_student_from_class(self, cid, student_code):
        with open(self.classes_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            
        if student_code in data["classes"][cid]["students"]:
            data["classes"][cid]["students"].remove(student_code)
            with open(self.classes_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            # Registro en el log administrativo
            self.persistence.log_admin_action(
                event_type="UNENROLL_STUDENT",
                target_id=cid,
                description=f"Estudiante {student_code} retirado de la clase"
            )
                
        self._manage_enrollment_view(cid)
        
    def _rename_class_logic(self, cid, old_name):
        dialog = ctk.CTkInputDialog(text=f"Nuevo nombre para '{old_name}':", title="Renombrar Materia")
        new_name = dialog.get_input()
        
        if new_name and new_name.strip() and new_name != old_name:
            new_name = new_name.strip()
            with open(self.classes_path, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            data["classes"][cid]["name"] = new_name
            
            with open(self.classes_path, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            # Registro de auditoría del cambio de nombre
            self.persistence.log_admin_action(
                event_type="RENAME_CLASS",
                target_id=cid,
                description=f"Cambio de nombre: '{old_name}' -> '{new_name}'"
            )
            
            # Si es la clase activa, actualizamos la sesión en vivo para que el log de asistencia use el nuevo nombre
            if self.current_session and self.current_session.get("class_id") == cid:
                self.current_session["clase"] = new_name
                with open(self.active_session_file, "w", encoding='utf-8') as f:
                    json.dump(self.current_session, f)
            
            self.show_classes()    
                

if __name__ == "__main__": 
    multiprocessing.freeze_support() 
    app = ReconApp(); app.mainloop()