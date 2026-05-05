import customtkinter as ctk
import json
import os
import datetime
import cv2
import numpy as np
import time

from PIL import Image
from deepface import DeepFace 
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager
from src.enrollment import EnrollmentManager

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ReconApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Asistencia Biométrica UT")
        self.geometry("1100x700")
        ctk.set_appearance_mode("Dark")
        
        self.security = SecurityManager()
        self.persistence = PersistenceManager(self.security)
        self.enrollment = EnrollmentManager()
        
        self.is_processing = False  # Flag para evitar saturación de DeepFace
        self.is_shutting_down = False # Flag para detener bucles de inmediato
        
        # Rutas técnicas
        self.active_session_file = "data/attendance/active_session.json"
        self.classes_path = "data/meta/classes.json"
        self.current_session = self._load_active_session()

        # Variables de control de cámara
        self.cap = None
        self.camera_loop_id = None
        self.enroll_vectors = []
        self.enroll_stage = 0
        self.last_capture_time = 0
        self.enroll_stages_text = ["Mire al frente", "Gire la cabeza a la izquierda", "Gire la cabeza a la derecha"]

        self._setup_ui()

    def _load_active_session(self):
        """Carga la sesión activa usando codificación UTF-8."""
        if os.path.exists(self.active_session_file):
            # Especificar encoding='utf-8' previene caracteres raros al leer
            with open(self.active_session_file, "r", encoding='utf-8') as f:
                return json.load(f)
        return None

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self._create_sidebar()
        
        # Vista Principal
        self.main_view = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.show_dashboard()

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="RECON-FACIAL", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        
        ctk.CTkButton(self.sidebar, text="Dashboard", command=self.show_dashboard).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Gestión de Clases", command=self.show_classes).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Registrar Estudiante", command=self.show_enrollment).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Tomar Asistencia", command=self.show_live_attendance).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Reportes", command=self.show_reports).pack(pady=5, padx=20)

    def show_dashboard(self):
        self._clear_view()
        status = f"Sesión Activa: {self.current_session['clase']}" if self.current_session else "No hay clase activa"
        color = "#2ecc71" if self.current_session else "#e74c3c"
        ctk.CTkLabel(self.main_view, text=f"ESTADO DEL SISTEMA\n\n{status}", 
                     font=ctk.CTkFont(size=24), text_color=color).pack(expand=True)

    def show_classes(self):
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="ADMINISTRACIÓN DE ASIGNATURAS", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=20)

        # Panel para crear nuevas clases
        create_frame = ctk.CTkFrame(self.main_view)
        create_frame.pack(fill="x", padx=40, pady=10)
        self.new_class_entry = ctk.CTkEntry(create_frame, placeholder_text="Nombre de la nueva materia")
        self.new_class_entry.pack(side="left", padx=20, pady=15, expand=True, fill="x")
        ctk.CTkButton(create_frame, text="Añadir", command=self._create_new_class_logic).pack(side="right", padx=20)
                
        session_frame = ctk.CTkFrame(self.main_view)
        session_frame.pack(fill="both", expand=True, padx=40, pady=20)

        if self.current_session:
            # La interfaz mostrará la tilde correctamente porque Python ya maneja el string en memoria
            ctk.CTkLabel(session_frame, text="SESIÓN EN CURSO", text_color="#2ecc71", font=ctk.CTkFont(weight="bold")).pack(pady=10)
            ctk.CTkLabel(session_frame, text=f"{self.current_session['clase']}\nID: {self.current_session['session_id']}").pack(pady=10)
            ctk.CTkButton(session_frame, text="FINALIZAR CLASE", fg_color="#e74c3c", command=self._close_class_logic).pack(pady=20)
        else:
            if os.path.exists(self.classes_path):
                # Abrir con UTF-8 para leer correctamente 'Mecánica' o 'Diseño'
                with open(self.classes_path, "r", encoding='utf-8') as f:
                    classes_data = json.load(f)
                for c_name in classes_data.keys():
                    ctk.CTkButton(session_frame, text=f"Activar: {c_name}", 
                                  command=lambda name=c_name: self._activate_class_logic(name)).pack(pady=5, padx=100, fill="x")

    def _activate_class_logic(self, class_name):
        """Genera el token de sesión con soporte para caracteres especiales."""
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        # Para el session_id (usado en nombres de archivos), es mejor limpiar la tilde
        s_id_clean = class_name.replace(' ', '_').replace('ñ', 'n').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        s_id = f"{s_id_clean}_{ts}"
        
        self.current_session = {"clase": class_name, "session_id": s_id}
        
        # ensure_ascii=False permite que 'ñ' y tildes se guarden como texto legible
        with open(self.active_session_file, "w", encoding='utf-8') as f:
            json.dump(self.current_session, f, ensure_ascii=False, indent=4)
        
        self.show_dashboard()

    def _close_class_logic(self):
        if os.path.exists(self.active_session_file):
            os.remove(self.active_session_file)
        self.current_session = None
        self.show_dashboard()

    def _create_new_class_logic(self):
        name = self.new_class_entry.get().strip()
        if not name: return
        data = {}
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r", encoding='utf-8') as f:
                data = json.load(f)
        if name not in data:
            data[name] = []
            # Guardar con ensure_ascii=False para que el JSON sea legible por humanos
            with open(self.classes_path, "w", encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            self.show_classes()

    def show_live_attendance(self):
        self._clear_view()
        
        if not self.current_session:
            ctk.CTkLabel(self.main_view, text="ERROR: NO HAY CLASE ACTIVA", 
                         text_color="#e74c3c", font=ctk.CTkFont(size=20, weight="bold")).pack(expand=True)
            return

        # Título de la Sesión
        ctk.CTkLabel(self.main_view, text=f"ASISTENCIA: {self.current_session['clase']}", 
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        # Contenedor de Video
        self.video_label = ctk.CTkLabel(self.main_view, text="")
        self.video_label.pack(pady=10)

        # Panel de Estado Inferior
        self.status_label = ctk.CTkLabel(self.main_view, text="Iniciando cámara...", 
                                         font=ctk.CTkFont(size=16))
        self.status_label.pack(pady=10)

        # Inicialización de captura y carga de datos
        self.cap = cv2.VideoCapture(0)
        self.all_profiles = self.persistence.load_profiles()
        
        # Cargar lista de matriculados para esta clase
        with open(self.classes_path, "r", encoding='utf-8') as f:
            classes_data = json.load(f)
        self.allowed_students = classes_data.get(self.current_session['clase'], [])

        self.process_frame_count = 0
        self._update_video_stream()
    
    def _update_enrollment_stream(self, name, code, profiles):
        if self.is_shutting_down or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret: return

        display_frame = frame.copy()
        self.process_frame_count = getattr(self, 'process_frame_count', 0) + 1
        
        face_ready = False
        if self.process_frame_count % 5 == 0:
            face_ready = self._draw_overlay(display_frame)
            self.last_face_ready = face_ready 
        else:
            face_ready = getattr(self, 'last_face_ready', False)
            self._draw_overlay_static(display_frame, face_ready) 

        self._display_frame(display_frame)

        if face_ready and not self.is_processing and self.enroll_stage < 3:
            if (time.time() - self.last_capture_time) > 2.5:
                self.is_processing = True
                self.after(10, lambda: self._run_heavy_analysis(frame, name, code, profiles))

        self.camera_loop_id = self.after(30, lambda: self._update_enrollment_stream(name, code, profiles))
    
    def _process_biometrics(self, frame):
        """Lógica de reconocimiento y validación de matrícula."""
        try:
            results = DeepFace.represent(
                img_path=frame, 
                model_name="ArcFace",
                detector_backend="mediapipe",
                enforce_detection=False,
                align=True
            )
        except:
            return

        for res in results:
            if res["facial_area"]["w"] < (frame.shape[1] * 0.20): continue
            
            current_embedding = np.array(res["embedding"])
            match_id, match_name = self._find_best_match(current_embedding)

            if match_id:
                # VALIDACIÓN DE MATRÍCULA
                if str(match_id) in self.allowed_students:
                    self.status_label.configure(text=f"IDENTIFICADO: {match_name}", text_color="#2ecc71")
                    # Registrar en el log de asistencia
                    self.persistence.log_attendance(match_id, match_name, self.current_session['session_id'])
                else:
                    self.status_label.configure(text=f"ALERTA: {match_name} NO MATRICULADO", text_color="#f1c40f")
            else:
                self.status_label.configure(text="Rostro no reconocido", text_color="#e74c3c")

    def _find_best_match(self, current_vec):
        """Búsqueda global corregida para evitar crashes por dimensiones."""
        best_dist = 0.0
        best_id, best_name = None, None

        for uid, data in self.all_profiles.items():
            stored = data.get("vector")
            if stored is None:
                stored = data.get(b"vector")
            
            if stored is None: continue
            
            # Aplicamos la misma corrección de dimensiones
            arr = np.array(stored)
            search_list = arr if arr.ndim > 1 else arr.reshape(1, -1)
            
            for vec in search_list:
                stored_emb = np.array(vec)
                dist = np.dot(current_vec, stored_emb) / (np.linalg.norm(current_vec) * np.linalg.norm(stored_emb))
                
                if dist > 0.82 and dist > best_dist:
                    best_dist = dist
                    best_id = uid
                    best_name = data.get("nombre") or data.get(b"nombre", "Estudiante")

        return best_id, best_name

    # --- ENROLAMIENTO ---
    def show_enrollment(self):
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="REGISTRO DE NUEVO ESTUDIANTE", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=10)

        # Formulario
        form_frame = ctk.CTkFrame(self.main_view)
        form_frame.pack(fill="x", padx=40, pady=10)
        
        self.ent_name = ctk.CTkEntry(form_frame, placeholder_text="Nombre Completo")
        self.ent_name.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.ent_code = ctk.CTkEntry(form_frame, placeholder_text="Código Estudiantil")
        self.ent_code.pack(side="left", padx=10, pady=10, expand=True, fill="x")

        self.btn_start_enroll = ctk.CTkButton(self.main_view, text="Iniciar Captura Biométrica", command=self._start_enrollment_logic)
        self.btn_start_enroll.pack(pady=10)

        self.video_label = ctk.CTkLabel(self.main_view, text="")
        self.video_label.pack(pady=10)

        self.enroll_status = ctk.CTkLabel(self.main_view, text="Ingrese datos para comenzar", font=ctk.CTkFont(size=16))
        self.enroll_status.pack(pady=10)

    def _draw_overlay(self, frame):
        """Dibuja la guía visual y detecta presencia básica de rostro."""
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        axes = (int(w * 0.22), int(h * 0.35))
        
        face_in_zone = False
        oval_color = (0, 0, 255) # Rojo por defecto

        try:
            # Detección rápida con mediapipe (muy ligera)
            faces = DeepFace.extract_faces(frame, detector_backend="mediapipe", enforce_detection=True)
            if faces:
                fw = faces[0]["facial_area"]['w']
                if fw > (w * 0.25): # Si está lo suficientemente cerca
                    oval_color = (0, 255, 0) # Verde
                    face_in_zone = True
        except: pass

        cv2.ellipse(frame, center, axes, 0, 0, 360, oval_color, 2)
        cv2.putText(frame, f"PASO {self.enroll_stage + 1}/3: {self.enroll_stages_text[self.enroll_stage]}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return face_in_zone
    
    def _draw_overlay_static(self, frame, face_ready):
        """Dibuja la guía visual sin realizar detección (Modo Pasivo)."""
        h, w, _ = frame.shape
        center = (w // 2, h // 2)
        axes = (int(w * 0.22), int(h * 0.35))
        oval_color = (0, 255, 0) if face_ready else (0, 0, 255)
        
        cv2.ellipse(frame, center, axes, 0, 0, 360, oval_color, 2)
        cv2.putText(frame, f"PASO {self.enroll_stage + 1}/3: {self.enroll_stages_text[self.enroll_stage]}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
        
    def _start_enrollment_logic(self):
        name = self.ent_name.get().strip()
        code = self.ent_code.get().strip()

        if not name or not code:
            self.enroll_status.configure(text="Error: Complete todos los campos", text_color="#e74c3c")
            return

        # Verificar duplicado administrativo
        profiles = self.persistence.load_profiles()
        if code in profiles:
            self.enroll_status.configure(text=f"Error: El código {code} ya existe", text_color="#e74c3c")
            return

        # Iniciar captura
        self.enroll_vectors = []
        self.enroll_stage = 0
        self.cap = cv2.VideoCapture(0)
        self.btn_start_enroll.configure(state="disabled")
        self._update_enrollment_stream(name, code, profiles)

    def _update_enrollment_stream(self, name, code, profiles):
        # 1. Protección de cierre y hardware
        if self.is_shutting_down or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret: return

        display_frame = frame.copy()
        
        # 2. Detección optimizada: Solo procesamos overlay cada 5 cuadros
        # Esto libera un 80% de carga de CPU en la UI
        face_ready = False
        if getattr(self, 'process_frame_count', 0) % 5 == 0:
            face_ready = self._draw_overlay(display_frame)
            self.last_face_ready = face_ready # Guardar estado
        else:
            face_ready = getattr(self, 'last_face_ready', False)
            self._draw_overlay_static(display_frame, face_ready) # Solo dibuja, no detecta

        self.process_frame_count = getattr(self, 'process_frame_count', 0) + 1
        self._display_frame(display_frame)

        # 3. Disparo de análisis pesado
        if face_ready and not self.is_processing and self.enroll_stage < 3:
            if (time.time() - self.last_capture_time) > 2.5:
                self.is_processing = True
                self.after(10, lambda: self._run_heavy_analysis(frame, name, code, profiles))

        self.camera_loop_id = self.after(30, lambda: self._update_enrollment_stream(name, code, profiles))

    def _run_heavy_analysis(self, frame, name, code, profiles):
        try:
            res = DeepFace.represent(frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=True)
            embedding = np.array(res[0]["embedding"])

            # Verificamos si el widget sigue vivo antes de interactuar
            if not hasattr(self, 'enroll_status') or not self.enroll_status.winfo_exists():
                return

            if self.enroll_stage == 0:
                if self._check_face_duplicate(embedding, profiles):
                    self.enroll_status.configure(text="BLOQUEADO: Rostro ya registrado", text_color="#e74c3c")
                    self._stop_camera()
                    return

            self.enroll_vectors.append(embedding)
            self.enroll_stage += 1
            self.last_capture_time = time.time()
            
            if self.enroll_stage == 3:
                self._save_enrollment(name, code, profiles)
            else:
                self.enroll_status.configure(text=f"¡Captura {self.enroll_stage}/3 lista!", text_color="#2ecc71")
        except Exception as e:
            print(f"[DEBUG] Reintentando captura... {e}")
        finally:
            self.is_processing = False

    def _check_face_duplicate(self, current_vec, profiles):
        """Bloqueo de seguridad: Detecta duplicados biométricos (Umbral 0.70)."""
        for uid, data in profiles.items():

            stored = data.get("vector")
            if stored is None:
                stored = data.get(b"vector")
            
            if stored is None: continue
            
            arr = np.array(stored)
            search_list = arr if arr.ndim > 1 else arr.reshape(1, -1)
            
            for vec in search_list:
                stored_emb = np.array(vec)
                # Similitud Coseno
                dist = np.dot(current_vec, stored_emb) / (np.linalg.norm(current_vec) * np.linalg.norm(stored_emb))
                
                # Umbral 0.70 para evitar que se registre dos veces
                if dist > 0.70: return True
        return False

    def _save_enrollment(self, name, code, profiles):
        try:
            profiles[code] = {
                "nombre": name,
                "vector": np.array(self.enroll_vectors, dtype=np.float32),
                "fecha_registro": str(np.datetime64('now'))
            }
            self.persistence.save_profiles(profiles)
            self._stop_camera() # Primero matamos el proceso
            self.video_label.configure(image="", text="✅ REGISTRO EXITOSO\n\nCámara liberada y datos cifrados.")
            self.enroll_status.configure(text=f"¡Estudiante {name} registrado!", text_color="#2ecc71")
        except Exception as e:
            self.enroll_status.configure(text=f"Error: {e}", text_color="#e74c3c")
    
    def show_reports(self):
        """Placeholder para la funcionalidad de reportes."""
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="MÓDULO DE REPORTES\n(Próximamente)", 
                     font=ctk.CTkFont(size=20)).pack(expand=True)
    
# --- MÉTODOS COMPARTIDOS Y OTROS ---
    def _display_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
        self.video_label.configure(image=img_ctk)
        self.video_label.image = img_ctk

    def _stop_camera(self):
        self.is_shutting_down = True
        if self.camera_loop_id:
            self.after_cancel(self.camera_loop_id)
            self.camera_loop_id = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Verificación estricta de existencia antes de configurar
        if hasattr(self, 'btn_start_enroll') and self.btn_start_enroll:
            try:
                if self.btn_start_enroll.winfo_exists():
                    self.btn_start_enroll.configure(state="normal")
            except: pass

    def _clear_view(self):
        self._stop_camera()
        # Esperar un breve ciclo de Tkinter para asegurar que la cámara cerró
        self.update_idletasks() 
        
        for w in self.main_view.winfo_children():
            w.destroy()
        
        # Reset de punteros críticos
        self.btn_start_enroll = None
        self.is_shutting_down = False

    def show_dashboard(self):
        self._clear_view()
        status = f"Sesión Activa: {self.current_session['clase']}" if self.current_session else "No hay clase activa"
        color = "#2ecc71" if self.current_session else "#e74c3c"
        ctk.CTkLabel(self.main_view, text=f"ESTADO DEL SISTEMA\n\n{status}", 
                     font=ctk.CTkFont(size=24), text_color=color).pack(expand=True)

if __name__ == "__main__":
    app = ReconApp()
    app.mainloop()