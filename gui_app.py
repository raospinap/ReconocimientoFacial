import customtkinter as ctk
import json
import os
import datetime
import cv2
import numpy as np 
from deepface import DeepFace 
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager
from src.enrollment import EnrollmentManager
from PIL import Image


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ReconApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Asistencia Biométrica")
        self.geometry("1100x600")
        ctk.set_appearance_mode("Dark")
        
        self.security = SecurityManager()
        self.persistence = PersistenceManager(self.security)
        self.enrollment = EnrollmentManager()
        
        # Rutas técnicas
        self.active_session_file = "data/attendance/active_session.json"
        self.classes_path = "data/meta/classes.json"
        
        self.current_session = self._load_active_session()
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
        self._create_sidebar()
        self.main_view = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.show_dashboard()

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="RECON-FACIAL", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        
        ctk.CTkButton(self.sidebar, text="Dashboard", command=self.show_dashboard).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Gestión de Clases", command=self.show_classes).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Registrar Estudiante", command=lambda: print("Enrolamiento")).pack(pady=5, padx=20)
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

        # ... (código previo del frame de creación) ...

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

    def _update_video_stream(self):
        """Bucle principal de la cámara integrado en la UI."""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if ret:
            self.process_frame_count += 1
            display_frame = frame.copy()
            
            # Procesar biometría cada 10 frames para no saturar la UI
            if self.process_frame_count % 10 == 0:
                self._process_biometrics(frame)

            # Convertir frame de OpenCV (BGR) a CTkImage (RGB)
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
            
            self.video_label.configure(image=img_ctk)
            self.video_label.image = img_ctk # Mantener referencia

        # Re-programar la siguiente actualización
        self.after(10, self._update_video_stream)

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
        """Búsqueda optimizada de identidad en el registro global."""
        best_dist = 0.0
        best_id = None
        best_name = None

        for uid, data in self.all_profiles.items():
            stored_vectors = data.get("vector")
            if stored_vectors is None:
                stored_vectors = data.get(b"vector")
            
            if stored_vectors is None: continue
            
            # Manejo de multi-vector (NumPy array o lista)
            search_list = stored_vectors if (isinstance(stored_vectors, np.ndarray) and stored_vectors.ndim > 1) else [stored_vectors]
            
            for vec in search_list:
                stored_emb = np.array(vec)
                # Similitud Coseno
                dist = np.dot(current_vec, stored_emb) / (np.linalg.norm(current_vec) * np.linalg.norm(stored_emb))
                
                if dist > 0.82 and dist > best_dist:
                    best_dist = dist
                    best_id = uid
                    best_name = data.get("nombre") or data.get(b"nombre", "Estudiante")

        return best_id, best_name

    def show_reports(self):
        """Placeholder para la funcionalidad de reportes."""
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="MÓDULO DE REPORTES\n(Próximamente)", 
                     font=ctk.CTkFont(size=20)).pack(expand=True)
    
    def _clear_view(self):
        """Detener cámara al cambiar de pestaña para liberar recursos."""
        if hasattr(self, 'cap'):
            self.cap.release()
        for widget in self.main_view.winfo_children():
            widget.destroy()
    
    """
    def _clear_view(self):
        for w in self.main_view.winfo_children(): w.destroy()
    """

if __name__ == "__main__":
    app = ReconApp()
    app.mainloop()