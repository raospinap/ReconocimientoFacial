import customtkinter as ctk
import json
import unicodedata
import os
import datetime
import cv2
import numpy as np
import time
import multiprocessing # Aislamiento total de la CPU para evitar bloqueos
from PIL import Image
from deepface import DeepFace 
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager

# Optimizaciones de entorno
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _clean_text(text):
    """Elimina tildes y eñes para compatibilidad con OpenCV."""
    if not text: return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).replace('ñ', 'n').replace('Ñ', 'N')

# ==========================================================
# PROCESO INDEPENDIENTE: MOTOR DE IA (SIDE-CAR)
# ==========================================================
def ai_camera_worker(mode, name, code, active_class_id, allowed_students, result_queue):
    """
    Este proceso corre en un núcleo de CPU distinto al de la GUI.
    Es 100% inmune a los bloqueos de Tkinter.
    """
    
    security = SecurityManager()
    persistence = PersistenceManager(security)
    cap = cv2.VideoCapture(0)
    
    enroll_vectors = []
    enroll_stage = 0
    stages_text = ["Mire al frente", "Gire a la izquierda", "Gire a la derecha"]
    win_title = "Registro - ESPACIO para capturar / ESC para salir" if mode == "enroll" else "Toma de Asistencia - ESC para salir"
    last_capture_time = 0
    is_processing = False
    
    # Carga de perfiles para validación
    all_profiles = persistence.load_profiles()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. Detección visual (Óvalo)
        face_ready = False
        try:
            faces = DeepFace.extract_faces(frame, detector_backend="mediapipe", enforce_detection=True)
            if faces and faces[0]["facial_area"]['w'] > (w * 0.25):
                face_ready = True
        except: pass

        color = (0, 255, 0) if face_ready else (0, 0, 255)
        cv2.ellipse(display_frame, (w//2, h//2), (int(w*0.22), int(h*0.35)), 0, 0, 360, color, 2)

        # 2. Lógica de Enrolamiento
        if mode == "enroll":
            if enroll_stage < 3:
                txt = f"PASO {enroll_stage+1}/3: {stages_text[enroll_stage]}"
                cv2.putText(display_frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_frame, "Presione ESPACIO para capturar", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and face_ready and not is_processing:
                    is_processing = True
                    try:
                        # 1. Extracción del vector actual
                        res = DeepFace.represent(frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=True)
                        emb = np.array(res[0]["embedding"])
                        
                        # 2. Validación de duplicados (Paso 1: Frente)
                        duplicate = False
                        if enroll_stage == 0:
                            for uid, data in all_profiles.items():
                                stored = data.get("vector")
                                if stored is None:
                                    stored = data.get(b"vector")
                                
                                if stored is not None:
                                    # Aseguramos que la comparación sea contra arreglos de NumPy
                                    search_list = np.array(stored)
                                    if search_list.ndim == 1:
                                        search_list = [search_list]
                                    
                                    for v in search_list:
                                        v_arr = np.array(v)
                                        # Similitud Coseno
                                        dist = np.dot(emb, v_arr) / (np.linalg.norm(emb) * np.linalg.norm(v_arr))
                                        if dist > 0.65: 
                                            duplicate = True
                                            break
                                if duplicate:
                                    #result_queue.put("DUPLICATE")
                                    break
                        
                        # 3. Respuesta al hallazgo de duplicado
                        if duplicate:
                            # Mostrar advertencia visual persistente antes de cerrar
                            for _ in range(120):
                                temp_frame = display_frame.copy()
                                cv2.putText(temp_frame, "ERROR: ROSTRO YA REGISTRADO", (w//10, h//2), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv2.imshow(win_title, temp_frame)
                                cv2.waitKey(10)
                            result_queue.put("DUPLICATE")
                            break # Cierra la ventana de la cámara
                        else:
                            enroll_vectors.append(emb)
                            enroll_stage += 1
                    except Exception as e:
                        print(f"Error en captura: {e}")
                    is_processing = False
            else:
                # Guardar Registro
                all_profiles[code] = {
                    "nombre": name,
                    "vector": np.array(enroll_vectors, dtype=np.float32),
                    "fecha_registro": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S")
                }
                persistence.save_profiles(all_profiles)
                for _ in range(120):
                    temp_f = display_frame.copy()
                    cv2.putText(temp_f, "REGISTRO EXITOSO", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow(win_title, temp_f)
                    cv2.waitKey(10)
                result_queue.put("SUCCESS")
                break

        # 3. Lógica de Asistencia
        elif mode == "attendance":
            cv2.putText(display_frame, f"ASISTENCIA: {active_class_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            current_t = int(time.time())
            if current_t > last_capture_time and face_ready:
                last_capture_time = current_t
                try:
                    res = DeepFace.represent(frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=False)
                    emb = np.array(res[0]["embedding"])
                    best_dist, match_id, match_name = 0.0, None, None
                    
                    for uid, data in all_profiles.items():
                        stored = data.get("vector")
                        if stored is None: stored = data.get(b"vector")
                        if stored is not None:
                            search_list = np.array(stored)
                            if search_list.ndim == 1: search_list = [search_list]
                            for v in search_list:
                                v_arr = np.array(v)
                                dist = np.dot(emb, v_arr) / (np.linalg.norm(emb) * np.linalg.norm(v_arr))
                                if dist > 0.82 and dist > best_dist:
                                    best_dist, match_id, match_name = dist, uid, data.get("nombre", "Estudiante")
                    
                    if match_id:
                        is_allowed = str(match_id) in allowed_students
                        # Llamada a la función global (sin self)
                        clean_name = _clean_text(match_name)
                        txt = f"{clean_name}: OK" if is_allowed else f"{clean_name}: NO MATRICULADO"
                        color_status = (0, 255, 0) if is_allowed else (0, 0, 255)
                        
                        if is_allowed:
                            # Se envían los parámetros como un único diccionario para evitar el error de argumentos
                            persistence.log_attendance({
                                "id": match_id, 
                                "name": match_name, 
                                "class": active_class_id
                            })
                        
                        # Persistencia visual para que el mensaje sea legible
                        for _ in range(60):
                            temp_f = display_frame.copy()
                            cv2.putText(temp_f, txt, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)
                            cv2.imshow(win_title, temp_f)
                            if (cv2.waitKey(1) & 0xFF) == 27: break
                except Exception as e:
                    print(f"Error en asistencia: {e}")

        cv2.imshow(win_title, display_frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            result_queue.put("CANCELLED")
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================================
# CLASE PRINCIPAL: GESTIÓN ADMINISTRATIVA (GUI)
# ==========================================================
class ReconApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Sistema de Asistencia Biométrica UT")
        self.geometry("1100x700")
        ctk.set_appearance_mode("Dark")
        
        # Canal de comunicación con el proceso de cámara
        self.result_queue = multiprocessing.Queue()
        
        self.security = SecurityManager()
        self.persistence = PersistenceManager(self.security)
        
        self.active_session_file = "data/attendance/active_session.json"
        self.classes_path = "data/meta/classes.json"
        self.current_session = self._load_active_session()
        self.enroll_status = None   
        self._setup_ui()

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
        ctk.CTkButton(self.sidebar, text="Dashboard", command=self.show_dashboard).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Gestión de Clases", command=self.show_classes).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Registrar Estudiante", command=self.show_enrollment).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Tomar Asistencia", command=self.show_live_attendance).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Reportes", command=self.show_reports).pack(pady=5, padx=20)
        ctk.CTkButton(self.sidebar, text="Salir", fg_color="#e74c3c", hover_color="#c0392b", command=self.quit).pack(pady=5, padx=20)

    def show_dashboard(self):
        self._clear_view()
        status = f"Sesión Activa: {self.current_session['clase']}" if self.current_session else "No hay clase activa"
        color = "#2ecc71" if self.current_session else "#e74c3c"
        ctk.CTkLabel(self.main_view, text=f"PANEL DE CONTROL\n\n{status}", font=ctk.CTkFont(size=24), text_color=color).pack(expand=True)

    def show_classes(self):
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="ADMINISTRACIÓN DE ASIGNATURAS", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=20)
        f = ctk.CTkFrame(self.main_view); f.pack(fill="x", padx=40, pady=10)
        self.new_class_entry = ctk.CTkEntry(f, placeholder_text="Nombre de la materia"); self.new_class_entry.pack(side="left", padx=20, expand=True, fill="x")
        ctk.CTkButton(f, text="Añadir", command=self._create_new_class_logic).pack(side="right", padx=20)
        s_frame = ctk.CTkFrame(self.main_view); s_frame.pack(fill="both", expand=True, padx=40, pady=20)
        if self.current_session:
            ctk.CTkLabel(s_frame, text=f"CLASE ACTUAL: {self.current_session['clase']}").pack(pady=10)
            ctk.CTkButton(s_frame, text="FINALIZAR SESIÓN", fg_color="#e74c3c", command=self._close_class_logic).pack(pady=10)
        else:
            if os.path.exists(self.classes_path):
                with open(self.classes_path, "r", encoding='utf-8') as f: data = json.load(f)
                for c in data.keys(): ctk.CTkButton(s_frame, text=f"Activar: {c}", command=lambda n=c: self._activate_class_logic(n)).pack(pady=5)

    def show_enrollment(self):
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="REGISTRO DE ESTUDIANTE", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=10)
        f = ctk.CTkFrame(self.main_view); f.pack(fill="x", padx=40, pady=10)
        self.ent_name = ctk.CTkEntry(f, placeholder_text="Nombre Completo"); self.ent_name.pack(side="left", padx=10, expand=True, fill="x")
        self.ent_code = ctk.CTkEntry(f, placeholder_text="Código"); self.ent_code.pack(side="left", padx=10, expand=True, fill="x")
        ctk.CTkButton(self.main_view, text="ABRIR CÁMARA DE REGISTRO", command=self._launch_enroll_worker).pack(pady=20)
        
        # AJUSTE TÉCNICO: Inicialización del widget de estatus para alarmas
        self.enroll_status = ctk.CTkLabel(self.main_view, text="", font=ctk.CTkFont(size=14))
        self.enroll_status.pack(pady=10)

    def _launch_enroll_worker(self):
        """Valida que el código no exista antes de abrir la cámara."""
        n, c = self.ent_name.get().strip(), self.ent_code.get().strip()
        if n and c:
            # 1. Cargar perfiles para verificación preventiva
            profiles = self.persistence.load_profiles()
            
            # 2. Búsqueda agnóstica de código (evita duplicados y sobrescritura)
            if any(str(k) == str(c) for k in profiles.keys()):
                # Mostrar alerta en la interfaz principal
                self.enroll_status.configure(
                    text=f"ERROR: EL CÓDIGO {c} YA ESTÁ EN USO", 
                    text_color="#e74c3c"
                )
                return

            # 3. Reiniciar estado visual y lanzar proceso de biometría
            self.enroll_status.configure(text="Iniciando cámara...", text_color="#3498db")
            multiprocessing.Process(target=ai_camera_worker, args=("enroll", n, c, None, None, self.result_queue)).start()
            
            # Comenzar a escuchar el resultado del proceso hijo
            self._listen_for_result()

    def _listen_for_result(self):
        """Escucha de forma asíncrona el canal de comunicación con la cámara."""
        try:
            res = self.result_queue.get_nowait()
            if res == "SUCCESS":
                self.enroll_status.configure(text="✅ REGISTRO COMPLETADO EXITOSAMENTE", text_color="#2ecc71")
                self.ent_name.delete(0, 'end')
                self.ent_code.delete(0, 'end')
            elif res == "DUPLICATE":
                self.enroll_status.configure(text="❌ ERROR: EL ROSTRO YA EXISTE EN EL SISTEMA", text_color="#e74c3c")
            elif res == "CANCELLED":
                msg = "⚠️ SESIÓN DE ASISTENCIA FINALIZADA" if not self.ent_name.get() else "⚠️ PROCESO CANCELADO"
                self.enroll_status.configure(text=msg, text_color="#f1c40f")

                
        except:
            # Si no hay mensaje aún, volver a consultar en 500ms
            self.after(500, self._listen_for_result)

    def show_live_attendance(self):
        self._clear_view()
        if not self.current_session:
            ctk.CTkLabel(self.main_view, text="ERROR: NO HAY CLASE ACTIVA", text_color="#e74c3c").pack(expand=True)
            return
        with open(self.classes_path, "r", encoding='utf-8') as f: data = json.load(f)
        allowed = [str(s) for s in data.get(self.current_session['clase'], [])]
        ctk.CTkButton(self.main_view, text="ABRIR CÁMARA DE ASISTENCIA", 
                     command=lambda: multiprocessing.Process(target=ai_camera_worker, 
                     args=("attendance", None, None, self.current_session['session_id'], allowed, self.result_queue)).start()).pack(expand=True)
        self._listen_for_result()
        
    def _clear_view(self):
        for w in self.main_view.winfo_children(): w.destroy()

    def _activate_class_logic(self, n):
        self.current_session = {"clase": n, "session_id": f"{n}_{int(time.time())}"}
        with open(self.active_session_file, "w", encoding='utf-8') as f: json.dump(self.current_session, f)
        self.show_dashboard()

    def _close_class_logic(self):
        if os.path.exists(self.active_session_file): os.remove(self.active_session_file)
        self.current_session = None; self.show_dashboard()

    def _create_new_class_logic(self):
        n = self.new_class_entry.get().strip()
        if not n: return
        data = {}
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r", encoding='utf-8') as f: data = json.load(f)
        data[n] = []; 
        with open(self.classes_path, "w", encoding='utf-8') as f: json.dump(data, f)
        self.show_classes()

    def show_reports(self): self._clear_view(); ctk.CTkLabel(self.main_view, text="REPORTES").pack(pady=20)

if __name__ == "__main__": 
    multiprocessing.freeze_support() 
    app = ReconApp(); app.mainloop()