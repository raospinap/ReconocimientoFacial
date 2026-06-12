#.\gui_app.py

"""
MÓDULO: Interfaz Gráfica y Orquestación
DESCRIPCIÓN: Controlador principal basado en CustomTkinter. Gestiona el ciclo de vida 
de la aplicación, navegación entre vistas y comunicación asíncrona con el motor de IA 
mediante multiprocessing.Queue para evitar bloqueos en la UI.
CUMPLIMIENTO: RNF-01 (Interfaz Intuitiva), RNF-06 (Estandarización)
"""

import os
# Silencia logs de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
#tf.get_logger().setLevel('ERROR')

import customtkinter as ctk
import json
import datetime
import cv2
import time
import pandas as pd
import multiprocessing
import queue

from src.security_manager import SecurityManager
from src.persistence import PersistenceManager
from src.ai_engine import ai_camera_worker
from tkinter import messagebox, filedialog

# Silencia advertencias de Tkinter y otros
#import warnings
#warnings.filterwarnings("ignore")


# ==========================================================
# GUI ADMINISTRATIVA
# ==========================================================
class ReconApp(ctk.CTk):
    """Controlador principal de la aplicación. 
    Mantiene el estado de sesión, gestión de workers y cola de resultados en memoria."""
    def __init__(self):
        """	Inicializa la ventana principal, configura el estado de sesión, instancia gestores de seguridad/persistencia y prepara los objetos de comunicación entre procesos. RNF-01, RNF-06, RNF-07"""
        super().__init__()
        self.title("Sistema Biométrico UT v1.0")
        self.geometry("1100x700+0+0")
        
        self.update_idletasks()
        
        self.after(0, lambda: self.state('zoomed'))
        
        self.sidebar_buttons = []
        
        self.console_logs = []
        
        self.result_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.worker_process = None
        
        self.security = SecurityManager()
        self.persistence = PersistenceManager(self.security)
        
        self.active_session_file = "data/attendance/active_session.json"
        self.classes_path = "data/meta/classes.json"
        self.settings_path = "data/meta/settings.json"
        self.performance_modes = {
            "Auto": "auto",
            "Lento": "lento",
            "Normal": "normal",
            "Rápido": "rapido"
        }
        self.performance_mode_var = ctk.StringVar(value=self._load_settings().get("performance_mode_label", "Auto"))
        self.current_session = self._load_active_session()
        
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _load_settings(self):
        """Carga preferencias locales de la interfaz, incluyendo el modo de rendimiento."""
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_settings(self):
        """Persiste preferencias locales no sensibles."""
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        data = {"performance_mode_label": self.performance_mode_var.get()}
        with open(self.settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _get_performance_mode(self):
        """Devuelve el identificador interno del modo de rendimiento seleccionado."""
        label = self.performance_mode_var.get()
        return self.performance_modes.get(label, "auto")

    def _on_performance_mode_change(self, _value):
        """Actualiza preferencia de rendimiento y deja trazabilidad visual en la consola."""
        self._save_settings()
        self._log_to_console(f"Modo de rendimiento: {self.performance_mode_var.get()}")

    def _load_active_session(self):
        """Recupera y deserializa el estado de la última clase activa desde active_session.json para mantener continuidad tras reinicios.
        RF-10, RNF-01 """
        if os.path.exists(self.active_session_file):
            with open(self.active_session_file, "r", encoding='utf-8') as f:
                return json.load(f)
        return None

    def _setup_ui(self):
        """Configura el layout principal de la aplicación, distribuyendo la barra lateral y el área de visualización dinámica.
        RNF-01, RNF-06 """
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._create_sidebar()
        self.main_view = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.show_dashboard()

    def _create_sidebar(self):
        """Genera el menú de navegación lateral con botones para acceder a cada módulo funcional del sistema.
        RNF-01, RNF-06 """
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="MENÚ", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        
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

        btn_profiles = ctk.CTkButton(self.sidebar, text="Perfiles Biométricos", command=self.show_profiles)
        btn_profiles.pack(pady=5, padx=20); self.sidebar_buttons.append(btn_profiles)
        
        self.btn_audit = ctk.CTkButton(self.sidebar, text="Sistema", command=self.show_admin_audit)
        self.btn_audit.pack(pady=5, padx=20); self.sidebar_buttons.append(self.btn_audit)

        ctk.CTkLabel(self.sidebar, text="Rendimiento", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(20, 4))
        self.performance_mode_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=list(self.performance_modes.keys()),
            variable=self.performance_mode_var,
            command=self._on_performance_mode_change
        )
        self.performance_mode_menu.pack(pady=5, padx=20)
        self.sidebar_buttons.append(self.performance_mode_menu)
        
        ctk.CTkButton(self.sidebar, text="Salir", fg_color="#e74c3c", command=self._on_closing).pack(pady=5, padx=20)

    def show_dashboard(self):
        """Renderiza la vista principal con telemetría (rostros, peso de BD, integridad), estado de sesión y consola de eventos en tiempo real.
        RF-09, RF-10, RNF-01, RNF-04 """
        self._clear_view()
        
        # Obtener datos de telemetría
        telemetry = self.persistence.get_system_telemetry()
        
        # Título y Estado de Sesión
        ctk.CTkLabel(self.main_view, text="DASHBOARD", 
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(20, 10))
        
        status_color = "#2ecc71" if self.current_session else "#95a5a6"
        status_text = f"Clase Activa para Asistencia: {self.current_session['clase']}" if self.current_session else "No hay ninguna clase Activa"
        
        ctk.CTkLabel(self.main_view, text=status_text, text_color=status_color,
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)

        # --- CONTENEDOR DE TARJETAS (TELEMETRÍA) ---
        cards_frame = ctk.CTkFrame(self.main_view, fg_color="transparent")
        cards_frame.pack(fill="both", expand=True, padx=40, pady=20)
        cards_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Tarjeta 1: Base de Datos Biométrica
        self._create_telemetry_card(cards_frame, 0, "BIOMETRÍA", [
            (f"Rostros: {telemetry['biometria']['rostros_registrados']}", None),
            (f"Peso: {telemetry['biometria']['peso_base_datos']}", None)
        ])

        # Tarjeta 2: Almacenamiento y Logs
        self._create_telemetry_card(cards_frame, 1, "PERSISTENCIA", [
            (f"Log Asistencia: {telemetry['logs']['asistencia']}", None),
            (f"Log Auditoría: {telemetry['logs']['auditoria']}", None)
        ])

        # Tarjeta 3: Integridad de Sistema (ISO 25012)
        integ = telemetry['integridad']
        self._create_telemetry_card(cards_frame, 2, "ESTADO CRÍTICO", [
            ("Registros Binarios", "OK" if integ['registry_ok'] else "ERROR"),
            ("Estructura Clases", "OK" if integ['classes_ok'] else "ERROR"),
            ("Historial Logs", "OK" if integ['logs_ok'] else "ERROR")
        ])

        # --- CONSOLA DE EVENTOS EN TIEMPO REAL ---
        ctk.CTkLabel(self.main_view, text="REGISTRO DE EVENTOS", 
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        self.console_box = ctk.CTkTextbox(self.main_view, height=150, width=800, 
                                          font=("Consolas", 12))
        self.console_box.pack(pady=10, padx=40)
        
        # Cargar historial existente
        for log in self.console_logs:
            self.console_box.insert("end", log + "\n")
        self.console_box.see("end") # Auto-scroll al final
        self.console_box.configure(state="disabled") # Solo lectura

        # Botón de cierre de sesión si existe una activa
        if self.current_session:
            ctk.CTkButton(self.main_view, text="Desactivar Clase Actual", fg_color="#e67e22",
                         command=self._close_class_logic).pack(pady=20) 

    def show_classes(self):
        """Muestra la interfaz de administración de asignaturas: creación, activación, matriculación, renombrado y eliminación lógica.
        RF-01, RF-10, RNF-01, RNF-09 """
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="ADMINISTRACIÓN DE CLASES", font=ctk.CTkFont(size=22)).pack(pady=20)
        f = ctk.CTkFrame(self.main_view); f.pack(fill="x", padx=40, pady=10)
        self.new_class_entry = ctk.CTkEntry(f, placeholder_text="Nombre de la materia"); self.new_class_entry.pack(side="left", padx=20, expand=True, fill="x")
        ctk.CTkButton(f, text="Añadir", command=self._create_new_class_logic).pack(side="right", padx=20)
        s_frame = ctk.CTkFrame(self.main_view); s_frame.pack(fill="both", expand=True, padx=40, pady=20)
        if self.current_session:
            ctk.CTkButton(s_frame, text="Desactivar Clase Actual", fg_color="#e74c3c", command=self._close_class_logic).pack(pady=10)
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
        """Presenta el formulario de registro de estudiantes, incluyendo campos de datos y el aviso legal de consentimiento informado (Ley 1581). RF-01, RNF-08, RNF-09 """
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="REGISTRO DE ESTUDIANTE", font=ctk.CTkFont(size=22)).pack(pady=10)
        self.ent_name = ctk.CTkEntry(self.main_view, placeholder_text="Nombre Completo", width=400); self.ent_name.pack(pady=5)
        self.ent_code = ctk.CTkEntry(self.main_view, placeholder_text="Código", width=400); self.ent_code.pack(pady=5)
        ctk.CTkButton(self.main_view, text="ABRIR CÁMARA DE REGISTRO", command=self._launch_enroll_worker).pack(pady=20)
        ctk.CTkLabel(self.main_view, text="Aviso de Privacidad y Consentimiento Informado", 
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(15, 5))

        consent_text = (
            "De conformidad con la Ley 1581 de 2012, se informa que este sistema constituye un prototipo de "
            "proyecto académico de la carrera de Ingeniería de Sistemas y su finalidad es exclusivamente el "
            "control de asistencia estudiantil. Para tal fin, se requiere el tratamiento de datos biométricos "
            "sensibles (características faciales), los cuales son transformados de forma inmediata en un vector "
            "numérico cifrado, eliminando la fotografía original de la memoria de forma automática sin "
            "almacenarla en disco duro, garantizando así la privacidad del usuario. Como titular de la información, "
            "usted tiene derecho a conocer, actualizar o solicitar la supresión de sus datos en cualquier momento; "
            "asimismo, al tratarse de datos sensibles, se le informa que su autorización es facultativa, pero "
            "necesaria para el registro en este entorno de aprendizaje. Al continuar con el proceso y solicitar "
            "la apertura de la cámara, usted manifiesta su consentimiento previo, expreso e informado para que "
            "sus rasgos faciales sean procesados bajo los términos éticos y legales aquí descritos para fines "
            "estrictamente académicos."
        )

        privacy_box = ctk.CTkTextbox(
            self.main_view, 
            width=550,             
            height=300, 
            corner_radius=10, 
            border_width=1,
            wrap="word",           
            font=ctk.CTkFont(size=12),
            spacing1=10,           
            spacing2=5,            
            spacing3=10            
        )
        privacy_box.insert("0.0", consent_text)
        privacy_box.configure(state="disabled", padx=15, pady=15) 
        privacy_box.pack(pady=10, padx=40)
        
        
        self.enroll_status = ctk.CTkLabel(self.main_view, text="")
        self.enroll_status.pack(pady=10)

    def _launch_enroll_worker(self):
        """Valida datos, verifica consentimiento, confirma unicidad de código y lanza el proceso hijo de IA en modo registro.
        RF-01, RF-07, RNF-08, RNF-09 """
        n = self.ent_name.get().strip()
        c = self.ent_code.get().strip()
        if not n or not c: 
            messagebox.showwarning("Datos Incompletos", "Debe ingresar nombre y código.")
            return
        
        # --- VALIDACIÓN PREVIA DE ID 
        all_profiles = self.persistence.load_profiles()
        if str(c) in all_profiles:
            messagebox.showerror("ID Duplicado", f"El código {c} ya está registrado a nombre de: {all_profiles[str(c)]['nombre']}")
            return
        
        confirm = messagebox.askyesno(
            "Confirmación de Seguridad", 
            "¿Ha leído el aviso de privacidad y autoriza de manera voluntaria el tratamiento de su dato biométrico para este proyecto académico?"
        )
        if not confirm:
            messagebox.showinfo("Proceso Cancelado", "No se ha realizado el registro al no contar con la autorización del titular.")
            return
        self.enroll_status.configure(text="Iniciando cámara...", text_color="#3498db")
        self.stop_event.clear()
        self._set_sidebar_state("disabled")
        self.worker_process = multiprocessing.Process(
            target=ai_camera_worker, 
            args=("enroll", n, c, None, None, None, self.result_queue, self.stop_event, self._get_performance_mode())
        )
        self.worker_process.start()
        self._listen_for_result()

    def _listen_for_result(self):
        """Sondea asíncronamente la cola de resultados del worker para actualizar la UI, limpiar campos y liberar recursos.
        RNF-03, RNF-06"""
        try:
            while True:
                self._handle_worker_result(self.result_queue.get_nowait())
        except queue.Empty:
            if self.worker_process and self.worker_process.is_alive():
                self.after(100, self._listen_for_result)
            else:
                self._set_sidebar_state("normal")
        except Exception as e:
            self._log_to_console(f"ERROR: Lectura de worker falló: {e}")
            self._set_sidebar_state("normal")

    def _handle_worker_result(self, res):
        """Procesa mensajes enviados por el worker de cámara."""
        if isinstance(res, str) and res.startswith("REG:"):
            nombre = res.split(":")[1]
            msg = f"REGISTRO EXITOSO: {nombre}"
            if hasattr(self, 'lbl_last_reg'):
                self.lbl_last_reg.configure(text=msg, text_color="#2ecc71")
            self._log_to_console(msg)
        
        elif isinstance(res, dict):
            status = res.get("status")
            if status == "success":
                msg = f"Asistencia: {res['nombre']} ({res['codigo']})"
                color = "#2ecc71"
            elif status == "already_marked":
                msg = f"Ya registrado: {res.get('nombre', 'N/A')} ({res.get('codigo', 'N/A')})"
                color = "#f1c40f"
            elif status == "not_enrolled_in_class":
                msg = f"Enrolado no matriculado: {res.get('nombre', 'N/A')} ({res.get('codigo', 'N/A')})"
                color = "#e67e22"
            elif status == "unknown_face":
                msg = "Rostro no registrado"
                color = "#e74c3c"
            elif status == "mode_selected":
                metrics = []
                if res.get("fps") is not None:
                    metrics.append(f"{res['fps']} FPS")
                if res.get("inference_ms") is not None:
                    metrics.append(f"{res['inference_ms']} ms IA")
                suffix = f" ({', '.join(metrics)})" if metrics else ""
                msg = f"Modo activo: {res.get('mode', 'auto')} - {res.get('resolution', '')}{suffix}"
                color = "#3498db"
            elif status == "camera_error":
                msg = f"Error de cámara: {res.get('message', 'No disponible')}"
                color = "#e74c3c"
                self.stop_event.set()
                self._set_sidebar_state("normal")
            else:
                msg = str(res)
                color = "#95a5a6"

            if hasattr(self, 'lbl_last_reg'):
                self.lbl_last_reg.configure(text=msg, text_color=color)
            self._log_to_console(msg)

        elif res in ["SUCCESS", "DUPLICATE", "CANCELLED"]:
            if res == "SUCCESS":
                if hasattr(self, 'enroll_status'):
                    self.enroll_status.configure(text="PROCESO COMPLETADO", text_color="#2ecc71")
                    self._log_to_console("Sistema: Proceso finalizado correctamente.")
            elif res == "DUPLICATE":
                if hasattr(self, 'enroll_status'):
                    self.enroll_status.configure(text="REGISTRO DUPLICADO", text_color="#f1c40f")
                    self._log_to_console("Sistema: Registro duplicado.")
            elif res == "CANCELLED":
                if hasattr(self, 'enroll_status'):
                    self.enroll_status.configure(text="CÁMARA CERRADA", text_color="#f1c40f")
                self._log_to_console("Sistema: Cámara cerrada por el usuario.")
            
            if hasattr(self, 'ent_code'):
                self.ent_code.delete(0, 'end')
            if hasattr(self, 'ent_name'):
                self.ent_name.delete(0, 'end')
                
            self._set_sidebar_state("normal")
            self.stop_event.set()

    def show_live_attendance(self):
        """Verifica sesión activa, carga lista blanca de matriculados, valida hardware y lanza el worker en modo asistencia con feedback inmediato. RF-02, RF-05, RF-07, RNF-01, RNF-03"""
        self._clear_view()
        if not self.current_session: 
            messagebox.showwarning(
            "Sesión Inactiva", 
            "No hay una clase activa.\n\n"
            "1. Vaya a 'Gestión de Clases'\n"
            "2. Seleccione una materia y haga clic en el botón de la clase\n"
            "3. Intente tomar asistencia nuevamente."
            )
            self._log_to_console("ERROR: Intento de asistencia sin clase activa.")
            return
        
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
        ctk.CTkLabel(
            self.main_view,
            text=f"Modo de rendimiento: {self.performance_mode_var.get()}",
            text_color="#3498db"
        ).pack(pady=(0, 10))
        
        self.lbl_last_reg = ctk.CTkLabel(self.main_view, text="Esperando registros...", 
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_last_reg.pack(pady=20)

        self._set_sidebar_state("disabled")
        self.stop_event.clear()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error de Hardware", "La cámara no está disponible.\nVerifique que no esté siendo usada por otra aplicación.")
            self._log_to_console("ERROR: Cámara no disponible o ocupada.")
            self._set_sidebar_state("normal")
            self.show_dashboard()
            return
        cap.release() # Soltamos rápido para que el worker pueda tomarla

        self._log_to_console(f"Iniciando captura para: {self.current_session['clase']}")
        
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
                self.stop_event,
                self._get_performance_mode()
            )
        )
        self.worker_process.start()
        self._listen_for_result()

    def _set_sidebar_state(self, state="normal"):
        """Bloquea o habilita la navegación lateral durante procesos de cámara para evitar colisiones de estado y fugas de recursos.
        RNF-01, RNF-03"""
        for btn in self.sidebar_buttons:
            btn.configure(state=state)
    
    def _clear_view(self):
        """Destruye dinámicamente los widgets del área principal para permitir renderización de nuevas vistas sin reiniciar la app.
        RNF-01, RNF-06 """
        for w in self.main_view.winfo_children(): w.destroy()

    def _activate_class_logic(self, cid, name):
        """Genera token de sesión, persiste el estado en JSON y registra el evento en la auditoría administrativa.
        RF-10, RNF-05, RNF-06 """
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
        """Finaliza la sesión activa, elimina el archivo de estado, registra el cierre en auditoría y retorna al dashboard.
        RF-10, RNF-06 """
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
        """Valida entrada, genera ID automático (CLS-XXX), persiste la nueva clase en classes.json y audita la creación.
        RF-10, RNF-06, RNF-07 """
        name = self.new_class_entry.get().strip()
        if not name: return

        # Estructura inicial si el archivo no existe
        data = {"classes": {}}
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r", encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if "classes" not in data: data = {"classes": {}}
                except json.JSONDecodeError:
                    data = {"classes": {}}

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

    def show_reports(self):
        """Renderiza el módulo de reportes con filtros multinivel (materia, fecha, código) y botones de exportación.
        RF-08, RNF-01 """
        self._clear_view()
        ctk.CTkLabel(self.main_view, text="MÓDULO DE REPORTES", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        # --- BARRA DE FILTROS ---
        filter_frame = ctk.CTkFrame(self.main_view)
        filter_frame.pack(fill="x", padx=20, pady=5)

        # Filtro de Materia (Col 0-1)
        ctk.CTkLabel(filter_frame, text="Materia:").grid(row=0, column=0, padx=10, pady=10)
        class_options = ["Todas"]
        if os.path.exists(self.classes_path):
            with open(self.classes_path, "r", encoding='utf-8') as f:
                c_data = json.load(f).get("classes", {})
                class_options += [f"{cid} | {info['name']}" for cid, info in c_data.items()]

        self.report_class_var = ctk.StringVar(value="Todas")
        self.combo_class = ctk.CTkComboBox(filter_frame, values=class_options, variable=self.report_class_var, width=200)
        self.combo_class.grid(row=0, column=1, padx=10, pady=10)

        # Filtro de Fecha (Col 2-3)
        ctk.CTkLabel(filter_frame, text="Fecha:").grid(row=0, column=2, padx=10, pady=10)
        self.date_filter_entry = ctk.CTkEntry(filter_frame, placeholder_text="AAAA-MM-DD", width=120)
        self.date_filter_entry.grid(row=0, column=3, padx=10, pady=10)
        self.date_filter_entry.insert(0, datetime.datetime.now().strftime('%Y-%m-%d'))

        # NUEVO: Filtro de Código (Col 4-5)
        ctk.CTkLabel(filter_frame, text="Código:").grid(row=0, column=4, padx=10, pady=10)
        self.code_filter_entry = ctk.CTkEntry(filter_frame, placeholder_text="Ej: 0854...", width=120)
        self.code_filter_entry.grid(row=0, column=5, padx=10, pady=10)

        # Botones (Col 6-7)
        ctk.CTkButton(filter_frame, text="Filtrar", width=100, command=self._generate_report_logic).grid(row=0, column=6, padx=10, pady=10)
        ctk.CTkButton(filter_frame, text="Exportar", fg_color="#27ae60", hover_color="#1e8449", width=100, command=self._export_report_to_excel).grid(row=0, column=7, padx=10, pady=10)

        # --- CABECERA Y TABLA --- (Se mantiene igual que tu versión anterior)
        header_frame = ctk.CTkFrame(self.main_view, fg_color="#2c3e50", height=30)
        header_frame.pack(fill="x", padx=20, pady=(10, 0))
        headers = ["Fecha / Hora", "Código", "Nombre", "Materia", "Conf."]
        widths = [180, 120, 200, 180, 60]
        for i, text in enumerate(headers):
            lbl = ctk.CTkLabel(header_frame, text=text, text_color="white", width=widths[i], font=ctk.CTkFont(weight="bold"))
            lbl.pack(side="left", padx=5)

        self.report_scroll = ctk.CTkScrollableFrame(self.main_view, fg_color="transparent")
        self.report_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self._generate_report_logic()
        
    def _generate_report_logic(self):
        """Aplica filtros en memoria sobre el log de asistencia, verifica integridad criptográfica (HMAC) por fila y renderiza tabla con alertas si hay manipulación. RF-08, RNF-05, RNF-06 """
        # Limpiar tabla anterior
        for widget in self.report_scroll.winfo_children():
            widget.destroy()

        df = self.persistence.get_attendance_data()
        if df.empty:
            ctk.CTkLabel(self.report_scroll, text="No hay registros de asistencia disponibles.").pack(pady=20)
            return

        # Aplicar Filtro de Fecha
        date_query = self.date_filter_entry.get().strip()
        if date_query:
            df = df[df['timestamp'].str.contains(date_query)]

        # Aplicar Filtro de Materia
        class_query = self.report_class_var.get()
        if class_query != "Todas":
            cid = class_query.split(" | ")[0]
            df = df[df['clase_id'] == cid]

        # Aplicar Filtro de Código de Estudiante
        code_query = self.code_filter_entry.get().strip()
        if code_query:
            df = df[df['codigo_estudiante'].astype(str).str.contains(code_query)]

        # Renderizar filas
        if df.empty:
            ctk.CTkLabel(self.report_scroll, text="No se encontraron registros para los filtros seleccionados.").pack(pady=20)
            return

        # Ordenar por el más reciente
        df = df.sort_values(by='timestamp', ascending=False)
        widths = [180, 120, 200, 180, 60]

        for _, row in df.iterrows():
            # [NUEVO] Determinamos el color según la integridad
            is_valid = row.get('integrity_ok', True)
            row_bg = "transparent" if is_valid else "#532222" # Fondo rojizo si hay alteración
            text_color = "white" if is_valid else "#e74c3c"
            
            f_row = ctk.CTkFrame(self.report_scroll, fg_color=row_bg)
            f_row.pack(fill="x", pady=1)
            
            ts = row['timestamp'].split('.')[0].replace('T', ' ')
            # Si no es válido, añadimos un prefijo de advertencia
            prefix = "" if is_valid else "⚠️ MODIFICADO: "
            
            data_cols = [ts, row['codigo_estudiante'], prefix + row['nombre'], row['clase_nombre'], f"{row['confianza_score']:.2f}"]
            
            for i, val in enumerate(data_cols):
                ctk.CTkLabel(f_row, text=val, width=widths[i], anchor="w", text_color=text_color).pack(side="left", padx=5)
            
            # Si el registro está alterado, registramos en el log administrativo de forma automática
            if not is_valid:
                self.persistence.log_admin_action(
                    "INTEGRITY_VIOLATION", 
                    row['codigo_estudiante'], 
                    f"Intento de manipulación detectado en registro del {ts}"
                )
            
            # Separador sutil
            ctk.CTkFrame(self.report_scroll, height=1, fg_color="#34495e").pack(fill="x", padx=5)

    def _on_closing(self):
        """Gestiona el cierre seguro de la aplicación, terminando workers activos y liberando memoria/cámara antes de destruir la ventana.
        RNF-03, RNF-06, RNF-07 """
        self.stop_event.set()
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate()
        self.destroy()
        
    def _delete_class_logic(self, cid):
        """Ejecuta eliminación lógica (Soft Delete) cambiando estado a hidden, conserva historial y audita la acción.
        RF-10, RNF-06, RNF-09"""
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
        """Muestra interfaz de matriculación específica por clase, listando estudiantes inscritos y permitiendo altas/bajas.
        RF-01, RF-10, RNF-01 """
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
        """Valida existencia biométrica del código, lo añade a la lista de la clase y registra la acción en auditoría.
        RF-01, RF-10, RNF-06 """
        code = self.enroll_code_entry.get().strip()
        if not code: return
        
        # Verificar si el estudiante existe en la base biométrica
        profiles = self.persistence.load_profiles()
        if code not in profiles:
            messagebox.showwarning("Estudiante no registrado", f"El código {code} no existe en el sistema biométrico.")
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
        """Retira estudiante de la clase activa, actualiza classes.json y genera log administrativo.
        RF-01, RF-10, RNF-06 """
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
        """Permite renombrar asignatura, actualiza metadatos y sincroniza el nombre en la sesión activa si está en curso.
        RF-10, RNF-06 """
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

    def _export_report_to_excel(self):
        """Exporta el reporte filtrado a formato .xlsx manteniendo integridad de IDs y registrando la acción en auditoría.
        RF-08, RNF-06, RNF-07"""
        df = self.persistence.get_attendance_data()
        
        if df.empty:
            messagebox.showwarning("Exportar", "No hay datos para exportar.")
            return

        # 1. Aplicar los mismos filtros de la vista
        date_query = self.date_filter_entry.get().strip()
        if date_query:
            df = df[df['timestamp'].str.contains(date_query)]

        class_query = self.report_class_var.get()
        if class_query != "Todas":
            cid = class_query.split(" | ")[0]
            df = df[df['clase_id'] == cid]

        code_query = self.code_filter_entry.get().strip()
        if code_query:
            df = df[df['codigo_estudiante'].astype(str).str.contains(code_query)]
        
        # 2. Solicitar ubicación de guardado
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"Reporte_Asistencia_{datetime.datetime.now().strftime('%Y%m%d')}"
        )

        if filename:
            try:
                # Ordenar cronológicamente antes de exportar
                df = df.sort_values(by='timestamp', ascending=True)
                
                # Exportación limpia (ISO 25012: mantenemos IDs como texto)
                df.to_excel(filename, index=False, engine='openpyxl')
                
                messagebox.showinfo("Éxito", f"Reporte exportado correctamente en:\n{filename}")
                
                # Log administrativo de la exportación
                self.persistence.log_admin_action(
                    event_type="EXPORT_REPORT",
                    target_id="SYSTEM",
                    description=f"Exportación de reporte Excel: {os.path.basename(filename)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el archivo: {e}")

    def show_profiles(self):
        """Lista todos los perfiles biométricos registrados con opción de eliminación segura mediante confirmación y auditoría.
        RF-01, RF-10, RNF-09"""
        self._clear_view()
        
        ctk.CTkLabel(self.main_view, text="GESTIÓN DE PERFILES BIOMÉTRICOS", 
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)

        # Contenedor de tabla
        header_frame = ctk.CTkFrame(self.main_view, fg_color="#2c3e50", height=30)
        header_frame.pack(fill="x", padx=40, pady=(10, 0))
        
        headers = ["Código Estudiante", "Nombre Completo", "Acciones"]
        widths = [150, 300, 100]
        
        for i, text in enumerate(headers):
            ctk.CTkLabel(header_frame, text=text, text_color="white", 
                         width=widths[i], font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)

        scroll_frame = ctk.CTkScrollableFrame(self.main_view, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=40, pady=(0, 20))

        profiles = self.persistence.load_profiles()
        
        if not profiles:
            ctk.CTkLabel(scroll_frame, text="No hay perfiles registrados.").pack(pady=20)
            return

        for code, info in profiles.items():
            row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row, text=code, width=150, anchor="w").pack(side="left", padx=10)
            ctk.CTkLabel(row, text=info.get('nombre', 'N/A'), width=300, anchor="w").pack(side="left", padx=10)
            
            # Botón de eliminación con confirmación
            ctk.CTkButton(row, text="Eliminar", fg_color="#c0392b", hover_color="#962d22", width=80,
                         command=lambda c=code, n=info.get('nombre'): self._delete_profile_logic(c, n)).pack(side="right", padx=10)
            
            ctk.CTkFrame(scroll_frame, height=1, fg_color="#34495e").pack(fill="x")

    def _delete_profile_logic(self, code, name):
        """Ejecuta eliminación física del vector cifrado, actualiza encrypted_registry.bin y genera registro forense.
        RF-01, RNF-05, RNF-09"""
        msg = f"¿Está seguro de eliminar el perfil de {name} ({code})?\nEsta acción no se puede deshacer."
        if not messagebox.askyesno("Confirmar Eliminación", msg):
            return

        if self.persistence.delete_profile(code):
            # Auditoría ISO 25012
            self.persistence.log_admin_action(
                event_type="DELETE_PROFILE",
                target_id=code,
                description=f"Perfil biométrico de {name} eliminado permanentemente."
            )
            messagebox.showinfo("Éxito", f"Perfil de {name} eliminado.")
            self.show_profiles()
        else:
            messagebox.showerror("Error", "No se pudo eliminar el perfil.")

    def _create_telemetry_card(self, parent, col, title, items):
        """Helper UI que renderiza tarjetas informativas en el dashboard con indicadores de estado y colores dinámicos.
        RNF-01, RNF-06 """
        card = ctk.CTkFrame(parent, border_width=2, border_color="#34495e")
        card.grid(row=0, column=col, padx=10, sticky="nsew")
        
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=14, weight="bold"), 
                     text_color="#3498db").pack(pady=10)
        
        for text, status in items:
            color = "white"
            if status == "OK": color = "#2ecc71"
            elif status == "ERROR": color = "#e74c3c"
            
            label_text = f"{text} {f'[{status}]' if status else ''}"
            ctk.CTkLabel(card, text=label_text, text_color=color).pack(pady=2)

    def _log_to_console(self, message):
        """Agrega mensajes con timestamp a la consola visual en tiempo real, manteniendo un buffer de 30 entradas para optimizar RAM.
        RNF-01, RNF-04, RNF-06"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.console_logs.append(formatted_msg)
        # Mantener solo los últimos 30 mensajes para no saturar RAM
        if len(self.console_logs) > 30: self.console_logs.pop(0)

    def show_admin_audit(self):
        """Descifra y muestra los últimos 30 eventos administrativos en formato compacto, con opciones de respaldo/restauración de llave maestra. RNF-05, RNF-06, RNF-09"""
        # Recuperar logs descifrados
        logs = self.persistence.get_admin_audit_logs()
        self._clear_view()
        self.update_idletasks()
        
        ctk.CTkLabel(self.main_view, text="REGISTRO DE AUDITORÍA", 
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        f_keys = ctk.CTkFrame(self.main_view, fg_color="transparent")
        f_keys.pack(pady=5)

        ctk.CTkButton(
            f_keys, 
            text="Exportar Llave",
            fg_color="#e67e22", 
            hover_color="#d35400",
            command=self._handle_key_backup,
            width=120,
            height=25
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            f_keys, 
            text="Restaurar Llave",
            fg_color="#27ae60", 
            hover_color="#219150",
            command=self._handle_key_restore,
            width=120,
            height=25
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(f_keys, text="Exportar Auditoría", fg_color="#3498db", hover_color="#2980b9",
                  command=self._export_audit_to_excel, width=160, height=25).pack(side="left", padx=5)
        
        # Frame compacto para la tabla
        scroll_frame = ctk.CTkScrollableFrame(self.main_view, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.update()
        if not logs:
            ctk.CTkLabel(scroll_frame, text="No hay eventos registrados", 
                         font=ctk.CTkFont(size=11)).pack(pady=10)
            return

        # Encabezados
        header_frame = ctk.CTkFrame(scroll_frame, fg_color="#2c3e50", height=25)
        header_frame.pack(fill="x", pady=(0, 2))
        
        headers = ["Fecha/Hora", "Evento", "Objeto", "Descripción"]
        widths = [130, 140, 100, 450]
        
        for i, text in enumerate(headers):
            ctk.CTkLabel(header_frame, text=text, text_color="white", 
                         width=widths[i], font=ctk.CTkFont(size=10, weight="bold"),
                         anchor="w").pack(side="left", padx=3)

        # Mostramos los 30 últimos logs,  del más reciente al más antiguo (Reverse)
        display_logs = logs[-30:]
        for log in reversed(display_logs):
            f_row = ctk.CTkFrame(scroll_frame, border_width=0, 
                                 fg_color="transparent")
            f_row.pack(fill="x", pady=0)
            
            # Formateo compacto en una sola línea
            ts = log['timestamp'].split('.')[0].replace('T', ' ')
            event_type = log['event_type']
            target_id = log['target_id']
            description = log['description'][:80] + "..." if len(log['description']) > 80 else log['description']
            
            # Columna 1: Timestamp
            ctk.CTkLabel(f_row, text=ts, width=widths[0], 
                         font=ctk.CTkFont(size=10), anchor="w",
                         text_color="#bdc3c7").pack(side="left", padx=3)
            
            # Columna 2: Evento (con color según tipo)
            event_color = "#3498db" if "CREATE" in event_type or "ENROLL" in event_type else \
                          "#e74c3c" if "DELETE" in event_type else \
                          "#f39c12" if "SESSION" in event_type else "#95a5a6"
            
            ctk.CTkLabel(f_row, text=event_type, width=widths[1], 
                         font=ctk.CTkFont(size=10, weight="bold"), anchor="w",
                         text_color=event_color).pack(side="left", padx=3)
            
            # Columna 3: Objeto
            ctk.CTkLabel(f_row, text=target_id, width=widths[2], 
                         font=ctk.CTkFont(size=10), anchor="w",
                         text_color="#ecf0f1").pack(side="left", padx=3)
            
            # Columna 4: Descripción
            ctk.CTkLabel(f_row, text=description, width=widths[3], 
                         font=ctk.CTkFont(size=10), anchor="w",
                         text_color="#95a5a6").pack(side="left", padx=3)

    def _export_audit_to_excel(self):
        """ Permite decargar todos los registros de auditoría administrativa en formato .xlsx. (altas, bajas, gestión de llaves).
        RF-08, RNF-06, RNF-07"""
        logs = self.persistence.get_admin_audit_logs()
        if not logs:
            messagebox.showwarning("Exportar", "No hay registros de auditoría para exportar.")
            return

        # Estandarización: Siempre trabajamos con DataFrame
        df = pd.DataFrame(logs)
        df = df.sort_values(by='timestamp', ascending=False)
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"Auditoria_Sistema_{datetime.datetime.now().strftime('%Y%m%d')}"
        )
        if filename:
            try:
                # Un solo punto de exportación
                df.to_excel(filename, index=False, engine='openpyxl')
                messagebox.showinfo("Éxito", f"Auditoría completa exportada a:\n{filename}")
                self.persistence.log_admin_action("EXPORT_AUDIT", "SYSTEM", f"Exportación completa del log de auditoría")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar el archivo: {e}")

    def _handle_key_backup(self):
        """Exporta la llave simétrica cifrada en un archivo .vault protegido por contraseña derivada con PBKDF2 (100k iteraciones).
        RNF-05, RNF-06"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".vault",
            filetypes=[("Security Vault", "*.vault")],
            title="Guardar respaldo de seguridad",
            initialfile="backup_recon_facial.vault"
        )
        if not filepath: return

        dialog = ctk.CTkInputDialog(text="Cree una contraseña para proteger este respaldo:", title="Cifrar Backup")
        pw = dialog.get_input()
        if not pw: return

        try:
            if self.security.export_key_backup(pw, filepath):
                messagebox.showinfo("Éxito", "Respaldo generado correctamente.\nGuarde este archivo y la contraseña en un lugar seguro.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar la llave: {str(e)}")

    def _handle_key_restore(self):
        """Restaura la llave maestra desde un respaldo .vault, verifica contraseña y reinicia la aplicación para aplicar cambios de seguridad. RNF-05, RNF-06"""
        filepath = filedialog.askopenfilename(
            filetypes=[("Security Vault", "*.vault")],
            title="Seleccione el archivo de respaldo"
        )
        if not filepath: return

        dialog = ctk.CTkInputDialog(text="Ingrese la contraseña de recuperación:", title="Descifrar Backup")
        pw = dialog.get_input()
        if not pw: return

        confirm = messagebox.askyesno("Restauración Crítica", 
            "Al restaurar una llave, se sobrescribirá la actual. "
            "Si la llave no coincide con la base de datos actual, perderá el acceso a los perfiles registrados.\n\n"
            "¿Desea continuar?")
        
        if not confirm: return

        try:
            if self.security.import_key_backup(pw, filepath):
                messagebox.showinfo("Éxito", "Llave restaurada correctamente.\nEl sistema se reiniciará para aplicar los cambios.")
                self._on_closing() # Cierra la app para forzar recarga con la nueva llave
            else:
                messagebox.showerror("Error", "Contraseña incorrecta o archivo de respaldo inválido.")
        except Exception as e:
            messagebox.showerror("Error Crítico", f"Fallo en la restauración: {str(e)}")

if __name__ == "__main__": 
    multiprocessing.freeze_support() 
    app = ReconApp(); app.mainloop()
   
