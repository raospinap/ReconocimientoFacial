# src/persistence.py

import msgpack
import msgpack_numpy as m
import os
import pandas as pd
import hmac
import hashlib
from datetime import datetime
import numpy as np

# Soporte para vectores de NumPy (Esencial para DeepFace)
m.patch()

class PersistenceManager:
    """
    Gestiona la persistencia con capas de seguridad HMAC y Cifrado AES.
    """
    
    def __init__(self, security_manager):
        self.sm = security_manager
        # Definición explícita de rutas
        self.profiles_path = 'data/profiles/encrypted_registry.bin'
        self.log_path = 'data/attendance/master_log.csv'
        self.audit_path = 'data/meta/admin_audit.bin'
        
        # Asegurar directorios
        for path in [self.profiles_path, self.log_path, self.audit_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    # --- PERFILES BIOMÉTRICOS ---
    def load_profiles(self):
        if not os.path.exists(self.profiles_path): return {}
        try:
            with open(self.profiles_path, 'rb') as f:
                encrypted_data = f.read()
            decrypted_data = self.sm.decrypt_data(encrypted_data)
            # Solo raw=False, sin use_bin_type aquí
            return msgpack.unpackb(decrypted_data, raw=False)
        except Exception as e:
            print(f"[ERROR PERSISTENCE] No se pudo cargar biometría: {e}")
            return {}

    def save_profiles(self, profiles_dict):
        # Convertir todos los vectores de NumPy a listas antes de serializar
        serializable_profiles = {}
        for uid, data in profiles_dict.items():
            # Convertimos el array de NumPy a lista para evitar errores de serialización
            vector_lista = data["vector"].tolist() if isinstance(data["vector"], np.ndarray) else data["vector"]
            serializable_profiles[str(uid)] = {
                "nombre": data["nombre"],
                "vector": vector_lista,
                "fecha_registro": data["fecha_registro"]
            }
        
        try:
            #packed_data = msgpack.packb(profiles_dict, use_bin_type=True)
            packed_data = msgpack.packb(serializable_profiles)
            encrypted_data = self.sm.encrypt_data(packed_data)
            with open(self.profiles_path, 'wb') as f:
                f.write(encrypted_data)
            return True    
        except Exception as e:
            print(f"[ERROR PERSISTENCE] No se pudo guardar biometría: {e}")
            return False

    def delete_profile(self, student_code):
        profiles = self.load_profiles()
        if student_code in profiles:
            del profiles[student_code]
            self.save_profiles(profiles)
            return True
        return False

    # --- ASISTENCIA (INTEGRIDAD) ---
    def _generate_signature(self, row_dict):
        """Sella la fila para detectar manipulaciones externas."""
        payload = f"{row_dict['timestamp']}|{row_dict['codigo_estudiante']}|{row_dict['clase_id']}"
        return hmac.new(self.sm.key, payload.encode(), hashlib.sha256).hexdigest()

    def get_attendance_data(self):
        """Carga el reporte. Si el archivo está vacío o es viejo, lo maneja sin crashear."""
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.log_path, dtype={'codigo_estudiante': str, 'clase_id': str})
            
            # Retrocompatibilidad: si no tiene firma, es válido (datos antiguos)
            if 'signature' not in df.columns:
                df['integrity_ok'] = True
                return df

            def verify(row):
                try:
                    if pd.isna(row['signature']): return False
                    expected = self._generate_signature(row)
                    return hmac.compare_digest(str(row['signature']), expected)
                except: return False

            df['integrity_ok'] = df.apply(verify, axis=1)
            return df
        except Exception as e:
            print(f"[ERROR PERSISTENCE] Error en reporte: {e}")
            return pd.DataFrame()

    def log_attendance(self, student_data):
        # Verificación rápida solo si el archivo ya existe y no está vacío
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            try:
                df = pd.read_csv(self.log_path, usecols=['codigo_estudiante', 'clase_id', 'timestamp'])
                current_date = datetime.now().strftime('%Y-%m-%d')
                mask = (df['codigo_estudiante'] == str(student_data['codigo'])) & \
                       (df['clase_id'] == str(student_data['clase_id'])) & \
                       (df['timestamp'].str.contains(current_date))
                if mask.any(): return False  # Ya existe hoy en esta clase
            except Exception: pass # Si falla la lectura, continúa (seguridad ante corrupciones menores)

        row = {
            'timestamp': datetime.now().isoformat(),
            'codigo_estudiante': str(student_data['codigo']),
            'nombre': student_data['nombre'],
            'confianza_score': student_data['confianza'],
            'clase_id': str(student_data['clase_id']),
            'clase_nombre': student_data.get('clase_nombre', 'N/A'),
            'sesion_id': student_data.get('session_id', datetime.now().strftime('%Y-%m-%d'))
        }
        row['signature'] = self._generate_signature(row)
        
        df = pd.DataFrame([row])
        df.to_csv(self.log_path, mode='a', index=False, header=not os.path.exists(self.log_path))
        return True

    # --- AUDITORÍA (CONFIDENCIALIDAD) ---
    def log_admin_action(self, event_type, target_id, description):
        audit_data = self.get_admin_audit_logs()
        new_entry = {
            'timestamp': datetime.now().isoformat(),
            'admin_user': 'Admin_Local',
            'event_type': str(event_type),
            'target_id': str(target_id),
            'description': str(description)
        }
        audit_data.append(new_entry)
        try:
            packed = msgpack.packb(audit_data, use_bin_type=True)
            encrypted = self.sm.encrypt_data(packed)
            with open(self.audit_path, 'wb') as f:
                f.write(encrypted)
        except Exception as e:
            print(f"[ERROR PERSISTENCE] Fallo en log administrativo: {e}")

    def get_admin_audit_logs(self):
        if not os.path.exists(self.audit_path): return []
        try:
            with open(self.audit_path, 'rb') as f:
                enc = f.read()
            dec = self.sm.decrypt_data(enc)
            # Solo raw=False, sin use_bin_type aquí
            return msgpack.unpackb(dec, raw=False)
        except: return []

    # --- TELEMETRÍA ---
    def get_system_telemetry(self):
        def get_size(path):
            return f"{os.path.getsize(path)/1024:.1f} KB" if os.path.exists(path) else "0 B"

        profiles = self.load_profiles()
        return {
            "biometria": {"rostros_registrados": len(profiles), "peso_base_datos": get_size(self.profiles_path)},
            "logs": {"asistencia": get_size(self.log_path), "auditoria": get_size(self.audit_path)},
            "integridad": {
                "registry_ok": os.path.exists(self.profiles_path),
                "classes_ok": os.path.exists('data/meta/classes.json'),
                "logs_ok": os.path.exists(self.log_path)
            }
        }
        
    def get_already_marked_today(self, active_class_id):
        """Retorna un conjunto de códigos de estudiantes que ya asistieron hoy."""
        df = self.get_attendance_data()
        if df.empty: return set()
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        try:
            # Filtramos por clase y fecha actual para evitar registros duplicados
            mask = (df['clase_id'] == str(active_class_id)) & (df['timestamp'].str.contains(current_date))
            return set(df.loc[mask, 'codigo_estudiante'].astype(str).unique())
        except:
            return set()