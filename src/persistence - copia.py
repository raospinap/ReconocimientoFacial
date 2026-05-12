# src/persistence.py

import msgpack
import msgpack_numpy as m
import os
import pandas as pd
import hmac
import hashlib
from datetime import datetime

# Habilitar soporte para arrays de NumPy en msgpack para optimizar vectores faciales
m.patch()

class PersistenceManager:
    """
    Gestiona la lectura y escritura de perfiles biométricos y logs de asistencia.
    Asegura que los datos sensibles nunca se guarden en texto plano.
    """
    
    def __init__(self, security_manager):
        self.sm = security_manager
        self.profiles_path = 'data/profiles/encrypted_registry.bin'
        self.log_path = 'data/attendance/master_log.csv'
        self.audit_path = 'data/meta/admin_audit.bin'
        
        # Asegurar la creación de la estructura de directorios
        os.makedirs(os.path.dirname(self.profiles_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)

    def save_profiles(self, profiles_dict):
        """Serializa con soporte para strings de Python 3."""
        # Usamos use_bin_type=True para diferenciar bytes de strings
        packed_data = msgpack.packb(profiles_dict, use_bin_type=True)
        encrypted_data = self.sm.encrypt_data(packed_data)
        with open(self.profiles_path, 'wb') as f:
            f.write(encrypted_data)

    def load_profiles(self):
        """Descifra y desempaqueta tratando bytes como strings (raw=False)."""
        if not os.path.exists(self.profiles_path):
            return {}
        try:
            with open(self.profiles_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Usamos self.sm que es como definiste el security_manager en el __init__
            decrypted_data = self.sm.decrypt_data(encrypted_data)
            
            # raw=False es CRÍTICO para evitar el KeyError: 'embedding'
            return msgpack.unpackb(decrypted_data, raw=False)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar perfiles: {e}")
            return {}
            
    def log_attendance(self, student_data):
        """Registra asistencia con sellado criptográfico por fila."""
        import pandas as pd
        from datetime import datetime
        
        file_exists = os.path.exists(self.log_path)
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        clase_id = student_data.get('clase_id', 'N/A')
        student_id = str(student_data['codigo'])

        # 1. Validación de duplicados (Igual que antes)
        if file_exists:
            try:
                df_existing = pd.read_csv(self.log_path, dtype={'codigo_estudiante': str, 'clase_id': str})
                mask = ((df_existing['codigo_estudiante'] == student_id) & 
                        (df_existing['clase_id'] == clase_id) &
                        (df_existing['timestamp'].str.contains(current_date)))
                if not df_existing[mask].empty: return False
            except: pass

        # 2. Preparación del registro
        row = {
            'timestamp': datetime.now().isoformat(),
            'codigo_estudiante': student_id,
            'nombre': student_data['nombre'],
            'confianza_score': student_data['confianza'],
            'clase_id': clase_id,
            'clase_nombre': student_data.get('clase_nombre', 'N/A'),
            'sesion_id': student_data.get('session_id', current_date.replace("-",""))
        }
        
        # 3. Generación de la firma [NUEVO]
        row['signature'] = self._generate_signature(row)
        
        df = pd.DataFrame([row])
        df.to_  (self.log_path, mode='a', index=False, header=not file_exists)
        return True
      
    def load_registry(self):
        """
        Carga y descifra la base de datos de perfiles biométricos 
        utilizando el bus de datos msgpack definido en la arquitectura.
        """
        if not os.path.exists(self.profiles_path):
            return {}

        try:
            # 1. Lectura del binario
            with open(self.profiles_path, "rb") as f:
                encrypted_data = f.read()

            # 2. Descifrado usando el alias correcto (self.sm)
            decrypted_data = self.sm.decrypt_data(encrypted_data)

            # 3. Deserialización usando msgpack para consistencia técnica
            # raw=False asegura que las llaves se recuperen como strings
            registry = msgpack.unpackb(decrypted_data, raw=False)
            return registry

        except Exception as e:
            print(f"[ERROR] Fallo en la carga del registro biométrico: {e}")
            return {}
    
    def get_already_marked_today(self, class_name):
        """Devuelve un set con los códigos que ya marcaron asistencia hoy en esta clase."""
        if not os.path.exists(self.log_path):
            return set()
        try:
            df = pd.read_csv(self.log_path, dtype={'codigo_estudiante': str})
            current_date = datetime.now().strftime('%Y-%m-%d')
            # Filtramos los registros de hoy para la clase activa
            mask = (df['clase_nombre'] == class_name) & (df['timestamp'].str.contains(current_date))
            return set(df.loc[mask, 'codigo_estudiante'].unique())
        except:
            return set()
            
    def log_admin_action(self, event_type, target_id, description):
        """Registra una acción cifrada en el log de auditoría binario (.bin)."""
        import msgpack
        from datetime import datetime
        
        # 1. Cargar historial existente (descifrado)
        audit_data = []
        if os.path.exists(self.audit_path):
            try:
                with open(self.audit_path, 'rb') as f:
                    encrypted_content = f.read()
                decrypted_content = self.sm.decrypt_data(encrypted_content)
                # raw=False asegura que los textos se carguen como strings de Python
                audit_data = msgpack.unpackb(decrypted_content, raw=False, use_bin_type=True)
            except Exception as e:
                print(f"Error cargando auditoría: {e}")
                audit_data = []

        # 2. Crear y añadir la nueva entrada
        new_entry = {
            'timestamp': datetime.now().isoformat(),
            'admin_user': 'Admin_Local',
            'event_type': str(event_type),
            'target_id': str(target_id),
            'description': str(description)
        }
        audit_data.append(new_entry)

        # 3. Serializar, Cifrar y Guardar
        packed_data = msgpack.packb(audit_data, use_bin_type=True)
        encrypted_data = self.sm.encrypt_data(packed_data)
        with open(self.audit_path, 'wb') as f:
            f.write(encrypted_data)

    def get_admin_audit_logs(self):
        """Recupera, descifra y devuelve los logs para la interfaz gráfica."""
        import msgpack
        if not os.path.exists(self.audit_path):
            return []
        try:
            with open(self.audit_path, 'rb') as f:
                encrypted_content = f.read()
            decrypted_content = self.sm.decrypt_data(encrypted_content)
            # raw=False es vital para que gui_app.py pueda leer los textos
            return msgpack.unpackb(decrypted_content, raw=False, use_bin_type=True)
        except Exception as e:
            print(f"Error leyendo auditoría: {e}")
            return []


    def get_attendance_data(self):
        """Carga el log y verifica la integridad de cada registro."""
        if not os.path.exists(self.log_path): return pd.DataFrame()
        try:
            df = pd.read_csv(self.log_path, dtype={'codigo_estudiante': str, 'clase_id': str, 'signature': str})
            
            # Verificación de integridad en tiempo real [NUEVO]
            def verify(row):
                if pd.isna(row['signature']): return False
                # Re-calculamos la firma con los datos actuales
                expected = self._generate_signature(row)
                return hmac.compare_digest(row['signature'], expected)

            df['integrity_ok'] = df.apply(verify, axis=1)
            return df
        except:
            return pd.DataFrame()
            
    def delete_profile(self, student_code):
        """Elimina un perfil biométrico del registro cifrado."""
        profiles = self.load_profiles()
        if student_code in profiles:
            del profiles[student_code]
            self.save_profiles(profiles)
            return True
        return False
    
    def get_system_telemetry(self):
        """Genera un reporte técnico del estado de los archivos y registros."""
        import os
        
        def get_size(path):
            if os.path.exists(path):
                size_bytes = os.path.getsize(path)
                for unit in ['B', 'KB', 'MB']:
                    if size_bytes < 1024: return f"{size_bytes:.1f} {unit}"
                    size_bytes /= 1024
            return "0 B"

        profiles = self.load_profiles()
        
        # Telemetría de archivos
        stats = {
            "biometria": {
                "rostros_registrados": len(profiles),
                "peso_base_datos": get_size(self.profiles_path)
            },
            "logs": {
                "asistencia": get_size(self.log_path),
                "auditoria": get_size(self.audit_path)
            },
            "integridad": {
                "registry_ok": os.path.exists(self.profiles_path),
                "classes_ok": os.path.exists('data/meta/classes.json'),
                "logs_ok": os.path.exists(self.log_path)
            }
        }
        return stats
        
    def _generate_signature(self, row_dict):
        """Genera una firma HMAC-SHA256 para una fila de datos."""
        # Concatenamos los campos críticos para la firma
        payload = f"{row_dict['timestamp']}|{row_dict['codigo_estudiante']}|{row_dict['clase_id']}"
        # Utilizamos la clave del SecurityManager para el sellado
        key = self.sm.key 
        return hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()

