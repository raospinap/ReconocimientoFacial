# src/persistence.py

import msgpack
import msgpack_numpy as m
import os
import pandas as pd
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
        
        # Asegurar la creación de la estructura de directorios
        os.makedirs(os.path.dirname(self.profiles_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

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
        """Registra asistencia evitando duplicados para el mismo estudiante, clase y fecha."""
        import pandas as pd
        from datetime import datetime
        
        file_exists = os.path.exists(self.log_path)
        current_date = datetime.now().strftime('%Y-%m-%d')
        class_name = student_data.get('clase', 'Sin Clase')
        student_id = str(student_data['codigo'])

        if file_exists:
            try:
                # Forzamos la lectura del código como string para evitar errores de comparación
                df_existing = pd.read_csv(self.log_path, dtype={'codigo_estudiante': str})
                
                # Validación de duplicado: mismo código, misma clase, misma fecha
                duplicates = df_existing[
                    (df_existing['codigo_estudiante'].astype(str) == student_id) & 
                    (df_existing['clase_nombre'] == class_name) &
                    (df_existing['timestamp'].str.contains(current_date))
                ]
                if not duplicates.empty:
                    return False # Ya existe registro hoy
            except: pass

        row = {
            'timestamp': datetime.now().isoformat(),
            'codigo_estudiante': student_id,
            'nombre': student_data['nombre'],
            'confianza_score': student_data['confianza'],
            'clase_nombre': class_name, 
            'sesion_id': student_data.get('session_id', current_date.replace("-",""))
        }
        
        df = pd.DataFrame([row])
        df.to_csv(self.log_path, mode='a', index=False, header=not file_exists)
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