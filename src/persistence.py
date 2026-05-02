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
        with open(self.profiles_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self.sm.decrypt_data(encrypted_data)
        # raw=False asegura que las llaves vuelvan como strings y no como b'...'
        return msgpack.unpackb(decrypted_data, raw=False)

    def log_attendance(self, student_data):
        """
        Registra la asistencia en el log maestro CSV (Fuente de verdad)[cite: 1].
        student_data debe contener: {'codigo', 'nombre', 'confianza'}
        """
        file_exists = os.path.isfile(self.log_path)
        
        # Estructura de datos según requerimiento RF-06[cite: 1]
        row = {
            'timestamp': datetime.now().isoformat(),
            'codigo_estudiante': student_data['codigo'],
            'nombre': student_data['nombre'],
            'confianza_score': student_data['confianza'],
            'sesion_id': datetime.now().strftime('%Y%m%d')
        }
        
        df = pd.DataFrame([row])
        # Añade al final del archivo sin sobreescribir[cite: 1]
        df.to_csv(self.log_path, mode='a', index=False, header=not file_exists)