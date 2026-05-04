# --- PUENTE DE COMPATIBILIDAD PARA DEEPFACE ---
import sys
try:
    import keras
    # Forzamos el mapeo del namespace que DeepFace busca
    sys.modules['tensorflow.keras'] = keras
except ImportError:
    pass
# ----------------------------------------------
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.security_manager import SecurityManager
from src.persistence import PersistenceManager
from src.enrollment import EnrollmentManager
import numpy as np

# ... el resto del código se mantiene exactamente igual ...

def main():
    # 1. Inicialización de la arquitectura técnica
    sm = SecurityManager()
    pm = PersistenceManager(sm)
    

    print("--- Sistema de Enrolamiento Local ---")
    nombre = input("Ingresa el nombre completo del estudiante: ")
    codigo = input("Ingresa el código estudiantil: ")

    # Cargar el registro actual desde el archivo  cifrado
    profiles = pm.load_profiles()

    # Comprobamos si el código ya existe en el diccionario antes de abrir la cámara
    if codigo in profiles:
        print(f"\n[ERROR] El código '{codigo}' ya se encuentra registrado en el sistema.")
        return # Finaliza el script inmediatamente

    em = EnrollmentManager()
    
    # 2. Captura y Extracción Biométrica (RF-01, RF-04)
    vector = em.enroll_student(nombre, codigo, profiles)

    if vector is not None:
        # 3. Carga del registro actual
        profiles = pm.load_profiles()
        
        # 4. Actualización del diccionario de perfiles
        profiles[codigo] = {
            "nombre": nombre,
            "vector": np.array(vector, dtype=np.float32),
            "fecha_registro": np.datetime64('now').astype(str)
        }

        # 5. Persistencia Cifrada (AES-256 + Msgpack)
        try:
            pm.save_profiles(profiles)
            print(f"\n[ÉXITO] Usuario {nombre} registrado correctamente.")
            print(f"Datos protegidos en: data/profiles/encrypted_registry.bin")
        except Exception as e:
            print(f"\n[ERROR] No se pudo guardar el registro: {e}")
    else:
        print("\n[CANCELADO] No se generó el vector biométrico.")

if __name__ == "__main__":
    main()