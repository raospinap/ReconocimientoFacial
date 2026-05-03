# --- PUENTE DE COMPATIBILIDAD PARA DEEPFACE ---
import sys
try:
    import keras
    # Forzamos el mapeo del namespace que DeepFace busca
    sys.modules['tensorflow.keras'] = keras
except ImportError:
    pass
# ----------------------------------------------

from src.security_manager import SecurityManager
from src.persistence import PersistenceManager
from src.enrollment import EnrollmentManager
import numpy as np

# ... el resto del código se mantiene exactamente igual ...

def main():
    # 1. Inicialización de la arquitectura técnica
    sm = SecurityManager()
    pm = PersistenceManager(sm)
    em = EnrollmentManager()

    print("--- Sistema de Enrolamiento Local ---")
    nombre = input("Ingresa el nombre completo del estudiante: ")
    codigo = input("Ingresa el código estudiantil: ")

    # 2. Captura y Extracción Biométrica (RF-01, RF-04)
    # Nota: La primera vez descargará el modelo ArcFace (~145MB)
    vector = em.enroll_student(nombre, codigo)

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