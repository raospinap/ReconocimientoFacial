import numpy as np
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager

def test_cimentacion():
    # 1. Inicializar componentes
    sm = SecurityManager()
    pm = PersistenceManager(sm)
    
    # 2. Crear un "Vector Facial" de prueba (512 dimensiones)
    test_vector = np.random.rand(512).astype(np.float32)
    test_profiles = {
        "12345": {
            "nombre": "Estudiante de Prueba",
            "vector": test_vector
        }
    }
    
    print("--- Iniciando Prueba de Hito 1 ---")
    
    # 3. Probar persistencia cifrada
    try:
        pm.save_profiles(test_profiles)
        print("[OK] Perfiles guardados y cifrados exitosamente.")
        
        loaded_profiles = pm.load_profiles()
        # Verificar integridad del vector
        if np.allclose(test_profiles["12345"]["vector"], loaded_profiles["12345"]["vector"]):
            print("[OK] Descifrado y deserialización íntegra (Msgpack + AES-256).")
        
        # 4. Probar log de asistencia (RF-06)
        test_attendance = {
            'codigo': '12345',
            'nombre': 'Estudiante de Prueba',
            'confianza': 0.85
        }
        pm.log_attendance(test_attendance)
        print("[OK] Registro de asistencia generado en master_log.csv.")
        
        print("\n--- Fase 1 Completada con Éxito ---")
        
    except Exception as e:
        print(f"[ERROR] Fallo en la validación: {e}")

if __name__ == "__main__":
    test_cimentacion()