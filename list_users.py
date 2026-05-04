from src.security_manager import SecurityManager
from src.persistence import PersistenceManager

def list_registered_students():
    # Inicialización de gestores para descifrar la data
    sm = SecurityManager()
    pm = PersistenceManager(sm)
    
    # Carga de perfiles[cite: 4]
    profiles = pm.load_profiles()
    
    if not profiles:
        print("\n[INFO] El registro está vacío.")
        return

    print(f"\n--- ESTUDIANTES REGISTRADOS ({len(profiles)}) ---")
    print(f"{'CÓDIGO':<15} | {'NOMBRE COMPLETO':<30} | {'FECHA REGISTRO'}")
    print("-" * 70)

    for codigo, data in profiles.items():
        # Recuperación segura de nombre sin operador 'or'
        nombre = data.get("nombre")
        if nombre is None:
            nombre = data.get(b"nombre", "Sin Nombre")
            
        fecha = data.get("fecha_registro")
        if fecha is None:
            fecha = data.get(b"fecha_registro", "N/A")

        print(f"{str(codigo):<15} | {str(nombre):<30} | {str(fecha)}")
    print("-" * 70)

if __name__ == "__main__":
    list_registered_students()