import os
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager

def main():
    security = SecurityManager()
    persistence = PersistenceManager(security)

    try:
        profiles = persistence.load_profiles()
    except Exception as e:
        print(f"Error al cargar el registro: {e}")
        return

    if not profiles:
        print("El registro está vacío.")
        return

    # Ciclo para permitir múltiples eliminaciones
    while True:
        print("\n--- REGISTROS ACTUALES ---")
        for k in profiles.keys():
            print(f"ID: [{k}] - Nombre: {profiles[k].get('nombre')}")
        
        code_to_delete = input("\nIngrese el CÓDIGO exacto a eliminar (o 'q' para salir): ").strip()
        
        if code_to_delete.lower() == 'q':
            break

        target_key = None
        # Normalización de búsqueda: comparamos todo como string limpio
        for key in profiles.keys():
            if str(key).strip() == code_to_delete:
                target_key = key
                break

        if target_key:
            nombre = profiles[target_key].get('nombre', 'Desconocido')
            confirm = input(f"¿Confirmar eliminación de '{nombre}'? (s/n): ")
            if confirm.lower() == 's':
                del profiles[target_key]
                persistence.save_profiles(profiles)
                print(f"✅ Registro {code_to_delete} eliminado exitosamente.")
            else:
                print("Operación cancelada.")
        else:
            print(f"❌ No se encontró el código: '{code_to_delete}'")
            print("Asegúrese de incluir caracteres especiales si los hay (ej. /).")

if __name__ == "__main__":
    main()