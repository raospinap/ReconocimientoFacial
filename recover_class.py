import json
import os
import datetime
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager

def main():
    # Inicialización del entorno técnico
    sm = SecurityManager()
    pm = PersistenceManager(sm)
    classes_path = 'data/meta/classes.json'

    if not os.path.exists(classes_path):
        print(f"❌ Error: No se encontró el archivo de configuración en {classes_path}")
        return

    try:
        with open(classes_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            classes_dict = data.get("classes", {})
    except Exception as e:
        print(f"❌ Error al leer las materias: {e}")
        return

    # Filtrar solo materias en estado 'hidden'
    hidden_classes = {cid: info for cid, info in classes_dict.items() if info.get("status") == "hidden"}

    if not hidden_classes:
        print("\n--- No hay materias ocultas (hidden) para recuperar ---")
        return

    while True:
        print("\n" + "="*50)
        print("SISTEMA DE RECUPERACIÓN DE MATERIAS (CLI)")
        print("="*50)
        print(f"{'ID':<10} | {'NOMBRE DE LA MATERIA':<30} | {'ELIMINADA EL'}")
        print("-"*70)

        for cid, info in hidden_classes.items():
            deactivated = info.get("deactivated_at", "N/A")
            # Recortar la fecha para legibilidad
            short_date = deactivated.split(".")[0].replace("T", " ") if deactivated != "N/A" else "N/A"
            print(f"{cid:<10} | {info['name'][:30]:<30} | {short_date}")

        print("-"*70)
        target_id = input("\nIngrese el ID exacto a recuperar (ej: CLS-001) o 'q' para salir: ").strip().upper()

        if target_id.lower() == 'q':
            break

        if target_id in hidden_classes:
            class_name = hidden_classes[target_id]['name']
            confirm = input(f"¿Restaurar '{class_name}' ({target_id}) a estado activo? (s/n): ").lower()
            
            if confirm == 's':
                # Cambio de estado lógico
                data["classes"][target_id]["status"] = "active"
                # Limpiar metadato de desactivación si existe
                if "deactivated_at" in data["classes"][target_id]:
                    del data["classes"][target_id]["deactivated_at"]

                # Persistencia en disco
                try:
                    with open(classes_path, "w", encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                    
                    # REGISTRO EN LOG ADMINISTRATIVO (ISO 25012)
                    pm.log_admin_action(
                        event_type="RECOVER_CLASS",
                        target_id=target_id,
                        description=f"Materia '{class_name}' restaurada a estado activo via CLI"
                    )
                    
                    print(f"\n✅ EXITO: La materia {target_id} vuelve a estar activa.")
                    # Actualizar lista local para el ciclo actual
                    del hidden_classes[target_id]
                    if not hidden_classes:
                        print("No quedan más materias para recuperar.")
                        break
                except Exception as e:
                    print(f"❌ Error al guardar los cambios: {e}")
            else:
                print("Operación cancelada.")
        else:
            print(f"❌ ID '{target_id}' no válido o no está en la lista de ocultos.")

if __name__ == "__main__":
    main()