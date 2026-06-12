#.\src\ai_engine.py

"""
MÓDULO: Motor de Inferencia (Side-Car Process)
DESCRIPCIÓN: Pipeline de Visión Artificial aislado de la GUI.
Ejecuta captura, detección y reconocimiento para evitar bloqueos.
CUMPLIMIENTO: RF-02, RF-03, RF-04, RF-05
"""
import cv2
import numpy as np
import time
import datetime
import unicodedata
import logging
import json
import os
from deepface import DeepFace
from src.security_manager import SecurityManager
from src.persistence import PersistenceManager

logger = logging.getLogger(__name__)

PERFORMANCE_CONFIG_PATH = "data/meta/performance_config.json"

DEFAULT_CONFIG = {
    "brightness_min": 75,
    "brightness_max": 230,
    "sharpness_min": 45,
    "face_size_ratio": 0.11,
    "display_resolution": (640, 480),
    "enroll_dup_threshold": 0.70,
    "attendance_match_threshold": 0.75,
    "cooldown_sec": 6,
    "debug_mode": False
}

DEFAULT_PERFORMANCE_MODES = {
    "lento": {
        "resolution": (320, 240),
        "detect_every": 8,
        "recognize_every": 12,
        "max_faces": 1,
        "match_enrolled_first": True,
        "global_fallback": True
    },
    "normal": {
        "resolution": (424, 240),
        "detect_every": 5,
        "recognize_every": 8,
        "max_faces": 2,
        "match_enrolled_first": True,
        "global_fallback": True
    },
    "rapido": {
        "resolution": (640, 480),
        "detect_every": 3,
        "recognize_every": 5,
        "max_faces": 4,
        "match_enrolled_first": True,
        "global_fallback": True
    }
}


def _normalize_resolution(value, fallback):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            width = int(value[0])
            height = int(value[1])
            if width > 0 and height > 0:
                return (width, height)
        except (TypeError, ValueError):
            pass
    return fallback


def _merge_performance_config(raw):
    config = DEFAULT_CONFIG.copy()
    modes = {name: mode.copy() for name, mode in DEFAULT_PERFORMANCE_MODES.items()}

    if isinstance(raw, dict):
        raw_config = raw.get("config", {})
        if isinstance(raw_config, dict):
            for key, default_value in DEFAULT_CONFIG.items():
                if key not in raw_config:
                    continue
                value = raw_config[key]
                if key == "display_resolution":
                    config[key] = _normalize_resolution(value, default_value)
                elif isinstance(default_value, bool):
                    config[key] = bool(value)
                elif isinstance(default_value, int):
                    try:
                        config[key] = int(value)
                    except (TypeError, ValueError):
                        pass
                elif isinstance(default_value, float):
                    try:
                        config[key] = float(value)
                    except (TypeError, ValueError):
                        pass

        raw_modes = raw.get("modes", {})
        if isinstance(raw_modes, dict):
            for mode_name, defaults in DEFAULT_PERFORMANCE_MODES.items():
                incoming = raw_modes.get(mode_name, {})
                if not isinstance(incoming, dict):
                    continue
                for key, default_value in defaults.items():
                    if key not in incoming:
                        continue
                    value = incoming[key]
                    if key == "resolution":
                        modes[mode_name][key] = _normalize_resolution(value, default_value)
                    elif isinstance(default_value, bool):
                        modes[mode_name][key] = bool(value)
                    elif isinstance(default_value, int):
                        try:
                            modes[mode_name][key] = max(1, int(value))
                        except (TypeError, ValueError):
                            pass

    return config, modes


def _serializable_performance_config(config, modes):
    data = {"config": {}, "modes": {}}
    for key, value in config.items():
        data["config"][key] = list(value) if key == "display_resolution" else value
    for mode_name, mode_config in modes.items():
        data["modes"][mode_name] = {}
        for key, value in mode_config.items():
            data["modes"][mode_name][key] = list(value) if key == "resolution" else value
    return data


def _load_performance_config():
    raw = {}
    if os.path.exists(PERFORMANCE_CONFIG_PATH):
        try:
            with open(PERFORMANCE_CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("No se pudo leer performance_config.json: %s", exc)

    config, modes = _merge_performance_config(raw)

    if not os.path.exists(PERFORMANCE_CONFIG_PATH):
        try:
            os.makedirs(os.path.dirname(PERFORMANCE_CONFIG_PATH), exist_ok=True)
            with open(PERFORMANCE_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(_serializable_performance_config(config, modes), f, indent=4)
        except OSError as exc:
            logger.debug("No se pudo crear performance_config.json: %s", exc)

    return config, modes


CONFIG, PERFORMANCE_MODES = _load_performance_config()


def _clean_text(text):
    """Elimina tildes y eñes para compatibilidad con OpenCV."""
    if not text:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).replace('ñ', 'n').replace('Ñ', 'N')


def _apply_camera_resolution(cap, resolution):
    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)


def _benchmark_camera(cap):
    """Mide FPS de lectura de cámara y latencia real aproximada de inferencia."""
    _apply_camera_resolution(cap, PERFORMANCE_MODES["normal"]["resolution"])
    start = time.time()
    frames = 0
    sample_frame = None
    while time.time() - start < 2:
        ret, frame = cap.read()
        if ret:
            frames += 1
            sample_frame = frame
    elapsed = max(time.time() - start, 0.001)
    fps = frames / elapsed
    inference_ms = None

    if sample_frame is not None:
        try:
            inference_frame = cv2.resize(sample_frame, PERFORMANCE_MODES["lento"]["resolution"], interpolation=cv2.INTER_AREA)
            infer_start = time.time()
            DeepFace.represent(
                inference_frame,
                model_name="ArcFace",
                detector_backend="mediapipe",
                enforce_detection=False
            )
            inference_ms = (time.time() - infer_start) * 1000
        except Exception as exc:
            logger.debug("Benchmark de inferencia omitido: %s", exc)

    if fps < 8 or (inference_ms is not None and inference_ms > 900):
        return "lento", {"fps": fps, "inference_ms": inference_ms}
    if fps < 15 or (inference_ms is not None and inference_ms > 450):
        return "normal", {"fps": fps, "inference_ms": inference_ms}
    return "rapido", {"fps": fps, "inference_ms": inference_ms}


def _resolve_performance_mode(cap, requested_mode):
    requested = (requested_mode or "auto").lower()
    if requested == "auto":
        selected, metrics = _benchmark_camera(cap)
        # Algunos drivers aplican crop/zoom al cambiar resolución con el stream abierto.
        # Reabrir la cámara hace que Auto use el mismo estado limpio que un modo manual.
        cap.release()
        cap = cv2.VideoCapture(0)
    else:
        selected, metrics = requested if requested in PERFORMANCE_MODES else "normal", {"fps": None, "inference_ms": None}
    cfg = PERFORMANCE_MODES[selected].copy()
    _apply_camera_resolution(cap, cfg["resolution"])
    return cap, selected, cfg, metrics


def _normalize_vector(vector):
    arr = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return None
    return arr / norm


def _build_profile_index(all_profiles, allowed_students):
    """Precalcula matrices normalizadas para comparar con NumPy."""
    allowed = {str(code) for code in (allowed_students or [])}
    global_rows, global_ids, global_names = [], [], []
    enrolled_rows, enrolled_ids, enrolled_names = [], [], []

    for uid, data in all_profiles.items():
        stored = data.get("vector")
        if stored is None:
            stored = data.get(b"vector")
        if stored is None:
            continue

        vectors = np.asarray(stored, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = np.asarray([vectors], dtype=np.float32)

        for vector in vectors:
            normalized = _normalize_vector(vector)
            if normalized is None:
                continue

            uid_str = str(uid)
            name = data.get("nombre", "N/A")
            global_rows.append(normalized)
            global_ids.append(uid_str)
            global_names.append(name)

            if uid_str in allowed:
                enrolled_rows.append(normalized)
                enrolled_ids.append(uid_str)
                enrolled_names.append(name)

    def pack(rows, ids, names):
        matrix = np.vstack(rows).astype(np.float32) if rows else np.empty((0, 0), dtype=np.float32)
        return {"matrix": matrix, "ids": ids, "names": names}

    return {
        "global": pack(global_rows, global_ids, global_names),
        "enrolled": pack(enrolled_rows, enrolled_ids, enrolled_names)
    }


def _best_match(embedding, profile_index, threshold):
    emb = _normalize_vector(embedding)
    matrix = profile_index["matrix"]
    if emb is None or matrix.size == 0:
        return None

    scores = matrix @ emb
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < threshold:
        return None

    return {
        "codigo": profile_index["ids"][best_idx],
        "nombre": profile_index["names"][best_idx],
        "score": best_score
    }


def _sort_relevant_faces(faces, max_faces):
    def area(face_data):
        facial_area = face_data.get("facial_area", {})
        return facial_area.get("w", 0) * facial_area.get("h", 0)

    return sorted(faces or [], key=area, reverse=True)[:max_faces]


def _display_metrics(frame):
    """Genera una copia escalada para UI y factores de conversión desde el frame procesado."""
    h, w = frame.shape[:2]
    display_w, display_h = CONFIG["display_resolution"]
    if w >= display_w and h >= display_h:
        return frame.copy(), 1.0, 1.0

    display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    return display_frame, display_w / w, display_h / h


def _scale_box(x, y, w_box, h_box, scale_x, scale_y, max_w, max_h):
    x1 = max(int(x * scale_x), 0)
    y1 = max(int(y * scale_y), 0)
    x2 = min(int((x + w_box) * scale_x), max_w)
    y2 = min(int((y + h_box) * scale_y), max_h)
    return x1, y1, x2, y2


def ai_camera_worker(
    mode,
    name,
    code,
    active_class_id,
    active_class_name,
    allowed_students,
    result_queue,
    stop_event,
    performance_mode="auto"
):
    """
    Ejecuta captura y reconocimiento en un proceso separado.
    El modo de rendimiento regula resolución, frecuencia de detección y frecuencia de reconocimiento.
    """
    security = SecurityManager()
    persistence = PersistenceManager(security)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        result_queue.put({"status": "camera_error", "message": "No se pudo abrir la cámara"})
        return

    cap, selected_mode, perf, benchmark_metrics = _resolve_performance_mode(cap, performance_mode)
    if not cap.isOpened():
        result_queue.put({"status": "camera_error", "message": "No se pudo reabrir la cámara"})
        return
    result_queue.put({
        "status": "mode_selected",
        "mode": selected_mode,
        "resolution": f"{perf['resolution'][0]}x{perf['resolution'][1]}",
        "fps": round(benchmark_metrics["fps"], 1) if benchmark_metrics["fps"] is not None else None,
        "inference_ms": round(benchmark_metrics["inference_ms"]) if benchmark_metrics["inference_ms"] is not None else None
    })

    for _ in range(5):
        cap.read()

    enroll_vectors = []
    enroll_stage = 0
    stages_text = ["Mire al frente  ", "Giro leve Izquierda  ", "Giro leve Derecha  "]
    win_title = "REGISTRO BIOMETRICO" if mode == "enroll" else "CONTROL DE ASISTENCIA"

    all_profiles = persistence.load_profiles()
    profile_indexes = _build_profile_index(all_profiles, allowed_students)
    already_marked = persistence.get_already_marked_today(active_class_id)
    session_cooldown = {}
    last_event_time = {}
    frame_counter = 0
    last_faces = []
    status_bar_msg = "ESPERANDO ROSTRO..."
    status_bar_color = (255, 255, 255)

    try:
        dummy_frame = np.zeros((perf["resolution"][1], perf["resolution"][0], 3), dtype=np.uint8)
        DeepFace.represent(dummy_frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=False)
    except Exception as exc:
        logger.debug("Warm-up de DeepFace omitido: %s", exc)

    def send_event(key, payload, interval=2):
        now = time.time()
        if now - last_event_time.get(key, 0) >= interval:
            result_queue.put(payload)
            last_event_time[key] = now

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            result_queue.put({"status": "camera_error", "message": "No se pudo leer frame de cámara"})
            break

        display_frame, scale_x, scale_y = _display_metrics(frame)
        h, w = frame.shape[:2]
        display_h, display_w = display_frame.shape[:2]
        key = cv2.waitKey(1) & 0xFF

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        light_ok = CONFIG["brightness_min"] < brightness < CONFIG["brightness_max"]
        sharp_ok = sharpness > CONFIG["sharpness_min"]

        if frame_counter % perf["detect_every"] == 0:
            try:
                detected = DeepFace.extract_faces(frame, detector_backend="mediapipe", enforce_detection=False)
                last_faces = _sort_relevant_faces(detected, perf["max_faces"])
            except Exception as exc:
                last_faces = []
                logger.debug("Fallo de detección facial: %s", exc)

        face_ready = bool(last_faces and last_faces[0]["facial_area"]["w"] > (w * CONFIG["face_size_ratio"]))

        if not light_ok:
            status_color = (0, 165, 255)
            cv2.putText(display_frame, "ADVERTENCIA: POCA LUZ", (20, display_h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        elif not sharp_ok:
            status_color = (0, 165, 255)
            cv2.putText(display_frame, "ADVERTENCIA: ENFOQUE POBRE", (20, display_h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            status_color = (0, 255, 0) if face_ready else (0, 0, 255)

        cv2.ellipse(
            display_frame,
            (display_w//2, display_h//2),
            (int(display_w*0.22), int(display_h*0.35)),
            0,
            0,
            360,
            status_color,
            2
        )

        if mode == "enroll":
            if enroll_stage < 3:
                txt_stage = f"ETAPA {enroll_stage+1}/3: {stages_text[enroll_stage]}"
                cv2.putText(display_frame, txt_stage, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                if light_ok and sharp_ok and face_ready:
                    cv2.putText(display_frame, "[PRESIONE ESPACIO PARA CAPTURAR]", (display_w//2 - 180, display_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if key == ord(' '):
                        try:
                            res = DeepFace.represent(frame, model_name="ArcFace", detector_backend="mediapipe", enforce_detection=True)
                            emb = np.array(res[0]["embedding"], dtype=np.float32)

                            duplicate = False
                            if enroll_stage == 0:
                                match = _best_match(emb, profile_indexes["global"], CONFIG["enroll_dup_threshold"])
                                duplicate = match is not None
                                logger.debug("Duplicado biométrico: %s", match)

                            if duplicate:
                                result_queue.put("DUPLICATE")
                                break

                            enroll_vectors.append(emb)
                            enroll_stage += 1
                        except Exception as exc:
                            logger.debug("Error técnico en represent durante enrolamiento: %s", exc)
                else:
                    msg = "ACERQUESE O MEJORE LUZ" if not face_ready else "ESTABILICE LA CAMARA"
                    cv2.putText(display_frame, msg, (20, display_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                current_registry = persistence.load_profiles()
                final_vector = np.mean(enroll_vectors, axis=0).astype(np.float32)
                final_vector = final_vector / np.linalg.norm(final_vector)

                current_registry[str(code)] = {
                    "nombre": name,
                    "vector": final_vector,
                    "fecha_registro": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S")
                }
                if persistence.save_profiles(current_registry):
                    result_queue.put("SUCCESS")
                break

        elif mode == "attendance":
            cv2.putText(display_frame, f"CLASE: {active_class_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.rectangle(display_frame, (0, display_h - 50), (display_w, display_h), (0, 0, 0), -1)

            if not last_faces:
                status_bar_msg, status_bar_color = "CAMARA VACIA", (200, 200, 200)

            for face_data in last_faces:
                facial_area = face_data["facial_area"]
                x = max(int(facial_area["x"]), 0)
                y = max(int(facial_area["y"]), 0)
                w_f = max(int(facial_area["w"]), 0)
                h_f = max(int(facial_area["h"]), 0)
                x2 = min(x + w_f, w)
                y2 = min(y + h_f, h)

                if x2 <= x or y2 <= y:
                    continue

                dx1, dy1, dx2, dy2 = _scale_box(x, y, w_f, h_f, scale_x, scale_y, display_w, display_h)
                cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

                should_recognize = (
                    w_f > (w * CONFIG["face_size_ratio"])
                    and light_ok
                    and sharp_ok
                    and frame_counter % perf["recognize_every"] == 0
                )
                if not should_recognize:
                    continue

                try:
                    res = DeepFace.represent(frame[y:y2, x:x2], model_name="ArcFace", detector_backend="skip", enforce_detection=False)
                    emb = np.array(res[0]["embedding"], dtype=np.float32)
                except Exception as exc:
                    logger.debug("Fallo al generar embedding de asistencia: %s", exc)
                    continue

                match = None
                if perf["match_enrolled_first"]:
                    match = _best_match(emb, profile_indexes["enrolled"], CONFIG["attendance_match_threshold"])

                if match:
                    clean_name = _clean_text(match["nombre"])
                    confianza = round(float(match["score"]), 2)
                    now = time.time()
                    last_log = session_cooldown.get(match["codigo"], 0)

                    if str(match["codigo"]) in already_marked or (now - last_log < CONFIG["cooldown_sec"]):
                        status_bar_msg = f"{clean_name} (Ya registrado)"
                        status_bar_color = (255, 255, 0)
                        send_event(
                            f"already:{match['codigo']}",
                            {"status": "already_marked", "codigo": match["codigo"], "nombre": match["nombre"]},
                            interval=CONFIG["cooldown_sec"]
                        )
                    else:
                        success = persistence.log_attendance({
                            "codigo": match["codigo"],
                            "nombre": match["nombre"],
                            "confianza": confianza,
                            "clase_id": active_class_id,
                            "clase_nombre": active_class_name
                        })
                        if success:
                            already_marked.add(str(match["codigo"]))
                            session_cooldown[match["codigo"]] = now
                            status_bar_msg = f"REGISTRADO: {clean_name} ({confianza:.2f})"
                            status_bar_color = (0, 255, 0)
                            result_queue.put({
                                "status": "success",
                                "codigo": match["codigo"],
                                "nombre": match["nombre"],
                                "score": confianza
                            })
                        else:
                            status_bar_msg = f"Error al guardar: {clean_name}"
                            status_bar_color = (0, 0, 255)
                    continue

                global_match = None
                if perf["global_fallback"]:
                    global_match = _best_match(emb, profile_indexes["global"], CONFIG["attendance_match_threshold"])

                if global_match:
                    clean_name = _clean_text(global_match["nombre"])
                    status_bar_msg = f"{clean_name}: NO MATRICULADO"
                    status_bar_color = (0, 0, 255)
                    send_event(
                        f"not_class:{global_match['codigo']}",
                        {
                            "status": "not_enrolled_in_class",
                            "codigo": global_match["codigo"],
                            "nombre": global_match["nombre"],
                            "score": round(float(global_match["score"]), 2)
                        }
                    )
                else:
                    status_bar_msg = "ROSTRO NO REGISTRADO"
                    status_bar_color = (0, 0, 255)
                    send_event("unknown_face", {"status": "unknown_face"})

            cv2.putText(display_frame, status_bar_msg, (20, display_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_bar_color, 2)

        cv2.imshow(win_title, display_frame)
        if key == 27 or cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
            result_queue.put("CANCELLED")
            break

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()
