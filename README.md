# PROTOTIPO FUNCIONAL DE SISTEMA DE CONTROL DE ASISTENCIA A CLASES MEDIANTE RECONOCIMIENTO FACIAL BIOMÉTRICO

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-orange.svg)
![DeepFace](https://img.shields.io/badge/IA-DeepFace_ArcFace-green.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)
![Modelo SOTA](https://img.shields.io/badge/Powered_by-ArcFace_(SOTA_2026)-brightgreen)
![Rendimiento](https://img.shields.io/badge/Optimized_for-Low_Resource_Hardware-orange)

> **Universidad del Tolima** | Facultad de Ingeniería | Ingeniería de Sistemas (7mo Semestre)  
> **Asignatura:** Aprendizaje Automático  
> **Estado:** 🟢 Versión 1.0 Estable

## 📖 Descripción General
Prototipo académico desarrollado bajo la **metodología de Aprendizaje Basado en Proyectos (ABP)** para la asignatura *Aprendizaje Automático*. Automatiza el control de asistencia mediante **visión artificial y redes neuronales profundas**, utilizando el modelo **ArcFace (DeepFace)** como enfoque SOTA 2026 para la extracción de embeddings biométricos de alta dimensionalidad. El sistema opera 100% en local, garantizando la privacidad mediante cifrado AES-256 y firmas de integridad HMAC, sin depender de motores de bases de datos ni conectividad externa.

### ✨ Características Principales
*   🧠 **SOTA & Redes Neuronales:** Pipeline basado en ArcFace para extracción de características biométricas, cumpliendo estándares de precisión académica y requisitos de la asignatura.
*   🖥️ **Optimizado para Hardware Limitado:** Inferencia ajustada con `frame-skipping`, downscaling adaptativo (640x480) y multiprocesamiento aislado para ejecución fluida en CPUs antiguos sin GPU.
*   🧩 **Arquitectura Modular:** Separación estricta de responsabilidades entre interfaz gráfica (`gui_app.py`) y motor de IA (`src/ai_engine.py`), siguiendo buenas prácticas de ingeniería de software y patrones MVC simplificados.
*   🔒 **Seguridad & Cumplimiento Normativo:** Diseño alineado a **ISO 25012** (Integridad de Datos) y **Ley 1581 de 2012** (Habeas Data). Cifrado AES-256 para biometría, firmas HMAC-SHA256 en logs y consentimiento informado integrado.
*   📊 **Auditoría y Reportes:** Trazabilidad inmutable de acciones administrativas y exportación de asistencia a Excel con validación criptográfica automática.

## 🛠️ Stack Tecnológico

| Capa | Tecnologías |
| :--- | :--- |
| **Lenguaje** | Python 3.10+ |
| **Visión Artificial** | DeepFace (Backend: ArcFace), MediaPipe, OpenCV |
| **Interfaz Gráfica** | CustomTkinter |
| **Datos & Integridad** | Pandas, HMAC-SHA256, Msgpack |
| **Seguridad** | Fernet (Cryptography), PBKDF2 |
| **Documentación** | Sphinx + reStructuredText |

## ✅ Cumplimiento de Requerimientos
El prototipo satisface integralmente los requisitos funcionales y no funcionales definidos en la propuesta académica:

| ID | Categoría | Evidencia de Implementación |
|---|---|---|
| **RF-01 a RF-10** | Funcionales | Enrolamiento biométrico, detección multirrostro en tiempo real, matching con ArcFace, control de unicidad por sesión, reportes filtrados y gestión local autónoma. |
| **RNF-01 a RNF-09** | No Funcionales | Interfaz intuitiva (<3 clics), precisión ≥70%, latencia de inferencia <4s, almacenamiento binario ligero, cifrado de datos sensibles, portabilidad sin BD externa y cumplimiento legal/ético. |

## 📂 Estructura del Proyecto

```text
ReconFacial/
│
├── .gitignore
├── gui_app.py              # Orquestador principal e Interfaz Gráfica
│
├──src/                    # Núcleo de Lógica e IA
│   ├── __init__.py
│   ├── ai_engine.py        # Motor de inferencia (Side-car process)
│   ├── persistence.py      # Gestión de datos y validación de integridad
│   └── security_manager.py # Cifrado, llaves y respaldo seguro
│
├── data/                   # Almacenamiento local
│   ├── attendance/
│   │   ├── active_session.json
│   │   └── master_log.csv
│   ├── meta/
│   │   ├── admin_audit.bin
│   │   └── classes.json
│   ├── profiles/
│   │   └── encrypted_registry.bin
│   └── security/
│       └── secret.key
└── docs/                   # Documentación técnica (Sphinx)                 
```

## 🚀 Instalación y Ejecución

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/raospinap/ReconocimientoFacial
   cd ReconFacial
   ```

2. **Crear entorno virtual e instalar dependencias:**
   ```bash
   python -3.10 -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Ejecutar la aplicación:**
   ```bash
   python gui_app.py
   ```

## 📄 Documentación Técnica

El proyecto incluye documentación técnica completa generada con Sphinx, que detalla la arquitectura, módulos de IA, capa de persistencia y gestión criptográfica.

*   🌐 **Ver Documentación Local:** Abre `docs/build/html/index.html` en tu navegador.
*    **Regenerar Documentación:**
   ```bash
   cd docs
   make html
   ```

## 👥 Equipo de Desarrollo

| Nombre | Rol Principal |
| :--- | :--- |
| **Daniela Martínez** | Desarrollo de Interfaz y Flujos de Usuario |
| **Santiago Quiroga** | Arquitectura de Software y Gestión de Datos |
| **Rolando Ospina** | Visión Artificial y Calibración de Modelos |
| **Juan Diego Amaya**   | Integración, Seguridad y Documentación |

## ⚖️ Licencia y Privacidad

Este software es un prototipo académico desarrollado en el marco del programa de Ingeniería de Sistemas de la Universidad del Tolima. Los datos biométricos son procesados y almacenados localmente bajo estándares de seguridad criptográfica. El uso del sistema implica la aceptación de las políticas de manejo de datos establecidas en el aviso de consentimiento informado integrado en la aplicación.

© 2026 Universidad del Tolima. Proyecto académico sin fines de lucro.
