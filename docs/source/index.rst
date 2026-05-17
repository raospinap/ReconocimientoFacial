.. PROTOTIPO FUNCIONAL DE SISTEMA DE CONTROL DE ASISTENCIA A CLASES MEDIANTE RECONOCIMIENTO FACIAL BIOMÉTRICO documentation master file, created by
   sphinx-quickstart on Fri May 15 15:43:40 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PROTOTIPO FUNCIONAL DE SISTEMA DE CONTROL DE ASISTENCIA A CLASES MEDIANTE RECONOCIMIENTO FACIAL BIOMÉTRICO documentation
========================================================================================================================

**Autores:** Daniela Martínez, Juan Diego Amaya, Santiago Quiroga, Rolando Ospina  
**Universidad del Tolima - Ingeniería de Sistemas**  
**Asignatura:** Aprendizaje Automático  

Descripción General
-------------------
Este proyecto desarrolla un prototipo funcional para la automatización del control de asistencia académica mediante técnicas de visión artificial y aprendizaje automático. El sistema opera bajo una arquitectura de procesamiento local, garantizando estándares de precisión confiables y la integridad técnica y ética de la información biométrica procesada, cumpliendo con la Ley 1581 de 2012 (Habeas Data).

Características Principales
---------------------------
* **Reconocimiento Facial en Tiempo Real:** Uso de ArcFace y MediaPipe para detección y extracción de embeddings.
* **Persistencia Local Segura:** Almacenamiento cifrado con AES-256 (Fernet) y firmas HMAC-SHA256 para integridad.
* **Interfaz Gráfica Intuitiva:** Desarrollada en CustomTkinter para facilitar la operación sin capacitación previa.
* **Auditoría Completa:** Registro inmutable de todas las acciones administrativas y de asistencia.

Documentación Técnica
---------------------

.. toctree::
   :maxdepth: 2
   :caption: Secciones:

   modules