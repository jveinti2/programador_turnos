# Programador de Turnos (Backend)

API construida con **FastAPI** y ejecutada mediante **uv**, diseñada para optimizar la asignación de turnos en centros de contacto utilizando el solver **Google OR-Tools (CP-SAT)** bajo una arquitectura modular basada en restricciones.

---

## ✨ Funcionalidades principales

- API REST para optimización de turnos
- Programación inteligente para equipos de 5 a 100 agentes
- Arquitectura modular basada en restricciones (9 reglas principales personalizables)
- Configuración de reglas vía archivos YAML (sin modificar código)
- Cumplimiento automático de reglas laborales:
  - Duración de turno efectiva: 4–9h
  - Máximo total (con almuerzo): 10h
  - Descanso mínimo entre turnos: 8h
  - Día libre obligatorio semanal
  - Pausas de 15min cada 3h (no en la primera ni última hora)
  - Almuerzo opcional de 30–60min
- Generación de reportes en JSON con estadísticas y coberturas
- Compatible con el frontend [programador_turnos_front](https://github.com/jveinti2/programador-turnos-front)

---

## 🚀 Getting Started

### Requisitos previos

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) instalado globalmente
- OR-Tools y FastAPI disponibles en entorno virtual

### Instalación

```bash
# Instalar dependencias con uv
uv sync

# O con pip si prefieres
pip install -e .
```

---

## 🧩 Ejecución del servidor

```bash
uv run uvicorn api_server:app --reload
```

Abrir [http://localhost:8000/docs](http://localhost:8000/docs) para ver la documentacion en el navegador.

---

## ⚙️ Configuración de entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```bash
OPENAI_API_KEY=sj-xxxxxxxxxxxxxxxxxxxxx
```

> 🔒 **Importante:** no subir este archivo al control de versiones.

## 🧠 Integración con IA

Este backend se conecta con un **LLM post-procesador** configurable, que optimiza y ajusta los resultados generados por el backend antes de mostrarlos en la interfaz.  
Desde el dashboard es posible personalizar prompts, reglas y comportamiento del modelo.

---

## 🧑‍💻 Contribuir

1. Crear una nueva rama desde `develop`
2. Hacer commit siguiendo las convenciones del proyecto
3. Abrir un Pull Request
4. La rama se eliminará automáticamente tras el merge

---

## 🪪 Licencia

MIT © jveinti2
