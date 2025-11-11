# API Endpoints Summary

## Endpoints Disponibles (14 total)

### 1. Gestión de Horarios
- **POST /generate-schedule** - Genera horarios usando CP-SAT solver
  - ✅ No requiere body
  - ✅ Incluye ejemplo de response exitoso
  - ✅ Incluye ejemplo de response con deficit

- **GET /schedules** - Obtiene todos los horarios generados
  - ✅ No requiere body
  - ✅ Incluye ejemplo completo de response

- **GET /schedule/{agent_id}** - Obtiene horario de un agente específico
  - ✅ Parámetro de ruta documentado
  - ✅ Incluye ejemplo de response

- **PUT /update-schedule** - Actualiza horarios manualmente
  - ✅ Incluye ejemplo de request body
  - ✅ Incluye ejemplo de response

- **POST /optimize-schedule-llm** - Post-procesa horarios con LLM
  - ✅ Parámetros query documentados
  - ✅ Incluye descripción del flujo

### 2. Gestión de Configuración
- **GET /get-rules** - Obtiene configuración de reglas
  - ✅ Response model: ReglasYAML con ejemplo completo
  - ✅ Incluye todos los campos con validación

- **POST /update-rules** - Actualiza configuración de reglas
  - ✅ Request body: ReglasYAML con ejemplo completo
  - ✅ Incluye ejemplo de response

- **GET /get-prompt** - Obtiene configuración de LLM
  - ✅ Response model: SystemPromptConfig con ejemplo

- **POST /update-prompt** - Actualiza configuración de LLM
  - ✅ Request body: SystemPromptConfig con ejemplo
  - ✅ Incluye descripción de campos

### 3. Gestión de Datos CSV
- **GET /get-agents-csv** - Obtiene CSV de agentes
  - ✅ Formato actualizado (id,nombre)
  - ✅ Response: text/plain
  - ✅ Incluye ejemplo del formato

- **POST /update-agents-csv** - Actualiza CSV de agentes
  - ✅ Request body: text/plain con validación
  - ✅ Formato actualizado (id,nombre)
  - ✅ Incluye ejemplos de curl y JavaScript fetch
  - ✅ Incluye ejemplo de response con estadísticas

- **GET /get-demand-csv** - Obtiene CSV de demanda
  - ✅ Formato documentado (dia,bloque,agentes_requeridos)
  - ✅ Response: text/plain
  - ✅ Incluye ejemplo del formato

- **POST /update-demand-csv** - Actualiza CSV de demanda
  - ✅ Request body: text/plain con validación
  - ✅ Incluye ejemplos de curl y JavaScript fetch
  - ✅ Incluye ejemplo de response con estadísticas

### 4. Utilidad
- **GET /health** - Health check
  - ✅ Simple respuesta de status

## Modelos Pydantic con Ejemplos

### ReglasYAML
```json
{
  "turnos": {
    "duracion_min_horas": 4,
    "duracion_max_horas": 9,
    "duracion_total_max_horas": 10,
    "descanso_entre_turnos_horas": 8,
    "dias_libres_min_por_semana": 1
  },
  "pausas_cortas": {
    "duracion_minutos": 15,
    "frecuencia_cada_horas": 3,
    "obligatorias": true,
    "separacion_minima_horas": 2.5,
    "prohibir_primera_hora": true,
    "prohibir_ultima_hora": true,
    "maximo_por_turno": 4
  },
  "almuerzo": {
    "duracion_min_minutos": 30,
    "duracion_max_minutos": 60,
    "obligatorio_si_turno_mayor_a_horas": 0,
    "maximo_por_turno": 1,
    "prohibir_primera_hora": true,
    "prohibir_ultima_hora": true
  },
  "solver": {
    "timeout_segundos": 300,
    "num_workers": 8
  },
  "cobertura": {
    "margen_seguridad": 1.2,
    "bloquear_si_deficit": true
  }
}
```

### SystemPromptConfig
```json
{
  "system_prompt": "You are an AI assistant...",
  "model": "gpt-4",
  "temperature": 0.7,
  "top_p": 1.0,
  "frecuency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

## Validaciones Incluidas

### Agentes CSV
- Headers: `id,nombre`
- Validación: IDs únicos, campos no vacíos

### Demanda CSV
- Headers: `dia,bloque,agentes_requeridos`
- Validación:
  - `dia`: 0-6 (Lunes-Domingo)
  - `bloque`: 0-47 (bloques de 30 minutos)
  - `agentes_requeridos`: >= 0
  - Sin duplicados (dia,bloque)

### Turnos Config
- `duracion_min_horas`: 1-24
- `duracion_max_horas`: 1-24
- `duracion_total_max_horas`: 1-24
- `descanso_entre_turnos_horas`: 0-24
- `dias_libres_min_por_semana`: 0-7

### Pausas Cortas Config
- `duracion_minutos`: 1-60
- `frecuencia_cada_horas`: 0.5-12
- `separacion_minima_horas`: 0-12
- `maximo_por_turno`: 0-10

### Almuerzo Config
- `duracion_min_minutos`: 0-120
- `duracion_max_minutos`: 0-120
- `obligatorio_si_turno_mayor_a_horas`: 0-12
- `maximo_por_turno`: 0-5

### Solver Config
- `timeout_segundos`: 1-3600
- `num_workers`: 1-128

### Cobertura Config
- `margen_seguridad`: 1.0-3.0

## Formato de Horarios (Response)

```json
[
  {
    "id": "A001",
    "name": "Juan Pérez",
    "schedule": {
      "lunes": {
        "start": "07:30",
        "end": "16:00",
        "break": [
          {
            "start": "12:30",
            "end": "12:45"
          }
        ],
        "disconnected": [
          {
            "start": "13:30",
            "end": "14:00"
          }
        ]
      },
      "martes": { ... }
    }
  }
]
```

## Conclusión

✅ **Todos los endpoints tienen documentación completa**
✅ **Todos los modelos incluyen ejemplos**
✅ **Todas las validaciones están especificadas**
✅ **OpenAPI schema generado (openapi.json) contiene toda la información**

El frontend puede usar el archivo `openapi.json` para:
1. Generar automáticamente clientes API
2. Validar requests antes de enviarlos
3. Mostrar ejemplos en la UI
4. Documentar el API en Swagger UI o ReDoc
