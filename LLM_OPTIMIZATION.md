# Optimización de Turnos con LLM

## Resumen

El endpoint `/optimize-schedule-llm` permite mejorar los turnos generados por CP-SAT usando OpenAI GPT-4o-mini. Procesa el schedule en chunks de 10 agentes para evitar límites de tokens y reduce el tiempo de procesamiento.

## Configuración

### 1. Configurar OpenAI API Key

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."

# Windows (CMD)
set OPENAI_API_KEY=sk-...

# Linux/Mac
export OPENAI_API_KEY=sk-...
```

### 2. Configurar Prompt del Sistema

Editar `config/system_prompt.yaml`:

```yaml
system_prompt: |
  [Instrucciones del LLM aquí]

model: gpt-4o-mini  # Modelo a usar
temperature: 0.7     # Creatividad (0-1)
```

## Uso

### Endpoint

```
POST /optimize-schedule-llm
```

### Parámetros Query (Opcionales)

- `temperature` (float): Temperatura del modelo (0-1, default: 0.3)
- `chunk_size` (int): Agentes por chunk (default: 10)
- `use_toon` (bool): Usar formato TOON (default: false, no mejora con esta estructura)

### Ejemplo de Llamada

```bash
# Curl
curl -X POST "http://localhost:8000/optimize-schedule-llm?temperature=0.3&chunk_size=10"

# Python
import requests
response = requests.post("http://localhost:8000/optimize-schedule-llm", params={
    "temperature": 0.3,
    "chunk_size": 10
})
print(response.json())
```

### Respuesta Exitosa

```json
{
  "status": "success",
  "message": "Schedule optimizado en 5 chunks",
  "total_agents": 50,
  "chunks_processed": 5,
  "chunks_optimized": 5,
  "chunks_fallback": 0,
  "chunk_details": [
    {
      "chunk": 1,
      "agents": 10,
      "status": "optimized"
    },
    ...
  ],
  "config": {
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "chunk_size": 10,
    "use_toon": false
  },
  "output_file": "output\\schedules.json"
}
```

## Flujo de Trabajo

1. **Pre-requisito**: Ejecutar `/optimize` primero para generar `output/schedules.json` con CP-SAT
2. **Optimización LLM**: Ejecutar `/optimize-schedule-llm` para mejorar el schedule
3. **Post-procesamiento**: El archivo `output/schedules.json` es sobrescrito con la versión optimizada

## Arquitectura de la Solución

### Problema Original

- **schedules.json** es muy grande (148KB, ~47k tokens con 50 agentes)
- Enviar todo de una vez excede límites de tokens del LLM
- LLM tarda mucho tiempo y puede devolver JSON malformado

### Solución Implementada: Chunking por Agentes

```
schedules.json (50 agentes)
    ↓
Dividir en 5 chunks de 10 agentes
    ↓
Para cada chunk:
  ├─ Construir contexto global (estadísticas del resto)
  ├─ Convertir a JSON compacto (~14k chars)
  ├─ Enviar a GPT-4o-mini (~15s por chunk)
  ├─ Validar respuesta JSON
  └─ Guardar chunk optimizado
    ↓
Fusionar chunks optimizados
    ↓
Validar schedule completo
    ↓
Guardar en output/schedules.json
```

### Ventajas del Chunking

- ✅ **Reduce tokens**: 14k chars/chunk vs 68k total (80% reducción)
- ✅ **Mantiene contexto semanal**: Cada agente se optimiza con sus 7 días completos
- ✅ **Tiempo predecible**: ~15s por chunk (75s para 50 agentes)
- ✅ **Fallback robusto**: Si un chunk falla, usa el original (no bloquea todo)
- ✅ **Validación por chunk**: Detecta errores temprano

### Manejo de Errores

El sistema es robusto ante fallos:

1. **JSON malformado**: Usa regex para extraer JSON del texto del LLM
2. **Chunk inválido**: Revierte al chunk original sin detener el proceso
3. **Error de OpenAI**: Captura y reporta, continúa con siguiente chunk
4. **Schedule final inválido**: Lanza error 500 (no sobrescribe el original)

## Estrategia de Contexto Global

Cada chunk recibe:

1. **Agentes del chunk**: Los 10 agentes a optimizar (estructura completa)
2. **Contexto global**: Estadísticas resumidas del resto del sistema
   - Total de agentes
   - Horas promedio trabajadas por otros agentes
   - Balance de carga esperado

Esto permite al LLM tomar decisiones informadas sin ver todo el schedule.

## Alternativas Consideradas (No Implementadas)

### 1. TOON Format
- **Problema**: Aumenta 3.5% el tamaño con esta estructura JSON
- **Razón**: El JSON ya está muy estructurado/repetitivo
- **Veredicto**: Deshabilitado por defecto

### 2. Chunking por Día
- **Problema**: Pierde contexto semanal (balance lunes-domingo)
- **Ventaja**: Menor tamaño por chunk
- **Veredicto**: Descartado (contexto semanal es crítico)

### 3. Batch API de OpenAI
- **Problema**: Latencia de hasta 24 horas
- **Ventaja**: 50% más barato
- **Veredicto**: No apto para UX en tiempo real

### 4. Vector Store + Assistants API
- **Problema**: Complejidad adicional + latencia variable
- **Ventaja**: Sin límites de contexto
- **Veredicto**: Overkill para 50 agentes

## Configuración del Prompt

### Recomendaciones

El `system_prompt.yaml` debe enfatizar:

1. **Contexto de chunk**: "Recibirás solo 10 agentes, no todos"
2. **Constraints críticos**: Listar reglas que DEBE respetar
3. **Formato de salida**: "Devolver JSON puro sin explicaciones"
4. **Ejemplo de estructura**: Mostrar formato esperado

### Prompt Actual

Ver `config/system_prompt.yaml` para el prompt completo.

## Costos Estimados

Con GPT-4o-mini:

- **Input**: ~14k tokens/chunk × 5 chunks = 70k tokens
- **Output**: ~14k tokens/chunk × 5 chunks = 70k tokens
- **Total**: ~140k tokens
- **Costo**: ~$0.02 USD por optimización (50 agentes)

## Troubleshooting

### Error: "OPENAI_API_KEY no configurada"

**Solución**: Configurar variable de entorno antes de iniciar el servidor.

### Error: "Archivo no encontrado: output/schedules.json"

**Solución**: Ejecutar `/optimize` primero para generar el schedule inicial.

### Error: "Expecting ',' delimiter: line X column Y"

**Causa**: LLM devolvió JSON malformado (raro con prompt actual)

**Solución**:
1. Revisar `system_prompt.yaml` para enfatizar JSON válido
2. Reducir `temperature` (ej: 0.1)
3. El regex fallback debería manejarlo automáticamente

### Chunking muy lento (>30s por chunk)

**Causa**: Modelo sobrecargado o temperatura alta

**Solución**:
1. Reducir `chunk_size` (ej: 5 agentes)
2. Reducir `temperature` (ej: 0.2)
3. Verificar status de OpenAI API

## Próximos Pasos (Mejoras Futuras)

### Estrategia 7: Sistema de Instrucciones

En lugar de regenerar todo el JSON, el LLM podría generar **instrucciones de cambio**:

```json
{
  "actions": [
    {"type": "move_shift", "agent": "A001", "day": "lunes", "offset_hours": 2},
    {"type": "swap_break", "agent": "A003", "day": "martes", "break_index": 1, "new_time": "14:00"}
  ]
}
```

**Ventajas**:
- Reduce tokens a ~5k por llamada
- Cambios incrementales (más controlables)
- Validación automática por Python

**Requiere**: Implementar motor de aplicación de cambios

## Archivos del Sistema

- `shift_optimizer/llm_optimizer.py` - Módulo principal de optimización LLM
- `api_server.py:584-731` - Endpoint `/optimize-schedule-llm`
- `config/system_prompt.yaml` - Configuración del prompt y modelo
- `output/schedules.json` - Schedule a optimizar (entrada/salida)
