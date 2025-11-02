# Shift Optimizer API

FastAPI-based REST API for optimizing call center shift assignments using Google OR-Tools CP-SAT solver with modular constraint-based architecture.

## Features

- **RESTful API** for shift optimization with simple GET endpoints
- **Smart shift scheduling** for call centers with 50+ agents across a week
- **Constraint-based architecture** with 9 modular rules (easy to customize)
- **YAML configuration** for business rules (no code changes needed)
- Respects individual agent availability
- Ensures compliance with labor rules:
  - 4-9 hours effective work per shift (max 10h total including lunch)
  - Minimum 8 hours rest between consecutive shifts
  - 1 mandatory day off per week
  - 15-minute break every 3 hours (not in first/last hour)
  - Optional 30-60 minute lunch break (not in first/last hour)
- Minimizes staffing gaps by matching coverage to demand
- Exports results to JSON files and HTTP responses

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Or if you prefer pip
pip install -e .
```

## Quick Start

### Starting the API Server

```bash
# Start the server with hot reload (development)
uv run uvicorn api_server:app --reload

# Or for production
uv run uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Root - API Information
```bash
GET http://localhost:8000/
```

Returns API information and available endpoints.

#### 2. Health Check
```bash
GET http://localhost:8000/health
```

Returns server health status.

#### 3. Run Optimization
```bash
GET http://localhost:8000/optimize
```

Executes shift optimization using example data and returns the summary.

**What it does:**
- Reads agents from `data/agentes_ejemplo.json`
- Reads demand from `data/demanda_ejemplo.json`
- Runs the CP-SAT solver with configuration from `config/reglas.yaml`
- Saves results to `output/schedules.json` and `output/resumen.json`
- Returns the optimization summary (resumen.json content)

**Example response:**
```json
{
  "status": "OPTIMAL",
  "tiempo_resolucion_segundos": 45.23,
  "costo_total": 123.45,
  "total_turnos": 35,
  "total_agentes_asignados": 5,
  "total_agentes_disponibles": 5,
  "demanda_total": 500,
  "cobertura_total": 498,
  "cobertura_por_dia": [...]
}
```

#### 4. Get Agent Schedule
```bash
GET http://localhost:8000/schedule/{agent_id}
```

Returns the schedule for a specific agent.

**Example:**
```bash
curl http://localhost:8000/schedule/AG001
```

**Response:**
```json
{
  "agente_id": "AG001",
  "nombre": "Juan Pérez",
  "schedule": {
    "lunes": {
      "start": "09:00",
      "end": "18:00",
      "break": ["11:00", "15:00"],
      "disconnected": ["13:00", "14:00"]
    },
    "martes": { ... }
  }
}
```

**Note:** You must call `/optimize` first to generate the schedules.

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Configuration

All optimization rules are configured in `config/reglas.yaml`. You can customize:

```yaml
shift_rules:
  duracion_efectiva:
    minimo_horas: 4        # Minimum effective work hours
    maximo_horas: 9        # Maximum effective work hours
  duracion_maxima_total:
    horas: 10              # Max total shift duration (including lunch)
  descanso_entre_turnos:
    horas: 8               # Minimum rest between shifts
  dia_libre_semanal_minimo: 1

breaks:
  frecuencia_horas: 3              # Break every 3 hours
  duracion_minutos: 15
  es_obligatorio: true             # true = mandatory breaks
  no_en_primera_hora: true
  no_en_ultima_hora: true
  separacion_minima_horas: 2.5     # Min 2.5h between consecutive breaks

lunch:
  duracion_minima_minutos: 30
  duracion_maxima_minutos: 60
  para_turnos_mayores_a_horas: 4
  no_en_primera_hora: true
  no_en_ultima_hora: true

solver:
  timeout_segundos: 300    # Solver timeout (affects solution quality)
  num_workers: 8           # Parallel search workers
```

### About Solver Timeout

The `timeout_segundos` parameter controls how long the solver runs:
- **Longer timeouts** (600+): Better chance of finding optimal solutions
- **Shorter timeouts** (60-300): Faster results, may be feasible but not optimal
- **Default (300s)**: Good balance between speed and quality

The solver will return one of these statuses:
- `OPTIMAL`: Best possible solution found
- `FEASIBLE`: Valid solution found within timeout, might improve with more time
- `INFEASIBLE`: No valid solution exists given the constraints

## Input File Formats

### Agents File (JSON)

```json
[
  {
    "id": "A001",
    "nombre": "Juan Pérez",
    "disponibilidad": [
      {"day": 0, "block": 18},
      {"day": 0, "block": 19},
      ...
    ]
  },
  ...
]
```

- `day`: 0-6 (Monday=0, Sunday=6)
- `block`: 0-47 (30-minute blocks in a day, starting at 00:00)
  - Example: block 18 = 09:00, block 19 = 09:30

### Demand File (JSON)

```json
[
  [0, 0, 0, 0, 0, 0, 5, 10, 15, ...],  // Monday (48 blocks)
  [0, 0, 0, 0, 0, 0, 5, 10, 15, ...],  // Tuesday (48 blocks)
  ...
]
```

- 7 arrays (one per day, Monday-Sunday)
- Each array has 48 elements (30-minute blocks)
- Values represent number of agents needed at that time

## Output Files

The optimizer generates 2 JSON files in the `output/` directory:

### 1. schedules.json

Agent schedules structured by agent for frontend consumption:

```json
{
  "A001": {
    "agente_id": "A001",
    "nombre": "Juan Pérez",
    "schedule": {
      "lunes": {
        "start": "09:00",
        "end": "18:00",
        "break": ["11:00", "15:00"],
        "disconnected": ["13:00", "14:00"]
      },
      "martes": { ... }
    }
  },
  ...
}
```

- `break`: 15-minute breaks (paid time, counts as effective work)
- `disconnected`: Lunch/hueco (unpaid, doesn't count as effective work)
- Only includes days where the agent has a shift

### 2. resumen.json

Summary statistics:

```json
{
  "status": "OPTIMAL",
  "tiempo_resolucion_segundos": 45.23,
  "costo_total": 123.45,
  "total_turnos": 35,
  "total_agentes_asignados": 5,
  "total_agentes_disponibles": 5,
  "demanda_total": 500,
  "cobertura_total": 498,
  "cobertura_por_dia": [...]
}
```

## Programmatic Usage (Python)

You can also use the optimizer directly in Python code:

```python
from shift_optimizer import ShiftOptimizer

# Create optimizer from configuration file
optimizer = ShiftOptimizer.from_config('config/reglas.yaml')

# Run optimization
result = optimizer.optimize(
    agents_path='data/agentes_ejemplo.json',
    demand_path='data/demanda_ejemplo.json'
)

print(f"Status: {result['status']}")
print(f"Total shifts: {result['summary']['total_turnos']}")
print(f"Agents assigned: {result['summary']['total_agentes_asignados']}")
```

## Architecture

### Modular Constraint-Based Design

The system uses 9 modular constraint classes, each encapsulating a specific business rule:

1. **AvailabilityConstraint** - Agents only work when available
2. **ShiftDurationConstraint** - 4-9 hours effective work
3. **ShiftSpanConstraint** - Max 10 hours total (start to end)
4. **RestBetweenShiftsConstraint** - Min 8 hours between shifts
5. **WeeklyDayOffConstraint** - At least 1 day off per week
6. **BreakConstraint** - 15-min break every 3 hours
   - Minimum: 1 break per 3 hours (obligatory)
   - Maximum: 4 breaks per shift
   - Separation: Min 2.5 hours between consecutive breaks
7. **LunchConstraint** - Optional 30-60 min lunch
8. **ShiftContinuityConstraint** - No gaps within shifts
9. **CoverageConstraint** - Calculate staffing levels

### Adding New Constraints

To add a new business rule:

1. Create a class in `shift_optimizer/constraints.py`:
```python
class MyNewConstraint(Constraint):
    @property
    def name(self) -> str:
        return "My New Rule"

    def apply(self, optimizer: ShiftOptimizer) -> None:
        # Add your constraint logic here
        optimizer.model.Add(...)
```

2. Add it to the constraint list in `shift_optimizer/solver.py`
3. Optionally add configuration to `config/reglas.yaml`

### Project Structure

```
programador_turnos/           # Project root
├── api_server.py             # FastAPI server (main entry point)
├── pyproject.toml            # Project dependencies (uv/pip)
├── config/
│   └── reglas.yaml          # Business rules configuration
├── data/
│   ├── agentes_ejemplo.json # Example agents (5)
│   └── demanda_ejemplo.json # Example demand (7 days)
├── shift_optimizer/         # Python package
│   ├── __init__.py          # Package exports
│   ├── api.py               # High-level ShiftOptimizer API
│   ├── config.py            # YAML configuration loading
│   ├── constraints.py       # Modular constraint classes (9)
│   ├── models.py            # Data models (Agent, Demanda, Shift, Solution)
│   ├── solver.py            # CP-SAT optimization engine
│   └── utils.py             # I/O utilities
└── output/
    ├── schedules.json       # Agent schedules (generated)
    └── resumen.json         # Summary statistics (generated)
```

## Key Implementation Details

### Time Blocks
- **30-minute blocks** (48 per day), not 15-minute
- Block 0 = 00:00-00:30, Block 1 = 00:30-01:00, ..., Block 47 = 23:30-00:00

### Break vs Lunch (Hueco)
- **Breaks (15 min)**: Count as effective work, reduce coverage by 0.5
  - **Obligatory**: 1 break every 3 hours of work
  - **Maximum**: 4 breaks per shift (prevents solver from adding unlimited breaks)
  - **Separation**: Minimum 2.5 hours between consecutive breaks
  - **Placement**: Cannot be in first or last hour of shift
- **Lunch/Hueco (30-60 min)**: Do NOT count as effective work, fully reduce coverage

### Effective Work Calculation
```
effective_work = total_work_blocks - lunch_blocks
```
Breaks are NOT subtracted because they're paid work time.

### Protected Hours
First and last hour of each shift cannot have breaks or lunch to protect productive time.

## Performance

- Typical solve time: 10-60 seconds for 5-50 agents
- Uses 8 parallel search workers (configurable)
- Solver may return OPTIMAL or FEASIBLE status (both are valid solutions)

## Requirements

- Python 3.12+
- FastAPI 0.119+
- OR-Tools 9.7+
- PyYAML 6.0+

## Documentation

For detailed architecture and development guidance, see `CLAUDE.md`.

## License

MIT License

## Author

Generated with Claude Code


# Plan general de organización de turnos

## Objetivo
Diseñar un flujo de planificación de turnos en tres fases.  
La prioridad absoluta es el **Punto 1: Organización de turnos con CP-SAT**.

---

## 1. Organización de turnos (fase principal con CP-SAT)

- Utilizar CP-SAT para **asignar los turnos de la mejor manera posible** cumpliendo todas las restricciones definidas.  
- Si **no es posible cumplir alguna restricción**, el modelo debe **indicar claramente la causa** (por ejemplo: falta de personal para cubrir la demanda).  
- No debe intentar forzar soluciones ni generar combinaciones innecesarias:
  - Si falta personal, debe reportarlo explícitamente.
  - La **demanda define la cantidad mínima necesaria** de agentes.
- El resultado debe ser conciso e incluir solo la información esencial:
  - **Estado del resultado:** óptimo, deficiente o superávit.
  - **Cantidad de personal faltante (si aplica).**
  - **Cobertura por día:** porcentaje o cantidad de demanda cubierta.

---

## 2. Postprocesamiento (fase opcional con LLM)

- El LLM puede **ajustar o mejorar la organización de los turnos** con base en preferencias individuales o de negocio.
- **No puede agregar ni eliminar personal.**
- Solo puede **reacomodar horarios** dentro de las restricciones y resultados obtenidos en el punto 1.

---

## 3. Configuración manual (fase final)

- Un responsable humano podrá revisar y modificar manualmente el resultado si lo considera necesario.
- Este paso será implementado posteriormente mediante un endpoint específico (aún no desarrollado).

---

## Resumen de prioridades

1. Cumplir correctamente el **punto 1 (CP-SAT)** sin sobreingeniería.  
2. Reportar resultados de forma clara y resumida.  
3. Mantener la lógica simple: **la demanda define el mínimo necesario**, y cualquier falta de personal debe ser informada, no compensada.
