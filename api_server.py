"""
FastAPI server for shift optimization system.
"""

from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from openai import OpenAI
import asyncio
import json
import re
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from shift_optimizer import ShiftOptimizer, DeficitAnalyzer
from shift_optimizer.utils import load_agents, load_demanda
from shift_optimizer.config import OptimizationConfig
from shift_optimizer.csv_parsers import parse_agents_csv, parse_schedules_csv, CSVParseError
from shift_optimizer.models import TimeBlock


from toon_format import encode, decode

# ===== PYDANTIC MODELS FOR CONFIGURATION =====

class TurnosConfig(BaseModel):
    """Shift duration and timing rules."""
    duracion_min_horas: int = Field(..., ge=1, le=24, description="Minimum effective work hours")
    duracion_max_horas: int = Field(..., ge=1, le=24, description="Maximum effective work hours")
    duracion_total_max_horas: int = Field(..., ge=1, le=24, description="Maximum total shift span")
    descanso_entre_turnos_horas: int = Field(..., ge=0, le=24, description="Minimum rest between shifts")
    dias_libres_min_por_semana: int = Field(..., ge=0, le=7, description="Minimum days off per week")
    horas_min_por_semana: int = Field(..., ge=0, le=168, description="Minimum weekly work hours")
    horas_max_por_semana: int = Field(..., ge=0, le=168, description="Maximum weekly work hours")

    @validator('duracion_max_horas')
    def max_greater_than_min(cls, v, values):
        if 'duracion_min_horas' in values and v < values['duracion_min_horas']:
            raise ValueError('duracion_max_horas debe ser >= duracion_min_horas')
        return v

    @validator('horas_max_por_semana')
    def weekly_max_greater_than_min(cls, v, values):
        if 'horas_min_por_semana' in values and v < values['horas_min_por_semana']:
            raise ValueError('horas_max_por_semana debe ser >= horas_min_por_semana')
        return v


class PausasCortasConfig(BaseModel):
    """Short break rules (15-min breaks)."""
    duracion_minutos: int = Field(..., ge=1, le=60, description="Break duration in minutes")
    frecuencia_cada_horas: float = Field(..., ge=0.5, le=12, description="Frequency: 1 break every X hours")
    obligatorias: bool = Field(..., description="Are breaks mandatory?")
    separacion_minima_horas: float = Field(..., ge=0, le=12, description="Minimum separation between breaks")
    prohibir_primera_hora: bool = Field(..., description="Prohibit breaks in first hour?")
    prohibir_ultima_hora: bool = Field(..., description="Prohibit breaks in last hour?")
    maximo_por_turno: int = Field(..., ge=0, le=10, description="Maximum breaks per shift")


class AlmuerzoConfig(BaseModel):
    """Lunch break rules (30-60 min)."""
    duracion_min_minutos: int = Field(..., ge=0, le=120, description="Minimum lunch duration")
    duracion_max_minutos: int = Field(..., ge=0, le=120, description="Maximum lunch duration")
    obligatorio_si_turno_mayor_a_horas: float = Field(..., ge=0, le=12, description="Mandatory if shift > X hours (0 = never)")
    maximo_por_turno: int = Field(..., ge=0, le=5, description="Maximum lunch breaks per shift")
    prohibir_primera_hora: bool = Field(..., description="Prohibit lunch in first hour?")
    prohibir_ultima_hora: bool = Field(..., description="Prohibit lunch in last hour?")


class SolverConfig(BaseModel):
    """Solver parameters."""
    timeout_segundos: int = Field(..., ge=1, le=3600, description="Maximum execution time in seconds")
    num_workers: int = Field(..., ge=1, le=128, description="Number of parallel workers")


class CoberturaConfig(BaseModel):
    """Coverage analysis configuration."""
    margen_seguridad: float = Field(..., ge=1.0, le=3.0, description="Recommended safety margin (e.g., 1.2 = 20% extra)")
    bloquear_si_deficit: bool = Field(..., description="Block optimization if critical deficit exists?")


class ReglasYAML(BaseModel):
    """Complete optimization rules configuration."""
    turnos: TurnosConfig
    pausas_cortas: PausasCortasConfig
    almuerzo: AlmuerzoConfig
    solver: SolverConfig
    cobertura: CoberturaConfig

    class Config:
        json_schema_extra = {
            "example": {
                "turnos": {
                    "duracion_min_horas": 4,
                    "duracion_max_horas": 9,
                    "duracion_total_max_horas": 10,
                    "descanso_entre_turnos_horas": 8,
                    "dias_libres_min_por_semana": 1,
                    "horas_min_por_semana": 35,
                    "horas_max_por_semana": 50
                },
                "pausas_cortas": {
                    "duracion_minutos": 15,
                    "frecuencia_cada_horas": 3,
                    "obligatorias": True,
                    "separacion_minima_horas": 2.5,
                    "prohibir_primera_hora": True,
                    "prohibir_ultima_hora": True,
                    "maximo_por_turno": 4
                },
                "almuerzo": {
                    "duracion_min_minutos": 30,
                    "duracion_max_minutos": 60,
                    "obligatorio_si_turno_mayor_a_horas": 0,
                    "maximo_por_turno": 1,
                    "prohibir_primera_hora": True,
                    "prohibir_ultima_hora": True
                },
                "solver": {
                    "timeout_segundos": 300,
                    "num_workers": 8
                },
                "cobertura": {
                    "margen_seguridad": 1.2,
                    "bloquear_si_deficit": True
                }
            }
        }


class SystemPromptConfig(BaseModel):
    """System prompt configuration for LLM."""
    system_prompt: str = Field(..., min_length=1, description="Multi-line system prompt text")
    model: str = Field(..., min_length=1, description="LLM model name (e.g., gpt-4)")
    temperature: float = Field(..., ge=0.0, le=2.0, description="Temperature parameter (0.0-2.0)")
    top_p: float = Field(..., ge=0.0, le=1.0, description="Top-p parameter (0.0-1.0)")
    frequency_penalty: float = Field(..., ge=-2.0, le=2.0, description="Frequency penalty (-2.0 to 2.0)")
    presence_penalty: float = Field(..., ge=-2.0, le=2.0, description="Presence penalty (-2.0 to 2.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "system_prompt": "You are an AI assistant...",
                "model": "gpt-4",
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        }


app = FastAPI(
    title="Shift Optimizer API",
    description="API for optimizing call center shift assignments using constraint programming",
    version="1.0.0"
)

# CORS Middleware (allow all origins for simplicity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ===== HELPER FUNCTIONS =====

def calculate_dashboard_metrics() -> List[Dict[str, Any]]:
    """
    Calculate dashboard metrics based on agents, demand, and schedules.

    Returns list of metric objects with title, value, description.
    Calculates weekly averages for all metrics.

    Raises:
        HTTPException: If required files are missing
    """
    agents_path = Path("data/agentes.csv")
    demand_path = Path("data/demanda.csv")
    schedules_path = Path("output/schedules.json")
    resumen_path = Path("output/resumen.json")

    if not agents_path.exists():
        raise HTTPException(status_code=404, detail="Agents file not found")
    if not demand_path.exists():
        raise HTTPException(status_code=404, detail="Demand file not found")
    if not schedules_path.exists():
        raise HTTPException(status_code=404, detail="Schedules not found. Run /generate-schedule first")

    agents = load_agents(str(agents_path))
    demanda = load_demanda(str(demand_path))

    with open(schedules_path, 'r', encoding='utf-8') as f:
        schedules = json.load(f)

    total_agentes = len(agents)

    max_demanda = max(
        demanda.data[day][block]
        for day in range(7)
        for block in range(48)
    )

    total_demanda = 0
    total_cobertura = 0

    if resumen_path.exists():
        with open(resumen_path, 'r', encoding='utf-8') as f:
            resumen = json.load(f)

        for dia_data in resumen.get('cobertura_por_dia', []):
            total_demanda += dia_data.get('demanda', 0)
            total_cobertura += dia_data.get('cobertura', 0)

    if total_demanda > 0:
        nivel_servicio = (total_cobertura / total_demanda) * 100
    else:
        nivel_servicio = 100.0

    return [
        {
            "title": "Agentes Disponibles",
            "value": str(total_agentes),
            "description": "Agentes activos en el sistema"
        },
        {
            "title": "Agentes Requeridos",
            "value": str(max_demanda),
            "description": "Según demanda semanal"
        },
        {
            "title": "Nivel de Servicio",
            "value": f"{nivel_servicio:.1f}%",
            "description": "Cobertura de demanda"
        }
    ]


def get_coverage_by_day(day: int) -> List[Dict[str, Any]]:
    """
    Get demand and coverage data for a specific day.

    Args:
        day: Day of week (0=Monday, 6=Sunday)

    Returns:
        List of 48 time blocks with demand and coverage data

    Raises:
        HTTPException: If required files are missing or day is invalid
    """
    if not 0 <= day <= 6:
        raise HTTPException(status_code=400, detail="Day must be between 0 (Monday) and 6 (Sunday)")

    demand_path = Path("data/demanda.csv")
    schedules_path = Path("output/schedules.json")
    resumen_path = Path("output/resumen.json")

    if not demand_path.exists():
        raise HTTPException(status_code=404, detail="Demand file not found")
    if not schedules_path.exists():
        raise HTTPException(status_code=404, detail="Schedules not found. Run /generate-schedule first")

    demanda = load_demanda(str(demand_path))

    cobertura_matrix = [[0 for _ in range(48)] for _ in range(7)]

    if resumen_path.exists():
        with open(schedules_path, 'r', encoding='utf-8') as f:
            schedules = json.load(f)

        day_names = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']

        for agent in schedules:
            schedule = agent.get('schedule', {})
            day_name = day_names[day]

            if day_name in schedule:
                day_schedule = schedule[day_name]

                if day_schedule and 'start' in day_schedule and 'end' in day_schedule:
                    start_time = day_schedule['start']
                    end_time = day_schedule['end']

                    start_hour, start_min = map(int, start_time.split(':'))
                    end_hour, end_min = map(int, end_time.split(':'))

                    start_block = (start_hour * 60 + start_min) // 30
                    end_block = (end_hour * 60 + end_min) // 30

                    breaks_set = set()
                    for brk in day_schedule.get('break', []):
                        brk_start = brk.get('start', '')
                        if brk_start:
                            brk_hour, brk_min = map(int, brk_start.split(':'))
                            brk_block = (brk_hour * 60 + brk_min) // 30
                            breaks_set.add(brk_block)

                    disconnected_set = set()
                    for disc in day_schedule.get('disconnected', []):
                        disc_start = disc.get('start', '')
                        disc_end = disc.get('end', '')
                        if disc_start and disc_end:
                            disc_start_hour, disc_start_min = map(int, disc_start.split(':'))
                            disc_end_hour, disc_end_min = map(int, disc_end.split(':'))
                            disc_start_block = (disc_start_hour * 60 + disc_start_min) // 30
                            disc_end_block = (disc_end_hour * 60 + disc_end_min) // 30
                            for blk in range(disc_start_block, disc_end_block):
                                disconnected_set.add(blk)

                    for block in range(start_block, end_block):
                        if block not in breaks_set and block not in disconnected_set:
                            cobertura_matrix[day][block] += 1

    result = []
    for block in range(48):
        tb = TimeBlock(day, block)
        block_time = tb.to_time()
        time_str = f"{block_time.hour:02d}:{block_time.minute:02d}"

        result.append({
            "time": time_str,
            "block": block,
            "demanda": demanda.data[day][block],
            "disponibilidad": cobertura_matrix[day][block]
        })

    return result


# ============================================================================
# SECTION 1: HEALTH & MONITORING
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================================================
# SECTION 2: DASHBOARD & ANALYTICS
# ============================================================================

@app.get("/dashboard")
async def get_dashboard(day: Optional[int] = None) -> Dict[str, Any]:
    """
    Get dashboard metrics and optionally chart data for a specific day.

    This endpoint unifies /dashboard-metrics and /demand-coverage-chart into a single call.
    Returns always the 3 main metrics, and optionally the 48 time blocks for a specific day.

    **Query Parameters:**
    - **day** (optional): Day of week (0=Monday, 6=Sunday). If provided, includes chart_data in response.

    **Example Request (metrics only):**
    ```
    GET /dashboard
    ```

    **Example Request (metrics + chart for Monday):**
    ```
    GET /dashboard?day=0
    ```

    **Example Response (without day parameter):**
    ```json
    {
      "metrics": [
        {
          "title": "Agentes Disponibles",
          "value": "8",
          "description": "Agentes activos en el sistema"
        },
        {
          "title": "Agentes Requeridos",
          "value": "24",
          "description": "Según demanda semanal"
        },
        {
          "title": "Nivel de Servicio",
          "value": "95.5%",
          "description": "Cobertura de demanda"
        }
      ]
    }
    ```

    **Example Response (with day=0):**
    ```json
    {
      "metrics": [
        {
          "title": "Agentes Disponibles",
          "value": "8",
          "description": "Agentes activos en el sistema"
        },
        {
          "title": "Agentes Requeridos",
          "value": "24",
          "description": "Según demanda semanal"
        },
        {
          "title": "Nivel de Servicio",
          "value": "95.5%",
          "description": "Cobertura de demanda"
        }
      ],
      "chart_data": {
        "day": 0,
        "day_name": "lunes",
        "blocks": [
          {
            "time": "00:00",
            "block": 0,
            "demanda": 3,
            "disponibilidad": 2
          },
          ...48 blocks total
        ]
      }
    }
    ```

    Returns:
        Dictionary with metrics (always) and chart_data (if day parameter provided)

    Raises:
        400: Invalid day parameter (must be 0-6)
        404: Required files not found
        500: Failed to calculate metrics or coverage
    """
    try:
        # Validate day parameter if provided
        if day is not None and (day < 0 or day > 6):
            raise HTTPException(
                status_code=400,
                detail="Day must be between 0 (Monday) and 6 (Sunday)"
            )

        # Calculate metrics (always included)
        metrics = calculate_dashboard_metrics()

        # Prepare base response
        response = {"metrics": metrics}

        # If day requested, add chart data
        if day is not None:
            day_names = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
            chart_blocks = get_coverage_by_day(day)

            response["chart_data"] = {
                "day": day,
                "day_name": day_names[day],
                "blocks": chart_blocks
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve dashboard data: {str(e)}"
        )



# ============================================================================
# SECTION 3: SCHEDULE GENERATION & OPTIMIZATION
# ============================================================================

def _build_unified_response(
    status: str,
    agents: List,
    demanda,
    resumen: Optional[Dict] = None,
    deficit_agents: int = 0
) -> Dict[str, Any]:
    """
    Construye respuesta unificada para /generate-schedule.

    Todos los casos de respuesta siguen el mismo esquema:
    - status: Estado del proceso
    - agentes_disponibles: Cantidad total de agentes
    - agentes_faltantes: Agentes adicionales necesarios
    - cobertura_por_dia: Array de cobertura por día (vacío si no hay resumen)

    Args:
        status: Estado del proceso
        agents: Lista de agentes disponibles
        demanda: Objeto de demanda
        resumen: Resumen del solver (None si no se ejecutó)
        deficit_agents: Agentes faltantes (para caso insufficient_coverage)
    """
    agentes_disponibles = len(agents)

    # Calcular agentes faltantes según el caso
    if status == "insufficient_agents":
        max_demand = max(max(day) for day in demanda.data)
        config = OptimizationConfig.from_yaml('config/reglas.yaml')
        agentes_faltantes = int(max_demand * config.coverage.margen_seguridad) - agentes_disponibles
    elif status == "insufficient_coverage":
        agentes_faltantes = deficit_agents
    elif resumen:
        agentes_faltantes = resumen.get('agentes_faltantes', 0)
    else:
        agentes_faltantes = 0

    # Cobertura por día (vacío si no hay resumen)
    cobertura_por_dia = []
    if resumen and 'cobertura_por_dia' in resumen:
        cobertura_por_dia = resumen['cobertura_por_dia']

    return {
        "status": status,
        "agentes_disponibles": agentes_disponibles,
        "agentes_faltantes": max(0, agentes_faltantes),
        "cobertura_por_dia": cobertura_por_dia
    }


async def _optimize_chunks_parallel(
    chunks: List[List[Dict[str, Any]]],
    original_schedule: List[Dict[str, Any]],
    system_prompt: str,
    api_key: str,
    model: str,
    temperature: float,
    use_toon: bool
) -> tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Optimiza múltiples chunks en paralelo usando asyncio.gather().

    Args:
        chunks: Lista de chunks de agentes
        original_schedule: Schedule completo original (para contexto global)
        system_prompt: Prompt del sistema
        api_key: OpenAI API key
        model: Modelo LLM a usar
        temperature: Temperatura del modelo
        use_toon: Si se usa formato TOON

    Returns:
        (optimized_chunks, chunk_stats): Chunks optimizados y estadísticas
    """
    from shift_optimizer.llm_optimizer import (
        optimize_chunk_with_llm,
        build_global_context,
        validate_schedule_structure
    )

    async def optimize_single_chunk(chunk_index: int, chunk: List[Dict[str, Any]]):
        """Optimiza un chunk individual y retorna resultado con estadísticas."""
        chunk_ids = [agent['id'] for agent in chunk]
        chunk_num = chunk_index + 1

        print(f"🚀 Chunk {chunk_num}/{len(chunks)}: Procesando {len(chunk)} agentes ({', '.join(chunk_ids[:3])}...)")

        try:
            # Construir contexto global
            global_context = build_global_context(original_schedule, chunk_ids)

            # Optimizar chunk (async)
            optimized_chunk = await optimize_chunk_with_llm(
                chunk_data=chunk,
                system_prompt=system_prompt,
                global_context=global_context,
                api_key=api_key,
                model=model,
                temperature=temperature,
                use_toon=use_toon
            )

            # Validar resultado
            is_valid, error_msg = validate_schedule_structure(optimized_chunk)
            if not is_valid:
                print(f"⚠️ Chunk {chunk_num} retornó estructura inválida: {error_msg}. Usando original.")
                return chunk, {
                    "chunk": chunk_num,
                    "agents": len(chunk),
                    "status": "fallback_to_original",
                    "reason": error_msg
                }

            print(f"✅ Chunk {chunk_num} optimizado exitosamente")
            return optimized_chunk, {
                "chunk": chunk_num,
                "agents": len(optimized_chunk),
                "status": "optimized"
            }

        except Exception as e:
            print(f"❌ Error en chunk {chunk_num}: {e}. Usando original.")
            return chunk, {
                "chunk": chunk_num,
                "agents": len(chunk),
                "status": "error",
                "error": str(e)
            }

    # Ejecutar todos los chunks en paralelo
    print(f"\n🔄 Procesando {len(chunks)} chunks en paralelo...")
    results = await asyncio.gather(*[
        optimize_single_chunk(i, chunk) for i, chunk in enumerate(chunks)
    ])

    # Separar chunks optimizados y estadísticas
    optimized_chunks = [result[0] for result in results]
    chunk_stats = [result[1] for result in results]

    return optimized_chunks, chunk_stats


@app.post("/generate-schedule")
async def generate_schedule() -> Dict[str, Any]:
    """
    Generate shift schedule using CP-SAT solver.

    Uses data/agentes.csv and data/demanda.csv with config/reglas.yaml.
    Performs deficit analysis first, then runs solver if feasible.

    **No Request Body Required** - Uses files from data/ directory.

    **Unified Response Schema** - All cases return the same structure:
    ```json
    {
      "status": "OPTIMAL | DEFICIT | SUPERHABIT | insufficient_agents | insufficient_coverage",
      "agentes_disponibles": 8,
      "agentes_faltantes": 0,
      "cobertura_por_dia": [
        {
          "dia": 0,
          "demanda": 57,
          "cobertura": 57
        }
      ]
    }
    ```

    **Status Values:**
    - `OPTIMAL`: Perfect coverage achieved
    - `DEFICIT`: Some time blocks have less coverage than demand
    - `SUPERHABIT`: More coverage than needed in some blocks
    - `insufficient_agents`: Not enough agents to meet minimum requirements
    - `insufficient_coverage`: Critical gaps detected before optimization

    **Fields:**
    - `status`: Current state of the schedule
    - `agentes_disponibles`: Total number of available agents
    - `agentes_faltantes`: Additional agents needed (0 if sufficient)
    - `cobertura_por_dia`: Coverage array (empty if generation failed)

    Returns:
        Unified schedule summary with status and coverage information

    Raises:
        404: Required files not found
        500: Optimization failed
    """
    try:
        agents_path = "data/agentes.csv"
        demand_path = "data/demanda.csv"
        config_path = "config/reglas.yaml"
        output_dir = "output/"

        # Verify files exist
        for path, name in [(agents_path, "Agents"), (demand_path, "Demand"), (config_path, "Config")]:
            if not Path(path).exists():
                raise HTTPException(status_code=404, detail=f"{name} file not found: {path}")

        # Load config and data
        config = OptimizationConfig.from_yaml(config_path)
        agents = load_agents(agents_path)
        demanda = load_demanda(demand_path)

        # STEP 0: Pre-validation - Check if enough agents exist
        max_demand = max(max(day) for day in demanda.data)
        min_agents_needed = int(max_demand * config.coverage.margen_seguridad)

        if len(agents) < min_agents_needed:
            return _build_unified_response(
                status="insufficient_agents",
                agents=agents,
                demanda=demanda
            )

        # STEP 1: Analyze coverage
        analyzer = DeficitAnalyzer(agents, demanda)
        deficit_report = analyzer.analyze()

        # STEP 2: Block if critical deficit
        if config.coverage.bloquear_si_deficit and deficit_report.has_critical_gaps:
            deficit_agents_needed = deficit_report.recommendation.get('additional_agents_needed', 0)
            return _build_unified_response(
                status="insufficient_coverage",
                agents=agents,
                demanda=demanda,
                deficit_agents=deficit_agents_needed
            )

        # STEP 3: Run CP-SAT solver
        optimizer = ShiftOptimizer.from_config(config_path)
        result = optimizer.optimize(
            agents_path=agents_path,
            demand_path=demand_path,
            output_dir=output_dir
        )

        # Read and return summary
        resumen_path = Path(output_dir) / "resumen.json"
        if resumen_path.exists():
            with open(resumen_path, 'r', encoding='utf-8') as f:
                resumen = json.load(f)

            return _build_unified_response(
                status=resumen.get('status', 'OPTIMAL'),
                agents=agents,
                demanda=demanda,
                resumen=resumen
            )
        else:
            return _build_unified_response(
                status=result.get('status', 'UNKNOWN'),
                agents=agents,
                demanda=demanda
            )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schedule generation failed: {str(e)}")



@app.post("/optimize-schedule-llm")
async def optimize_schedule_llm(
    temperature: float = 0.3,
    chunk_size: int = 10,
    use_toon: bool = False
) -> Dict[str, Any]:
    """
    Optimiza el schedule usando OpenAI procesando chunks en paralelo.

    Args:
        temperature: Temperatura del modelo LLM (0-1, default: 0.3)
        chunk_size: Número de agentes por chunk (default: 10)
        use_toon: Si True, usa formato TOON (default: False, no mejora con esta estructura JSON)

    Returns:
        Resumen de la optimización con estadísticas

    Nota:
        - Procesa schedules.json en chunks de N agentes para evitar límites de tokens
        - Procesa todos los chunks EN PARALELO usando asyncio.gather()
        - Tiempo estimado: ~18-22s para 50 agentes (5 chunks) - 73% más rápido que secuencial
        - Cada chunk se optimiza independientemente manteniendo contexto global
    """
    try:
        from shift_optimizer.llm_optimizer import (
            chunk_agents,
            merge_optimized_chunks,
            validate_schedule_structure,
            save_optimized_schedule
        )

        schedule_path = Path("output/schedules.json")
        prompt_path = Path("config/system_prompt.yaml")

        if not schedule_path.exists():
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {schedule_path}")
        if not prompt_path.exists():
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {prompt_path}")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

        # Cargar configuración y schedule original
        prompt_config = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        system_prompt = prompt_config.get("system_prompt", "")
        model = prompt_config.get("model", "gpt-4o-mini")

        with open(schedule_path, 'r', encoding='utf-8') as f:
            original_schedule = json.load(f)

        # Validar estructura original
        is_valid, error_msg = validate_schedule_structure(original_schedule)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Schedule original inválido: {error_msg}")

        # Dividir en chunks
        chunks = chunk_agents(original_schedule, chunk_size=chunk_size)
        total_chunks = len(chunks)

        print(f"📦 Dividiendo {len(original_schedule)} agentes en {total_chunks} chunks de ~{chunk_size} agentes")
        print(f"🔧 Usando modelo: {model}, temperatura: {temperature}, TOON: {use_toon}")

        # Optimizar chunks en paralelo usando asyncio
        optimized_chunks, chunk_stats = await _optimize_chunks_parallel(
            chunks=chunks,
            original_schedule=original_schedule,
            system_prompt=system_prompt,
            api_key=api_key,
            model=model,
            temperature=temperature,
            use_toon=use_toon
        )

        # Fusionar chunks optimizados
        print("\n🔗 Fusionando chunks optimizados...")
        final_schedule = merge_optimized_chunks(optimized_chunks)

        # Validar schedule final
        is_valid, error_msg = validate_schedule_structure(final_schedule)
        if not is_valid:
            raise HTTPException(
                status_code=500,
                detail=f"Schedule fusionado es inválido: {error_msg}"
            )

        # Guardar schedule optimizado
        save_optimized_schedule(final_schedule, schedule_path)
        print(f"💾 Schedule optimizado guardado en {schedule_path}")

        # Preparar respuesta
        successful_optimizations = sum(1 for stat in chunk_stats if stat['status'] == 'optimized')

        return {
            "status": "success",
            "message": f"Schedule optimizado en {total_chunks} chunks",
            "total_agents": len(final_schedule),
            "chunks_processed": total_chunks,
            "chunks_optimized": successful_optimizations,
            "chunks_fallback": total_chunks - successful_optimizations,
            "chunk_details": chunk_stats,
            "config": {
                "model": model,
                "temperature": temperature,
                "chunk_size": chunk_size,
                "use_toon": use_toon
            },
            "output_file": str(schedule_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ Error inesperado: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Fallo al optimizar con LLM: {e}")



@app.post("/update-schedule")
async def update_schedule(csv_content: str = Body(..., media_type="text/plain")) -> Dict[str, Any]:
    """
    Manually update schedule from CSV.

    Receives schedule in CSV format and overwrites output/schedules.json.

    **Request Body (text/plain):**
    Send the CSV content as plain text in the request body.

    **CSV Format (Long Format):**
    ```
    agent_id,agent_name,day,shift_start,shift_end,break_start,break_end,disconnected_start,disconnected_end
    A001,Juan Pérez,lunes,08:00,17:30,09:30,09:45,,
    A001,Juan Pérez,lunes,08:00,17:30,13:00,13:15,,
    A001,Juan Pérez,martes,09:00,18:30,09:00,09:15,,
    A002,María García,lunes,10:00,17:00,10:00,10:15,,
    ...
    ```

    **Column Descriptions:**
    - **agent_id**: Agent unique identifier (e.g., "A001")
    - **agent_name**: Agent full name
    - **day**: Day name (lunes, martes, miercoles, jueves, viernes, sabado, domingo)
    - **shift_start**: Shift start time (HH:MM format)
    - **shift_end**: Shift end time (HH:MM format)
    - **break_start**: Break start time (optional, leave empty if no break in this row)
    - **break_end**: Break end time (optional, leave empty if no break in this row)
    - **disconnected_start**: Disconnected period start time (optional, leave empty if no lunch in this row)
    - **disconnected_end**: Disconnected period end time (optional, leave empty if no lunch in this row)

    **Data Model:**
    - Each row can represent: shift info only, shift + break, or shift + disconnected period
    - Multiple rows with same agent_id + day = same shift with multiple breaks/disconnected periods
    - Empty break/disconnected fields = no break/disconnected in that row

    **Validation Rules:**
    - Required headers: agent_id, agent_name, day, shift_start, shift_end
    - **agent_id**: Non-empty string
    - **agent_name**: Non-empty string
    - **day**: One of: lunes, martes, miercoles, jueves, viernes, sabado, domingo
    - **shift_start**: Non-empty time in HH:MM format
    - **shift_end**: Non-empty time in HH:MM format

    **Example Request (curl):**
    ```bash
    curl -X POST http://localhost:8000/update-schedule \\
      -H "Content-Type: text/plain" \\
      --data-binary @schedules.csv
    ```

    **Example Request (JavaScript fetch):**
    ```javascript
    const csvContent = "agent_id,agent_name,day,shift_start,shift_end,break_start,break_end,disconnected_start,disconnected_end\\nA001,Juan Pérez,lunes,08:00,17:30,09:30,09:45,,\\n";

    fetch('http://localhost:8000/update-schedule', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: csvContent
    });
    ```

    **Example Response:**
    ```json
    {
      "status": "success",
      "message": "Schedule updated successfully",
      "statistics": {
        "total_agents": 8,
        "total_shifts": 30
      }
    }
    ```

    Returns:
        Success message with statistics

    Raises:
        400: Invalid CSV format or data
        500: Failed to write file
    """
    try:
        # Validate CSV using parser
        try:
            schedule = parse_schedules_csv(csv_content)
        except CSVParseError as e:
            raise HTTPException(
                status_code=400,
                detail=f"CSV validation error: {str(e)}"
            )

        # Save schedule as JSON
        schedules_path = Path("output/schedules.json")
        schedules_path.parent.mkdir(exist_ok=True)
        with open(schedules_path, 'w', encoding='utf-8') as f:
            json.dump(schedule, f, ensure_ascii=False, indent=2)

        # Calculate stats
        total_agents = len(schedule)
        total_shifts = sum(
            len([day for day in agent.get('schedule', {}).values() if day])
            for agent in schedule
        )

        return {
            "status": "success",
            "message": "Schedule updated successfully",
            "statistics": {
                "total_agents": total_agents,
                "total_shifts": total_shifts
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schedule update failed: {str(e)}")


@app.get("/schedules")
async def get_schedules() -> List[Dict[str, Any]]:
    """
    Get all agent schedules.

    Returns the complete schedule from output/schedules.json which must be
    generated first by calling /generate-schedule.

    **No Request Body Required**

    **Example Response:**
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
          "martes": {
            "start": "08:00",
            "end": "17:30",
            "break": [
              {
                "start": "10:30",
                "end": "10:45"
              }
            ],
            "disconnected": []
          }
        }
      },
      {
        "id": "A002",
        "name": "María García",
        "schedule": {
          "lunes": {
            "start": "09:00",
            "end": "18:00",
            "break": [],
            "disconnected": []
          }
        }
      }
    ]
    ```

    Returns:
        List of all agent schedules

    Raises:
        404: Schedules not found (run /generate-schedule first)
        500: Failed to read schedules
    """
    try:
        schedules_path = Path("output/schedules.json")

        # Check if schedules file exists
        if not schedules_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Schedules not found. Please run /generate-schedule first."
            )

        # Load and return all schedules
        with open(schedules_path, 'r', encoding='utf-8') as f:
            schedules = json.load(f)

        return schedules

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid schedules file format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve schedules: {str(e)}"
        )



# ============================================================================
# SECTION 4: CONFIGURATION MANAGEMENT
# ============================================================================

@app.get("/get-rules", response_model=ReglasYAML)
async def get_rules() -> ReglasYAML:
    """
    Get current optimization rules configuration from config/reglas.yaml.

    Returns the complete configuration as JSON, excluding the read-only
    'time_blocks' section which should never be modified.

    Returns:
        ReglasYAML: Current optimization rules configuration

    Example Response:
        {
          "turnos": {
            "duracion_min_horas": 4,
            "duracion_max_horas": 9,
            ...
          },
          "pausas_cortas": {...},
          "almuerzo": {...},
          "solver": {...},
          "cobertura": {...}
        }
    """
    try:
        config_path = Path("config/reglas.yaml")

        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Configuration file not found: config/reglas.yaml"
            )

        # Load YAML file
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Extract configuration sections (exclude time_blocks)
        config_data = {
            'turnos': data.get('turnos', {}),
            'pausas_cortas': data.get('pausas_cortas', {}),
            'almuerzo': data.get('almuerzo', {}),
            'solver': data.get('solver', {}),
            'cobertura': data.get('cobertura', {})
        }

        # Validate and return using Pydantic model
        return ReglasYAML(**config_data)

    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing YAML file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load configuration: {str(e)}"
        )


@app.post("/update-rules")
async def update_rules(config: ReglasYAML) -> Dict[str, Any]:
    """
    Update optimization rules configuration in config/reglas.yaml.

    This endpoint validates the new configuration and overwrites the
    existing reglas.yaml file. The 'time_blocks' section is preserved
    and never modified.

    **Request Body (JSON):**
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

    Returns:
        Success message with updated configuration

    Raises:
        400: Invalid configuration data
        500: Failed to write file
    """
    try:
        config_path = Path("config/reglas.yaml")

        # Read existing YAML to preserve comments and time_blocks
        existing_data = {}
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f)

        # Build new YAML data (preserve time_blocks from existing)
        new_data = {
            'turnos': config.turnos.dict(),
            'pausas_cortas': config.pausas_cortas.dict(),
            'almuerzo': config.almuerzo.dict(),
            'solver': config.solver.dict(),
            'cobertura': config.cobertura.dict(),
            'time_blocks': existing_data.get('time_blocks', {'duracion_minutos': 30})
        }

        # Write updated YAML file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "file": str(config_path),
            "config": config.dict()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )


@app.get("/get-prompt", response_model=SystemPromptConfig)
async def get_prompt() -> SystemPromptConfig:
    """
    Get current system prompt configuration from config/system_prompt.yaml.

    Returns the LLM configuration including system prompt text and model parameters.

    Returns:
        SystemPromptConfig: Current system prompt configuration

    Example Response:
        {
          "system_prompt": "You are an AI assistant...",
          "model": "gpt-4",
          "temperature": 0.7,
          "top_p": 1.0,
          "frequency_penalty": 0.0,
          "presence_penalty": 0.0
        }
    """
    try:
        config_path = Path("config/system_prompt.yaml")

        if not config_path.exists():
            raise HTTPException(
                status_code=404,
                detail="System prompt file not found: config/system_prompt.yaml"
            )

        # Load YAML file
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Validate and return using Pydantic model
        return SystemPromptConfig(**data)

    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing YAML file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load system prompt: {str(e)}"
        )


@app.post("/update-prompt")
async def update_prompt(config: SystemPromptConfig) -> Dict[str, Any]:
    """
    Update system prompt configuration in config/system_prompt.yaml.

    This endpoint validates the new configuration and overwrites the
    existing system_prompt.yaml file.

    **Request Body (JSON):**
    ```json
    {
      "system_prompt": "You are an advanced AI assistant...",
      "model": "gpt-4",
      "temperature": 0.7,
      "top_p": 1.0,
      "frequency_penalty": 0.0,
      "presence_penalty": 0.0
    }
    ```

    **Field Descriptions:**
    - **system_prompt**: Multi-line text with instructions for the LLM
    - **model**: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
    - **temperature**: Randomness (0.0 = deterministic, 2.0 = very random)
    - **top_p**: Nucleus sampling (0.0-1.0)
    - **frequency_penalty**: Penalize frequent tokens (-2.0 to 2.0)
    - **presence_penalty**: Penalize repeated topics (-2.0 to 2.0)

    Returns:
        Success message with updated configuration

    Raises:
        400: Invalid configuration data
        500: Failed to write file
    """
    try:
        config_path = Path("config/system_prompt.yaml")

        # Build YAML data with literal block scalar for multi-line prompt
        data = {
            'system_prompt': config.system_prompt,
            'model': config.model,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'frequency_penalty': config.frequency_penalty,
            'presence_penalty': config.presence_penalty
        }

        # Write YAML file with literal style for system_prompt
        with open(config_path, 'w', encoding='utf-8') as f:
            # Write system_prompt as literal block scalar
            f.write("system_prompt: |\n")
            for line in config.system_prompt.split('\n'):
                f.write(f"  {line}\n")
            f.write("\n")

            # Write other parameters
            f.write(f"model: {config.model}\n")
            f.write(f"temperature: {config.temperature}\n")
            f.write(f"top_p: {config.top_p}\n")
            f.write(f"frequency_penalty: {config.frequency_penalty}\n")
            f.write(f"presence_penalty: {config.presence_penalty}\n")

        return {
            "status": "success",
            "message": "System prompt updated successfully",
            "file": str(config_path),
            "config": config.dict()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update system prompt: {str(e)}"
        )



# ============================================================================
# SECTION 5: DATA MANAGEMENT (CSV)
# ============================================================================

@app.get("/get-agents-csv", response_class=PlainTextResponse)
async def get_agents_csv() -> str:
    """
    Get current agents data from data/agentes.csv.

    Returns the CSV file content as plain text.

    **CSV Format (Simple Format):**
    ```csv
    id,nombre
    A001,Juan Pérez
    A002,María García
    A003,Carlos Rodríguez
    ...
    ```

    **Column Descriptions:**
    - **id**: Agent unique identifier (e.g., "A001")
    - **nombre**: Agent full name

    **Data Model:**
    - Each row represents one agent (available 24/7 by default)
    - One row per agent

    Returns:
        Plain text CSV content

    Raises:
        404: CSV file not found
        500: Failed to read file
    """
    try:
        csv_path = Path("data/agentes.csv")

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Agents CSV file not found: data/agentes.csv"
            )

        # Read and return CSV content
        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return content

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read agents CSV: {str(e)}"
        )


@app.post("/update-agents-csv")
async def update_agents_csv(csv_content: str = Body(..., media_type="text/plain")) -> Dict[str, Any]:
    """
    Update agents data in data/agentes.csv.

    This endpoint validates the CSV format and data, then overwrites the
    existing agentes.csv file. All agents are considered available 24/7.

    **Request Body (text/plain):**
    Send the CSV content as plain text in the request body.

    **CSV Format (Simple Format):**
    ```
    id,nombre
    A001,Juan Pérez
    A002,María García
    A003,Carlos Rodríguez
    ...
    ```

    **Validation Rules:**
    - Required headers: id, nombre
    - **id**: Non-empty string
    - **nombre**: Non-empty string
    - No duplicate agent IDs

    **Example Request (curl):**
    ```bash
    curl -X POST http://localhost:8000/update-agents-csv \\
      -H "Content-Type: text/plain" \\
      --data-binary @agentes.csv
    ```

    **Example Request (JavaScript fetch):**
    ```javascript
    const csvContent = "id,nombre\\nA001,Juan Pérez\\nA002,María García\\n";

    fetch('http://localhost:8000/update-agents-csv', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: csvContent
    });
    ```

    **Example Response:**
    ```json
    {
      "status": "success",
      "message": "Agents CSV updated successfully",
      "file": "data/agentes.csv",
      "statistics": {
        "total_agents": 8,
        "agent_ids": ["A001", "A002", "A003", "A004", "A005", "A006", "A007", "A008"]
      }
    }
    ```

    Returns:
        Success message with statistics about uploaded data

    Raises:
        400: Invalid CSV format or data
        500: Failed to write file
    """
    try:
        # Validate CSV using existing parser
        try:
            agents = parse_agents_csv(csv_content)
        except CSVParseError as e:
            raise HTTPException(
                status_code=400,
                detail=f"CSV validation error: {str(e)}"
            )

        # Write CSV to file
        csv_path = Path("data/agentes.csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)

        # Calculate statistics
        agent_ids = [agent.id for agent in agents]

        return {
            "status": "success",
            "message": "Agents CSV updated successfully",
            "file": str(csv_path),
            "statistics": {
                "total_agents": len(agents),
                "agent_ids": agent_ids
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update agents CSV: {str(e)}"
        )


@app.get("/get-demand-csv", response_class=PlainTextResponse)
async def get_demand_csv() -> str:
    """
    Get current demand data from data/demanda.csv.

    Returns the CSV file content as plain text.

    **CSV Format (Long Format):**
    ```csv
    dia,bloque,agentes_requeridos
    0,14,1
    0,15,1
    0,16,2
    ...
    ```

    **Column Descriptions:**
    - **dia**: Day of week (0=Monday, 6=Sunday)
    - **bloque**: 30-minute time block (0-47, where 0=00:00-00:30, 47=23:30-24:00)
    - **agentes_requeridos**: Number of agents needed in that time block

    **Data Model:**
    - Each row represents demand for ONE 30-minute block
    - Only blocks with demand > 0 need to be included (blocks with 0 demand can be omitted)

    Returns:
        Plain text CSV content

    Raises:
        404: CSV file not found
        500: Failed to read file
    """
    try:
        csv_path = Path("data/demanda.csv")

        if not csv_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Demand CSV file not found: data/demanda.csv"
            )

        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return content

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read demand CSV: {str(e)}"
        )


@app.post("/update-demand-csv")
async def update_demand_csv(csv_content: str = Body(..., media_type="text/plain")) -> Dict[str, Any]:
    """
    Update demand data in data/demanda.csv.

    This endpoint validates the CSV format and data, then overwrites the
    existing demanda.csv file.

    **Request Body (text/plain):**
    Send the CSV content as plain text in the request body.

    **CSV Format (Long Format):**
    ```
    dia,bloque,agentes_requeridos
    0,14,1
    0,15,1
    0,16,2
    0,17,2
    ...
    ```

    **Validation Rules:**
    - Required headers: dia, bloque, agentes_requeridos
    - **dia**: Integer between 0-6 (0=Monday, 6=Sunday)
    - **bloque**: Integer between 0-47 (0=00:00, 47=23:30)
    - **agentes_requeridos**: Non-negative integer
    - No duplicate rows (same dia, bloque combination)

    **Example Request (curl):**
    ```bash
    curl -X POST http://localhost:8000/update-demand-csv \\
      -H "Content-Type: text/plain" \\
      --data-binary @demanda.csv
    ```

    **Example Request (JavaScript fetch):**
    ```javascript
    const csvContent = "dia,bloque,agentes_requeridos\\n0,14,1\\n0,15,1\\n...";

    fetch('http://localhost:8000/update-demand-csv', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: csvContent
    });
    ```

    **Example Response:**
    ```json
    {
      "status": "success",
      "message": "Demand CSV updated successfully",
      "file": "data/demanda.csv",
      "statistics": {
        "total_demand_blocks": 143,
        "total_agents_required": 347,
        "days_with_demand": 7
      }
    }
    ```

    Returns:
        Success message with statistics about uploaded data

    Raises:
        400: Invalid CSV format or data
        500: Failed to write file
    """
    try:
        # Validate CSV using existing parser
        try:
            from shift_optimizer.csv_parsers import parse_demanda_csv
            demanda = parse_demanda_csv(csv_content)
        except CSVParseError as e:
            raise HTTPException(
                status_code=400,
                detail=f"CSV validation error: {str(e)}"
            )

        # Write CSV to file
        csv_path = Path("data/demanda.csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)

        # Calculate statistics
        total_demand = sum(sum(day) for day in demanda.data)
        days_with_demand = sum(1 for day in demanda.data if sum(day) > 0)

        # Count non-zero blocks
        total_blocks = sum(
            1 for day in demanda.data for block in day if block > 0
        )

        return {
            "status": "success",
            "message": "Demand CSV updated successfully",
            "file": str(csv_path),
            "statistics": {
                "total_demand_blocks": total_blocks,
                "total_agents_required": total_demand,
                "days_with_demand": days_with_demand
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update demand CSV: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
