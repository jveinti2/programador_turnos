"""
FastAPI server for shift optimization system.
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import json
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from shift_optimizer import ShiftOptimizer, DeficitAnalyzer
from shift_optimizer.utils import load_agents, load_demanda
from shift_optimizer.config import OptimizationConfig
from shift_optimizer.csv_parsers import parse_agents_csv, CSVParseError


# ===== PYDANTIC MODELS FOR CONFIGURATION =====

class TurnosConfig(BaseModel):
    """Shift duration and timing rules."""
    duracion_min_horas: int = Field(..., ge=1, le=24, description="Minimum effective work hours")
    duracion_max_horas: int = Field(..., ge=1, le=24, description="Maximum effective work hours")
    duracion_total_max_horas: int = Field(..., ge=1, le=24, description="Maximum total shift span")
    descanso_entre_turnos_horas: int = Field(..., ge=0, le=24, description="Minimum rest between shifts")
    dias_libres_min_por_semana: int = Field(..., ge=0, le=7, description="Minimum days off per week")


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
                    "dias_libres_min_por_semana": 1
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
    temperature: float = Field(..., ge=0.0, le=2.0, description="Temperature parameter")
    top_p: float = Field(..., ge=0.0, le=1.0, description="Top-p parameter")
    frecuency_penalty: float = Field(..., ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(..., ge=-2.0, le=2.0, description="Presence penalty")

    class Config:
        json_schema_extra = {
            "example": {
                "system_prompt": "You are an AI assistant...",
                "model": "gpt-4",
                "temperature": 0.7,
                "top_p": 1.0,
                "frecuency_penalty": 0.0,
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


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/generate-schedule")
async def generate_schedule() -> Dict[str, Any]:
    """
    Generate shift schedule using CP-SAT solver.

    Uses data/agentes.csv and data/demanda.csv with config/reglas.yaml.
    Performs deficit analysis first, then runs solver if feasible.

    **No Request Body Required** - Uses files from data/ directory.

    **Example Response (Success):**
    ```json
    {
      "status": "OPTIMAL",
      "agentes_disponibles": 8,
      "agentes_faltantes": 0,
      "cobertura_por_dia": [
        {
          "dia": 0,
          "demanda": 57,
          "cobertura": 57
        },
        {
          "dia": 1,
          "demanda": 57,
          "cobertura": 57
        }
      ]
    }
    ```

    **Example Response (Insufficient Coverage):**
    ```json
    {
      "status": "insufficient_coverage",
      "message": "No hay suficientes agentes para cubrir la demanda",
      "deficit_analysis": {
        "has_deficit": true,
        "has_critical_gaps": true,
        "total_deficit_hours": 15.5,
        "gaps": [
          {
            "day": 6,
            "day_name": "Domingo",
            "start_time": "07:00",
            "end_time": "22:30",
            "required_agents": 3,
            "available_agents": 0,
            "deficit": 3
          }
        ]
      },
      "recommendation": {
        "message": "Se necesitan 3 agentes adicionales (CRÍTICO)",
        "additional_agents_needed": 3,
        "problematic_days": ["Domingo"],
        "severity": "critical"
      }
    }
    ```

    Returns:
        Schedule summary with status and coverage information

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

        # STEP 1: Analyze coverage
        analyzer = DeficitAnalyzer(agents, demanda)
        deficit_report = analyzer.analyze()

        # STEP 2: Block if critical deficit
        if config.coverage.bloquear_si_deficit and deficit_report.has_critical_gaps:
            return {
                "status": "insufficient_coverage",
                "message": "No hay suficientes agentes para cubrir la demanda",
                "deficit_analysis": deficit_report.to_dict(),
                "recommendation": deficit_report.recommendation
            }

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
                summary = json.load(f)

            if deficit_report.has_deficit:
                summary['coverage_warnings'] = {
                    'has_gaps': True,
                    'total_deficit_hours': deficit_report.total_deficit_hours,
                    'recommendation': deficit_report.recommendation
                }

            return summary
        else:
            return {
                "status": result['status'],
                "summary": result.get('summary'),
                "message": "Optimization completed but summary file not found"
            }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schedule generation failed: {str(e)}")


@app.post("/optimize-schedule-llm")
async def optimize_schedule_llm(
    temperature_override: Optional[float] = None,
    custom_instructions: Optional[str] = None
):
    """
    Post-process and improve existing schedule using LLM.

    Reads output/schedules.json, applies LLM-based optimizations to fix
    constraint violations or improve quality, and saves the improved schedule.

    This endpoint does NOT return the schedule. Use GET /schedules to retrieve it.
    """
    try:
        schedules_path = Path("output/schedules.json")
        prompt_config_path = Path("config/system_prompt.yaml")
        config_path = Path("config/reglas.yaml")

        # Verify schedule exists
        if not schedules_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No schedule found. Run /generate-schedule first."
            )

        # Load schedule
        with open(schedules_path, 'r', encoding='utf-8') as f:
            schedule = json.load(f)

        # Load LLM config
        if not prompt_config_path.exists():
            raise HTTPException(status_code=404, detail="LLM config not found: config/system_prompt.yaml")

        with open(prompt_config_path, 'r', encoding='utf-8') as f:
            llm_config = yaml.safe_load(f)

        # Load rules for context
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Rules config not found: config/reglas.yaml")

        with open(config_path, 'r', encoding='utf-8') as f:
            reglas = yaml.safe_load(f)

        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )

        # Import OpenAI (lazy import to avoid dependency if not used)
        try:
            from openai import OpenAI
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="OpenAI library not installed. Run: pip install openai"
            )

        # Build prompt with rules context
        system_prompt = llm_config.get('system_prompt', '')
        user_prompt = f"""Analiza el siguiente schedule de turnos y mejóralo post-procesándolo.

OBJETIVO:
- Validar que se cumplan TODAS las restricciones/reglas
- Corregir violaciones de constraints si las hay
- Mejorar la calidad del schedule (mejor distribución de descansos, almuerzos, etc.)

REGLAS DE NEGOCIO (deben cumplirse):
{json.dumps(reglas, indent=2, ensure_ascii=False)}

SCHEDULE ACTUAL:
{json.dumps(schedule, indent=2, ensure_ascii=False)}

INSTRUCCIONES ADICIONALES:
{custom_instructions or 'Verifica constraints y optimiza la distribución de descansos y almuerzos.'}

Retorna SOLO el schedule mejorado en formato JSON idéntico al original (array de agentes)."""

        # Call OpenAI
        client = OpenAI(api_key=api_key)
        temperature = temperature_override if temperature_override is not None else llm_config.get('temperature', 0.7)

        response = client.chat.completions.create(
            model=llm_config.get('model', 'gpt-4'),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            top_p=llm_config.get('top_p', 1.0),
            frequency_penalty=llm_config.get('frecuency_penalty', 0.0),
            presence_penalty=llm_config.get('presence_penalty', 0.0)
        )

        # Extract improved schedule
        llm_response = response.choices[0].message.content

        # Try to parse JSON from response
        try:
            # Find JSON in response (LLM might add explanatory text)
            import re
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                improved_schedule = json.loads(json_match.group(0))
            else:
                improved_schedule = json.loads(llm_response)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"LLM response is not valid JSON: {llm_response[:200]}"
            )

        # Save improved schedule (overwrite original)
        with open(schedules_path, 'w', encoding='utf-8') as f:
            json.dump(improved_schedule, f, ensure_ascii=False, indent=2)

        # Return 204 No Content (success but no body)
        from fastapi.responses import Response
        return Response(status_code=204)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM optimization failed: {str(e)}")


@app.put("/update-schedule")
async def update_schedule(schedule: List[Dict[str, Any]] = Body(...)) -> Dict[str, Any]:
    """
    Manually update schedule.

    Receives complete schedule and overwrites output/schedules.json.

    **Request Body (JSON):**
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
            "disconnected": []
          }
        }
      }
    ]
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
        400: Invalid schedule format
        500: Failed to write file
    """
    try:
        schedules_path = Path("output/schedules.json")

        # Basic validation
        if not isinstance(schedule, list):
            raise HTTPException(status_code=400, detail="Schedule must be an array")

        for agent in schedule:
            if 'id' not in agent or 'name' not in agent or 'schedule' not in agent:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid agent format. Required: id, name, schedule"
                )

        # Save schedule
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


@app.get("/schedule/{agent_id}")
async def get_agent_schedule(agent_id: str) -> Dict[str, Any]:
    """
    Get the schedule for a specific agent.

    This endpoint reads the agent's schedule from output/schedules.json
    which must be generated first by calling the /optimize endpoint.

    Args:
        agent_id: The agent's unique identifier (e.g., "AG001")

    Returns:
        Agent schedule with shifts for the week

    Example:
        GET /schedule/AG001
    """
    try:
        schedules_path = Path("output/schedules.json")

        # Check if schedules file exists
        if not schedules_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Schedules not found. Please run /optimize first to generate schedules."
            )

        # Load schedules (it's a list of agent objects)
        with open(schedules_path, 'r', encoding='utf-8') as f:
            schedules = json.load(f)

        # Find the agent's schedule by searching in the list
        for agent in schedules:
            if agent.get('id') == agent_id:
                return agent

        # Agent not found
        available_ids = [agent.get('id') for agent in schedules]
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found in schedules. Available agents: {available_ids}"
        )

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
            detail=f"Failed to retrieve schedule: {str(e)}"
        )


# ===== CONFIGURATION MANAGEMENT ENDPOINTS =====

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
          "frecuency_penalty": 0.0,
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
      "frecuency_penalty": 0.0,
      "presence_penalty": 0.0
    }
    ```

    **Field Descriptions:**
    - **system_prompt**: Multi-line text with instructions for the LLM
    - **model**: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
    - **temperature**: Randomness (0.0 = deterministic, 2.0 = very random)
    - **top_p**: Nucleus sampling (0.0-1.0)
    - **frecuency_penalty**: Penalize frequent tokens (-2.0 to 2.0)
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
            'frecuency_penalty': config.frecuency_penalty,
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
            f.write(f"frecuency_penalty: {config.frecuency_penalty}\n")
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
