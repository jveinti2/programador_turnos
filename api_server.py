"""
FastAPI server for shift optimization system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import json
from pathlib import Path

from shift_optimizer import ShiftOptimizer, DeficitAnalyzer
from shift_optimizer.utils import load_agents, load_demanda
from shift_optimizer.config import OptimizationConfig



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


@app.get("/analyze-coverage")
async def analyze_coverage() -> Dict[str, Any]:
    """
    Analyze coverage gaps between agent availability and demand.

    This endpoint performs a pre-optimization analysis to detect if there
    are sufficient agents to cover the demand. It returns:
    - Whether there are deficits
    - Specific time periods with gaps
    - Recommendations for additional agents

    Returns:
        DeficitReport with detailed analysis
    """
    try:
        # Load data
        agents_path = "data/agentes_ejemplo_completo.json"
        demand_path = "data/demanda_ejemplo.json"

        agents = load_agents(agents_path)
        demanda = load_demanda(demand_path)

        # Analyze coverage
        analyzer = DeficitAnalyzer(agents, demanda)
        report = analyzer.analyze()

        return report.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Coverage analysis failed: {str(e)}"
        )


@app.get("/optimize")
async def optimize_shifts() -> Dict[str, Any]:
    """
    Run shift optimization using example data and return summary.

    This endpoint:
    1. Analyzes coverage for deficits (NEW)
    2. Blocks optimization if critical deficit exists (configurable)
    3. Reads agents from data/agentes_ejemplo_completo.json
    4. Reads demand from data/demanda_ejemplo.json
    5. Runs the CP-SAT solver with configuration from config/reglas.yaml
    6. Saves results to output/schedules.json and output/resumen.json
    7. Returns the optimization summary (resumen.json content)

    Returns:
        Summary with status, total shifts, costs, and coverage statistics
    """
    try:
        # Define paths
        agents_path = "data/agentes_ejemplo_completo.json"
        demand_path = "data/demanda_ejemplo.json"
        config_path = "config/reglas.yaml"
        output_dir = "output/"

        # Verify input files exist
        if not Path(agents_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Agents file not found: {agents_path}"
            )
        if not Path(demand_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Demand file not found: {demand_path}"
            )
        if not Path(config_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Configuration file not found: {config_path}"
            )

        # Load configuration
        config = OptimizationConfig.from_yaml(config_path)

        # STEP 1: Analyze coverage BEFORE optimizing
        agents = load_agents(agents_path)
        demanda = load_demanda(demand_path)
        analyzer = DeficitAnalyzer(agents, demanda)
        deficit_report = analyzer.analyze()

        # STEP 2: Check if we should block optimization due to critical deficit
        if config.coverage.bloquear_si_deficit and deficit_report.has_critical_gaps:
            return {
                "status": "insufficient_coverage",
                "message": "No hay suficientes agentes para cubrir la demanda",
                "deficit_analysis": deficit_report.to_dict(),
                "recommendation": deficit_report.recommendation
            }

        # STEP 3: If OK (or blocking disabled), proceed with optimization
        optimizer = ShiftOptimizer.from_config(config_path)
        result = optimizer.optimize(
            agents_path=agents_path,
            demand_path=demand_path,
            output_dir=output_dir
        )

        # Read and return the summary (resumen.json)
        resumen_path = Path(output_dir) / "resumen.json"
        if resumen_path.exists():
            with open(resumen_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            # Add deficit info to summary if there were any gaps
            if deficit_report.has_deficit:
                summary['coverage_warnings'] = {
                    'has_gaps': True,
                    'total_deficit_hours': deficit_report.total_deficit_hours,
                    'recommendation': deficit_report.recommendation
                }

            return summary
        else:
            # Fallback: return basic result info
            return {
                "status": result['status'],
                "summary": result.get('summary'),
                "message": "Optimization completed but summary file not found"
            }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
