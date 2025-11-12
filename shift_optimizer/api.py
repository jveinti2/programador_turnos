"""
Simplified API for shift optimization.

Provides a high-level interface for optimizing shift schedules.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any

from .config import OptimizationConfig
from .models import Agent, Demanda, Solution
from .solver import optimize_shifts
from .utils import load_agents, load_demanda, export_schedules_json, export_resumen_json


class ShiftOptimizer:
    """
    High-level API for shift optimization.

    Example usage:
        optimizer = ShiftOptimizer.from_config('config/reglas.yaml')
        result = optimizer.optimize(
            agents_path='data/agentes.json',
            demand_path='data/demanda.json'
        )
        print(f"Status: {result['status']}")
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize optimizer with configuration.

        Args:
            config: Optimization configuration
        """
        self.config = config

    @classmethod
    def from_config(cls, config_path: str = 'config/reglas.yaml') -> 'ShiftOptimizer':
        """
        Create optimizer from configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            ShiftOptimizer instance

        Example:
            optimizer = ShiftOptimizer.from_config('config/reglas.yaml')
        """
        config = OptimizationConfig.from_yaml(config_path)
        return cls(config)

    def optimize(
        self,
        agents_path: str,
        demand_path: str,
        output_dir: str = 'output',
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize shift assignments and export results.

        Args:
            agents_path: Path to agents JSON file
            demand_path: Path to demand JSON file
            output_dir: Directory for output files (default: 'output')
            timeout: Solver timeout in seconds (overrides config if provided)

        Returns:
            Dictionary with:
                - status: Solver status ('OPTIMAL', 'FEASIBLE', etc.)
                - schedules: List of agent schedules
                - summary: Summary statistics
                - solution_time: Time taken to solve (seconds)
                - output_files: Dict with paths to generated files

        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If input data is invalid

        Example:
            result = optimizer.optimize(
                agents_path='data/agentes.json',
                demand_path='data/demanda.json'
            )
            print(f"Optimized {result['summary']['total_turnos']} shifts")
        """
        # Validate input files
        if not os.path.exists(agents_path):
            raise FileNotFoundError(f"Agents file not found: {agents_path}")
        if not os.path.exists(demand_path):
            raise FileNotFoundError(f"Demand file not found: {demand_path}")

        # Load data
        print(f"Loading agents from {agents_path}...")
        agents = load_agents(agents_path)
        print(f"Loaded {len(agents)} agents.")

        print(f"Loading demand from {demand_path}...")
        demanda = load_demanda(demand_path)
        total_demand = sum(sum(day) for day in demanda.data)
        print(f"Loaded demand for 7 days (total demand: {total_demand}).")

        # Run optimization
        solver_timeout = timeout if timeout is not None else self.config.solver.timeout_segundos
        print(f"\nOptimizing with timeout of {solver_timeout} seconds...")
        print(f"Using configuration: {self.config.to_dict()}\n")

        solution = optimize_shifts(agents, demanda, solver_timeout, self.config)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Export results
        print("\nExporting results...")
        schedules_path = os.path.join(output_dir, 'schedules.json')
        export_schedules_json(solution, agents, schedules_path)
        print(f"  Agent schedules exported to: {schedules_path}")

        resumen_path = os.path.join(output_dir, 'resumen.json')
        export_resumen_json(solution, demanda, agents, resumen_path)
        print(f"  Summary exported to: {resumen_path}")

        # Build response
        return {
            'status': solution.status,
            'schedules': solution.shifts,
            'summary': {
                'status': solution.status,
                'solution_time': solution.tiempo_resolucion,
                'total_cost': solution.costo_total,
                'total_turnos': len(solution.shifts),
                'total_agentes_asignados': len(set(s.agente_id for s in solution.shifts)),
                'total_agentes_disponibles': len(agents),
                'demanda_total': total_demand,
                'cobertura_total': sum(sum(day) for day in solution.cobertura)
            },
            'solution_time': solution.tiempo_resolucion,
            'output_files': {
                'schedules': schedules_path,
                'summary': resumen_path
            }
        }

    # def optimize_from_data(
    #     self,
    #     agents: list,
    #     demanda: list,
    #     timeout: Optional[int] = None
    # ) -> Solution:
    #     """
    #     Optimize shift assignments from in-memory data.

    #     Args:
    #         agents: List of Agent objects
    #         demanda: Demanda object or list of lists (7 days x 48 blocks)
    #         timeout: Solver timeout in seconds (overrides config if provided)

    #     Returns:
    #         Solution object with assigned shifts

    #     Example:
    #         from shift_optimizer.models import Agent, Demanda, TimeBlock

    #         agents = [
    #             Agent(id="A001", nombre="Juan", disponibilidad=[...]),
    #             Agent(id="A002", nombre="María", disponibilidad=[...])
    #         ]
    #         demanda = Demanda(data=[[0]*48 for _ in range(7)])

    #         solution = optimizer.optimize_from_data(agents, demanda)
    #     """
    #     from .models import Demanda as DemandaModel

    #     # Convert demanda to Demanda object if needed
    #     if isinstance(demanda, list):
    #         demanda = DemandaModel(data=demanda)

    #     solver_timeout = timeout if timeout is not None else self.config.solver.timeout_segundos
    #     return optimize_shifts(agents, demanda, solver_timeout, self.config)
