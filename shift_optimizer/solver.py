"""
OR-Tools CP-SAT solver for shift optimization.
"""
import time as time_module
from datetime import time
from typing import List, Dict, Tuple, Optional
from ortools.sat.python import cp_model
from .models import Agent, Demanda, Solution, Shift, TimeBlock
from .config import OptimizationConfig
from .constraints import (
    Constraint,
    AvailabilityConstraint,
    ShiftDurationConstraint,
    ShiftSpanConstraint,
    RestBetweenShiftsConstraint,
    WeeklyDayOffConstraint,
    BreakConstraint,
    LunchConstraint,
    ShiftContinuityConstraint,
    CoverageConstraint
)


class ShiftOptimizer:
    """Optimizes shift assignments using OR-Tools CP-SAT solver."""

    def __init__(self, agents: List[Agent], demanda: Demanda, timeout: int = 300, config: Optional[OptimizationConfig] = None):
        """
        Initialize the optimizer.

        Args:
            agents: List of available agents
            demanda: Staffing demand requirements
            timeout: Solver timeout in seconds
            config: Optimization configuration (if None, uses defaults)
        """
        self.agents = agents
        self.demanda = demanda
        self.timeout = timeout
        self.config = config if config is not None else self._default_config()
        self.model = cp_model.CpModel()

        # Decision variables
        self.x = {}  # x[agent_id, day, block] = 1 if agent works at that time (48 blocks)
        self.shift_active = {}  # shift_active[agent_id, day] = 1 if agent has shift that day
        self.day_off = {}  # day_off[agent_id, day] = 1 if agent has day off
        self.breaks = {}  # breaks[agent_id, day, block] = 1 if agent has 15min break in that 30min block
        self.hueco = {}  # hueco[agent_id, day, block] = 1 if agent has lunch break at that time
        self.shift_start = {}  # shift_start[agent_id, day] = block where shift starts
        self.shift_end = {}  # shift_end[agent_id, day] = block where shift ends
        self.coverage = {}  # coverage[day, block] = number of agents working
        self.coverage_diff = {}  # Difference between demand and coverage

        # Constraint instances
        self.constraints: List[Constraint] = self._build_constraints()

    def _default_config(self) -> OptimizationConfig:
        """Create default configuration if none provided."""
        return OptimizationConfig.from_yaml('config/reglas.yaml')

    def _build_constraints(self) -> List[Constraint]:
        """Build list of all constraints to be applied."""
        return [
            AvailabilityConstraint(self.config),
            ShiftDurationConstraint(self.config),
            ShiftSpanConstraint(self.config),
            RestBetweenShiftsConstraint(self.config),
            WeeklyDayOffConstraint(self.config),
            BreakConstraint(self.config),
            LunchConstraint(self.config),
            ShiftContinuityConstraint(self.config),
            CoverageConstraint(self.config)
        ]

    def build_model(self):
        """Build the constraint programming model using configured constraints."""
        print("Building optimization model...")

        # Create decision variables
        self._create_variables()

        # Apply all constraints
        print("Applying constraints:")
        for constraint in self.constraints:
            print(f"  - {constraint.name}")
            constraint.apply(self)

        # Set objective: minimize sum of absolute coverage differences
        self._set_objective()

        print("Model built successfully.")

    def _create_variables(self):
        """Create all decision variables."""
        # Work variables
        for agent in self.agents:
            for day in range(7):
                self.shift_active[agent.id, day] = self.model.NewBoolVar(f'shift_{agent.id}_{day}')
                self.day_off[agent.id, day] = self.model.NewBoolVar(f'dayoff_{agent.id}_{day}')
                self.shift_start[agent.id, day] = self.model.NewIntVar(0, 47, f'start_{agent.id}_{day}')
                self.shift_end[agent.id, day] = self.model.NewIntVar(0, 47, f'end_{agent.id}_{day}')

                for block in range(48):  # 48 blocks of 30 minutes
                    self.x[agent.id, day, block] = self.model.NewBoolVar(f'x_{agent.id}_{day}_{block}')
                    self.breaks[agent.id, day, block] = self.model.NewBoolVar(f'break_{agent.id}_{day}_{block}')
                    self.hueco[agent.id, day, block] = self.model.NewBoolVar(f'hueco_{agent.id}_{day}_{block}')

        # Coverage variables
        for day in range(7):
            for block in range(48):  # 48 blocks of 30 minutes
                max_coverage = len(self.agents)
                self.coverage[day, block] = self.model.NewIntVar(0, max_coverage, f'cov_{day}_{block}')
                self.coverage_diff[day, block] = self.model.NewIntVar(
                    -max_coverage, max_coverage, f'diff_{day}_{block}'
                )

    def _set_objective(self):
        """Minimize sum of absolute coverage differences."""
        abs_diffs = []
        for day in range(7):
            for block in range(48):  # 48 blocks of 30 minutes
                abs_diff = self.model.NewIntVar(0, len(self.agents), f'abs_diff_{day}_{block}')
                self.model.AddAbsEquality(abs_diff, self.coverage_diff[day, block])
                abs_diffs.append(abs_diff)

        self.model.Minimize(sum(abs_diffs))

    def solve(self) -> Solution:
        """
        Solve the optimization model.

        Returns:
            Solution object with assigned shifts and coverage
        """
        print(f"Solving with timeout of {self.timeout} seconds...")
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.timeout
        solver.parameters.num_search_workers = self.config.solver.num_workers  # Use configured workers

        start_time = time_module.time()
        status = solver.Solve(self.model)
        elapsed_time = time_module.time() - start_time

        status_name = solver.StatusName(status)
        print(f"Solver status: {status_name}")
        print(f"Solution time: {elapsed_time:.2f} seconds")

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver, elapsed_time, status_name)
        else:
            print("No solution found!")
            return Solution(
                shifts=[],
                cobertura=[[0 for _ in range(48)] for _ in range(7)],
                costo_total=float('inf'),
                tiempo_resolucion=elapsed_time,
                status=status_name
            )

    def _extract_solution(self, solver: cp_model.CpSolver, elapsed_time: float, status: str) -> Solution:
        """Extract solution from solver."""
        print("Extracting solution...")

        shifts = []
        cobertura = [[0 for _ in range(48)] for _ in range(7)]

        # Extract shifts for each agent
        for agent in self.agents:
            for day in range(7):
                if solver.Value(self.shift_active[agent.id, day]):
                    shift = self._extract_shift(solver, agent.id, day)
                    if shift:
                        shifts.append(shift)

        # Extract coverage
        for day in range(7):
            for block in range(48):  # 48 blocks of 30 minutes
                cobertura[day][block] = solver.Value(self.coverage[day, block])

        # Calculate total cost
        total_cost = 0.0
        for day in range(7):
            for block in range(48):  # 48 blocks of 30 minutes
                diff = solver.Value(self.coverage_diff[day, block])
                total_cost += diff ** 2

        print(f"Total shifts assigned: {len(shifts)}")
        print(f"Total cost (sum of squared differences): {total_cost:.2f}")

        return Solution(
            shifts=shifts,
            cobertura=cobertura,
            costo_total=total_cost,
            tiempo_resolucion=elapsed_time,
            status=status
        )

    def _extract_shift(self, solver: cp_model.CpSolver, agent_id: str, day: int) -> Shift:
        """Extract shift details for a specific agent and day."""
        work_blocks = []
        break_blocks = []
        hueco_blocks = []

        for block in range(48):  # 48 blocks of 30 minutes
            if solver.Value(self.x[agent_id, day, block]):
                work_blocks.append(block)
            if solver.Value(self.breaks[agent_id, day, block]):
                break_blocks.append(block)
            if solver.Value(self.hueco[agent_id, day, block]):
                hueco_blocks.append(block)

        if not work_blocks:
            return None

        # Calculate start and end times
        start_block = min(work_blocks)
        end_block = max(work_blocks) + 1  # +1 because end time is exclusive

        hora_inicio = TimeBlock(day, start_block).to_time()
        hora_fin = TimeBlock(day, end_block).to_time()

        # Extract break times (15min breaks)
        breaks = [TimeBlock(day, b).to_time() for b in break_blocks]

        # Extract hueco times
        hueco = None
        if hueco_blocks:
            hueco_inicio = TimeBlock(day, min(hueco_blocks)).to_time()
            hueco_fin = TimeBlock(day, max(hueco_blocks) + 1).to_time()
            hueco = (hueco_inicio, hueco_fin)

        return Shift(
            agente_id=agent_id,
            dia=day,
            hora_inicio=hora_inicio,
            hora_fin=hora_fin,
            breaks=breaks,
            hueco=hueco
        )


def optimize_shifts(agents: List[Agent], demanda: Demanda, timeout: int = 300, config: Optional[OptimizationConfig] = None) -> Solution:
    """
    Main function to optimize shift assignments.

    Args:
        agents: List of available agents
        demanda: Staffing demand requirements
        timeout: Solver timeout in seconds
        config: Optimization configuration (if None, uses defaults)

    Returns:
        Solution object with assigned shifts and coverage
    """
    optimizer = ShiftOptimizer(agents, demanda, timeout, config)
    optimizer.build_model()
    return optimizer.solve()
