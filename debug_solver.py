"""Debug script to understand why break constraints aren't working."""
from shift_optimizer import ShiftOptimizer
from shift_optimizer.utils import load_agents, load_demanda

# Load data
agents = load_agents('data/agentes_ejemplo_completo.json')
demanda = load_demanda('data/demanda_ejemplo.json')

# Create optimizer
optimizer = ShiftOptimizer.from_config('config/reglas.yaml')

# Build model (this applies all constraints)
print("Building model...")
from shift_optimizer.solver import ShiftOptimizer as SolverClass
solver_instance = SolverClass(agents, demanda, 300, optimizer.config)
solver_instance.build_model()

# Check constraints for A001 on day 0 (Monday)
agent_id = 'A001'
day = 0

print(f"\nChecking constraints for {agent_id} day {day}:")
print("=" * 60)

# Get the constraint from breaks
from ortools.sat.python import cp_model
model = solver_instance.model

# Try to find the specific constraints
# We can't directly inspect individual constraints, but we can check if model is solvable

print("Model has been built with all constraints.")
print("Attempting to solve...")

status = solver_instance.solver.Solve(model)

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print(f"\nSolution status: {solver_instance.solver.StatusName(status)}")

    # Check A001 Monday
    total_work = sum(solver_instance.solver.Value(solver_instance.x[agent_id, day, block]) for block in range(48))
    total_hueco = sum(solver_instance.solver.Value(solver_instance.hueco[agent_id, day, block]) for block in range(48))
    total_breaks = sum(solver_instance.solver.Value(solver_instance.breaks[agent_id, day, block]) for block in range(48))

    effective_work = total_work - total_hueco

    print(f"\n{agent_id} Monday:")
    print(f"  total_work blocks: {total_work}")
    print(f"  total_hueco blocks: {total_hueco}")
    print(f"  effective_work blocks: {effective_work}")
    print(f"  total_breaks: {total_breaks}")
    print(f"  effective_work hours: {effective_work * 0.5}")
    print(f"  required breaks (ceil({effective_work}/6)): {(effective_work + 5) // 6}")
    print(f"  Constraint: total_breaks * 6 >= effective_work")
    print(f"  Check: {total_breaks} * 6 = {total_breaks * 6} >= {effective_work}? {total_breaks * 6 >= effective_work}")

else:
    print(f"Solver failed with status: {solver_instance.solver.StatusName(status)}")
