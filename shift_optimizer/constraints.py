"""
Constraint classes for shift optimization.

Each constraint encapsulates a specific rule that must be enforced
in the shift scheduling problem.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solver import ShiftOptimizer
    from .config import OptimizationConfig


class Constraint(ABC):
    """Base class for all constraints."""

    def __init__(self, config: 'OptimizationConfig'):
        """
        Initialize constraint with configuration.

        Args:
            config: Optimization configuration
        """
        self.config = config

    @abstractmethod
    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        Apply this constraint to the optimization model.

        Args:
            optimizer: ShiftOptimizer instance with model and variables
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable name of this constraint."""
        pass


class AvailabilityConstraint(Constraint):
    """Agents can only work when available (no-op since all agents are 24/7)."""

    @property
    def name(self) -> str:
        return "Agent Availability"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """No constraints needed - all agents available 24/7."""
        pass


class ShiftDurationConstraint(Constraint):
    """Enforces shift duration limits (4-9 hours effective work)."""

    @property
    def name(self) -> str:
        return "Shift Duration (Effective Work)"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        Shifts must be 4-9 hours effective work.
        BREAKS count as effective work.
        HUECO (lunch) does NOT count as effective work.
        """
        min_blocks = int(self.config.shift_rules.duracion_min_horas * 2)  # 2 blocks = 1 hour
        max_blocks = int(self.config.shift_rules.duracion_max_horas * 2)

        for agent in optimizer.agents:
            for day in range(7):
                work_blocks = []
                hueco_blocks = []

                for block in range(48):
                    work_blocks.append(optimizer.x[agent.id, day, block])
                    hueco_blocks.append(optimizer.hueco[agent.id, day, block])

                total_work = sum(work_blocks)
                total_hueco = sum(hueco_blocks)

                # Effective work = work blocks - hueco (breaks NOT subtracted)
                optimizer.model.Add(total_work - total_hueco >= min_blocks).OnlyEnforceIf(
                    optimizer.shift_active[agent.id, day]
                )
                optimizer.model.Add(total_work - total_hueco <= max_blocks).OnlyEnforceIf(
                    optimizer.shift_active[agent.id, day]
                )

                # If no shift, no work
                optimizer.model.Add(total_work == 0).OnlyEnforceIf(
                    optimizer.shift_active[agent.id, day].Not()
                )

                # Link shift_active and day_off
                optimizer.model.Add(
                    optimizer.shift_active[agent.id, day] + optimizer.day_off[agent.id, day] <= 1
                )

                # Breaks and hueco only when working
                for block in range(48):
                    optimizer.model.Add(
                        optimizer.breaks[agent.id, day, block] <= optimizer.x[agent.id, day, block]
                    )
                    optimizer.model.Add(
                        optimizer.hueco[agent.id, day, block] <= optimizer.x[agent.id, day, block]
                    )

                # Link shift_start and shift_end with work blocks
                for block in range(48):
                    optimizer.model.Add(optimizer.shift_start[agent.id, day] <= block).OnlyEnforceIf(
                        optimizer.x[agent.id, day, block]
                    )
                    optimizer.model.Add(optimizer.shift_end[agent.id, day] >= block).OnlyEnforceIf(
                        optimizer.x[agent.id, day, block]
                    )


class ShiftSpanConstraint(Constraint):
    """Enforces maximum shift span (start to end)."""

    @property
    def name(self) -> str:
        return "Maximum Shift Span"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """Maximum shift span (start to end) is configured hours."""
        max_span_blocks = int(self.config.shift_rules.duracion_total_max_horas * 2) - 1

        for agent in optimizer.agents:
            for day in range(7):
                span = optimizer.model.NewIntVar(0, 47, f'span_{agent.id}_{day}')
                optimizer.model.Add(span == optimizer.shift_end[agent.id, day] - optimizer.shift_start[agent.id, day])
                optimizer.model.Add(span <= max_span_blocks).OnlyEnforceIf(
                    optimizer.shift_active[agent.id, day]
                )


class RestBetweenShiftsConstraint(Constraint):
    """Enforces minimum rest period between consecutive shifts."""

    @property
    def name(self) -> str:
        return "Rest Between Shifts"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """Minimum rest hours between end of one shift and start of next."""
        min_rest_blocks = int(self.config.shift_rules.descanso_entre_turnos_horas * 2)

        for agent in optimizer.agents:
            for day in range(6):  # Days 0-5
                both_days_active = optimizer.model.NewBoolVar(f'both_{agent.id}_{day}')
                optimizer.model.AddBoolAnd([
                    optimizer.shift_active[agent.id, day],
                    optimizer.shift_active[agent.id, day + 1]
                ]).OnlyEnforceIf(both_days_active)

                # If both days active: end_day + min_rest <= start_day+1 + 48
                optimizer.model.Add(
                    optimizer.shift_end[agent.id, day] + min_rest_blocks <=
                    optimizer.shift_start[agent.id, day + 1] + 48
                ).OnlyEnforceIf(both_days_active)


class WeeklyDayOffConstraint(Constraint):
    """Enforces minimum days off per week."""

    @property
    def name(self) -> str:
        return "Weekly Day Off"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """Each agent must have at least configured days off per week."""
        min_days_off = self.config.shift_rules.dias_libres_min_por_semana

        for agent in optimizer.agents:
            days_off = [optimizer.day_off[agent.id, day] for day in range(7)]
            optimizer.model.Add(sum(days_off) >= min_days_off)


class BreakConstraint(Constraint):
    """Enforces break rules (15-min breaks every X hours) - ROBUST IMPLEMENTATION."""

    @property
    def name(self) -> str:
        return "Break Periods (Mandatory)"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        ROBUST break constraint implementation:

        1. First/last hour protection: Direct constraints (no weak boolean logic)
        2. Break frequency: Global constraint based on total effective work hours
        3. Minimum separation: Prevent breaks from clustering
        4. Maximum breaks: Configurable limit per shift

        Mathematical formulation:
        - effective_work_blocks = sum(x[block]) - sum(hueco[block])
        - min_breaks_required = ceil(effective_work_blocks / freq_blocks) if obligatory
        - breaks only allowed in [shift_start + 2, shift_end - 2]
        """
        freq_blocks = int(self.config.breaks.frecuencia_cada_horas * 2)  # 3 hours = 6 blocks
        separation_blocks = int(self.config.breaks.separacion_minima_horas * 2)  # 2.5 hours = 5 blocks
        max_breaks = self.config.breaks.maximo_por_turno

        for agent in optimizer.agents:
            for day in range(7):
                # === CONSTRAINT 1: First/Last Hour Protection - REINTRODUCED ===
                # Breaks cannot occur in first or last hour of shift.
                # Using direct constraint approach: breaks only allowed in [start+2, end-2]
                if self.config.breaks.prohibir_primera_hora or self.config.breaks.prohibir_ultima_hora:
                    for block in range(48):
                        # If break at this block, ensure it's not in first/last hour
                        break_at_block = optimizer.breaks[agent.id, day, block]

                        if self.config.breaks.prohibir_primera_hora:
                            # Break at block => block >= shift_start + 2
                            optimizer.model.Add(
                                block >= optimizer.shift_start[agent.id, day] + 2
                            ).OnlyEnforceIf(break_at_block)

                        if self.config.breaks.prohibir_ultima_hora:
                            # Break at block => block <= shift_end - 2
                            optimizer.model.Add(
                                block <= optimizer.shift_end[agent.id, day] - 2
                            ).OnlyEnforceIf(break_at_block)

                # === CONSTRAINT 2: Maximum Breaks Per Shift ===
                total_breaks = sum(optimizer.breaks[agent.id, day, block] for block in range(48))
                optimizer.model.Add(total_breaks <= max_breaks)

                # === CONSTRAINT 2.5: Breaks and Lunch are Mutually Exclusive ===
                # A block cannot have both a break AND lunch
                for block in range(48):
                    optimizer.model.Add(
                        optimizer.breaks[agent.id, day, block] + optimizer.hueco[agent.id, day, block] <= 1
                    )

                # === CONSTRAINT 3: Minimum Separation Between Consecutive Breaks ===
                # If break at block i, then no breaks at blocks [i+1, i+separation_blocks]
                for block in range(48 - separation_blocks):
                    for next_block in range(1, separation_blocks + 1):
                        if block + next_block < 48:
                            optimizer.model.Add(
                                optimizer.breaks[agent.id, day, block + next_block] == 0
                            ).OnlyEnforceIf(optimizer.breaks[agent.id, day, block])

                # === CONSTRAINT 4: Mandatory Break Frequency (DIRECT FORMULATION) ===
                # SIMPLIFIED: Use direct mathematical relationship without intermediate booleans
                if self.config.breaks.obligatorias:
                    total_work = sum(optimizer.x[agent.id, day, block] for block in range(48))
                    total_hueco = sum(optimizer.hueco[agent.id, day, block] for block in range(48))

                    # Effective work blocks (excludes lunch, includes breaks since they're paid time)
                    effective_work = total_work - total_hueco

                    # DIRECT FORMULATION: total_breaks * freq_blocks >= effective_work
                    # This ensures: required_breaks = ceil(effective_work / freq_blocks)
                    #
                    # Equivalent to: if work > 0, then breaks >= ceil(work / freq)
                    # Mathematical: freq * breaks >= work
                    #                  breaks >= ceil(work / freq)
                    #
                    # We add freq-1 to ensure ceil behavior:
                    # freq * breaks >= work - 1 + freq
                    # breaks >= (work + freq - 1) / freq  (which is ceil division)

                    # To ensure breaks >= ceil(effective_work / freq_blocks):
                    # Multiply both sides by freq_blocks:
                    # breaks * freq_blocks >= effective_work
                    # This is equivalent but works directly in CP-SAT
                    # Create explicit IntVar for effective work
                    effective_work_var = optimizer.model.NewIntVar(0, 48, f'eff_work_{agent.id}_{day}')
                    optimizer.model.Add(effective_work_var == effective_work)

                    # EXPLICIT CASCADING CONSTRAINTS to force ceil behavior
                    # If work >= threshold, MUST have at least N breaks
                    # This prevents solver from reducing work to avoid breaks
                    for n_breaks in range(1, 5):
                        threshold = (n_breaks - 1) * freq_blocks + 1
                        # If effective_work >= threshold, force total_breaks >= n_breaks
                        work_exceeds_threshold = optimizer.model.NewBoolVar(f'work_exceeds_{n_breaks}_{agent.id}_{day}')
                        optimizer.model.Add(effective_work_var >= threshold).OnlyEnforceIf(work_exceeds_threshold)
                        optimizer.model.Add(effective_work_var < threshold).OnlyEnforceIf(work_exceeds_threshold.Not())
                        optimizer.model.Add(total_breaks >= n_breaks).OnlyEnforceIf(work_exceeds_threshold)


class BreakDistributionConstraint(Constraint):
    """Enforces maximum gap between breaks (sliding window approach)."""

    @property
    def name(self) -> str:
        return "Break Distribution (Max 3h gap)"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        ROBUST break distribution constraint using sliding window.

        Prevents long gaps without breaks by enforcing:
        - Every 3-hour window with continuous work must have at least 1 break

        Mathematical formulation:
        - Window size: 6 blocks (3 hours)
        - For each window [i, i+6]: if working continuously, must have >= 1 break
        - Working continuously means: sum(x[blocks]) - sum(hueco[blocks]) >= 4 blocks (2h minimum)

        This prevents scenarios like: break at 06:00, then work until 11:00 without break (5h gap).
        """
        window_size = 6  # 3 hours = 6 blocks of 30 min
        min_work_threshold = 4  # Need at least 2 hours of work to require break

        for agent in optimizer.agents:
            for day in range(7):
                # For each possible window position
                for start_block in range(48 - window_size + 1):
                    window_blocks = range(start_block, start_block + window_size)

                    # Count work blocks in window (excluding lunch)
                    work_in_window = sum(
                        optimizer.x[agent.id, day, block] for block in window_blocks
                    )
                    hueco_in_window = sum(
                        optimizer.hueco[agent.id, day, block] for block in window_blocks
                    )
                    effective_work_in_window = work_in_window - hueco_in_window

                    # Count breaks in window
                    breaks_in_window = sum(
                        optimizer.breaks[agent.id, day, block] for block in window_blocks
                    )

                    # If working >= threshold in this window, must have at least 1 break
                    window_requires_break = optimizer.model.NewBoolVar(
                        f'window_requires_break_{agent.id}_{day}_{start_block}'
                    )

                    # Create IntVar for effective work (needed for comparison)
                    effective_work_var = optimizer.model.NewIntVar(
                        0, window_size,
                        f'effective_work_window_{agent.id}_{day}_{start_block}'
                    )
                    optimizer.model.Add(effective_work_var == effective_work_in_window)

                    # window_requires_break = (effective_work >= threshold)
                    optimizer.model.Add(
                        effective_work_var >= min_work_threshold
                    ).OnlyEnforceIf(window_requires_break)
                    optimizer.model.Add(
                        effective_work_var < min_work_threshold
                    ).OnlyEnforceIf(window_requires_break.Not())

                    # If window requires break, enforce at least 1 break
                    optimizer.model.Add(
                        breaks_in_window >= 1
                    ).OnlyEnforceIf(window_requires_break)


class LunchConstraint(Constraint):
    """Enforces lunch break rules (30-60 min) - ROBUST IMPLEMENTATION."""

    @property
    def name(self) -> str:
        return "Lunch Break (Almuerzo/Hueco)"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        ROBUST lunch constraint implementation:

        1. Duration limits: BIDIRECTIONAL implication (if hueco > 0, must be in [min, max])
        2. First/last hour protection: Direct constraints (no weak boolean logic)
        3. Consecutivity: All hueco blocks must be consecutive (no gaps)
        4. Maximum per shift: Configurable limit

        Mathematical formulation:
        - total_hueco = sum(hueco[block])
        - total_hueco = 0 OR (min_blocks <= total_hueco <= max_blocks)
        - hueco only allowed in [shift_start + 2, shift_end - 2]
        - If hueco[i] = 1 and hueco[i+1] = 0, then hueco[j] = 0 for all j > i+1 (consecutivity)
        """
        min_lunch_blocks = self.config.lunch.duracion_min_minutos // 30  # 30 min = 1 block
        max_lunch_blocks = self.config.lunch.duracion_max_minutos // 30  # 60 min = 2 blocks
        max_lunches = self.config.lunch.maximo_por_turno

        for agent in optimizer.agents:
            for day in range(7):
                total_hueco = sum(optimizer.hueco[agent.id, day, block] for block in range(48))

                # === CONSTRAINT 1: Duration Limits (SIMPLIFIED & ROBUST) ===
                # total_hueco = 0 OR (min_blocks <= total_hueco <= max_blocks)
                # Enforced using direct constraint on allowable values

                # Approach: For each possible value, create boolean indicating if total = value
                # Then ensure only valid values can be true

                # Simpler approach: Use AddLinearConstraintWithBounds or direct constraints
                # total_hueco is either 0, or between min and max

                # Create boolean: has_any_lunch
                has_any_lunch = optimizer.model.NewBoolVar(f'has_any_lunch_{agent.id}_{day}')

                # has_any_lunch ⟷ (total_hueco > 0)
                optimizer.model.Add(total_hueco >= 1).OnlyEnforceIf(has_any_lunch)
                optimizer.model.Add(total_hueco == 0).OnlyEnforceIf(has_any_lunch.Not())

                # FORCE LUNCH if shift > obligatorio_si_turno_mayor_a_horas
                min_hours_for_lunch = self.config.lunch.obligatorio_si_turno_mayor_a_horas
                if min_hours_for_lunch > 0:
                    min_blocks_for_lunch = int(min_hours_for_lunch * 2)

                    # Calculate total work blocks in the day
                    total_day_work = sum(optimizer.x[agent.id, day, block] for block in range(48))

                    # Create boolean: work_exceeds_threshold
                    work_exceeds = optimizer.model.NewBoolVar(f'work_exceeds_lunch_{agent.id}_{day}')
                    optimizer.model.Add(total_day_work >= min_blocks_for_lunch).OnlyEnforceIf(work_exceeds)
                    optimizer.model.Add(total_day_work < min_blocks_for_lunch).OnlyEnforceIf(work_exceeds.Not())

                    # If work_exceeds AND shift_active, FORCE has_any_lunch = True
                    optimizer.model.Add(has_any_lunch == 1).OnlyEnforceIf(
                        [optimizer.shift_active[agent.id, day], work_exceeds]
                    )

                # If has_any_lunch, then min <= total <= max (CRUCIAL: use both bounds)
                # IMPORTANT: These MUST be enforced or solver will violate
                optimizer.model.Add(total_hueco >= min_lunch_blocks).OnlyEnforceIf(has_any_lunch)
                optimizer.model.Add(total_hueco <= max_lunch_blocks).OnlyEnforceIf(has_any_lunch)

                # ADDITIONAL SAFETY: Global upper bound (even when no lunch)
                # This prevents solver from assigning huge lunch blocks
                optimizer.model.Add(total_hueco <= max_lunch_blocks)

                # === CONSTRAINT 2: Maximum Lunches Per Shift ===
                # This is implicitly enforced by consecutivity, but we can add explicit constraint
                # Since lunches must be consecutive, having 1 lunch means one contiguous block
                # For now, we rely on consecutivity constraint below

                # === CONSTRAINT 3: First/Last Hour Protection for Lunch ===
                # Lunch cannot occur in first or last hour of shift.
                # Using direct constraint approach: lunch only allowed in [start+2, end-2]
                if self.config.lunch.prohibir_primera_hora or self.config.lunch.prohibir_ultima_hora:
                    for block in range(48):
                        # If lunch at this block, ensure it's not in first/last hour
                        lunch_at_block = optimizer.hueco[agent.id, day, block]

                        if self.config.lunch.prohibir_primera_hora:
                            # Lunch at block => block >= shift_start + 2
                            optimizer.model.Add(
                                block >= optimizer.shift_start[agent.id, day] + 2
                            ).OnlyEnforceIf(lunch_at_block)

                        if self.config.lunch.prohibir_ultima_hora:
                            # Lunch at block => block <= shift_end - 2 - max_lunch_blocks
                            # (to ensure entire lunch fits before last hour)
                            optimizer.model.Add(
                                block <= optimizer.shift_end[agent.id, day] - 2 - max_lunch_blocks
                            ).OnlyEnforceIf(lunch_at_block)

                # === CONSTRAINT 4: Consecutivity (SIMPLIFIED ROBUST VERSION) ===
                # All hueco blocks must be consecutive - no gaps allowed
                #
                # SIMPLE APPROACH: If hueco ends at block i (hueco[i]=1, hueco[i+1]=0),
                # then there can be no more hueco blocks after i+1.
                #
                # This ensures: 000011110000 (valid) but prevents: 00011100110 (invalid - gap)

                for block in range(47):
                    # Detect when lunch ends: hueco[block]=1 AND hueco[block+1]=0
                    lunch_ends_here = optimizer.model.NewBoolVar(f'lunch_ends_{agent.id}_{day}_{block}')

                    # lunch_ends_here = (hueco[block] AND NOT hueco[block+1])
                    optimizer.model.AddBoolAnd([
                        optimizer.hueco[agent.id, day, block],
                        optimizer.hueco[agent.id, day, block + 1].Not()
                    ]).OnlyEnforceIf(lunch_ends_here)

                    # If lunch ends here, all future blocks must have no lunch
                    for future_block in range(block + 2, 48):
                        optimizer.model.Add(
                            optimizer.hueco[agent.id, day, future_block] == 0
                        ).OnlyEnforceIf(lunch_ends_here)


class ShiftContinuityConstraint(Constraint):
    """Ensures shifts are continuous (no empty gaps)."""

    @property
    def name(self) -> str:
        return "Shift Continuity"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        Between shift_start and shift_end, all blocks must be work/break/hueco.
        No empty gaps.
        """
        for agent in optimizer.agents:
            for day in range(7):
                for block in range(48):
                    within_shift = optimizer.model.NewBoolVar(f'within_{agent.id}_{day}_{block}')

                    optimizer.model.Add(optimizer.shift_start[agent.id, day] <= block).OnlyEnforceIf(within_shift)
                    optimizer.model.Add(optimizer.shift_end[agent.id, day] >= block).OnlyEnforceIf(within_shift)
                    optimizer.model.Add(optimizer.x[agent.id, day, block] == 1).OnlyEnforceIf(within_shift)


class CoverageConstraint(Constraint):
    """Calculates coverage and tracks demand vs. coverage difference."""

    @property
    def name(self) -> str:
        return "Coverage Calculation"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        Calculate coverage for each time block.
        Coverage = agents working (not on break or lunch).
        """
        for day in range(7):
            for block in range(48):
                agents_working = []
                for agent in optimizer.agents:
                    working = optimizer.model.NewBoolVar(f'working_{agent.id}_{day}_{block}')
                    optimizer.model.Add(
                        optimizer.x[agent.id, day, block] -
                        optimizer.breaks[agent.id, day, block] -
                        optimizer.hueco[agent.id, day, block] == 1
                    ).OnlyEnforceIf(working)
                    optimizer.model.Add(
                        optimizer.x[agent.id, day, block] -
                        optimizer.breaks[agent.id, day, block] -
                        optimizer.hueco[agent.id, day, block] <= 0
                    ).OnlyEnforceIf(working.Not())
                    agents_working.append(working)

                optimizer.model.Add(optimizer.coverage[day, block] == sum(agents_working))

                # Coverage difference
                demand = optimizer.demanda.get_demand(day, block)
                optimizer.model.Add(
                    optimizer.coverage_diff[day, block] == optimizer.coverage[day, block] - demand
                )


class WeeklyHoursConstraint(Constraint):
    """
    Enforce minimum and maximum weekly hours per agent.

    Legal requirement: Minimum 46 hours per week per agent.
    Maximum 48 hours per week to prevent overwork.

    This ensures equitable distribution of work hours across all agents
    and prevents scenarios where some agents work very little while others
    are overloaded.
    """

    @property
    def name(self) -> str:
        return "Weekly Hours (Min/Max)"

    def apply(self, optimizer: 'ShiftOptimizer') -> None:
        """
        Apply weekly hours constraints.

        For each agent:
        - Total weekly work blocks >= min_hours * 2 (46h = 92 blocks)
        - Total weekly work blocks <= max_hours * 2 (48h = 96 blocks)

        Note: Blocks are 30 minutes each, so 1 hour = 2 blocks.
        """
        min_hours = self.config.shift_rules.horas_min_por_semana
        max_hours = self.config.shift_rules.horas_max_por_semana

        # Convert to blocks (30 min each)
        min_blocks = min_hours * 2  # 46h = 92 blocks
        max_blocks = max_hours * 2  # 48h = 96 blocks

        for agent in optimizer.agents:
            # Calculate EFFECTIVE work blocks (excluding breaks and lunch)
            # Effective work = x - breaks - lunch
            effective_blocks = []

            for day in range(7):
                for block in range(48):
                    # Create variable for effective work at this block
                    # effective = 1 if agent is working (not on break or lunch), 0 otherwise
                    effective = optimizer.model.NewIntVar(
                        0, 1,
                        f'effective_{agent.id}_{day}_{block}'
                    )

                    # Effective work = assigned block - break - lunch
                    optimizer.model.Add(
                        effective == optimizer.x[agent.id, day, block] -
                                     optimizer.breaks[agent.id, day, block] -
                                     optimizer.hueco[agent.id, day, block]
                    )

                    effective_blocks.append(effective)

            # Sum all effective work blocks
            total_effective_work = sum(effective_blocks)

            # Apply constraints on EFFECTIVE hours (not total span)
            optimizer.model.Add(total_effective_work >= min_blocks)  # Minimum 46h effective
            optimizer.model.Add(total_effective_work <= max_blocks)  # Maximum 48h effective
