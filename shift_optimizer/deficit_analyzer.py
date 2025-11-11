"""
Deficit analyzer for shift optimization.

Analyzes coverage gaps between agent availability and demand.
Provides recommendations for additional agents needed.
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
from .models import Agent, Demanda


@dataclass
class CoverageGap:
    """Represents a coverage gap at a specific time."""
    day: int
    day_name: str
    start_block: int
    end_block: int
    start_time: str
    end_time: str
    required_agents: int
    available_agents: int
    deficit: int


@dataclass
class DeficitReport:
    """Complete deficit analysis report."""
    has_deficit: bool
    has_critical_gaps: bool  # True if any period has 0 agents available
    total_deficit_hours: float
    gaps: List[CoverageGap]
    recommendation: Dict[str, any]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'has_deficit': self.has_deficit,
            'has_critical_gaps': self.has_critical_gaps,
            'total_deficit_hours': self.total_deficit_hours,
            'gaps': [
                {
                    'day': gap.day,
                    'day_name': gap.day_name,
                    'start_time': gap.start_time,
                    'end_time': gap.end_time,
                    'required_agents': gap.required_agents,
                    'available_agents': gap.available_agents,
                    'deficit': gap.deficit
                }
                for gap in self.gaps
            ],
            'recommendation': self.recommendation
        }


class DeficitAnalyzer:
    """Analyzes coverage deficits and provides recommendations."""

    DAY_NAMES = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']

    def __init__(self, agents: List[Agent], demanda: Demanda):
        """
        Initialize analyzer.

        Args:
            agents: List of available agents
            demanda: Demand requirements
        """
        self.agents = agents
        self.demanda = demanda

        # Build availability matrix: [day][block] -> count of available agents
        self.availability = self._build_availability_matrix()

    def _build_availability_matrix(self) -> List[List[int]]:
        """
        Build matrix of agent availability.

        Returns:
            7x48 matrix where [day][block] = number of available agents
        """
        num_agents = len(self.agents)
        matrix = [[num_agents for _ in range(48)] for _ in range(7)]
        return matrix

    def analyze(self) -> DeficitReport:
        """
        Analyze coverage and detect deficits.

        Returns:
            DeficitReport with detailed analysis
        """
        gaps = []
        total_deficit_hours = 0.0
        has_critical_gaps = False

        for day in range(7):
            day_gaps = self._analyze_day(day)

            for gap in day_gaps:
                gaps.append(gap)
                # Each block is 0.5 hours
                gap_hours = (gap.end_block - gap.start_block + 1) * 0.5
                total_deficit_hours += gap_hours * gap.deficit

                if gap.available_agents == 0 and gap.required_agents > 0:
                    has_critical_gaps = True

        has_deficit = len(gaps) > 0
        recommendation = self._generate_recommendation(gaps, has_critical_gaps)

        return DeficitReport(
            has_deficit=has_deficit,
            has_critical_gaps=has_critical_gaps,
            total_deficit_hours=total_deficit_hours,
            gaps=gaps,
            recommendation=recommendation
        )

    def _analyze_day(self, day: int) -> List[CoverageGap]:
        """
        Analyze coverage gaps for a single day.

        Args:
            day: Day index (0-6)

        Returns:
            List of gaps found in this day
        """
        gaps = []
        current_gap = None

        for block in range(48):
            available = self.availability[day][block]
            required = self.demanda.get_demand(day, block)

            if required > available:
                # We have a deficit
                deficit = required - available

                if current_gap is None:
                    # Start new gap
                    current_gap = {
                        'day': day,
                        'start_block': block,
                        'end_block': block,
                        'required': required,
                        'available': available,
                        'deficit': deficit
                    }
                else:
                    # Extend current gap if similar deficit
                    current_gap['end_block'] = block
                    # Update to max deficit in range
                    current_gap['deficit'] = max(current_gap['deficit'], deficit)
                    current_gap['required'] = max(current_gap['required'], required)
                    current_gap['available'] = min(current_gap['available'], available)
            else:
                # No deficit, close current gap if exists
                if current_gap is not None:
                    gaps.append(self._create_gap_object(current_gap))
                    current_gap = None

        # Close final gap if exists
        if current_gap is not None:
            gaps.append(self._create_gap_object(current_gap))

        return gaps

    def _create_gap_object(self, gap_data: dict) -> CoverageGap:
        """Create CoverageGap object from gap data."""
        day = gap_data['day']
        start_block = gap_data['start_block']
        end_block = gap_data['end_block']

        # Convert blocks to time strings
        start_hours = (start_block * 30) // 60
        start_mins = (start_block * 30) % 60
        end_hours = ((end_block + 1) * 30) // 60
        end_mins = ((end_block + 1) * 30) % 60

        return CoverageGap(
            day=day,
            day_name=self.DAY_NAMES[day],
            start_block=start_block,
            end_block=end_block,
            start_time=f"{start_hours:02d}:{start_mins:02d}",
            end_time=f"{end_hours:02d}:{end_mins:02d}",
            required_agents=gap_data['required'],
            available_agents=gap_data['available'],
            deficit=gap_data['deficit']
        )

    def _generate_recommendation(self, gaps: List[CoverageGap], has_critical: bool) -> Dict:
        """
        Generate recommendations for addressing deficits.

        Args:
            gaps: List of coverage gaps
            has_critical: Whether there are critical gaps (0 availability)

        Returns:
            Recommendation dictionary
        """
        if not gaps:
            return {
                'message': 'Cobertura adecuada - no se necesitan agentes adicionales',
                'additional_agents_needed': 0
            }

        # Group gaps by day to find patterns
        gaps_by_day = {}
        for gap in gaps:
            if gap.day not in gaps_by_day:
                gaps_by_day[gap.day] = []
            gaps_by_day[gap.day].append(gap)

        # Calculate max deficit per day
        max_deficit_by_day = {}
        for day, day_gaps in gaps_by_day.items():
            max_deficit_by_day[day] = max(g.deficit for g in day_gaps)

        # Find the most problematic days
        total_agents_needed = sum(max_deficit_by_day.values())

        # Suggest availability patterns
        suggested_days = [self.DAY_NAMES[day] for day in sorted(max_deficit_by_day.keys())]

        message = f"Se necesitan {total_agents_needed} agentes adicionales"
        if has_critical:
            message += " (CRÍTICO: algunos períodos tienen 0 agentes disponibles)"

        return {
            'message': message,
            'additional_agents_needed': total_agents_needed,
            'problematic_days': suggested_days,
            'suggested_availability': f"Días: {', '.join(suggested_days)}. Horarios variables según demanda.",
            'severity': 'critical' if has_critical else 'warning'
        }
