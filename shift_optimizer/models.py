"""
Data models for shift optimization.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import time


@dataclass
class TimeBlock:
    """Represents a 30-minute time block."""
    day: int  # 0-6 (Monday-Sunday)
    block: int  # 0-47 (30-min blocks in a day)

    def to_time(self) -> time:
        """Convert block index to time."""
        hours = (self.block * 30) // 60
        minutes = (self.block * 30) % 60
        return time(hours, minutes)

    @staticmethod
    def from_time(day: int, hour: int, minute: int) -> 'TimeBlock':
        """Create TimeBlock from day, hour, and minute."""
        block = (hour * 60 + minute) // 30
        return TimeBlock(day, block)


@dataclass
class Agent:
    """Represents a shift agent."""
    id: str
    nombre: str
    disponibilidad: List[TimeBlock] = field(default_factory=list)

    def is_available(self, day: int, block: int) -> bool:
        """Check if agent is available at given time."""
        return TimeBlock(day, block) in self.disponibilidad


@dataclass
class Shift:
    """Represents an assigned shift."""
    agente_id: str
    dia: int
    hora_inicio: time
    hora_fin: time
    breaks: List[time] = field(default_factory=list)
    hueco: Tuple[time, time] = None  # (start, end) of lunch break

    def duracion_efectiva(self) -> float:
        """Calculate effective hours worked."""
        total_minutes = (self.hora_fin.hour * 60 + self.hora_fin.minute) - \
                       (self.hora_inicio.hour * 60 + self.hora_inicio.minute)

        # Subtract breaks (15 min each)
        total_minutes -= len(self.breaks) * 15

        # Subtract lunch break if exists
        if self.hueco:
            hueco_minutes = (self.hueco[1].hour * 60 + self.hueco[1].minute) - \
                           (self.hueco[0].hour * 60 + self.hueco[0].minute)
            total_minutes -= hueco_minutes

        return total_minutes / 60.0


@dataclass
class Demanda:
    """Represents staffing demand."""
    data: List[List[int]]  # 7 days x 48 blocks (30-min each)

    def get_demand(self, day: int, block: int) -> int:
        """Get demand for specific day and block."""
        if 0 <= day < 7 and 0 <= block < 48:
            return self.data[day][block]
        return 0

    @staticmethod
    def create_empty() -> 'Demanda':
        """Create empty demand matrix."""
        return Demanda([[0 for _ in range(48)] for _ in range(7)])


@dataclass
class Solution:
    """Represents the optimization solution."""
    shifts: List[Shift]
    cobertura: List[List[int]]  # 7 days x 48 blocks - actual coverage
    costo_total: float  # Total cost (sum of squared differences)
    tiempo_resolucion: float  # Solution time in seconds
    status: str  # Solver status

    def get_coverage(self, day: int, block: int) -> int:
        """Get coverage for specific day and block."""
        if 0 <= day < 7 and 0 <= block < 48:
            return self.cobertura[day][block]
        return 0
