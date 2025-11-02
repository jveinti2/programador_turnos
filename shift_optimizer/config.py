"""
Configuration management for shift optimization.

Loads and validates configuration from YAML files.
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShiftRulesConfig:
    """Configuration for shift duration and timing rules."""
    duracion_min_horas: float  # Minimum effective work hours
    duracion_max_horas: float  # Maximum effective work hours
    duracion_total_max_horas: float  # Maximum total shift span (including breaks and lunch)
    descanso_entre_turnos_horas: float  # Minimum rest between shifts
    dias_libres_min_por_semana: int  # Minimum days off per week


@dataclass
class BreakConfig:
    """Configuration for short break rules (15-min breaks)."""
    duracion_minutos: int  # Break duration in minutes (fixed at 15)
    frecuencia_cada_horas: float  # Frequency: 1 break every X hours
    obligatorias: bool  # Are breaks mandatory?
    separacion_minima_horas: float  # Minimum separation between consecutive breaks
    prohibir_primera_hora: bool  # Prohibit breaks in first hour?
    prohibir_ultima_hora: bool  # Prohibit breaks in last hour?
    maximo_por_turno: int  # Maximum breaks per shift


@dataclass
class LunchConfig:
    """Configuration for lunch break rules (30-60 min)."""
    duracion_min_minutos: int  # Minimum lunch duration
    duracion_max_minutos: int  # Maximum lunch duration
    obligatorio_si_turno_mayor_a_horas: float  # Mandatory if shift > X hours (0 = never mandatory)
    maximo_por_turno: int  # Maximum lunch breaks per shift
    prohibir_primera_hora: bool  # Prohibit lunch in first hour?
    prohibir_ultima_hora: bool  # Prohibit lunch in last hour?


@dataclass
class SolverConfig:
    """Configuration for solver parameters."""
    timeout_segundos: int
    num_workers: int


@dataclass
class TimeBlockConfig:
    """Configuration for time block settings."""
    duracion_minutos: int


@dataclass
class CoverageConfig:
    """Configuration for coverage analysis and deficit detection."""
    margen_seguridad: float  # Recommended safety margin (e.g., 1.2 = 20% extra capacity)
    bloquear_si_deficit: bool  # Block optimization if critical deficit exists


class OptimizationConfig:
    """
    Main configuration class for shift optimization.

    Loads and validates configuration from YAML file.
    """

    def __init__(
        self,
        shift_rules: ShiftRulesConfig,
        breaks: BreakConfig,
        lunch: LunchConfig,
        solver: SolverConfig,
        time_blocks: TimeBlockConfig,
        coverage: CoverageConfig
    ):
        self.shift_rules = shift_rules
        self.breaks = breaks
        self.lunch = lunch
        self.solver = solver
        self.time_blocks = time_blocks
        self.coverage = coverage

    @classmethod
    def from_yaml(cls, config_path: str = 'config/reglas.yaml') -> 'OptimizationConfig':
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            OptimizationConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Parse shift rules (new flatter structure)
        turnos_data = data.get('turnos', {})
        shift_rules = ShiftRulesConfig(
            duracion_min_horas=turnos_data.get('duracion_min_horas', 4),
            duracion_max_horas=turnos_data.get('duracion_max_horas', 9),
            duracion_total_max_horas=turnos_data.get('duracion_total_max_horas', 10),
            descanso_entre_turnos_horas=turnos_data.get('descanso_entre_turnos_horas', 8),
            dias_libres_min_por_semana=turnos_data.get('dias_libres_min_por_semana', 1)
        )

        # Parse breaks config (new flatter structure)
        pausas_data = data.get('pausas_cortas', {})
        breaks = BreakConfig(
            duracion_minutos=pausas_data.get('duracion_minutos', 15),
            frecuencia_cada_horas=pausas_data.get('frecuencia_cada_horas', 3),
            obligatorias=pausas_data.get('obligatorias', True),
            separacion_minima_horas=pausas_data.get('separacion_minima_horas', 2.5),
            prohibir_primera_hora=pausas_data.get('prohibir_primera_hora', True),
            prohibir_ultima_hora=pausas_data.get('prohibir_ultima_hora', True),
            maximo_por_turno=pausas_data.get('maximo_por_turno', 4)
        )

        # Parse lunch config (new flatter structure)
        almuerzo_data = data.get('almuerzo', {})
        lunch = LunchConfig(
            duracion_min_minutos=almuerzo_data.get('duracion_min_minutos', 30),
            duracion_max_minutos=almuerzo_data.get('duracion_max_minutos', 60),
            obligatorio_si_turno_mayor_a_horas=almuerzo_data.get('obligatorio_si_turno_mayor_a_horas', 0),
            maximo_por_turno=almuerzo_data.get('maximo_por_turno', 1),
            prohibir_primera_hora=almuerzo_data.get('prohibir_primera_hora', True),
            prohibir_ultima_hora=almuerzo_data.get('prohibir_ultima_hora', True)
        )

        # Parse solver config
        solver_data = data.get('solver', {})
        solver = SolverConfig(
            timeout_segundos=solver_data.get('timeout_segundos', 300),
            num_workers=solver_data.get('num_workers', 8)
        )

        # Parse time block config
        time_blocks_data = data.get('time_blocks', {})
        time_blocks = TimeBlockConfig(
            duracion_minutos=time_blocks_data.get('duracion_minutos', 30)
        )

        # Parse coverage config
        coverage_data = data.get('cobertura', {})
        coverage = CoverageConfig(
            margen_seguridad=coverage_data.get('margen_seguridad', 1.2),
            bloquear_si_deficit=coverage_data.get('bloquear_si_deficit', True)
        )

        return cls(shift_rules, breaks, lunch, solver, time_blocks, coverage)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary (new structure)."""
        return {
            'turnos': {
                'duracion_min_horas': self.shift_rules.duracion_min_horas,
                'duracion_max_horas': self.shift_rules.duracion_max_horas,
                'duracion_total_max_horas': self.shift_rules.duracion_total_max_horas,
                'descanso_entre_turnos_horas': self.shift_rules.descanso_entre_turnos_horas,
                'dias_libres_min_por_semana': self.shift_rules.dias_libres_min_por_semana
            },
            'pausas_cortas': {
                'duracion_minutos': self.breaks.duracion_minutos,
                'frecuencia_cada_horas': self.breaks.frecuencia_cada_horas,
                'obligatorias': self.breaks.obligatorias,
                'separacion_minima_horas': self.breaks.separacion_minima_horas,
                'prohibir_primera_hora': self.breaks.prohibir_primera_hora,
                'prohibir_ultima_hora': self.breaks.prohibir_ultima_hora,
                'maximo_por_turno': self.breaks.maximo_por_turno
            },
            'almuerzo': {
                'duracion_min_minutos': self.lunch.duracion_min_minutos,
                'duracion_max_minutos': self.lunch.duracion_max_minutos,
                'obligatorio_si_turno_mayor_a_horas': self.lunch.obligatorio_si_turno_mayor_a_horas,
                'maximo_por_turno': self.lunch.maximo_por_turno,
                'prohibir_primera_hora': self.lunch.prohibir_primera_hora,
                'prohibir_ultima_hora': self.lunch.prohibir_ultima_hora
            },
            'solver': {
                'timeout_segundos': self.solver.timeout_segundos,
                'num_workers': self.solver.num_workers
            },
            'time_blocks': {
                'duracion_minutos': self.time_blocks.duracion_minutos
            },
            'cobertura': {
                'margen_seguridad': self.coverage.margen_seguridad,
                'bloquear_si_deficit': self.coverage.bloquear_si_deficit
            }
        }
