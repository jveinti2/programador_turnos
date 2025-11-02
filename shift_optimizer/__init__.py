"""
Shift Optimizer - Sistema de optimización de turnos para call centers.

Este package contiene toda la lógica de optimización usando Google OR-Tools CP-SAT.
"""

from .api import ShiftOptimizer
from .models import Agent, Shift, Demanda, Solution, TimeBlock
from .config import OptimizationConfig
from .deficit_analyzer import DeficitAnalyzer, DeficitReport, CoverageGap

__version__ = '2.0.0'
__all__ = [
    'ShiftOptimizer',
    'Agent',
    'Shift',
    'Demanda',
    'Solution',
    'TimeBlock',
    'OptimizationConfig',
    'DeficitAnalyzer',
    'DeficitReport',
    'CoverageGap'
]
