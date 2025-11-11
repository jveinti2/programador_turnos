"""
Utility functions for loading and exporting data.
"""
import json
import csv
from typing import List
from datetime import time
from pathlib import Path
from .models import Agent, TimeBlock, Demanda, Solution, Shift
from .csv_parsers import parse_agents_csv, parse_demanda_csv, CSVParseError


def load_agents(filepath: str) -> List[Agent]:
    """
    Load agents from CSV or JSON file (auto-detected by extension).

    CSV format (long):
        id,nombre,dia,bloque
        A001,Juan Pérez,0,14
        A001,Juan Pérez,0,15
        ...

    JSON format:
        [
            {
                "id": "A001",
                "nombre": "Juan Pérez",
                "disponibilidad": [
                    {"day": 0, "block": 0},
                    {"day": 0, "block": 1},
                    ...
                ]
            },
            ...
        ]
    """
    file_ext = Path(filepath).suffix.lower()

    if file_ext == '.csv':
        # Load CSV format
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            return parse_agents_csv(content)
        except CSVParseError as e:
            raise ValueError(f"Error al cargar agentes desde CSV: {str(e)}")

    elif file_ext == '.json':
        # Load JSON format (legacy)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        agents = []
        for agent_data in data:
            disponibilidad = [
                TimeBlock(day=tb['day'], block=tb['block'])
                for tb in agent_data.get('disponibilidad', [])
            ]
            agents.append(Agent(
                id=agent_data['id'],
                nombre=agent_data['nombre'],
                disponibilidad=disponibilidad
            ))

        return agents

    else:
        raise ValueError(
            f"Formato de archivo no soportado: {file_ext}. "
            "Use .csv o .json"
        )


def load_demanda(filepath: str) -> Demanda:
    """
    Load demand from CSV or JSON file (auto-detected by extension).

    CSV format (long):
        dia,bloque,agentes_requeridos
        0,14,5
        0,15,10
        ...

    JSON format:
        [
            [0, 0, 0, ..., 5, 10, 15, ...],  # Day 0 (48 blocks of 30min)
            [0, 0, 0, ..., 5, 10, 15, ...],  # Day 1 (48 blocks of 30min)
            ...
        ]
    """
    file_ext = Path(filepath).suffix.lower()

    if file_ext == '.csv':
        # Load CSV format
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            return parse_demanda_csv(content)
        except CSVParseError as e:
            raise ValueError(f"Error al cargar demanda desde CSV: {str(e)}")

    elif file_ext == '.json':
        # Load JSON format (legacy)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate format
        if len(data) != 7:
            raise ValueError(f"Expected 7 days, got {len(data)}")

        for day_idx, day_data in enumerate(data):
            if len(day_data) != 48:
                raise ValueError(f"Day {day_idx}: Expected 48 blocks (30min each), got {len(day_data)}")

        return Demanda(data=data)

    else:
        raise ValueError(
            f"Formato de archivo no soportado: {file_ext}. "
            "Use .csv o .json"
        )


def export_turnos_csv(solution: Solution, filepath: str) -> None:
    """
    Export shifts to CSV file.

    Format:
    agente_id,dia,hora_inicio,hora_fin,breaks,hueco_inicio,hueco_fin,duracion_efectiva
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'agente_id', 'dia', 'hora_inicio', 'hora_fin',
            'breaks', 'hueco_inicio', 'hueco_fin', 'duracion_efectiva'
        ])

        for shift in solution.shifts:
            breaks_str = ';'.join([b.strftime('%H:%M') for b in shift.breaks])
            hueco_inicio = shift.hueco[0].strftime('%H:%M') if shift.hueco else ''
            hueco_fin = shift.hueco[1].strftime('%H:%M') if shift.hueco else ''

            writer.writerow([
                shift.agente_id,
                shift.dia,
                shift.hora_inicio.strftime('%H:%M'),
                shift.hora_fin.strftime('%H:%M'),
                breaks_str,
                hueco_inicio,
                hueco_fin,
                f"{shift.duracion_efectiva():.2f}"
            ])


def export_cobertura_csv(solution: Solution, demanda: Demanda, filepath: str) -> None:
    """
    Export coverage analysis to CSV file.

    Format:
    dia,bloque,hora,demanda,cobertura,diferencia
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dia', 'bloque', 'hora', 'demanda', 'cobertura', 'diferencia'])

        for day in range(7):
            for block in range(48):  # 48 blocks of 30 minutes
                hours = (block * 30) // 60
                minutes = (block * 30) % 60
                hora_str = f"{hours:02d}:{minutes:02d}"

                dem = demanda.get_demand(day, block)
                cob = solution.get_coverage(day, block)
                diff = cob - dem

                writer.writerow([day, block, hora_str, dem, cob, diff])


def export_resumen_json(solution: Solution, demanda: Demanda, agents: List[Agent], filepath: str) -> None:
    """
    Export solution summary to JSON file.

    Simplified format with 3 status levels:
    - SUPERHABIT: Coverage met with spare capacity
    - OPTIMAL: Coverage met, all agents utilized
    - DEFICIT: Cannot meet coverage (insufficient agents)
    """
    # Calculate statistics
    total_demanda = sum(sum(day) for day in demanda.data)
    total_cobertura = sum(sum(day) for day in solution.cobertura)

    # Coverage by day (simplified: no 'diferencia' field)
    cobertura_por_dia = []
    for day in range(7):
        dem_dia = sum(demanda.data[day])
        cob_dia = sum(solution.cobertura[day])
        cobertura_por_dia.append({
            'dia': day,
            'demanda': dem_dia,
            'cobertura': cob_dia
        })

    # Determine status and calculate missing agents
    agentes_asignados = len(set(shift.agente_id for shift in solution.shifts))
    agentes_disponibles = len(agents)
    agentes_faltantes = 0

    if solution.status == 'INFEASIBLE' or total_cobertura < total_demanda:
        # DEFICIT: Cannot meet demand
        status = 'DEFICIT'
        # Estimate missing agents based on uncovered demand
        deficit = total_demanda - total_cobertura
        if agentes_asignados > 0:
            avg_coverage_per_agent = total_cobertura / agentes_asignados
            if avg_coverage_per_agent > 0:
                agentes_faltantes = int((deficit / avg_coverage_per_agent) + 0.5)  # round up
        else:
            agentes_faltantes = 1  # At least 1 agent needed
    elif agentes_asignados < agentes_disponibles:
        # SUPERHABIT: Not all agents were needed
        status = 'SUPERHABIT'
    else:
        # OPTIMAL: All agents utilized to meet demand
        status = 'OPTIMAL'

    # Simplified output format
    resumen = {
        'status': status,
        'agentes_disponibles': agentes_disponibles,
        'agentes_faltantes': agentes_faltantes,
        'cobertura_por_dia': cobertura_por_dia
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


def time_to_string(t: time) -> str:
    """Convert time to HH:MM string."""
    return t.strftime('%H:%M')


def string_to_time(s: str) -> time:
    """Convert HH:MM string to time."""
    parts = s.split(':')
    return time(int(parts[0]), int(parts[1]))


def export_schedules_json(solution: Solution, agents: List[Agent], filepath: str) -> None:
    """
    Export agent schedules in a frontend-friendly JSON format.

    Format:
    [
      {
        "id": "A001",
        "name": "Juan Pérez",
        "schedule": {
          "lunes": {
            "start": "09:00",
            "end": "18:00",
            "break": [{"start": "11:00", "end": "11:15"}],
            "disconnected": [{"start": "13:00", "end": "14:00"}]
          },
          ...
        }
      },
      ...
    ]
    """
    # Day number to Spanish name mapping
    day_names = {
        0: 'lunes',
        1: 'martes',
        2: 'miercoles',
        3: 'jueves',
        4: 'viernes',
        5: 'sabado',
        6: 'domingo'
    }

    # Group shifts by agent
    shifts_by_agent = {}
    for shift in solution.shifts:
        if shift.agente_id not in shifts_by_agent:
            shifts_by_agent[shift.agente_id] = []
        shifts_by_agent[shift.agente_id].append(shift)

    # Create agent map for quick lookup
    agent_map = {agent.id: agent for agent in agents}

    # Build schedules array
    schedules = []
    for agent in agents:
        agent_schedule = {
            'id': agent.id,
            'name': agent.nombre,
            'schedule': {}
        }

        # Add shifts for this agent
        if agent.id in shifts_by_agent:
            for shift in shifts_by_agent[agent.id]:
                day_name = day_names[shift.dia]

                # Build break list (15-min breaks)
                breaks_list = []
                for break_time in shift.breaks:
                    # Each break is 15 minutes
                    break_end_minutes = break_time.hour * 60 + break_time.minute + 15
                    break_end_hour = break_end_minutes // 60
                    break_end_minute = break_end_minutes % 60

                    breaks_list.append({
                        'start': break_time.strftime('%H:%M'),
                        'end': f"{break_end_hour:02d}:{break_end_minute:02d}"
                    })

                # Build disconnected list (lunch break/hueco)
                disconnected_list = []
                if shift.hueco:
                    disconnected_list.append({
                        'start': shift.hueco[0].strftime('%H:%M'),
                        'end': shift.hueco[1].strftime('%H:%M')
                    })

                # Add day schedule
                agent_schedule['schedule'][day_name] = {
                    'start': shift.hora_inicio.strftime('%H:%M'),
                    'end': shift.hora_fin.strftime('%H:%M'),
                    'break': breaks_list,
                    'disconnected': disconnected_list
                }

        schedules.append(agent_schedule)

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(schedules, f, indent=2, ensure_ascii=False)
