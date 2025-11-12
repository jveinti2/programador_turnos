"""
CSV parsers for agents and demand data.

Supports long-format CSV files for agent availability and demand.
"""
import csv
from typing import List, Dict, Set
from io import StringIO
from .models import Agent, TimeBlock, Demanda


class CSVParseError(Exception):
    """Custom exception for CSV parsing errors."""
    pass


def parse_agents_csv(content: str) -> List[Agent]:
    """
    Parse agents from CSV in simple format.

    Expected format:
        id,nombre
        A001,Juan Pérez
        A002,María García
        ...

    Each row represents one agent (available 24/7 by default).

    Args:
        content: CSV file content as string

    Returns:
        List of Agent objects

    Raises:
        CSVParseError: If CSV format is invalid or data is inconsistent
    """
    try:
        reader = csv.DictReader(StringIO(content))
    except Exception as e:
        raise CSVParseError(f"Failed to parse CSV: {str(e)}")

    # Validate headers
    required_headers = {'id', 'nombre'}
    if not reader.fieldnames:
        raise CSVParseError("CSV file is empty or has no headers")

    actual_headers = set(reader.fieldnames)
    if not required_headers.issubset(actual_headers):
        missing = required_headers - actual_headers
        raise CSVParseError(f"Missing required headers: {missing}")

    # Parse rows
    agents = []
    seen_ids: Set[str] = set()
    row_num = 1

    for row in reader:
        row_num += 1

        # Validate non-empty fields
        agent_id = row.get('id', '').strip()
        nombre = row.get('nombre', '').strip()

        if not agent_id:
            raise CSVParseError(f"Row {row_num}: 'id' cannot be empty")
        if not nombre:
            raise CSVParseError(f"Row {row_num}: 'nombre' cannot be empty")

        # Check for duplicate IDs
        if agent_id in seen_ids:
            raise CSVParseError(f"Row {row_num}: Duplicate agent ID: {agent_id}")
        seen_ids.add(agent_id)

        # Create agent (available 24/7 by default)
        agents.append(Agent(id=agent_id, nombre=nombre))

    # Verify we have at least one agent
    if not agents:
        raise CSVParseError("No agent data found in CSV")

    return agents


def parse_demanda_csv(content: str) -> Demanda:
    """
    Parse demand from CSV in long format.

    Expected format:
        dia,bloque,agentes_requeridos
        0,14,5
        0,15,10
        ...

    Each row represents the required agents for ONE 30-minute block.

    Args:
        content: CSV file content as string

    Returns:
        Demanda object with 7x48 matrix

    Raises:
        CSVParseError: If CSV format is invalid or data is inconsistent
    """
    try:
        reader = csv.DictReader(StringIO(content))
    except Exception as e:
        raise CSVParseError(f"Failed to parse CSV: {str(e)}")

    # Validate headers
    required_headers = {'dia', 'bloque', 'agentes_requeridos'}
    if not reader.fieldnames:
        raise CSVParseError("CSV file is empty or has no headers")

    actual_headers = set(reader.fieldnames)
    if not required_headers.issubset(actual_headers):
        missing = required_headers - actual_headers
        raise CSVParseError(f"Missing required headers: {missing}")

    # Initialize empty demand matrix (7 days x 48 blocks)
    demand_matrix = [[0 for _ in range(48)] for _ in range(7)]
    seen_blocks: Set[tuple] = set()
    row_num = 1

    for row in reader:
        row_num += 1

        # Validate non-empty fields
        dia_str = row.get('dia', '').strip()
        bloque_str = row.get('bloque', '').strip()
        agentes_str = row.get('agentes_requeridos', '').strip()

        if not dia_str:
            raise CSVParseError(f"Row {row_num}: 'dia' cannot be empty")
        if not bloque_str:
            raise CSVParseError(f"Row {row_num}: 'bloque' cannot be empty")
        if not agentes_str:
            raise CSVParseError(f"Row {row_num}: 'agentes_requeridos' cannot be empty")

        # Parse integers
        try:
            dia = int(dia_str)
        except ValueError:
            raise CSVParseError(f"Row {row_num}: 'dia' must be an integer, got '{dia_str}'")

        try:
            bloque = int(bloque_str)
        except ValueError:
            raise CSVParseError(f"Row {row_num}: 'bloque' must be an integer, got '{bloque_str}'")

        try:
            agentes = int(agentes_str)
        except ValueError:
            raise CSVParseError(f"Row {row_num}: 'agentes_requeridos' must be an integer, got '{agentes_str}'")

        # Validate ranges
        if not (0 <= dia <= 6):
            raise CSVParseError(f"Row {row_num}: 'dia' must be between 0-6 (Monday-Sunday), got {dia}")
        if not (0 <= bloque <= 47):
            raise CSVParseError(f"Row {row_num}: 'bloque' must be between 0-47 (30-min blocks), got {bloque}")
        if agentes < 0:
            raise CSVParseError(f"Row {row_num}: 'agentes_requeridos' cannot be negative, got {agentes}")

        # Check for duplicate blocks
        block_key = (dia, bloque)
        if block_key in seen_blocks:
            raise CSVParseError(f"Row {row_num}: Duplicate block: day={dia}, block={bloque}")
        seen_blocks.add(block_key)

        # Set demand
        demand_matrix[dia][bloque] = agentes

    # Verify we have at least some data
    total_demand = sum(sum(day) for day in demand_matrix)
    if total_demand == 0:
        raise CSVParseError("No demand data found in CSV (all values are 0)")

    return Demanda(data=demand_matrix)


def parse_schedules_csv(content: str) -> List[Dict]:
    """
    Parse schedules from CSV in long format.

    Expected format:
        agent_id,agent_name,day,shift_start,shift_end,break_start,break_end,disconnected_start,disconnected_end
        A001,Juan Pérez,lunes,08:00,17:30,09:30,09:45,,
        A001,Juan Pérez,lunes,08:00,17:30,13:00,13:15,,
        A001,Juan Pérez,martes,09:00,18:30,09:00,09:15,,
        ...

    Each row represents one break or disconnected period within a shift.
    Multiple rows with the same agent_id + day = same shift with multiple breaks/disconnected.

    Args:
        content: CSV file content as string

    Returns:
        List of agent schedule dictionaries (same format as schedules.json)

    Raises:
        CSVParseError: If CSV format is invalid or data is inconsistent
    """
    try:
        reader = csv.DictReader(StringIO(content))
    except Exception as e:
        raise CSVParseError(f"Failed to parse CSV: {str(e)}")

    # Validate headers
    required_headers = {'agent_id', 'agent_name', 'day', 'shift_start', 'shift_end'}
    if not reader.fieldnames:
        raise CSVParseError("CSV file is empty or has no headers")

    actual_headers = set(reader.fieldnames)
    if not required_headers.issubset(actual_headers):
        missing = required_headers - actual_headers
        raise CSVParseError(f"Missing required headers: {missing}")

    # Build schedule structure
    agents_data = {}
    row_num = 1

    for row in reader:
        row_num += 1

        # Validate required fields
        agent_id = row.get('agent_id', '').strip()
        agent_name = row.get('agent_name', '').strip()
        day = row.get('day', '').strip()
        shift_start = row.get('shift_start', '').strip()
        shift_end = row.get('shift_end', '').strip()

        if not agent_id:
            raise CSVParseError(f"Row {row_num}: 'agent_id' cannot be empty")
        if not agent_name:
            raise CSVParseError(f"Row {row_num}: 'agent_name' cannot be empty")
        if not day:
            raise CSVParseError(f"Row {row_num}: 'day' cannot be empty")
        if not shift_start:
            raise CSVParseError(f"Row {row_num}: 'shift_start' cannot be empty")
        if not shift_end:
            raise CSVParseError(f"Row {row_num}: 'shift_end' cannot be empty")

        # Validate day name
        valid_days = {'lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo'}
        if day.lower() not in valid_days:
            raise CSVParseError(f"Row {row_num}: Invalid day '{day}'. Must be one of: {valid_days}")

        # Initialize agent if not exists
        if agent_id not in agents_data:
            agents_data[agent_id] = {
                'id': agent_id,
                'name': agent_name,
                'schedule': {}
            }

        # Initialize day if not exists
        if day not in agents_data[agent_id]['schedule']:
            agents_data[agent_id]['schedule'][day] = {
                'start': shift_start,
                'end': shift_end,
                'break': [],
                'disconnected': []
            }

        # Add breaks (if present)
        break_start = row.get('break_start', '').strip()
        break_end = row.get('break_end', '').strip()
        if break_start and break_end:
            agents_data[agent_id]['schedule'][day]['break'].append({
                'start': break_start,
                'end': break_end
            })

        # Add disconnected periods (if present)
        disconnected_start = row.get('disconnected_start', '').strip()
        disconnected_end = row.get('disconnected_end', '').strip()
        if disconnected_start and disconnected_end:
            agents_data[agent_id]['schedule'][day]['disconnected'].append({
                'start': disconnected_start,
                'end': disconnected_end
            })

    # Convert to list
    schedules = list(agents_data.values())

    # Verify we have at least one agent
    if not schedules:
        raise CSVParseError("No schedule data found in CSV")

    return schedules
