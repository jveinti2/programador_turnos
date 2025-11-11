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
    Parse agents from CSV in long format.

    Expected format:
        id,nombre,dia,bloque
        A001,Juan Pérez,0,14
        A001,Juan Pérez,0,15
        ...

    Each row represents ONE available 30-minute block for an agent.

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
    required_headers = {'id', 'nombre', 'dia', 'bloque'}
    if not reader.fieldnames:
        raise CSVParseError("CSV file is empty or has no headers")

    actual_headers = set(reader.fieldnames)
    if not required_headers.issubset(actual_headers):
        missing = required_headers - actual_headers
        raise CSVParseError(f"Missing required headers: {missing}")

    # Parse rows and group by agent
    agents_data: Dict[str, Dict] = {}
    seen_blocks: Dict[str, Set[tuple]] = {}
    row_num = 1

    for row in reader:
        row_num += 1

        # Validate non-empty fields
        agent_id = row.get('id', '').strip()
        nombre = row.get('nombre', '').strip()
        dia_str = row.get('dia', '').strip()
        bloque_str = row.get('bloque', '').strip()

        if not agent_id:
            raise CSVParseError(f"Row {row_num}: 'id' cannot be empty")
        if not nombre:
            raise CSVParseError(f"Row {row_num}: 'nombre' cannot be empty")
        if not dia_str:
            raise CSVParseError(f"Row {row_num}: 'dia' cannot be empty")
        if not bloque_str:
            raise CSVParseError(f"Row {row_num}: 'bloque' cannot be empty")

        # Parse integers
        try:
            dia = int(dia_str)
        except ValueError:
            raise CSVParseError(f"Row {row_num}: 'dia' must be an integer, got '{dia_str}'")

        try:
            bloque = int(bloque_str)
        except ValueError:
            raise CSVParseError(f"Row {row_num}: 'bloque' must be an integer, got '{bloque_str}'")

        # Validate ranges
        if not (0 <= dia <= 6):
            raise CSVParseError(f"Row {row_num}: 'dia' must be between 0-6 (Monday-Sunday), got {dia}")
        if not (0 <= bloque <= 47):
            raise CSVParseError(f"Row {row_num}: 'bloque' must be between 0-47 (30-min blocks), got {bloque}")

        # Check for duplicate blocks
        if agent_id not in seen_blocks:
            seen_blocks[agent_id] = set()
        block_key = (dia, bloque)
        if block_key in seen_blocks[agent_id]:
            raise CSVParseError(f"Row {row_num}: Duplicate block for agent {agent_id}: day={dia}, block={bloque}")
        seen_blocks[agent_id].add(block_key)

        # Group by agent
        if agent_id not in agents_data:
            agents_data[agent_id] = {
                'id': agent_id,
                'nombre': nombre,
                'disponibilidad': []
            }
        else:
            # Verify nombre consistency
            if agents_data[agent_id]['nombre'] != nombre:
                raise CSVParseError(
                    f"Row {row_num}: Inconsistent name for agent {agent_id}. "
                    f"Expected '{agents_data[agent_id]['nombre']}', got '{nombre}'"
                )

        # Add time block
        agents_data[agent_id]['disponibilidad'].append(TimeBlock(day=dia, block=bloque))

    # Convert to Agent objects
    if not agents_data:
        raise CSVParseError("No agent data found in CSV")

    agents = [
        Agent(
            id=data['id'],
            nombre=data['nombre'],
            disponibilidad=data['disponibilidad']
        )
        for data in agents_data.values()
    ]

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
