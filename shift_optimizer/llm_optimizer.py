import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from openai import OpenAI
from toon_format import encode, decode


def chunk_agents(agents_data: List[Dict[str, Any]], chunk_size: int = 10) -> List[List[Dict[str, Any]]]:
    """
    Divide la lista de agentes en chunks de tamaño especificado.

    Args:
        agents_data: Lista completa de agentes con sus schedules
        chunk_size: Número de agentes por chunk (default: 10)

    Returns:
        Lista de chunks, donde cada chunk es una lista de agentes
    """
    chunks = []
    for i in range(0, len(agents_data), chunk_size):
        chunks.append(agents_data[i:i + chunk_size])
    return chunks


def convert_to_toon(data: List[Dict[str, Any]]) -> str:
    """
    Convierte datos JSON a formato TOON (reduce ~50% tokens).

    Args:
        data: Datos en formato Python (dict/list)

    Returns:
        String en formato TOON
    """
    return encode(data)


def convert_from_toon(toon_str: str) -> List[Dict[str, Any]]:
    """
    Convierte formato TOON de vuelta a Python dict/list.

    Args:
        toon_str: String en formato TOON

    Returns:
        Datos en formato Python
    """
    return decode(toon_str)


def parse_llm_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Extrae y valida JSON del response del LLM.

    El LLM puede devolver:
    1. JSON puro: [...]
    2. JSON con explicaciones: "Aquí está el resultado: [...]"
    3. JSON en markdown: ```json [...] ```

    Args:
        response_text: Texto completo del LLM

    Returns:
        Lista de agentes validada

    Raises:
        ValueError: Si no se puede extraer JSON válido
    """
    # Estrategia 1: Intentar parsear directamente
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Estrategia 2: Buscar JSON en markdown code block
    markdown_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except json.JSONDecodeError:
            pass

    # Estrategia 3: Buscar array JSON en el texto (más permisivo)
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Si todo falla, lanzar error con preview del response
    preview = response_text[:500] if len(response_text) > 500 else response_text
    raise ValueError(f"No se pudo extraer JSON válido del LLM. Preview: {preview}...")


def build_global_context(all_agents: List[Dict[str, Any]], current_chunk_ids: List[str]) -> str:
    """
    Genera un resumen estadístico de los agentes NO incluidos en el chunk actual.
    Esto da contexto al LLM sobre el resto del schedule.

    Args:
        all_agents: Lista completa de todos los agentes
        current_chunk_ids: IDs de los agentes en el chunk actual

    Returns:
        String con contexto global (estadísticas resumidas)
    """
    other_agents = [a for a in all_agents if a['id'] not in current_chunk_ids]

    if not other_agents:
        return "Este es el único grupo de agentes en el sistema."

    total_agents = len(all_agents)
    chunk_size = len(current_chunk_ids)

    # Calcular horas trabajadas por el resto de agentes
    days = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']
    total_hours_others = 0

    for agent in other_agents:
        for day in days:
            if day in agent['schedule']:
                day_schedule = agent['schedule'][day]
                if 'start' in day_schedule and 'end' in day_schedule:
                    start_h, start_m = map(int, day_schedule['start'].split(':'))
                    end_h, end_m = map(int, day_schedule['end'].split(':'))
                    hours = (end_h * 60 + end_m - start_h * 60 - start_m) / 60
                    total_hours_others += hours

    avg_hours_per_agent = total_hours_others / len(other_agents) if other_agents else 0

    context = f"""
CONTEXTO GLOBAL:
- Total de agentes en el sistema: {total_agents}
- Agentes en este grupo: {chunk_size}
- Agentes restantes: {len(other_agents)}
- Horas semanales promedio (otros agentes): {avg_hours_per_agent:.1f}h

Tu tarea es optimizar SOLO los {chunk_size} agentes en este grupo, manteniendo balance con el resto del equipo.
"""
    return context


def optimize_chunk_with_llm(
    chunk_data: List[Dict[str, Any]],
    system_prompt: str,
    global_context: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    use_toon: bool = True
) -> List[Dict[str, Any]]:
    """
    Optimiza un chunk de agentes usando OpenAI API.

    Args:
        chunk_data: Lista de agentes a optimizar (máximo 10)
        system_prompt: Instrucciones del sistema (desde YAML)
        global_context: Contexto estadístico del resto de agentes
        api_key: OpenAI API key
        model: Modelo a usar (default: gpt-4o-mini)
        temperature: Temperatura del modelo (default: 0.3)
        use_toon: Si True, envía datos en formato TOON (default: True)

    Returns:
        Lista de agentes optimizados

    Raises:
        Exception: Si la llamada a OpenAI falla o el response es inválido
    """
    client = OpenAI(api_key=api_key)

    # Convertir a TOON si está habilitado (reduce ~50% tokens)
    if use_toon:
        schedule_content = convert_to_toon(chunk_data)
        format_instruction = "Los datos están en formato TOON. Debes devolver JSON estándar."
    else:
        schedule_content = json.dumps(chunk_data, indent=2, ensure_ascii=False)
        format_instruction = "Los datos están en formato JSON. Debes devolver JSON estándar."

    # Construir prompt completo
    user_prompt = f"""{global_context}

{format_instruction}

SCHEDULE A OPTIMIZAR:
{schedule_content}

INSTRUCCIONES:
{system_prompt}

IMPORTANTE: Devuelve ÚNICAMENTE un array JSON válido con los agentes optimizados, sin explicaciones adicionales.
"""

    # Llamar a OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Eres un experto en optimización de turnos de call center. Respondes ÚNICAMENTE con JSON válido."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=16000  # Suficiente para 10 agentes en JSON
    )

    # Extraer y parsear response
    llm_response = response.choices[0].message.content
    optimized_chunk = parse_llm_response(llm_response)

    # Validación básica
    if not isinstance(optimized_chunk, list):
        raise ValueError(f"LLM no devolvió un array. Tipo: {type(optimized_chunk)}")

    if len(optimized_chunk) != len(chunk_data):
        raise ValueError(
            f"LLM devolvió {len(optimized_chunk)} agentes pero se enviaron {len(chunk_data)}"
        )

    return optimized_chunk


def merge_optimized_chunks(chunks: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Fusiona chunks optimizados en un solo schedule completo.

    Args:
        chunks: Lista de chunks, donde cada chunk es una lista de agentes

    Returns:
        Lista completa de agentes fusionada
    """
    merged = []
    for chunk in chunks:
        merged.extend(chunk)
    return merged


def validate_schedule_structure(schedule_data: List[Dict[str, Any]]) -> tuple[bool, Optional[str]]:
    """
    Valida que el schedule tenga estructura correcta.

    Args:
        schedule_data: Lista de agentes con schedules

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(schedule_data, list):
        return False, f"Schedule debe ser lista, no {type(schedule_data)}"

    required_fields = ['id', 'name', 'schedule']
    days = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']

    for i, agent in enumerate(schedule_data):
        # Validar campos requeridos
        for field in required_fields:
            if field not in agent:
                return False, f"Agente {i} falta campo '{field}'"

        # Validar estructura de schedule
        if not isinstance(agent['schedule'], dict):
            return False, f"Agente {agent['id']}: schedule debe ser dict"

        # Validar cada día
        for day, day_data in agent['schedule'].items():
            if day not in days:
                return False, f"Agente {agent['id']}: día inválido '{day}'"

            if not isinstance(day_data, dict):
                return False, f"Agente {agent['id']} {day}: debe ser dict"

            # Validar horarios si existen
            if 'start' in day_data and 'end' in day_data:
                time_pattern = re.compile(r'^\d{2}:\d{2}$')
                if not time_pattern.match(day_data['start']):
                    return False, f"Agente {agent['id']} {day}: start inválido '{day_data['start']}'"
                if not time_pattern.match(day_data['end']):
                    return False, f"Agente {agent['id']} {day}: end inválido '{day_data['end']}'"

    return True, None


def save_optimized_schedule(schedule_data: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Guarda el schedule optimizado en formato JSON.

    Args:
        schedule_data: Lista de agentes optimizados
        output_path: Ruta donde guardar el archivo
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schedule_data, f, indent=2, ensure_ascii=False)
