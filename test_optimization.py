"""
Script de prueba para optimizacion con datos realistas.
"""
from shift_optimizer import ShiftOptimizer
from shift_optimizer.utils import load_agents, load_demanda
from shift_optimizer.config import OptimizationConfig
import json

print("="*60)
print("PRUEBA DE OPTIMIZACION CON DEMANDA REALISTA 24/7")
print("="*60)

# Cargar configuracion
config = OptimizationConfig.from_yaml('config/reglas.yaml')
print(f"\nConfiguracion cargada:")
print(f"  - Timeout: {config.solver.timeout_segundos}s")
print(f"  - Workers: {config.solver.num_workers}")
print(f"  - Domingos max por agente: {getattr(config.shift_rules, 'domingos_max_por_agente', 'No definido')}")

# Cargar agentes y demanda
agents = load_agents('data/agentes.csv')
demanda = load_demanda('data/demanda.csv')

print(f"\nDatos cargados:")
print(f"  - Agentes disponibles: {len(agents)}")
print(f"  - Demanda maxima: {max(max(day) for day in demanda.data)} agentes")
print(f"  - Demanda promedio: {sum(sum(day) for day in demanda.data) / (7*48):.1f} agentes")

# Analizar demanda por bloques horarios
print(f"\nPatron de demanda (promedio semanal):")
demanda_promedio = [sum(demanda.data[day][block] for day in range(7)) / 7 for block in range(48)]
pico_block = demanda_promedio.index(max(demanda_promedio))
valle_block = demanda_promedio.index(min(demanda_promedio))

def block_to_time_str(block):
    hours = (block * 30) // 60
    minutes = (block * 30) % 60
    return f"{hours:02d}:{minutes:02d}"

print(f"  - Pico: {max(demanda_promedio):.1f} agentes a las {block_to_time_str(pico_block)}")
print(f"  - Valle: {min(demanda_promedio):.1f} agentes a las {block_to_time_str(valle_block)}")

# Ejecutar optimizacion
print(f"\n{'='*60}")
print("INICIANDO OPTIMIZACION")
print("="*60)

# Usar la API de alto nivel
optimizer_api = ShiftOptimizer(config)
result = optimizer_api.optimize(
    agents_path='data/agentes.csv',
    demand_path='data/demanda.csv',
    output_dir='output/'
)

solution = result['solution']

print(f"\n{'='*60}")
print("RESULTADOS")
print("="*60)
print(f"Status: {solution.status}")
print(f"Tiempo de resolucion: {solution.tiempo_resolucion:.2f}s")
print(f"Costo total: {solution.costo_total}")

# Estadisticas de turnos
print(f"\nTurnos generados: {len(solution.shifts)}")
turnos_por_dia = [0] * 7
for shift in solution.shifts:
    turnos_por_dia[shift.dia] += 1

dias = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
print("Turnos por dia:")
for dia, count in enumerate(turnos_por_dia):
    print(f"  {dias[dia]}: {count} turnos")

# Analizar cobertura domingo
print(f"\nCobertura Domingo (dia critico):")
demanda_domingo = demanda.data[6]
cobertura_domingo = solution.cobertura[6]
for block in [0, 12, 20, 28, 40]:  # Muestreo de bloques
    time_str = block_to_time_str(block)
    print(f"  {time_str}: Demanda={demanda_domingo[block]}, Cobertura={cobertura_domingo[block]}, Diff={cobertura_domingo[block] - demanda_domingo[block]}")

# Verificar descansos
print(f"\nVerificacion de constraints (muestra - primeros 5 turnos):")
agentes_con_breaks_en_borde = 0
agentes_con_gaps_largos = 0

def time_to_block(t):
    return (t.hour * 60 + t.minute) // 30

from datetime import datetime, timedelta

for idx, shift in enumerate(solution.shifts[:5]):  # Revisar primeros 5 turnos
    print(f"\nTurno {idx+1} - Agente {shift.agente_id} - {dias[shift.dia]}:")
    print(f"  Horario: {shift.hora_inicio.strftime('%H:%M')} - {shift.hora_fin.strftime('%H:%M')}")

    # Verificar breaks en primera/ultima hora
    if shift.breaks:
        print(f"  Breaks: {len(shift.breaks)}")
        first_break_time = min(b[0] for b in shift.breaks)
        last_break_time = max(b[0] for b in shift.breaks)

        start_plus_1h = (datetime.combine(datetime.today(), shift.hora_inicio) + timedelta(hours=1)).time()
        end_minus_1h = (datetime.combine(datetime.today(), shift.hora_fin) - timedelta(hours=1)).time()

        if first_break_time < start_plus_1h:
            print(f"  ⚠️ Break en primera hora: {first_break_time.strftime('%H:%M')}")
            agentes_con_breaks_en_borde += 1
        if last_break_time > end_minus_1h:
            print(f"  ⚠️ Break en ultima hora: {last_break_time.strftime('%H:%M')}")
            agentes_con_breaks_en_borde += 1

        # Verificar gaps largos
        break_blocks = sorted([time_to_block(b[0]) for b in shift.breaks])
        for i in range(len(break_blocks) - 1):
            gap_blocks = break_blocks[i+1] - break_blocks[i]
            if gap_blocks > 6:  # > 3 horas
                print(f"  ⚠️ Gap largo sin descanso: {gap_blocks * 0.5:.1f} horas")
                agentes_con_gaps_largos += 1
                break

print(f"\n{'='*60}")
print(f"RESUMEN DE VALIDACION (primeros 5 turnos):")
print(f"  - Breaks en bordes: {agentes_con_breaks_en_borde}/5")
print(f"  - Gaps largos (>3h): {agentes_con_gaps_largos}/5")

print(f"\n{'='*60}")
print("PRUEBA COMPLETADA")
print("="*60)
