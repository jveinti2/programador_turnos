"""Quick validation script to check constraints."""
import json
import math

def time_to_block(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 2 + (m // 30)

def calculate_hours(start, end):
    return (time_to_block(end) - time_to_block(start)) * 0.5

# Load schedules
with open('output/schedules.json', 'r', encoding='utf-8') as f:
    schedules = json.load(f)

print("=" * 80)
print("CONSTRAINT VALIDATION")
print("=" * 80)

lunch_violations = []
break_violations = []

for agent in schedules:
    for day_name, shift in agent['schedule'].items():
        start = shift['start']
        end = shift['end']
        breaks = shift['break']
        disconnected = shift['disconnected']

        # Calculate effective hours
        total_hours = calculate_hours(start, end)
        lunch_hours = sum(calculate_hours(l['start'], l['end']) for l in disconnected)
        effective_hours = total_hours - lunch_hours

        # Check break frequency (ceil logic)
        required_breaks = math.ceil(effective_hours / 3) if effective_hours > 0 else 0
        actual_breaks = len(breaks)

        if actual_breaks < required_breaks:
            break_violations.append({
                'agent': agent['id'],
                'day': day_name,
                'effective_hours': effective_hours,
                'required': required_breaks,
                'actual': actual_breaks
            })

        # Check lunch at boundaries
        start_block = time_to_block(start)
        end_block = time_to_block(end)

        for lunch in disconnected:
            lunch_start_block = time_to_block(lunch['start'])
            lunch_end_block = time_to_block(lunch['end'])

            if lunch_start_block == start_block:
                lunch_violations.append({
                    'agent': agent['id'],
                    'day': day_name,
                    'issue': f"Lunch starts at shift start ({lunch['start']})"
                })

            if lunch_end_block == end_block:
                lunch_violations.append({
                    'agent': agent['id'],
                    'day': day_name,
                    'issue': f"Lunch ends at shift end ({lunch['end']})"
                })

print(f"\nBreak Frequency Violations: {len(break_violations)}")
if break_violations:
    for v in break_violations[:5]:
        print(f"  - {v['agent']} {v['day']}: {v['effective_hours']:.1f}h requires {v['required']} breaks, has {v['actual']}")
    if len(break_violations) > 5:
        print(f"  ... and {len(break_violations) - 5} more")

print(f"\nLunch Boundary Violations: {len(lunch_violations)}")
if lunch_violations:
    for v in lunch_violations[:5]:
        print(f"  - {v['agent']} {v['day']}: {v['issue']}")
    if len(lunch_violations) > 5:
        print(f"  ... and {len(lunch_violations) - 5} more")

if not break_violations and not lunch_violations:
    print("\n[SUCCESS] All constraints satisfied!")
else:
    print(f"\n[ISSUES] {len(break_violations)} break + {len(lunch_violations)} lunch violations")

print("=" * 80)
