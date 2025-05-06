import streamlit as st
import pandas as pd
import io
import zipfile
from datetime import datetime
from fpdf import FPDF

# --- Data and Logic ---
strip_options = {59: 43.26, 118: 74.63, 236: 139.06}
power_specs = [
    {'W': 36, 'cost': 26.16},
    {'W': 96, 'cost': 43.04},
]
power_specs.sort(key=lambda s: s['cost']/s['W'])

@st.cache_data
def optimized_allocation(runs, opts, max_connections):
    runs_left = runs.copy()
    allocations = []
    strip_types = sorted(opts.items(), key=lambda x: x[1]/x[0])
    while runs_left:
        best_pair = None
        best_strip = None
        for i, r1 in enumerate(runs_left):
            for j, r2 in enumerate(runs_left):
                if i == j:
                    continue
                total = r1 + r2
                for length, cost in strip_types:
                    if total <= length and (best_pair is None or cost < opts[best_strip]):
                        best_pair = (r1, r2)
                        best_strip = length
        if best_pair:
            allocations.append({'strip_length': best_strip, 'used': best_pair, 'waste': best_strip - sum(best_pair), 'cost': opts[best_strip]})
            runs_left.remove(best_pair[0])
            runs_left.remove(best_pair[1])
        else:
            r = max(runs_left)
            candidates = [(L, C) for L, C in opts.items() if L >= r]
            if candidates:
                length, cost = min(candidates, key=lambda x: x[1])
            else:
                length, cost = max(opts.items(), key=lambda x: x[0])
            allocations.append({'strip_length': length, 'used': (r,), 'waste': length - r, 'cost': cost})
            runs_left.remove(r)
        if sum(len(a['used']) for a in allocations) > max_connections:
            break
    total_conns = sum(len(a['used']) for a in allocations)
    total_cost = sum(a['cost'] for a in allocations)
    total_waste = sum(a['waste'] for a in allocations)
    return allocations, {'connections': total_conns, 'led_cost': total_cost, 'waste': total_waste}


def compute_power(allocations, watt_per_foot, power_specs):
    # Enforce ~15% headroom across each power supply
    headroom_factor = 1.15
    # Slot capacities per wattage
    slot_limits = {s['W']: (10 if s['W']==36 else 30) for s in power_specs}
    # Compute segment loads in watts
    segment_watts = [(length/12) * watt_per_foot for alloc in allocations for length in alloc['used']]

    bins = []
    # Place each load into existing supply or open a new one
    for load in sorted(segment_watts, reverse=True):
        placed = False
        for b in bins:
            # Check slot availability and headroom capacity
            if b['slots'] > 0 and b['used_sum'] + load <= b['capacity']:
                b['loads'].append(load)
                b['used_sum'] += load
                b['remaining'] = b['spec_W'] - b['used_sum']
                b['slots'] -= 1
                placed = True
                break
        if placed:
            continue
        # Need to create a new supply
        # Choose the most cost-efficient spec that meets headroom for this load
        suitable = [s for s in power_specs if s['W'] >= load * headroom_factor]
        if suitable:
            spec = min(suitable, key=lambda s: s['cost']/s['W'])
        else:
            spec = max(power_specs, key=lambda s: s['W'])
        spec_W = spec['W']
        # Effective capacity considering headroom
        capacity = spec_W / headroom_factor
        bins.append({
            'spec_W': spec_W,
            'W': spec_W,
            'Cost': spec['cost'],
            'capacity': capacity,
            'used_sum': load,
            'remaining': spec_W - load,
            'slots': slot_limits.get(spec_W, 0) - 1,
            'loads': [load]
        })

    # Build DataFrame for display
    df = pd.DataFrame([
        {
            'Supply #': i+1,
            'Wattage': b['W'],
            'Cost': b['Cost'],
            'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']),
            'Remaining (W)': round(b['remaining'], 1)
        }
        for i, b in enumerate(bins)
    ])
    total_cost = df['Cost'].sum()
    counts = df['Wattage'].value_counts().to_dict()
    return df, total_cost, counts

# --- Configuration Settings ---
if "watt_per_foot" not in st.session_state:
    st.session_state["watt_per_foot"] = 3.0
if "strip_options" not in st.session_state:
    st.session_state["strip_options"] = strip_options.copy()
if "power_specs" not in st.session_state:
    st.session_state["power_specs"] = power_specs.copy()

# Sidebar form for configuration and the rest of the app remains unchanged
# ... [omitted for brevity]
