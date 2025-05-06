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

@st.cache_data
def compute_power(allocations, watt_per_foot, power_specs):
     import math
     # Power supply sizing with ~10% headroom
     headroom_factor = 1.10  # 10% headroom
     # Slot capacities per wattage
     slot_limits = {s['W']: (10 if s['W']==36 else 30) for s in power_specs}
     # Compute loads in watts for LED segments
     segment_watts = [(length/12)*watt_per_foot for alloc in allocations for length in alloc['used']]
     # Prepare supply bins
     bins = []
     # Identify highest capacity spec
     max_spec = max(power_specs, key=lambda s: s['W'])
     for load in sorted(segment_watts, reverse=True):
         placed = False
         required = load * headroom_factor
         # Try placing into existing bins
         for b in bins:
             if b['slots'] > 0 and b['remaining'] >= required:
                 b['remaining'] -= load
                 b['slots'] -= 1
                 b['loads'].append(load)
                 placed = True
                 break
         if placed:
             continue
         # Need new supply
         suitable = [s for s in power_specs if s['W'] >= required]
         if suitable:
             # Pick cheapest absolute cost
             spec = min(suitable, key=lambda s: s['cost'])
         else:
             spec = max_spec
         bins.append({
             'W': spec['W'],
             'cost': spec['cost'],
             'remaining': spec['W'] - load,
             'slots': slot_limits.get(spec['W'], 0) - 1,
             'loads': [load]
         })
     # Build DataFrame
     df = pd.DataFrame([
         {
             'Supply #': i+1,
             'Wattage': b['W'],
             'Cost': b['cost'],
             'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']),
             'Remaining (W)': round(b['remaining'], 1)
         }
         for i, b in enumerate(bins)
     ])
     total_cost = df['Cost'].sum()
     counts = df['Wattage'].value_counts().to_dict()
     return df, total_cost, counts

# --- Configuration Settings ---
