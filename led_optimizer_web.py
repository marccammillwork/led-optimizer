import streamlit as st
import pandas as pd

# --- Data and Logic ---
strip_options = {59: 43.26, 118: 74.63, 236: 139.06}
power_specs = [
    {'W': 24, 'cost': 50.41},
    {'W': 36, 'cost': 26.16},
    {'W': 60, 'cost': 82.72},
    {'W': 96, 'cost': 98.85},
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
            allocations.append({
                'strip_length': best_strip,
                'used': best_pair,
                'waste': best_strip - sum(best_pair),
                'cost': opts[best_strip]
            })
            runs_left.remove(best_pair[0])
            runs_left.remove(best_pair[1])
        else:
            r = max(runs_left)
            length, cost = min((s for s in opts.items() if s[0] >= r), key=lambda x: x[1])
            allocations.append({
                'strip_length': length,
                'used': (r,),
                'waste': length - r,
                'cost': cost
            })
            runs_left.remove(r)
        if sum(len(a['used']) for a in allocations) > max_connections:
            break

    total_conns = sum(len(a['used']) for a in allocations)
    total_led_cost = sum(a['cost'] for a in allocations)
    total_waste = sum(a['waste'] for a in allocations)
    return allocations, {
        'connections': total_conns,
        'led_cost': total_led_cost,
        'waste': total_waste
    }

@st.cache_data
def compute_power(allocations):
    segment_watts = [(l/12)*3 for alloc in allocations for l in alloc['used']]
    bins = []
    for load in sorted(segment_watts, reverse=True):
        placed = False
        for b in bins:
            if b['slots'] > 0 and b['remaining'] >= load:
                b['remaining'] -= load
                b['slots'] -= 1
                b['loads'].append(load)
                placed = True
                break
        if placed:
            continue
        chosen = next((s for s in power_specs if s['W'] >= load * 1.2), None)
        if chosen is None:
            chosen = next((s for s in power_specs if s['W'] >= load), None)
        bins.append({
            'W': chosen['W'],
            'cost': chosen['cost'],
            'remaining': chosen['W'] - load,
            'slots': 9,
            'loads': [load]
        })
    df = pd.DataFrame([{   
        'Supply #': i+1,
        'Wattage': b['W'],
        'Cost': b['cost'],
        'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']),
        'Remaining (W)': round(b['remaining'], 1)
    } for i, b in enumerate(bins)])
    total_cost = sum(b['cost'] for b in bins)
    return df, total_cost

# --- Streamlit UI ---
st.title("LED Strip & Power Supply Optimizer")

if 'runs' not in st.session_state:
    st.session_state.runs = ["" for _ in range(7)]

st.subheader("Enter LED runs (inches):")

new_runs = []
for idx, run_value in enumerate(st.session_state.runs):
    value = st.text_input(f"Run {idx+1}", value=run_value, key=f"run_{idx}")
    new_runs.append(value)

st.session_state.runs = new_runs

# If the last box has value, automatically add a new empty box
if st.session_state.runs and st.session_state.runs[-1].strip():
    st.session_state.runs.append("")

if st.button("Optimize"):
    try:
        runs = [float(x) for x in st.session_state.runs if x.strip()]
    except ValueError:
        st.error("Enter valid numeric run lengths.")
    else:
        allocations, summary = optimized_allocation(runs, strip_options, max_connections=10)
        st.subheader("LED Allocation")
        df_led = pd.DataFrame([{
            'Strip Length': a['strip_length'],
            'Used Runs': a['used'],
            'Waste (in)': a['waste'],
            'Cost': a['cost']
        } for a in allocations])
        st.dataframe(df_led)
        st.write(f"**Total Connections:** {summary['connections']}")
        st.write(f"**Total LED Cost:** ${summary['led_cost']:.2f}")
        st.write(f"**Total Waste:** {summary['waste']} in")

        st.subheader("Power Supply Plan")
        df_ps, ps_cost = compute_power(allocations)
        st.dataframe(df_ps)
        st.write(f"**Total Supply Cost:** ${ps_cost:.2f}")

st.markdown("---")
st.write("*Optimized for cost with 20% oversize for supplies.*")
