import streamlit as st
import pandas as pd
import io
from datetime import datetime

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
                if i == j: continue
                total = r1 + r2
                for length, cost in strip_types:
                    if total <= length and (best_pair is None or cost < opts[best_strip]):
                        best_pair = (r1, r2)
                        best_strip = length
        if best_pair:
            allocations.append({'strip_length': best_strip, 'used': best_pair,
                                'waste': best_strip - sum(best_pair), 'cost': opts[best_strip]})
            runs_left.remove(best_pair[0])
            runs_left.remove(best_pair[1])
        else:
            r = max(runs_left)
            candidates = [(L, C) for L, C in opts.items() if L >= r]
            if candidates:
                length, cost = min(candidates, key=lambda x: x[1])
            else:
                length, cost = max(opts.items(), key=lambda x: x[0])
            allocations.append({'strip_length': length, 'used': (r,),
                                'waste': length - r, 'cost': cost})
            runs_left.remove(r)
        if sum(len(a['used']) for a in allocations) > max_connections:
            break
    total_conns = sum(len(a['used']) for a in allocations)
    total_cost = sum(a['cost'] for a in allocations)
    total_waste = sum(a['waste'] for a in allocations)
    return allocations, {'connections': total_conns, 'led_cost': total_cost, 'waste': total_waste}

@st.cache_data
def compute_power(allocations):
    segment_watts = [(l/12)*3 for a in allocations for l in a['used']]
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
        if placed: continue
        spec = next((s for s in power_specs if s['W'] >= load*1.2), None)
        if not spec:
            spec = next((s for s in power_specs if s['W'] >= load), power_specs[-1])
        bins.append({'W': spec['W'], 'cost': spec['cost'],
                     'remaining': spec['W'] - load, 'slots': 9, 'loads': [load]})
    df = pd.DataFrame([{'Supply #': i+1,
                        'Wattage': b['W'],
                        'Cost': b['cost'],
                        'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']),
                        'Remaining (W)': round(b['remaining'], 1)}
                       for i, b in enumerate(bins)])
    total_cost = df['Cost'].sum()
    counts = df['Wattage'].value_counts().to_dict()
    return df, total_cost, counts

# --- UI: Batch Orders ---
st.title("LED Strip & Power Supply Optimizer (Batch)")
cols = ["Order"] + [f"Run{i+1}" for i in range(10)]
if "df_orders" not in st.session_state:
    st.session_state.df_orders = pd.DataFrame([ [""]*len(cols) for _ in range(5)], columns=cols)

st.subheader("Enter Orders and Runs (Tab to navigate, paste rows)")
df_edited = st.data_editor(st.session_state.df_orders, num_rows="dynamic", use_container_width=True)
df_clean = df_edited.replace({None: "", "None": ""}).fillna("")
st.session_state.df_orders = df_clean

if st.button("Optimize All Orders"):
    # Parse and optimize
    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in['Order'].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row['Order']).strip()
        runs = [float(row[c]) for c in cols[1:] if row[c] not in ("", None)]
        orders.append({'order': o_no, 'runs': runs})

    global_runs = [r for o in orders for r in o['runs']]
    alloc_all, sum_all = optimized_allocation(global_runs, strip_options, max_connections=len(global_runs))
    df_led = pd.DataFrame(alloc_all)
    df_power_all, ps_cost, ps_counts = compute_power(alloc_all)

    # Track per-order and reuse
    order_details, total_unit_waste = [], 0
    for o in orders:
        alloc, summ = optimized_allocation(o['runs'], strip_options, max_connections=10)
        total_unit_waste += summ['waste']
        order_details.append({'order': o['order'], 'alloc': alloc, 'sum': summ})
    waste_used = total_unit_waste - sum_all['waste']

    # Compute dollars saved
    scrap_pre, scrap_post = {}, {}
    for od in order_details:
        for a in od['alloc']:
            scrap_pre[a['strip_length']] = scrap_pre.get(a['strip_length'], 0) + a['waste']
    for a in alloc_all:
        scrap_post[a['strip_length']] = scrap_post.get(a['strip_length'], 0) + a['waste']
    dollars_saved = sum(
        max(0, scrap_pre[L] - scrap_post.get(L, 0)) * (strip_options[L]/L)
        for L in scrap_pre
    )

    # Display Order Details
    st.header("Order Details")
    seen = set()
    for od in order_details:
        if od['order'] in seen: continue
        seen.add(od['order'])
        with st.expander(f"Order {od['order']}"):
            df_o = pd.DataFrame(od['alloc'])
            df_o.index += 1
            df_o['cost'] = df_o['cost'].apply(lambda x: f"${x:.2f}")
            st.dataframe(df_o, use_container_width=True)
            ps_o, cost_o, _ = compute_power(od['alloc'])
            ps_o['Cost'] = ps_o['Cost'].apply(lambda x: f"${x:.2f}")
            ps_o['Remaining (W)'] = ps_o['Remaining (W)'].apply(lambda x: f"{x:.1f}W")
            st.dataframe(ps_o.drop(columns=['Supply #']), use_container_width=True, hide_index=True)
            st.write(f"**Supply Cost:** ${cost_o:.2f}")

    # Cutoffs expander
    scrap_list = [a['waste'] for a in alloc_all if a['waste']>0]
    df_cutoffs = pd.DataFrame({'Cutoff Number': range(1, len(scrap_list)+1), 'Length': scrap_list})
    total_cut_len = df_cutoffs['Length'].sum()
    total_row = pd.DataFrame([{'Cutoff Number': 'Total', 'Length': total_cut_len}])
    df_cutoffs = pd.concat([df_cutoffs, total_row], ignore_index=True)
    with st.expander("Cutoffs"):
        st.dataframe(df_cutoffs, use_container_width=True)
