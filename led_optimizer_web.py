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
        best_pair = None; best_strip = None
        for i, r1 in enumerate(runs_left):
            for j, r2 in enumerate(runs_left):
                if i == j: continue
                total = r1 + r2
                for length, cost in strip_types:
                    if total <= length and (best_pair is None or cost < opts[best_strip]):
                        best_pair, best_strip = (r1, r2), length
        if best_pair:
            allocations.append({'order_runs': best_pair, 'strip_length': best_strip,
                                'used': best_pair, 'waste': best_strip - sum(best_pair),
                                'cost': opts[best_strip]})
            runs_left.remove(best_pair[0]); runs_left.remove(best_pair[1])
        else:
            r = max(runs_left)
            length, cost = min((s for s in opts.items() if s[0] >= r), key=lambda x: x[1])
            allocations.append({'order_runs': (r,), 'strip_length': length,
                                'used': (r,), 'waste': length - r,
                                'cost': cost})
            runs_left.remove(r)
        if sum(len(a['used']) for a in allocations) > max_connections: break
    total_conns = sum(len(a['used']) for a in allocations)
    total_led_cost = sum(a['cost'] for a in allocations)
    total_waste = sum(a['waste'] for a in allocations)
    return allocations, {'connections': total_conns, 'led_cost': total_led_cost, 'waste': total_waste}

@st.cache_data
def compute_power(allocations):
    segment_watts = [(l/12)*3 for a in allocations for l in a['used']]
    bins = []
    for load in sorted(segment_watts, reverse=True):
        placed = False
        for b in bins:
            if b['slots']>0 and b['remaining']>=load:
                b['remaining']-=load; b['slots']-=1; b['loads'].append(load); placed=True; break
        if placed: continue
        chosen = next((s for s in power_specs if s['W']>=load*1.2), None)
        if not chosen: chosen = next((s for s in power_specs if s['W']>=load), None)
        bins.append({'W': chosen['W'], 'cost': chosen['cost'],
                     'remaining': chosen['W']-load, 'slots':9, 'loads':[load]})
    df = pd.DataFrame([{ 'Supply #': i+1, 'Wattage':b['W'], 'Cost':b['cost'],
                         'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']),
                         'Remaining (W)': round(b['remaining'],1) }
                       for i,b in enumerate(bins)])
    total_cost = sum(b['cost'] for b in bins)
    counts = df['Wattage'].value_counts().to_dict()
    return df, total_cost, counts

# --- UI: Batch Orders ---
st.title("LED Strip & Power Supply Optimizer (Batch Mode)")

# Init 5 blank orders of 10 runs
if 'orders' not in st.session_state:
    st.session_state.orders = [{'order':'','runs':['']*10} for _ in range(5)]

# Input grid headers
cols = st.columns([1]+[1]*10)
cols[0].write("**Order #**")
for i in range(10): cols[i+1].write(f"**Run {i+1} (in)**")

# Input rows
new_orders=[]
for idx, o in enumerate(st.session_state.orders):
    row = st.columns([1]+[1]*10)
    order_no = row[0].text_input(f"ord_{idx}", o['order'], key=f"ord_{idx}")
    runs=[]
    for j in range(10):
        val = row[j+1].text_input(f"run_{idx}_{j}", o['runs'][j], key=f"run_{idx}_{j}")
        runs.append(val)
    new_orders.append({'order':order_no,'runs':runs})
    # auto-add if last row populated
    if idx==len(st.session_state.orders)-1 and (order_no.strip() or any(r.strip() for r in runs)):
        st.session_state.orders.append({'order':'','runs':['']*10})
st.session_state.orders=new_orders

# Optimize button
event = st.button("Optimize All Orders")
if event:
    # Collect global runs
    try:
        global_runs = [float(r) for o in st.session_state.orders for r in o['runs'] if r.strip()]
    except:
        st.error("Please ensure all run inputs are numeric.")
    else:
        # Global optimization
        global_alloc, global_sum = optimized_allocation(global_runs, strip_options, max_connections=10)
        df_led = pd.DataFrame(global_alloc)
        df_ps, tot_ps_cost, ps_counts = compute_power(global_alloc)
        # Per-order details and wasted inches
        total_unit_waste=0
        order_details=[]
        for o in st.session_state.orders:
            runs = [float(r) for r in o['runs'] if r.strip()]
            if not runs: continue
            alloc, summ = optimized_allocation(runs, strip_options, max_connections=10)
            total_unit_waste+=summ['waste']
            order_details.append({'order':o['order'],'alloc':alloc,'sum':summ})
        waste_used = total_unit_waste - global_sum['waste']

        # Overall summary
        st.subheader("Overall Summary")
        rolls = df_led['strip_length'].value_counts().reindex([59,118,236],fill_value=0)
        led_costs={L:rolls[L]*strip_options[L] for L in rolls.index}
        df_rolls=pd.DataFrame({'Count':rolls,'Cost':pd.Series(led_costs)})
        st.dataframe(df_rolls)
        st.write(f"**Total LED Cost:** ${global_sum['led_cost']:.2f}")
        # Power summary
        df_power_summary=pd.DataFrame(
            [(W,ps_counts.get(W,0),ps_counts.get(W,0)*next(s['cost'] for s in power_specs if s['W']==W))
             for W in sorted(ps_counts)],
            columns=['Wattage','Count','Total Cost']
        )
        st.dataframe(df_power_summary)
        st.write(f"**Total Supply Cost:** ${tot_ps_cost:.2f}")
        st.write(f"**Total Waste (in):** {global_sum['waste']:.2f}")
        st.write(f"**Inches Used from Waste:** {waste_used:.2f}")

                # Per-order summary table
        st.subheader("Orders Summary")
        rows = []
        for od in order_details:
            # collect runs for this order
            runs_list = [float(r) for o in st.session_state.orders if o['order']==od['order'] for r in o['runs'] if r.strip()]
            counts59 = sum(1 for a in od['alloc'] if a['strip_length']==59)
            counts118 = sum(1 for a in od['alloc'] if a['strip_length']==118)
            counts236 = sum(1 for a in od['alloc'] if a['strip_length']==236)
            _, _, counts_sup = compute_power(od['alloc'])
            # build row dict
            row = {'Order': od['order']}
            for i in range(10):
                key = f'Run{i+1}'
                row[key] = runs_list[i] if i < len(runs_list) else ''
            row['59"'] = counts59
            row['118"'] = counts118
            row['236"'] = counts236
            row['36W'] = counts_sup.get(36, 0)
            row['60W'] = counts_sup.get(60, 0)
            row['96W'] = counts_sup.get(96, 0)
            rows.append(row)
        df_orders = pd.DataFrame(rows).fillna('')
        df_orders.index += 1
        st.dataframe(df_orders)

        # Expanders for details
        st.subheader("Order Details")
        for od in order_details:
            with st.expander(f"Order {od['order']}"):
                df_o=pd.DataFrame(od['alloc'])
                df_o.index+=1
                st.dataframe(df_o)
                ps_o,c_o,_=compute_power(od['alloc'])
                ps_o=ps_o.drop(columns=['Supply #']).set_index(ps_o['Wattage'])
                st.dataframe(ps_o)
                st.write(f"**Supply Cost:** ${c_o:.2f}")

st.markdown("---")
st.write("*Batch optimization across orders with global and per-order summaries.*")("---")
st.write("*Batch optimization across orders, global and per-order breakdown.*")
