import streamlit as st
import pandas as pd
import io
import zipfile
from datetime import datetime
from fpdf import FPDF

# --- Data and Logic ---
strip_options = {59: 43.26, 118: 74.63, 236: 139.06}
power_specs = [
    {'W': 24, 'cost': 50.41},
    {'W': 36, 'cost': 26.16},
    {'W': 60, 'cost': 82.72},
    {'W': 96, 'cost': 93.91},
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
        # Try pairing two runs
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
            # Single-run
            r = max(runs_left)
            candidates = [(L, C) for L, C in opts.items() if L >= r]
            if candidates:
                length, cost = min(candidates, key=lambda x: x[1])
            else:
                length, cost = max(opts.items(), key=lambda x: x[0])
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
    total_cost = sum(a['cost'] for a in allocations)
    total_waste = sum(a['waste'] for a in allocations)
    return allocations, {
        'connections': total_conns,
        'led_cost': total_cost,
        'waste': total_waste
    }

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
        if placed:
            continue
        spec = next((s for s in power_specs if s['W'] >= load*1.2), None)
        if not spec:
            spec = next((s for s in power_specs if s['W'] >= load), power_specs[-1])
        bins.append({
            'W': spec['W'],
            'cost': spec['cost'],
            'remaining': spec['W'] - load,
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
    total_cost = df['Cost'].sum()
    counts = df['Wattage'].value_counts().to_dict()
    return df, total_cost, counts

# --- UI: Batch Orders ---
st.title("LED Strip & Power Supply Optimizer (Batch)")

# Initialize DataFrame
cols = ["Order"] + [f"Run{i+1}" for i in range(10)]
if "df_orders" not in st.session_state:
    st.session_state.df_orders = pd.DataFrame(
        [[""] * len(cols) for _ in range(5)], columns=cols
    )

st.subheader("Enter Orders and Runs (Tab to navigate, paste rows)")
df_edited = st.data_editor(
    st.session_state.df_orders,
    num_rows="dynamic",
    use_container_width=True
)
# Clean pasted/edited data
st.session_state.df_orders = df_edited.replace({None: "", "None": ""}).fillna("")

if st.button("Optimize All Orders"):
    # Build orders list
    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in["Order"].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row["Order"]).strip()
        runs = []
        for c in cols[1:]:
            val = row[c]
            if val in ("", None):
                continue
            try:
                runs.append(float(val))
            except:
                st.error(f"Invalid run value '{val}' in order {o_no}")
                st.stop()
        orders.append({"order": o_no, "runs": runs})

    # Global optimization
    global_runs = [r for o in orders for r in o["runs"]]
    alloc_all, sum_all = optimized_allocation(
        global_runs, strip_options, max_connections=len(global_runs)
    )
    df_led = pd.DataFrame(alloc_all)
    df_ps, ps_cost, ps_counts = compute_power(alloc_all)

    # Per-order details & waste calculation
    order_details = []
    total_unit_waste = 0
    for o in orders:
        alloc, summ = optimized_allocation(
            o["runs"], strip_options, max_connections=10
        )
        total_unit_waste += summ["waste"]
        order_details.append({"order": o["order"], "alloc": alloc, "sum": summ})
    waste_used = total_unit_waste - sum_all["waste"]

    # Order-level statistics
    total_orders = len(order_details)
    order_costs = []
    for od in order_details:
        led_cost = sum(item['cost'] for item in od['alloc'])
        _, supply_cost, _ = compute_power(od['alloc'])
        order_costs.append((od['order'], led_cost + supply_cost))
    if total_orders:
        avg_cost = sum(t for _, t in order_costs) / total_orders
        min_order, min_cost = min(order_costs, key=lambda x: x[1])
        max_order, max_cost = max(order_costs, key=lambda x: x[1])
    else:
        avg_cost, min_order, min_cost, max_order, max_cost = 0.0, "N/A", 0.0, "N/A", 0.0

    st.markdown("**Order-level Summary**")
    st.write(f"- **Total Orders:** {total_orders}")
    st.write(f"- **Average Cost per Order:** ${avg_cost:.2f}")
    st.write(f"- **Minimum Order Cost:** ${min_cost:.2f} (Order {min_order})")
    st.write(f"- **Maximum Order Cost:** ${max_cost:.2f} (Order {max_order})")
    st.markdown("---")

    # Overall Summary
    st.header("Overall Summary")
    st.subheader("LEDS")
    rolls = df_led["strip_length"].value_counts().reindex([59,118,236], fill_value=0)
    costs = {L: rolls[L] * strip_options[L] for L in rolls.index}
    df_rolls = pd.DataFrame({"Count": rolls, "Cost": pd.Series(costs)})
    df_rolls["Cost"] = df_rolls["Cost"].apply(lambda x: f"${x:.2f}")
    df_rolls["Count"] = df_rolls["Count"].replace(0, "")
    df_rolls["Cost"] = df_rolls["Cost"].replace("$0.00", "")
    st.dataframe(df_rolls, use_container_width=True)

    st.write(f"**Total LED Cost:** ${sum_all['led_cost']:.2f}")
