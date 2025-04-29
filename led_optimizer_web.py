import streamlit as st
import pandas as pd
import io
import zipfile
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
        # try pairing two runs
        best_pair = None
        best_strip = None
        for i, r1 in enumerate(runs_left):
            for j, r2 in enumerate(runs_left):
                if i == j: continue
                total = r1 + r2
                for length, cost in strip_types:
                    if total <= length and (best_pair is None or cost < opts[best_strip]):
                        best_pair, best_strip = (r1, r2), length
        if best_pair:
            allocations.append({'strip_length': best_strip,
                                'used': best_pair,
                                'waste': best_strip - sum(best_pair),
                                'cost': opts[best_strip]})
            runs_left.remove(best_pair[0])
            runs_left.remove(best_pair[1])
        else:
            # single-run
            r = max(runs_left)
            candidates = [(L, C) for L, C in opts.items() if L >= r]
            if candidates:
                length, cost = min(candidates, key=lambda x: x[1])
            else:
                length, cost = max(opts.items(), key=lambda x: x[0])
            allocations.append({'strip_length': length,
                                'used': (r,),
                                'waste': length - r,
                                'cost': cost})
            runs_left.remove(r)
        if sum(len(a['used']) for a in allocations) > max_connections:
            break
    total_conns = sum(len(a['used']) for a in allocations)
    total_led_cost = sum(a['cost'] for a in allocations)
    total_waste = sum(a['waste'] for a in allocations)
    return allocations, {'connections': total_conns,
                         'led_cost': total_led_cost,
                         'waste': total_waste}

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
            spec = next((s for s in power_specs if s['W'] >= load),                               power_specs[-1])
        bins.append({'W': spec['W'],
                     'cost': spec['cost'],
                     'remaining': spec['W']-load,
                     'slots': 9,
                     'loads': [load]})
    df = pd.DataFrame([{
        'Supply #': i+1,
        'Wattage': b['W'],
        'Cost': f"${b['cost']:.2f}",
        'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']),
        'Remaining (W)': f"{b['remaining']:.1f}W"
    } for i, b in enumerate(bins)])
    total_cost = sum(b['cost'] for b in bins)
    counts = df['Wattage'].value_counts().to_dict()
    return df, total_cost, counts

# --- UI: Batch Orders ---
st.title("LED Runs and Power Supply Wattage Optimizer")

# Initialize DataFrame
cols = ["Order"] + [f"Run{i+1}" for i in range(10)]
if "df_orders" not in st.session_state:
    st.session_state.df_orders = pd.DataFrame([[""]*len(cols) for _ in range(5)], columns=cols)

st.subheader("Enter order number and runs (inches), then optimize")
# Spreadsheet-like editor
df_edited = st.data_editor(
    st.session_state.df_orders,
    num_rows="dynamic",
    use_container_width=True
)
# Clean data: blank stays blank
df_clean = df_edited.replace({None: "", 'None': ""}).fillna("")
st.session_state.df_orders = df_clean

# Optimize button\if st.button("Optimize All Orders"):
    # Parse orders
    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in['Order'].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row['Order']).strip()
        runs = []
        for c in cols[1:]:
            v = row[c]
            if v is None or v == "":
                continue
            runs.append(float(v))
        orders.append({'order': o_no, 'runs': runs})

    # Global optimization
    all_runs = [r for o in orders for r in o['runs']]
    alloc_all, sum_all = optimized_allocation(all_runs, strip_options,                                max_connections=len(all_runs))
    df_led = pd.DataFrame(alloc_all)

    # Overall Summary
    st.header("Overall Summary")
    rolls = df_led['strip_length'].value_counts().reindex([59,118,236], fill_value=0)
    costs = {L: rolls[L]*strip_options[L] for L in rolls.index}
    df_rolls = pd.DataFrame({'Count': rolls, 'Cost': pd.Series(costs)})
    df_rolls['Cost'] = df_rolls['Cost'].apply(lambda x: f"${x:.2f}")
    df_rolls['Count'] = df_rolls['Count'].replace(0, "")
    df_rolls['Cost'] = df_rolls['Cost'].replace("$0.00", "")
    st.dataframe(df_rolls, use_container_width=True)
    st.write(f"**Total LED Cost:** ${sum_all['led_cost']:.2f}")

    # Power summary
    df_ps, ps_cost, ps_counts = compute_power(alloc_all)
    st.subheader("Power Summary")
    df_ps_disp = df_ps.copy()
    df_ps_disp['Count'] = df_ps_disp['Wattage'].map(ps_counts).replace(0, "")
    df_ps_disp['Cost'] = df_ps_disp['Cost'].apply(lambda x: f"${float(x.strip('$')):.2f}")
    st.dataframe(df_ps_disp, use_container_width=True)
    st.write(f"**Total Supply Cost:** ${ps_cost:.2f}")

    # Order Details
    st.header("Order Details")
    seen = set()
    for od in orders:
        if od['order'] in seen:
            continue
        seen.add(od['order'])
        alloc, _ = optimized_allocation(od['runs'], strip_options, max_connections=10)
        df_o = pd.DataFrame(alloc)
        df_o['cost'] = df_o['cost'].apply(lambda x: f"${x:.2f}")
        st.subheader(f"Order {od['order']}")
        st.dataframe(df_o, use_container_width=True)
        ps_o, cost_o, _ = compute_power(alloc)
        st.dataframe(ps_o, use_container_width=True)
        st.write(f"**Supply Cost:** ${cost_o:.2f}")

    # Export ZIP of CSVs
    buf = io.BytesIO()
    folder = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f"{folder}/OverallRolls.csv", df_rolls.to_csv())
        zf.writestr(f"{folder}/OverallPower.csv", df_ps.to_csv(index=False))
        zf.writestr(f"{folder}/GlobalSummary.csv", pd.DataFrame([sum_all]).to_csv(index=False))
    buf.seek(0)
    st.download_button("Export Data", data=buf.getvalue(), file_name=f"{folder}.zip", mime='application/zip')

st.markdown("---")
st.write("*Optimized for cost and waste; Power Supplies sized with 20–25% headroom.*")
