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
    """
    Allocate LED runs to strips, returning list of allocations and summary.
    Ensures no ValueError by safe fallback for single runs.
    """
    runs_left = runs.copy()
    allocations = []
    # sort strip types by cost per inch
    strip_types = sorted(opts.items(), key=lambda x: x[1]/x[0])
    while runs_left:
        # try to pair two runs greedily
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
                'order_runs': best_pair,
                'strip_length': best_strip,
                'used': best_pair,
                'waste': best_strip - sum(best_pair),
                'cost': opts[best_strip]
            })
            runs_left.remove(best_pair[0])
            runs_left.remove(best_pair[1])
        else:
            # single-run allocation
            r = max(runs_left)
            # find candidates that fit
            candidates = [(length, cost) for length, cost in opts.items() if length >= r]
            if candidates:
                # cheapest roll among candidates
                length, cost = min(candidates, key=lambda x: x[1])
            else:
                # no roll fits: pick largest roll length
                length, cost = max(opts.items(), key=lambda x: x[0])
            allocations.append({
                'order_runs': (r,),
                'strip_length': length,
                'used': (r,),
                'waste': length - r,
                'cost': cost
            })
            runs_left.remove(r)
        # respect max_connections
        if sum(len(a['used']) for a in allocations) > max_connections:
            break
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

# --- UI: Batch Orders (Spreadsheet-like) ---
# Define initial DataFrame with 5 blank rows and 10 run columns
cols = ["Order"] + [f"Run{i+1}" for i in range(10)]
if "df_orders" not in st.session_state:
    st.session_state.df_orders = pd.DataFrame([[""]*(len(cols)) for _ in range(5)], columns=cols)

st.subheader("Orders (Edit directly like a spreadsheet)")
# Spreadsheet-like input editor
try:
    df_edited = st.data_editor(
        st.session_state.df_orders,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
except AttributeError:
    st.error("Your Streamlit version does not support data_editor. Please upgrade to ≥1.19.")
    st.stop()
# Persist edits
st.session_state.df_orders = df_edited
st.session_state.df_orders = df_edited

# Optimize button triggers calculations
if st.button("Optimize All Orders"):
    # Parse orders DataFrame
    df_in = st.session_state.df_orders.copy()
    # Filter out empty rows (no Order#)
    df_in = df_in[df_in['Order'].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        order_no = str(row['Order']).strip()
        runs = []
        runs_valid = True
        # parse run columns
        for c in cols[1:]:
            val = str(row[c]).strip()
            if val:
                try:
                    runs.append(float(val))
                except ValueError:
                    st.error(f"Invalid run value '{val}' in order {order_no}")
                    runs_valid = False
                    break
        if not runs_valid:
            st.stop()
        orders.append({'order': order_no, 'runs': runs})

    # Global optimization
    global_runs = [r for o in orders for r in o['runs']]
    global_alloc, global_sum = optimized_allocation(global_runs, strip_options, max_connections=len(global_runs))
    df_led = pd.DataFrame(global_alloc)
    df_ps, tot_ps_cost, ps_counts = compute_power(global_alloc)

    # Per-order details
    order_details = []
    total_unit_waste = 0
    for o in orders:
        alloc, summ = optimized_allocation(o['runs'], strip_options, max_connections=10)
        total_unit_waste += summ['waste']
        order_details.append({'order': o['order'], 'alloc': alloc, 'sum': summ})
    waste_used = total_unit_waste - global_sum['waste']

    # Overall summary
    st.subheader("Overall Summary")
    rolls = df_led['strip_length'].value_counts().reindex([59,118,236], fill_value=0)
    led_costs = {L: rolls[L]*strip_options[L] for L in rolls.index}
    df_rolls = pd.DataFrame({'Count': rolls, 'Cost': pd.Series(led_costs)})
    st.dataframe(df_rolls)
    st.write(f"**Total LED Cost:** ${global_sum['led_cost']:.2f}")
    # Power summary
    df_power_summary = pd.DataFrame(
        [(W, ps_counts.get(W,0), ps_counts.get(W,0)*next(s['cost'] for s in power_specs if s['W']==W))
         for W in sorted(ps_counts)],
        columns=['Wattage','Count','Total Cost']
    )
    st.dataframe(df_power_summary)
    st.write(f"**Total Supply Cost:** ${tot_ps_cost:.2f}")
    st.write(f"**Total Waste (in):** {global_sum['waste']:.2f}")
    st.write(f"**Inches Used from Waste:** {waste_used:.2f}")

    # Orders summary table
    st.subheader("Orders Summary")
    summary_rows = []
    for od in order_details:
        row = {'Order': od['order']}
        # count rolls
        summary_rows.append(row)
    # Display empty, detailed below or skip if not needed
    # Per-order details
    st.subheader("Order Details")
    for od in order_details:
        with st.expander(f"Order {od['order']}"):
            df_o = pd.DataFrame(od['alloc'])
            df_o.index += 1
            st.dataframe(df_o)
            ps_o, c_o, _ = compute_power(od['alloc'])
            ps_o = ps_o.drop(columns=['Supply #']).set_index('Wattage')
            st.dataframe(ps_o)
            st.write(f"**Supply Cost:** ${c_o:.2f}")

    # Export Data as ZIP (CSV files)
    buffer = io.BytesIO()
    folder_name = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    with zipfile.ZipFile(buffer, 'w') as zf:
        # overall rolls
        zf.writestr(f"{folder_name}/OverallRolls.csv", df_rolls.to_csv(index=True))
        zf.writestr(f"{folder_name}/OverallPower.csv", df_power_summary.to_csv(index=False))
        zf.writestr(f"{folder_name}/GlobalSummary.csv", pd.DataFrame([global_sum]).to_csv(index=False))
        # per-order files
        for od in order_details:
            zf.writestr(f"{folder_name}/{od['order']}_alloc.csv", pd.DataFrame(od['alloc']).to_csv(index=False))
            zf.writestr(f"{folder_name}/{od['order']}_summary.csv", pd.DataFrame([od['sum']]).to_csv(index=False))
    buffer.seek(0)
    st.download_button("Export Data", data=buffer.getvalue(), file_name=f"{folder_name}.zip", mime='application/zip')

st.markdown("---")
st.write("*This data is optimized for reducing cost and waste. Power Supply requirements are calculated with headroom of between 20%-25%*")
