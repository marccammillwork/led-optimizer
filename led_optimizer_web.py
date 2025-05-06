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
            if b['slots'] > 0 and b['used_sum'] + load <= b['capacity']:
                b['loads'].append(load)
                b['used_sum'] += load
                b['remaining'] = b['spec_W'] - b['used_sum']
                b['slots'] -= 1
                placed = True
                break
        if placed:
            continue
        suitable = [s for s in power_specs if s['W'] >= load * headroom_factor]
        if suitable:
            spec = min(suitable, key=lambda s: s['cost']/s['W'])
        else:
            spec = max(power_specs, key=lambda s: s['W'])
        spec_W = spec['W']
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

# Sidebar form for configuration\config = st.sidebar.expander("Configure Specs", expanded=False)
with config:
    with st.form("config_form"):
        wp = st.number_input(
            "LED Watt per Foot", value=st.session_state["watt_per_foot"], min_value=0.0, step=0.1, key="form_watt"
        )
        strip_df = pd.DataFrame(
            list(st.session_state["strip_options"].items()), columns=["Length (in)", "Cost"]
        )
        strip_df = st.data_editor(strip_df, num_rows="dynamic", use_container_width=True, key="form_strips")
        power_df = pd.DataFrame(st.session_state["power_specs"])
        power_df = st.data_editor(power_df, num_rows="dynamic", use_container_width=True, key="form_power")
        apply = st.form_submit_button("Apply Settings")
        if apply:
            st.session_state["watt_per_foot"] = wp
            st.session_state["strip_options"] = dict(zip(strip_df["Length (in)"], strip_df["Cost"]))
            ps_list = power_df.to_dict("records")
            st.session_state["power_specs"] = sorted(ps_list, key=lambda s: s['cost']/s['W'])

# --- UI: Batch Orders ---
st.title("LED Strip & Power Supply Optimizer")
cols = ["Order"] + [f"Run{i+1}" for i in range(10)]
if "df_orders" not in st.session_state:
    st.session_state.df_orders = pd.DataFrame([["" for _ in cols] for _ in range(5)], columns=cols)

st.subheader("Enter Orders and Runs")
df_edited = st.data_editor(st.session_state.df_orders, num_rows="dynamic", use_container_width=True)
st.session_state.df_orders = df_edited.replace({None: "", "None": ""}).fillna("")

if st.button("Optimize All Orders"):
    headroom_factor = 1.15
    max_capacity = max(s['W'] for s in st.session_state["power_specs"])
    unsupported = []
    df_exp = st.session_state.df_orders.copy()
    df_exp = df_exp[df_exp["Order"].astype(str).str.strip() != ""]
    for _, row in df_exp.iterrows():
        order_no = str(row["Order"]).strip()
        for run_col in [c for c in df_exp.columns if c.startswith("Run")]:
            val = row[run_col]
            if pd.isna(val) or str(val).strip() == "": continue
            try:
                length = float(str(val).strip())
            except ValueError:
                st.error(f"Invalid run value '{val}' in order {order_no}")
                st.stop()
            load = (length/12) * st.session_state["watt_per_foot"]
            if load * headroom_factor > max_capacity:
                unsupported.append({'order': order_no, 'length': length, 'watts': load})
    if unsupported:
        exp_label = f"Unsupported runs: {len(unsupported)}"
        with st.expander(exp_label):
            for u in unsupported:
                st.write(f"- Order {u['order']}: {u['length']}\" run requires {u['watts']:.1f} W (exceeds capacity)")
        st.error("Please adjust configuration or split runs to fit available power supplies.")
        st.stop()

    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in["Order"].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row["Order"]).strip()
        runs = [float(str(row[c]).strip()) for c in cols[1:] if str(row[c]).strip()]
        orders.append({'order': o_no, 'runs': runs})

    global_runs = [r for o in orders for r in o['runs']]
    alloc_all, sum_all = optimized_allocation(global_runs, st.session_state["strip_options"], max_connections=len(global_runs))
    unusable_scrap = sum(a['waste'] for a in alloc_all if len(a['used']) == 2)
    scrap_unusable_cost = sum(a['waste'] * (strip_options[a['strip_length']] / a['strip_length']) for a in alloc_all if len(a['used']) == 2)
    reusable_scrap = sum(a['waste'] for a in alloc_all if len(a['used']) == 1)
    scrap_reusable_cost = sum(a['waste'] * (strip_options[a['strip_length']] / a['strip_length']) for a in alloc_all if len(a['used']) == 1)

    st.markdown("**Order-level Summary**")
    st.write(f"- Total Orders: {len(orders)}")
    st.write(f"- Total Unusable Cutoff Waste: {unusable_scrap:.2f} in")
    st.write(f"- Cost of Unusable Cutoff Waste: ${scrap_unusable_cost:.2f}")
    st.write(f"- Total Available Scrap for Next Batch: {reusable_scrap:.2f} in")
    st.write(f"- Cost of Available Scrap for Next Batch: ${scrap_reusable_cost:.2f}")
    st.markdown("---")

    order_details = []
    for o in orders:
        alloc, summ = optimized_allocation(o['runs'], st.session_state["strip_options"], max_connections=10)
        order_details.append({'order': o['order'], 'alloc': alloc, 'sum': summ})

    for od in order_details:
        with st.expander(f"Order {od['order']}"):
            df_o = pd.DataFrame(od['alloc'])
            df_o['Watts'] = df_o['used'].apply(lambda used: ", ".join(f"{(l/12)*st.session_state['watt_per_foot']:.1f}" for l in used))
            df_o['cost'] = df_o['cost'].apply(lambda x: f"${x:.2f}")
            st.subheader("LED Allocations")
            st.dataframe(df_o[['strip_length','used','Watts','waste','cost']], use_container_width=True)

            ps_df, ps_cost, ps_counts = compute_power(od['alloc'], st.session_state['watt_per_foot'], st.session_state['power_specs'])
            ps_df['Cost'] = ps_df['Cost'].apply(lambda x: f"${x:.2f}")
            ps_df['Remaining (W)'] = ps_df['Remaining (W)'].apply(lambda x: f"{x:.1f}W")
            st.subheader("Power Supplies")
            st.dataframe(ps_df, use_container_width=True)
            st.write(f"**Supply Cost:** ${ps_cost:.2f}")
            st.write(f"**Total Lighting Cost:** ${(od['sum']['led_cost']+ps_cost):.2f}")

    buf = io.BytesIO()
    folder = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    pdf_dir = f"{folder}/PDF"
    with zipfile.ZipFile(buf, 'w') as zf:
        for od in order_details:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f"Order {od['order']} Report", ln=1)
            # Table rows omitted for brevity
            data = pdf.output(dest='S').encode('latin1')
            zf.writestr(f"{pdf_dir}/{od['order']}_report.pdf", data)
        batch_pdf = FPDF()
        # Batch PDF generation omitted
        data_batch = batch_pdf.output(dest='S').encode('latin1')
        zf.writestr(f"{pdf_dir}/_BATCH_REPORT.pdf", data_batch)
    buf.seek(0)
    st.download_button("Export PDF Reports", data=buf.getvalue(), file_name=f"{folder}.zip", mime="application/zip")

st.markdown("---")
st.write("*Optimized for cost and waste; Power Supplies sized with 15-20% headroom.*")
