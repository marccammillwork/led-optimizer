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
    {'W': 60, 'cost': 82.72},
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
    # Calculate segment watt loads using configurable watt_per_foot
    segment_watts = [(l/12)*watt_per_foot for a in allocations for l in a['used']]
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
    return df, df['Cost'].sum(), df['Wattage'].value_counts().to_dict()

# --- Configuration Settings ---
# Initialize settings in session state
if "watt_per_foot" not in st.session_state:
    st.session_state["watt_per_foot"] = 3.0
if "strip_options" not in st.session_state:
    st.session_state["strip_options"] = strip_options.copy()
if "power_specs" not in st.session_state:
    st.session_state["power_specs"] = power_specs.copy()

# Sidebar form for configuration
config = st.sidebar.expander("Configure Specs", expanded=False)
with config:
    with st.form("config_form"):
        wp = st.number_input(
            "LED Watt per Foot",
            value=st.session_state["watt_per_foot"],
            min_value=0.0,
            step=0.1,
            key="form_watt"
        )
        # LED strip lengths and costs
        strip_df = pd.DataFrame(
            list(st.session_state["strip_options"].items()),
            columns=["Length (in)", "Cost"]
        )
        strip_df = st.data_editor(
            strip_df,
            num_rows="dynamic",
            use_container_width=True,
            key="form_strips"
        )
        # Power supply specs
        power_df = pd.DataFrame(st.session_state["power_specs"])
        power_df = st.data_editor(
            power_df,
            num_rows="dynamic",
            use_container_width=True,
            key="form_power"
        )
        apply = st.form_submit_button("Apply Settings")
        if apply:
            # Save updated settings
            st.session_state["watt_per_foot"] = wp
            st.session_state["strip_options"] = dict(zip(strip_df["Length (in)"], strip_df["Cost"]))
            ps_list = power_df.to_dict("records")
            st.session_state["power_specs"] = sorted(ps_list, key=lambda s: s['cost']/s['W'])

# Load updated settings
watt_per_foot = st.session_state["watt_per_foot"]
strip_options = st.session_state["strip_options"]
power_specs = st.session_state["power_specs"]

# --- UI: Batch Orders ---
st.title("LED Strip & Power Supply Optimizer")
cols = ["Order"] + [f"Run{i+1}" for i in range(10)]
if "df_orders" not in st.session_state:
    st.session_state.df_orders = pd.DataFrame([["" for _ in cols] for _ in range(5)], columns=cols)

st.subheader("Enter Orders and Runs")
df_edited = st.data_editor(st.session_state.df_orders, num_rows="dynamic", use_container_width=True)
st.session_state.df_orders = df_edited.replace({None: "", "None": ""}).fillna("")

if st.button("Optimize All Orders"):
    # Parse orders
    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in["Order"].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row["Order"]).strip()
        runs = []
        for c in cols[1:]:
            raw = row[c]
            val_str = str(raw).strip()
            if val_str:
                runs.append(float(val_str))
        orders.append({'order': o_no, 'runs': runs})

    # Global optimization
    global_runs = [r for o in orders for r in o['runs']]
    alloc_all, sum_all = optimized_allocation(
        global_runs, strip_options, max_connections=len(global_runs)
    )
    # Determine cutoff usability
    unusable_scrap = sum(a['waste'] for a in alloc_all if len(a['used']) == 2)
    scrap_unusable_cost = sum(
        a['waste'] * (strip_options[a['strip_length']] / a['strip_length'])
        for a in alloc_all if len(a['used']) == 2
    )
    reusable_scrap = sum(a['waste'] for a in alloc_all if len(a['used']) == 1)
    scrap_reusable_cost = sum(
        a['waste'] * (strip_options[a['strip_length']] / a['strip_length'])
        for a in alloc_all if len(a['used']) == 1
    )
    df_led = pd.DataFrame(alloc_all)
    df_ps, ps_cost, ps_counts = compute_power(alloc_all, watt_per_foot, power_specs)

    # Per-order details
    order_details = []
    total_unit_waste = 0
    for o in orders:
        alloc, summ = optimized_allocation(o['runs'], strip_options, max_connections=10)
        total_unit_waste += summ['waste']
        order_details.append({'order': o['order'], 'alloc': alloc, 'sum': summ})
    waste_used = total_unit_waste - sum_all['waste']

    # Order-level Summary
    st.markdown("**Batch level Summary**")
    st.write(f"- Cost of Available Scrap for Next Batch: ${scrap_reusable_cost:.2f}")
    st.markdown("---")

    # --- Export PDF Reports ---
    buf = io.BytesIO()
    folder = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    pdf_dir = f"{folder}/PDF"
    with zipfile.ZipFile(buf, 'w') as zf:
        # Individual PDFs
        for od in order_details:
            order = od['order']
            df_o = pd.DataFrame(od['alloc'])
            _, _, ps_counts = compute_power(od['alloc'], watt_per_foot, power_specs)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, f"Order {order} Report", ln=1)
            pdf.set_font('Arial', '', 10)
            cols_headers = ['strip_length', 'used', 'waste', 'cost', 'supplies']
            for h in cols_headers:
                pdf.cell(40, 8, h, border=1)
            pdf.ln()
            for r in df_o.itertuples(index=False):
                for val in r:
                    pdf.cell(40, 8, str(val), border=1)
                pdf.cell(40, 8, '', border=1)
                pdf.ln()
            counts_str = ", ".join(f"{w}W:{cnt}" for w,cnt in ps_counts.items())
            pdf.cell(40, 8, 'Power Supplies', border=1)
            pdf.cell(120, 8, counts_str, border=1)
            pdf.ln()
            data = pdf.output(dest='S').encode('latin1')
            zf.writestr(f"{pdf_dir}/{order}_report.pdf", data)
        # Batch PDF
        batch_pdf = FPDF()
        batch_pdf.set_auto_page_break(auto=True, margin=15)
        cols_headers = ['strip_length', 'used', 'waste', 'cost', 'supplies']
        for idx, od in enumerate(order_details):
            if idx % 5 == 0:
                batch_pdf.add_page()
                batch_pdf.set_font('Arial', 'B', 14)
                batch_pdf.cell(0, 10, 'Batch Order Report', ln=1)
                batch_pdf.set_font('Arial', 'B', 12)
                for h in cols_headers:
                    batch_pdf.cell(40, 8, h, border=1)
                batch_pdf.ln()
            df_b = pd.DataFrame(od['alloc'])
            _, _, ps_counts_b = compute_power(od['alloc'], watt_per_foot, power_specs)
            batch_pdf.set_font('Arial', '', 10)
            for r in df_b.itertuples(index=False):
                for val in r:
                    batch_pdf.cell(40, 8, str(val), border=1)
                batch_pdf.cell(40, 8, '', border=1)
                batch_pdf.ln()
            counts_str_b = ", ".join(f"{w}W:{cnt}" for w,cnt in ps_counts_b.items())
            batch_pdf.cell(40, 8, 'Power Supplies', border=1)
            batch_pdf.cell(120, 8, counts_str_b, border=1)
            batch_pdf.ln(4)
        data_batch = batch_pdf.output(dest='S').encode('latin1')
        zf.writestr(f"{pdf_dir}/_BATCH_REPORT.pdf", data_batch)
    buf.seek(0)
    st.download_button("Export PDF Reports", data=buf.getvalue(), file_name=f"{folder}.zip", mime="application/zip")

st.markdown("---")
st.write("*Optimized for cost and waste; Power Supplies sized with 20-25% headroom.*")
