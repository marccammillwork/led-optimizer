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
    headroom_factor = 1.10
    # Slot capacities per wattage
    slot_limits = {s['W']: (10 if s['W']==36 else 30) for s in power_specs}
    # Compute loads in watts for LED segments
    segment_watts = [(length/12)*watt_per_foot for alloc in allocations for length in alloc['used']]
    # Prepare supply bins
    bins = []
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
    df = pd.DataFrame([
        {'Supply #': i+1, 'Wattage': b['W'], 'Cost': b['cost'], 'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']), 'Remaining (W)': round(b['remaining'], 1)}
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
config = st.sidebar.expander("Configure Specs", expanded=False)
with config:
    with st.form("config_form"):
        wp = st.number_input("LED Watt per Foot", value=st.session_state["watt_per_foot"], min_value=0.0, step=0.1, key="form_watt")
        strip_df = pd.DataFrame(list(st.session_state["strip_options"].items()), columns=["Length (in)", "Cost"])
        strip_df = st.data_editor(strip_df, num_rows="dynamic", use_container_width=True, key="form_strips")
        power_df = pd.DataFrame(st.session_state["power_specs"])
        power_df = st.data_editor(power_df, num_rows="dynamic", use_container_width=True, key="form_power")
        apply = st.form_submit_button("Apply Settings")
        if apply:
            st.session_state["watt_per_foot"] = wp
            st.session_state["strip_options"] = dict(zip(strip_df["Length (in)"], strip_df["Cost"]))
            ps_list = power_df.to_dict("records")
            st.session_state["power_specs"] = sorted(ps_list, key=lambda s: s['cost']/s['W'])

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
    headroom_factor = 1.10
    max_capacity = max(s['W'] for s in power_specs)
    unsupported = []
    df_exp = st.session_state.df_orders.copy()
    df_exp = df_exp[df_exp["Order"].astype(str).str.strip() != ""]
    for _, row in df_exp.iterrows():
        order_no = str(row["Order"]).strip()
        for c in cols[1:]:
            val = row[c]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                length = float(str(val).strip())
            except ValueError:
                st.error(f"Invalid run value '{val}' in order {order_no}")
                st.stop()
            load = (length/12)*watt_per_foot
            if load*headroom_factor > max_capacity:
                unsupported.append({'order': order_no, 'length': length, 'watts': load})
    if unsupported:
        exp_label = f"Unsupported runs: {len(unsupported)}"
        exp_html = f"<span style='color:red'>{exp_label}</span>"
        with st.expander(exp_html, unsafe_allow_html=True):
            for u in unsupported:
                st.write(f"- Order {u['order']}: {u['length']}\" requires {u['watts']:.1f} W")
        st.error("Please adjust or split unsupported runs.")
        st.stop()

    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in["Order"].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row["Order"]).strip()
        runs = []
        for c in cols[1:]:
            if pd.notna(row[c]) and str(row[c]).strip():
                runs.append(float(str(row[c]).strip()))
        orders.append({'order': o_no, 'runs': runs})
    global_runs = [r for o in orders for r in o['runs']]
    alloc_all, sum_all = optimized_allocation(global_runs, strip_options, max_connections=len(global_runs))
    df_led = pd.DataFrame(alloc_all)
    df_ps, ps_cost, ps_counts = compute_power(alloc_all, watt_per_foot, power_specs)

    st.markdown("**Order-level Summary**")
    total_orders = len(orders)
    unusable = sum(a['waste'] for a in alloc_all if len(a['used'])==2)
    cost_unusable = sum(a['waste']*(strip_options[a['strip_length']]/a['strip_length']) for a in alloc_all if len(a['used'])==2)
    reusable = sum(a['waste'] for a in alloc_all if len(a['used'])==1)
    cost_reusable = sum(a['waste']*(strip_options[a['strip_length']]/a['strip_length']) for a in alloc_all if len(a['used'])==1)
    st.write(f"- Total Orders: {total_orders}")
    st.write(f"- Unusable Cutoff Waste: {unusable:.2f} in | ${cost_unusable:.2f}")
    st.write(f"- Reusable Scrap for Next Batch: {reusable:.2f} in | ${cost_reusable:.2f}")
    st.markdown("---")

    st.header("Order Details")
    for od in orders:
            order = od['order']
            alloc, summ = optimized_allocation(od['runs'], strip_options, max_connections=10)
            df_o = pd.DataFrame(alloc)
            df_o['Watts'] = df_o['used'].apply(lambda u: ", ".join(f"{(l/12)*watt_per_foot:.1f}" for l in u))
            _, cost_o, counts_o = compute_power(alloc, watt_per_foot, power_specs)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial','B',12)
            pdf.cell(0,10,f"Order {order} Report",ln=1)
            pdf.set_font('Arial','',10)
            headers=['strip_length','used','Watts','waste','cost','supplies']
            for h in headers:
                pdf.cell(30,8,h,border=1)
            pdf.ln()
            for row in df_o.itertuples(index=False):
                pdf.cell(30, 8, str(row.strip_length), border=1)
                pdf.cell(30, 8, str(row.used), border=1)
                pdf.cell(30, 8, row.Watts, border=1)
                pdf.cell(30, 8, str(row.waste), border=1)
                pdf.cell(30, 8, f"${row.cost:.2f}", border=1)
                supplies_str = ", ".join(f"{w}W:{cnt}" for w,cnt in counts_o.items())
                pdf.cell(30, 8, supplies_str, border=1)
                pdf.ln()
            buf_pdf = io.BytesIO(pdf.output(dest='S').encode('latin1'))
            zf.writestr(f"{pdf_dir}/{order}_report.pdf", buf_pdf.read())
        
        # ----- Batch PDF report for all orders -----
            batch_pdf = FPDF()
            batch_pdf.set_auto_page_break(auto=True, margin=15)
            headers = ['strip_length','used','Watts','waste','cost','supplies']
            for idx, od in enumerate(orders):
                if idx % 5 == 0:
                    batch_pdf.add_page()
                    batch_pdf.set_font('Arial','B',14)
                    batch_pdf.cell(0,10,'Batch Order Report',ln=1)
                    batch_pdf.set_font('Arial','B',12)
                for h in headers:
                    batch_pdf.cell(30,8,h,border=1)
                batch_pdf.ln()
            batch_pdf.set_font('Arial','',10)
            alloc_b, _ = optimized_allocation(od['runs'], strip_options, max_connections=10)
            df_b = pd.DataFrame(alloc_b)
            df_b['Watts'] = df_b['used'].apply(lambda u: ", ".join(f"{(l/12)*watt_per_foot:.1f}" for l in u))
            _, cost_b, counts_b = compute_power(alloc_b, watt_per_foot, power_specs)
            for row in df_b.itertuples(index=False):
                batch_pdf.cell(30,8,str(row.strip_length),border=1)
                batch_pdf.cell(30,8,str(row.used),border=1)
                batch_pdf.cell(30,8,row.Watts,border=1)
                batch_pdf.cell(30,8,str(row.waste),border=1)
                batch_pdf.cell(30,8,f"${row.cost:.2f}",border=1)
                batch_pdf.cell(30,8", ".join(f"{w}W:{cnt}" for w,cnt in counts_b.items()),border=1)
                batch_pdf.ln()
            batch_pdf.ln(4)
        buf_batch = io.BytesIO(batch_pdf.output(dest='S').encode('latin1'))
        zf.writestr(f"{pdf_dir}/_BATCH_REPORT.pdf", buf_batch.read())
    # finalize ZIP and present download button
    buf.seek(0)
    st.download_button(
        "Export PDF Reports",
        data=buf.getvalue(),
        file_name=f"{folder}.zip",
        mime="application/zip"
    )
    # after zf context, add download button
    buf.seek(0)
    st.download_button(
        "Export PDF Reports",
        data=buf.getvalue(),
        file_name=f"{folder}.zip",
        mime="application/zip"
    )

