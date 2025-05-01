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
            key="form_strips",
            column_config={
                "Length (in)": st.column_config.NumberColumn("Strip length in inches", width="small"),
                "Cost": st.column_config.NumberColumn("Cost per strip", width="small")
            }
        )
        # Power supply specs
        power_df = pd.DataFrame(st.session_state["power_specs"])
        power_df = st.data_editor(
            power_df,
            num_rows="dynamic",
            use_container_width=True,
            key="form_power",
            column_config={
                "W": st.column_config.NumberColumn("Supply Wattage", width="small"),
                "cost": st.column_config.NumberColumn("Cost per supply", width="small")
            }
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
    alloc_all, sum_all = optimized_allocation(global_runs, strip_options, max_connections=len(global_runs))
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
    st.markdown("**Order-level Summary**")
    total_orders = len(order_details)
    order_costs = []
    for od in order_details:
        led_cost = sum(item['cost'] for item in od['alloc'])
        _, supply_cost, _ = compute_power(od['alloc'], watt_per_foot, power_specs)
        order_costs.append((od['order'], led_cost + supply_cost))
    if total_orders:
        avg_cost = sum(c for _, c in order_costs) / total_orders
        min_order, min_cost = min(order_costs, key=lambda x: x[1])
        max_order, max_cost = max(order_costs, key=lambda x: x[1])
    else:
        avg_cost = min_cost = max_cost = 0.0
        min_order = max_order = "N/A"
    st.write(f"- Total Orders: {total_orders}")
    st.write(f"- Average Cost: ${avg_cost:.2f}")
    st.write(f"- Min Order: {min_order} (${min_cost:.2f})")
    st.write(f"- Max Order: {max_order} (${max_cost:.2f})")
    st.markdown("---")

    # Order Details
    st.header("Order Details")
    for od in order_details:
        with st.expander(f"Order {od['order']}"):
            df_o = pd.DataFrame(od['alloc'])
            df_o_disp = df_o.copy()
            df_o_disp['cost'] = df_o_disp['cost'].apply(lambda x: f"${x:.2f}")
            st.subheader("LED Allocations")
            st.dataframe(df_o_disp, use_container_width=True)
            # Total LED Allocation Cost per order
            led_alloc_cost = sum(item['cost'] for item in od['alloc'])
            st.write(f"**Total LED Allocation Cost:** ${led_alloc_cost:.2f}")
            ps_o, cost_o, _ = compute_power(od['alloc'], watt_per_foot, power_specs)
            ps_o['Cost'] = ps_o['Cost'].apply(lambda x: f"${x:.2f}")
            ps_o['Remaining (W)'] = ps_o['Remaining (W)'].apply(lambda x: f"{x:.1f}W")
            st.subheader("Power Supplies")
            st.dataframe(ps_o, use_container_width=True)
            st.write(f"**Supply Cost:** ${cost_o:.2f}")

    # Export ZIP
    buf = io.BytesIO()
    folder = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    csv_dir, excel_dir, pdf_dir = f"{folder}/CSV", f"{folder}/Excel", f"{folder}/PDF"
    with zipfile.ZipFile(buf, 'w') as zf:
        # Individual order exports
        for od in order_details:
            order = od['order']
            df_o = pd.DataFrame(od['alloc'])
            summ = od['sum']

            # Excel (CSV format)
            zf.writestr(f"{excel_dir}/{order}_LED_OPT.csv", df_o.to_csv(index=False))

            # Individual PDF with full table and summary
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Order {order} Report", ln=1)
            pdf.set_font("Arial", "", 10)
            # Table header
            table_cols = ['strip_length','used','waste','cost']
            for col in table_cols:
                pdf.cell(45, 8, col, border=1)
            pdf.ln()
            # Data rows
            for row in df_o.itertuples(index=False):
                for cell in row:
                    pdf.cell(45, 8, str(cell), border=1)
                pdf.ln()
            # Summary rows
            total_supply = compute_power(od['alloc'], watt_per_foot, power_specs)[1]
            total_combined = summ['led_cost'] + total_supply
            summary_rows = [
                ("Total LED Cost", f"${summ['led_cost']:.2f}"),
                ("Total Power Supply Cost", f"${total_supply:.2f}"),
                ("Total Lighting Cost", f"${total_combined:.2f}"),
                ("Total Waste (in)", f"{summ['waste']:.2f}")
            ]
            for label, value in summary_rows:
                pdf.cell(45, 8, label, border=1)
                pdf.cell(45, 8, value, border=1)
                pdf.ln()
            buf_pdf = io.BytesIO(pdf.output(dest='S').encode('latin1'))
            zf.writestr(f"{pdf_dir}/{order}_report.pdf", buf_pdf.read())

        # Batch PDF report for all orders (5 per page)
        batch_pdf = FPDF()
        batch_pdf.set_auto_page_break(auto=True, margin=15)
        for idx, od in enumerate(order_details):
            # Start new page every 5 orders
            if idx % 5 == 0:
                batch_pdf.add_page()
                batch_pdf.set_font("Arial", "B", 14)
                batch_pdf.cell(0, 10, "Batch Order Report", ln=1)
            # Order header above each table
            batch_pdf.set_font("Arial", "B", 12)
            batch_pdf.cell(0, 8, f"Order {od['order']}", ln=1)
            # Table header
            batch_pdf.set_font("Arial", "B", 12)
            for col in table_cols:
                batch_pdf.cell(45, 8, col, border=1)
            batch_pdf.ln()
            # Order data rows
            df_batch = pd.DataFrame(od['alloc'])
            batch_pdf.set_font("Arial", "", 10)
            for row in df_batch.itertuples(index=False):
                for cell in row:
                    batch_pdf.cell(45, 8, str(cell), border=1)
                batch_pdf.ln()
            # Batch summary rows
            total_supply = compute_power(od['alloc'], watt_per_foot, power_specs)[1]
            total_combined = od['sum']['led_cost'] + total_supply
            summary_rows = [
                ("Total LED Cost", f"${od['sum']['led_cost']:.2f}"),
                ("Total Power Supply Cost", f"${total_supply:.2f}"),
                ("Total Lighting Cost", f"${total_combined:.2f}"),
                ("Total Waste (in)", f"{od['sum']['waste']:.2f}")
            ]
            for label, value in summary_rows:
                batch_pdf.cell(45, 8, label, border=1)
                batch_pdf.cell(45, 8, value, border=1)
                batch_pdf.ln()
            batch_pdf.ln(4)
        buf_batch = io.BytesIO(batch_pdf.output(dest='S').encode('latin1'))
        zf.writestr(f"{pdf_dir}/_BATCH_REPORT.pdf", buf_batch.read())
    
    buf.seek(0)
    st.download_button("Export Data", data=buf.getvalue(), file_name=f"{folder}.zip", mime="application/zip")

st.markdown("---")
st.write("*Optimized for cost and waste; Power Supplies sized with 20-25% headroom.*")
