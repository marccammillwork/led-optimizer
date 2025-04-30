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
df_clean = df_edited.replace({None: "", "None": ""}).fillna("")
st.session_state.df_orders = df_clean

# Optimize button
if st.button("Optimize All Orders"):
    # Build orders list
    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in["Order"].astype(str).str.strip() != ""]
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row["Order"]).strip()
        runs = []
        for c in cols[1:]:
            raw = row[c]
            val_str = str(raw).strip()
            if val_str == "":
                continue
            try:
                runs.append(float(val_str))
            except ValueError:
                st.error(f"Invalid run value '{raw}' in order {o_no}")
                st.stop()
        orders.append({"order": o_no, "runs": runs})
    # Global optimization({"order": o_no, "runs": runs})

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

    # Power summary
    df_power = pd.DataFrame(
        [
            (W,
             ps_counts.get(W,0),
             ps_counts.get(W,0) * next(s['cost'] for s in power_specs if s['W']==W))
            for W in sorted(ps_counts)
        ],
        columns=["Wattage","Count","Total Cost"]
    )
    df_power["Total Cost"] = df_power["Total Cost"].apply(lambda x: f"${x:.2f}")
    df_power["Count"] = df_power["Count"].replace(0,"")
    df_power["Total Cost"] = df_power["Total Cost"].replace("$0.00","")
    st.dataframe(df_power, use_container_width=True, hide_index=True)

    st.write(f"**Total Supply Cost:** ${ps_cost:.2f}")
    st.write(f"**Total Cutoffs (in):** {sum_all['waste']:.2f}")
    st.write(f"**Inches Used from Cutoffs:** {waste_used:.2f}")

    # Order Details
    st.header("Order Details")
    seen = set()
    for od in order_details:
        if od["order"] in seen:
            continue
        seen.add(od["order"])
        with st.expander(f"Order {od['order']}"):
            df_o = pd.DataFrame(od["alloc"])
            df_o.index += 1
            df_o_disp = df_o.copy()
            df_o_disp["cost"] = df_o_disp["cost"].apply(lambda x: f"${x:.2f}")
            st.dataframe(df_o_disp, use_container_width=True)

            ps_o, cost_o, _ = compute_power(od["alloc"])
            ps_o_disp = ps_o.drop(columns=["Supply #"]).copy()
            ps_o_disp["Cost"] = ps_o_disp["Cost"].apply(lambda x: f"${x:.2f}")
            ps_o_disp["Remaining (W)"] = ps_o_disp["Remaining (W)"].apply(lambda x: f"{x:.1f}W")
            ps_o_disp = ps_o_disp.replace({"Count": {0:""}, "Loads (W)": {"":""}})
            st.dataframe(ps_o_disp, use_container_width=True)
            st.write(f"**Supply Cost:** ${cost_o:.2f}")

        # Cutoffs expander
    scrap_list = [a["waste"] for a in alloc_all if a["waste"] > 0]
    df_cutoffs = pd.DataFrame({
        "Cutoff Number": list(range(1, len(scrap_list) + 1)),
        "Length": [round(w, 2) for w in scrap_list],
    })
    # add total row
    df_cutoffs_disp = df_cutoffs.copy()
    total_length = df_cutoffs_disp['Length'].sum()
    df_cutoffs_disp.loc[len(df_cutoffs_disp)] = ['Total', round(total_length, 2)]
    with st.expander("Cutoffs"):
        st.dataframe(df_cutoffs_disp, use_container_width=True)

                            # Export ZIP with CSV, Excel, and PDF
    buf = io.BytesIO()
    folder = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    csv_dir = f"{folder}/CSV"
    excel_dir = f"{folder}/Excel"
    pdf_dir = f"{folder}/PDF"
    with zipfile.ZipFile(buf, "w") as zf:
        for od in order_details:
            order = od['order']
            df_o = pd.DataFrame(od['alloc'])
            summ = od['sum']

            # CSV export
            zf.writestr(f"{csv_dir}/{order}_alloc.csv", df_o.to_csv(index=False))
            zf.writestr(f"{csv_dir}/{order}_summary.csv", pd.DataFrame([summ]).to_csv(index=False))

            # Excel export (CSV format)
            zf.writestr(f"{excel_dir}/{order}_LED_OPT.csv", df_o.to_csv(index=False))

            # PDF export
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Order {order} Report", ln=1)
            pdf.set_font("Arial", "", 10)
            # Table header
            for col in df_o.columns:
                pdf.cell(40, 8, str(col), border=1)
            pdf.ln()
            # Data rows
            for row in df_o.itertuples(index=False):
                for cell in row:
                    pdf.cell(40, 8, str(cell), border=1)
                pdf.ln()
            pdf.ln(4)
            pdf.cell(0, 8, f"Total LED Cost: ${summ['led_cost']:.2f}", ln=1)
            pdf.cell(0, 8, f"Total Supply Cost: ${compute_power(od['alloc'])[1]:.2f}", ln=1)
            pdf.cell(0, 8, f"Total Waste: {summ['waste']:.2f} in", ln=1)
            pdf_buf = io.BytesIO(pdf.output(dest='S').encode('latin1'))
            zf.writestr(f"{pdf_dir}/{order}_report.pdf", pdf_buf.read())
    buf.seek(0)
    st.download_button(
        "Export Data",
        data=buf.getvalue(),
        file_name=f"{folder}.zip",
        mime="application/zip"
    ),
        file_name=f"{folder}.zip",
        mime="application/zip"
    ),
        file_name=f"{folder}.zip",
        mime="application/zip"
    )

st.markdown("---")
st.write("*Optimized for cost and waste; Power Supplies sized with 20-25% headroom.*")
