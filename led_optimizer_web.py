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

# --- UI: Batch Orders ---
import re

# Paste row(s) from Excel
st.subheader("Paste rows from Excel")
paste = st.text_area(
    "Paste entire row(s) (Order # followed by runs, separated by tabs or commas):",
    key="paste_area",
    help="Copy a row from Excel (e.g. 25706<TAB>131.5<TAB>68.5 ...) and paste here"
)
if st.button("Parse Paste"):
    for line in st.session_state.paste_area.splitlines():
        parts = [p.strip() for p in re.split(r'	|,', line) if p.strip()]
        if not parts:
            continue
        order = parts[0]
        runs_list = parts[1:]
        # find first empty order row
        for o in st.session_state.orders:
            if not o['order'] and all(not r for r in o['runs']):
                o['order'] = order
                # fill runs
                for j, run_val in enumerate(runs_list):
                    if j < len(o['runs']):
                        o['runs'][j] = run_val
                break
    st.session_state.paste_area = ""

# Init 5 blank orders of 10 runs

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
    order_no = row[0].text_input("", o['order'], key=f"ord_{idx}")
    runs=[]
    for j in range(10):
        val = row[j+1].text_input("", o['runs'][j], key=f"run_{idx}_{j}")
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
        global_alloc, global_sum = optimized_allocation(global_runs, strip_options, max_connections=len(global_runs))
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
            runs_raw = []
            for o in st.session_state.orders:
                if o['order'] == od['order']:
                    runs_raw = [r for r in o['runs'] if r.strip()]
                    break
            counts59 = sum(1 for a in od['alloc'] if a['strip_length']==59)
            counts118 = sum(1 for a in od['alloc'] if a['strip_length']==118)
            counts236 = sum(1 for a in od['alloc'] if a['strip_length']==236)
            _, _, counts_sup = compute_power(od['alloc'])
            # build row dict
            row = {'Order': od['order']}
            for i in range(10):
                key = f'Run{i+1}'
                if i < len(runs_raw):
                    row[key] = str(runs_raw[i])
                else:
                    row[key] = ''
            row['59"'] = counts59
            row['118"'] = counts118
            row['236"'] = counts236
            row['36W'] = counts_sup.get(36, 0)
            row['60W'] = counts_sup.get(60, 0)
            row['96W'] = counts_sup.get(96, 0)
            rows.append(row)
        df_orders = pd.DataFrame(rows).fillna('')
        df_orders.index += 1
        # ensure run columns are strings
        run_cols = [f'Run{i+1}' for i in range(10)]
        for col in run_cols:
            df_orders[col] = df_orders[col].astype(str)
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

# Export Data as ZIP
        if order_details:
            buffer = io.BytesIO()
            folder_name = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
            with zipfile.ZipFile(buffer, 'w') as zf:
                # overall summary
                df_summary = df_rolls.copy()
                with io.BytesIO() as xls_buf:
                    with pd.ExcelWriter(xls_buf, engine='xlsxwriter') as writer:
                        df_summary.to_excel(writer, sheet_name='Overall LED')
                        df_power_summary.to_excel(writer, sheet_name='Overall Power', index=False)
                        pd.DataFrame([global_sum]).to_excel(writer, sheet_name='Global Summary', index=False)
                    xls_buf.seek(0)
                    zf.writestr(f"{folder_name}/Overall.xlsx", xls_buf.read())
                # per-order files
                for od in order_details:
                    with io.BytesIO() as xls_buf:
                        with pd.ExcelWriter(xls_buf, engine='xlsxwriter') as writer:
                            pd.DataFrame(od['alloc']).to_excel(writer, sheet_name='Allocations', index=False)
                            pd.DataFrame([od['sum']]).to_excel(writer, sheet_name='Summary', index=False)
                        xls_buf.seek(0)
                        zf.writestr(f"{folder_name}/{od['order']}.xlsx", xls_buf.read())
            buffer.seek(0)
            st.download_button("Export Data", data=buffer.getvalue(), file_name=f"{folder_name}.zip", mime='application/zip')

st.markdown("---")
st.write("*This data is optimized for reducing cost and waste. Power Supply requirements are calculated with headroom of between 20%-25%*")
