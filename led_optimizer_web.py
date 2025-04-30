import streamlit as st
import pandas as pd
import io
import zipfile
from datetime import datetime

# Attempt PDF support via fpdf
try:
    from fpdf import FPDF
    pdf_enabled = True
except ImportError:
    pdf_enabled = False

# --- Data and Logic ---
strip_options = {59: 43.26, 118: 74.63, 236: 139.06}
power_specs = [
    {'W': 24, 'cost': 50.41},
    {'W': 36, 'cost': 26.16},
    {'W': 60, 'cost': 82.72},
    {'W': 96, 'cost': 93.91},
]
power_specs.sort(key=lambda s: s['cost'] / s['W'])

@st.cache_data
def optimized_allocation(runs, opts, max_connections):
    runs_left = runs.copy()
    allocations = []
    strip_types = sorted(opts.items(), key=lambda x: x[1] / x[0])
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
            runs_left.remove(best_pair[0]); runs_left.remove(best_pair[1])
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
def compute_power(allocations):
    segment_watts = [(l/12)*3 for a in allocations for l in a['used']]
    bins = []
    for load in sorted(segment_watts, reverse=True):
        placed = False
        for b in bins:
            if b['slots'] > 0 and b['remaining'] >= load:
                b['remaining'] -= load; b['slots'] -= 1; b['loads'].append(load); placed = True; break
        if placed:
            continue
        spec = next((s for s in power_specs if s['W'] >= load*1.2), None)
        if not spec:
            spec = next((s for s in power_specs if s['W'] >= load), power_specs[-1])
        bins.append({'W': spec['W'], 'cost': spec['cost'], 'remaining': spec['W'] - load, 'slots': 9, 'loads': [load]})
    df = pd.DataFrame([{'Supply #': i+1, 'Wattage': b['W'], 'Cost': b['cost'], 'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']), 'Remaining (W)': round(b['remaining'], 1)} for i, b in enumerate(bins)])
    return df, df['Cost'].sum(), df['Wattage'].value_counts().to_dict()

# --- UI: Batch Orders ---
st.title('LED Strip & Power Supply Optimizer (Batch)')
cols = ['Order'] + [f'Run{i+1}' for i in range(10)]
if 'df_orders' not in st.session_state:
    st.session_state.df_orders = pd.DataFrame([['' for _ in cols] for _ in range(5)], columns=cols)
st.subheader('Enter Orders and Runs (Tab to navigate, paste rows)')
df_edited = st.data_editor(st.session_state.df_orders, num_rows='dynamic', use_container_width=True)
df_clean = df_edited.replace({None: '', 'None': ''}).fillna('')
st.session_state.df_orders = df_clean

if st.button('Optimize All Orders'):
    # Build orders list
    df_in = st.session_state.df_orders.copy()
    df_in = df_in[df_in['Order'].astype(str).str.strip() != '']
    orders = []
    for _, row in df_in.iterrows():
        o_no = str(row['Order']).strip()
        runs = []
        for c in cols[1:]:
            raw = row[c]
            val_str = str(raw).strip()
            if val_str == '':
                continue
            try:
                num = float(val_str)
            except ValueError:
                st.error(f"Invalid run value '{raw}' (interpreted as '{val_str}') in order {o_no}")
                st.stop()
            runs.append(num)
        orders.append({'order': o_no, 'runs': runs})
    
    # Per-order allocations
    order_details = []
    for o in orders:
        alloc, summ = optimized_allocation(o['runs'], strip_options, max_connections=10)
        order_details.append({'order': o['order'], 'alloc': alloc, 'sum': summ})

    # ZIP export
    buf = io.BytesIO()
    folder = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    excel_dir = f"{folder}/Excel"
    pdf_dir = f"{folder}/PDF"
    with zipfile.ZipFile(buf, 'w') as zf:
        for od in order_details:
            order = od['order']
            df_o = pd.DataFrame(od['alloc'])
            # Excel export (fallback to CSV if openpyxl missing)
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_o.to_excel(writer, index=False, sheet_name='Allocations')
                excel_buffer.seek(0)
                zf.writestr(f"{excel_dir}/{order}_LED_OPT.xlsx", excel_buffer.read())
            except Exception:
                csv_buffer = io.StringIO()
                df_o.to_csv(csv_buffer, index=False)
                zf.writestr(f"{excel_dir}/{order}_LED_OPT.csv", csv_buffer.getvalue())

            # PDF export or placeholder
            if pdf_enabled:
                pdf = FPDF(); pdf.add_page(); pdf.set_font('Arial', size=12)
                for col in df_o.columns:
                    pdf.cell(40, 10, str(col), border=1)
                pdf.ln()
                for row in df_o.itertuples(index=False):
                    for cell in row:
                        pdf.cell(40, 10, str(cell), border=1)
                    pdf.ln()
                pdf_buffer = io.BytesIO(pdf.output(dest='S').encode('latin1'))
                zf.writestr(f"{pdf_dir}/{order}_LED_OPT.pdf", pdf_buffer.read())
            else:
                zf.writestr(f"{pdf_dir}/README.txt", 'Install the `fpdf` library to enable PDF export.')
    buf.seek(0)
    st.download_button('Export Data', data=buf.getvalue(), file_name=f"{folder}.zip", mime='application/zip')

st.markdown('---')
st.write("*Optimized for cost and waste; Power Supplies sized with 20-25% headroom.*")
