import streamlit as st
import pandas as pd
import io
import zipfile
from datetime import datetime

# For PDF generation
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

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
    # existing logic left unchanged
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
            allocations.append({
                'strip_length': best_strip,
                'used': best_pair,
                'waste': best_strip - sum(best_pair),
                'cost': opts[best_strip]
            })
            runs_left.remove(best_pair[0])
            runs_left.remove(best_pair[1])
        else:
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
    return allocations, {'connections': total_conns, 'led_cost': total_cost, 'waste': total_waste}

@st.cache_data
def compute_power(allocations):
    # unchanged
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
        bins.append({'W': spec['W'], 'cost': spec['cost'], 'remaining': spec['W'] - load, 'slots': 9, 'loads': [load]})
    df = pd.DataFrame([{'Supply #': i+1, 'Wattage': b['W'], 'Cost': b['cost'], 'Loads (W)': ", ".join(f"{l:.1f}" for l in b['loads']), 'Remaining (W)': round(b['remaining'], 1)} for i, b in enumerate(bins)])
    total_cost = df['Cost'].sum()
    counts = df['Wattage'].value_counts().to_dict()
    return df, total_cost, counts

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
    # parse and optimize logic unchanged up to ZIP export
    # ... [existing parsing/allocation/computations here] ...
    
    # ZIP export into two folders: Excel and PDF
    buf = io.BytesIO()
    folder = f"LED_OPT_{datetime.now().strftime('%m%d%y')}"
    excel_dir = f"{folder}/Excel"
    pdf_dir = f"{folder}/PDF"
    with zipfile.ZipFile(buf, 'w') as zf:
        for od in order_details:
            order = od['order']
            # Excel
            excel_buffer = io.BytesIO()
            df_o = pd.DataFrame(od['alloc'])
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_o.to_excel(writer, index=False, sheet_name='Allocations')
                writer.save()
            excel_buffer.seek(0)
            zf.writestr(f"{excel_dir}/{order}_LED_OPT.xlsx", excel_buffer.read())

            # PDF
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            data = [list(df_o.columns)] + df_o.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            doc.build([table])
            pdf_buffer.seek(0)
            zf.writestr(f"{pdf_dir}/{order}_LED_OPT.pdf", pdf_buffer.read())

    buf.seek(0)
    st.download_button('Export Data', data=buf.getvalue(), file_name=f"{folder}.zip", mime='application/zip')

st.markdown('---')
st.write('*Optimized for cost and waste; Power Supplies sized with 20–25% headroom.*')
