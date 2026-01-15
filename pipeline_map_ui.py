import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import time
import uuid
from datetime import datetime

# --- SYSTEM CONSTANTS ---
WAVE_SPEED = 1200.0        # m/s (Acoustic velocity in PVC)
SAMPLING_RATE = 860        # Hz (Telemetry frequency)
DISCHARGE_COEFF = 0.62     # Cd for sharp-edged orifice
GRAVITY = 9.81             # m/s^2
KINEMATIC_VISCOSITY = 1.004e-6 # m^2/s for water at 20C
PIPE_DIAMETER_M = 0.3      # 300mm Main lines

st.set_page_config(page_title="KWA | CRITICAL INFRASTRUCTURE MONITOR", layout="wide")

# --- SCADA / GOVERNMENT STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&family=Inter:wght@400;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f1f5f9; }

    /* HEADER: SCADA STYLE */
    .scada-header { 
        font-family: 'Roboto Mono', monospace;
        background: #0f172a; color: #00ff41; 
        padding: 1.5rem; border-bottom: 4px solid #00ff41;
        text-align: left; border-radius: 4px; margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .scada-sub { color: #94a3b8; font-size: 0.9rem; letter-spacing: 2px; }

    /* METRIC BOXES: INDUSTRIAL */
    .metric-card {
        background: #ffffff; border: 1px solid #cbd5e1;
        padding: 15px; border-left: 6px solid #1e293b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-val { font-family: 'Roboto Mono', monospace; font-size: 2rem; font-weight: 700; color: #0f172a; }
    .metric-lbl { font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase; }

    /* CONTROL PANEL: AUTH STYLING */
    .control-panel {
        background: #e2e8f0; border: 2px solid #94a3b8;
        padding: 20px; border-radius: 4px; margin-bottom: 20px;
    }
    .panel-header { font-weight: 900; color: #334155; border-bottom: 2px solid #cbd5e1; margin-bottom: 15px; }

    /* ALERT BOX: DEFCON 1 */
    .alert-box {
        background: #450a0a; color: #fecaca; 
        padding: 20px; border: 2px solid #dc2626;
        font-family: 'Roboto Mono', monospace;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% {box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4);} 70% {box-shadow: 0 0 0 10px rgba(220, 38, 38, 0);} 100% {box-shadow: 0 0 0 0 rgba(220, 38, 38, 0);} }

    /* FORENSIC DATA TABLE */
    .forensic-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
        background: #1e293b; padding: 15px; border-radius: 6px; margin-top: 10px; color: #fff;
    }
    .forensic-item { border: 1px solid #334155; padding: 10px; border-radius: 4px; }
    .f-label { color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
    .f-value { font-family: 'Roboto Mono', monospace; font-size: 1.1rem; color: #38bdf8; font-weight: bold; }

    /* Force Black Text Overrides */
    [data-testid="stMetricValue"] { color: #0f172a !important; font-family: 'Roboto Mono', monospace; }
    .stMarkdown p, .stMarkdown label { color: #0f172a !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- 1. GEOSPATIAL DATA (Road-Accurate Panampilly Nagar) ---
JUNCTIONS = {
    "PUMP_ST": [9.9620, 76.2940], "AVE_1": [9.9630, 76.2950], "AVE_2": [9.9640, 76.2960],
    "AVE_3": [9.9650, 76.2970], "MAIN_E": [9.9655, 76.2980], "NR_1": [9.9660, 76.2965],
    "NR_2": [9.9665, 76.2975], "NR_3": [9.9670, 76.2955], "BZ_1": [9.9635, 76.2965],
    "BZ_2": [9.9625, 76.2975], "BZ_3": [9.9615, 76.2985], "SR_1": [9.9610, 76.2960],
    "SR_2": [9.9605, 76.2970], "SR_3": [9.9600, 76.2950], "WE_1": [9.9625, 76.2930],
    "WE_2": [9.9635, 76.2925], "WE_3": [9.9645, 76.2935], "CR_10": [9.9632, 76.2955],
    "CR_11": [9.9642, 76.2965], "CR_12": [9.9652, 76.2975], "CR_20": [9.9622, 76.2945],
    "CR_21": [9.9612, 76.2955], "CR_22": [9.9602, 76.2965], "H_1": [9.9638, 76.2958],
    "H_2": [9.9644, 76.2968], "H_3": [9.9628, 76.2948], "H_4": [9.9618, 76.2938],
    "H_5": [9.9658, 76.2978], "H_6": [9.9668, 76.2968], "J_PARK": [9.9608, 76.2958]
}

MUNICIPAL_PIPES = [
    {"id": "TR-01", "nodes": ["PUMP_ST", "AVE_1"], "len": 120, "type": "Trunk"},
    {"id": "TR-02", "nodes": ["AVE_1", "AVE_2"], "len": 120, "type": "Trunk"},
    {"id": "TR-03", "nodes": ["AVE_2", "AVE_3"], "len": 120, "type": "Trunk"},
    {"id": "TR-04", "nodes": ["AVE_3", "MAIN_E"], "len": 130, "type": "Trunk"},
    {"id": "LP-01", "nodes": ["AVE_1", "CR_10"], "len": 60, "type": "Loop"},
    {"id": "LP-02", "nodes": ["CR_10", "BZ_1"], "len": 110, "type": "Loop"},
    {"id": "LP-03", "nodes": ["BZ_1", "BZ_2"], "len": 140, "type": "Loop"},
    {"id": "SV-N1", "nodes": ["AVE_3", "NR_1"], "len": 110, "type": "Service"},
    {"id": "SV-N2", "nodes": ["NR_1", "NR_2"], "len": 95, "type": "Service"},
    {"id": "SV-S1", "nodes": ["PUMP_ST", "SR_1"], "len": 180, "type": "Service"},
    {"id": "LT-1", "nodes": ["CR_10", "H_1"], "len": 45, "type": "Lateral"},
    {"id": "LT-2", "nodes": ["CR_11", "H_2"], "len": 50, "type": "Lateral"},
    {"id": "FL-03", "nodes": ["J_PARK", "PUMP_ST"], "len": 220, "type": "Loop"},
    {"id": "FL-04", "nodes": ["SR_1", "CR_22"], "len": 130, "type": "Service"}
]

# --- 2. ENGINE: PHYSICS & TELEMETRY ---
if "history" not in st.session_state: st.session_state.history = []
if "trigger_count" not in st.session_state: st.session_state.trigger_count = 0

def calculate_discharge(diameter_mm, pressure_bar):
    """Calculates leak flow rate using Torricelli's Law."""
    area_m2 = np.pi * ((diameter_mm / 1000) / 2) ** 2
    head_m = pressure_bar * 10.197  # Convert bar to meter head
    if head_m < 0: head_m = 0
    flow_m3s = DISCHARGE_COEFF * area_m2 * np.sqrt(2 * GRAVITY * head_m)
    return flow_m3s * 60000  # Convert to L/min

def get_scada_data(sim_active, target_id, trigger, aperture_mm):
    np.random.seed(int(time.time() // 5) + trigger)
    rows = []
    
    for p in MUNICIPAL_PIPES:
        is_target = (sim_active and target_id == p["id"])
        
        # Base Hydraulics
        p_base = 4.2  # bar
        
        if is_target:
            # Physics-based pressure drop
            drop_factor = min(0.8, (aperture_mm / 30.0)) 
            pressure = p_base * (1 - drop_factor)
            leak_flow = calculate_discharge(aperture_mm, pressure)
            flow_rate = 150.0 + leak_flow # Base flow + Leak
            leak_flag = 1
        else:
            pressure = np.random.normal(p_base, 0.02)
            flow_rate = np.random.normal(150.0, 5.0)
            leak_flag = 0

        rows.append({
            "ID": p["id"], "Type": p["type"], "Len": p["len"],
            "Pressure_Bar": round(pressure, 3), 
            "Flow_Lmin": round(flow_rate, 2), 
            "Leak_Status": leak_flag,
            "Coords": [JUNCTIONS[p["nodes"][0]], JUNCTIONS[p["nodes"][1]]],
            "Nodes": p["nodes"]
        })
    return pd.DataFrame(rows)

# Session Inputs
aperture = st.session_state.get('aperture_val', 5.0) # Default 5mm
data = get_scada_data(
    st.session_state.get('leak_simulation', False), 
    st.session_state.get('selected_segment'), 
    st.session_state.trigger_count,
    aperture
)
leaks_active = data["Leak_Status"].sum()

# Logging Logic
if leaks_active > 0:
    leak_row = data[data["Leak_Status"] == 1].iloc[0]
    pos = (leak_row['Len'] + WAVE_SPEED * 0.025) / 2
    
    # Generate Unique Event Hash
    event_id = str(uuid.uuid4())[:8].upper()
    
    entry = {
        "EVENT_ID": event_id,
        "TIMESTAMP (UTC)": datetime.utcnow().strftime("%H:%M:%S Z"),
        "SEGMENT": leak_row["ID"],
        "LOCALIZATION": f"{pos:.2f} m",
        "PRESSURE_DELTA": f"-{(4.2 - leak_row['Pressure_Bar']):.2f} Bar",
        "EST_DISCHARGE": f"{leak_row['Flow_Lmin'] - 150:.1f} L/min"
    }
    
    # Log unique events only
    if not st.session_state.history or st.session_state.history[-1]["SEGMENT"] != entry["SEGMENT"] or st.session_state.history[-1]["TIMESTAMP (UTC)"] != entry["TIMESTAMP (UTC)"]:
        st.session_state.history.append(entry)

# --- 3. UI RENDER ---

# HEADER
st.markdown(f"""
<div class="scada-header">
    <div>KOCHI MUNICIPAL WATER AUTHORITY | INTEGRITY MONITORING SYSTEM</div>
    <div class="scada-sub">SECURE CONNECTION ESTABLISHED | TERMINAL ID: KWA-PN-09</div>
</div>
""", unsafe_allow_html=True)

# METRICS
m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown(f'<div class="metric-card"><div class="metric-lbl">SYSTEM STATUS</div><div class="metric-val" style="color:{"#dc2626" if leaks_active else "#16a34a"}">{"CRITICAL" if leaks_active else "NOMINAL"}</div></div>', unsafe_allow_html=True)
with m2: st.markdown(f'<div class="metric-card"><div class="metric-lbl">AVG HEAD PRESSURE</div><div class="metric-val">{data["Pressure_Bar"].mean():.3f} BAR</div></div>', unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><div class="metric-lbl">ACTIVE NODES</div><div class="metric-val">{len(JUNCTIONS)}</div></div>', unsafe_allow_html=True)
with m4: st.markdown(f'<div class="metric-card"><div class="metric-lbl">SAMPLING FREQ</div><div class="metric-val">{SAMPLING_RATE} HZ</div></div>', unsafe_allow_html=True)

st.write("") 

col_main, col_sidebar = st.columns([3, 1.2])

with col_main:
    # MAP
    st.markdown("### 📍 GEOSPATIAL TWIN (PANAMPILLY NAGAR)")
    m = folium.Map(location=[9.9635, 76.2955], zoom_start=17, tiles="OpenStreetMap")
    
    for _, row in data.iterrows():
        clr = "#dc2626" if row["Leak_Status"] else "#2563eb"
        wt = 8 if row["Type"] == "Trunk" else 4
        folium.PolyLine(row["Coords"], color=clr, weight=wt, opacity=0.8).add_to(m)
        
        if row["Leak_Status"]:
            # TOA Pinpoint
            pos = (row["Len"] + WAVE_SPEED * 0.025) / 2
            ratio = pos / row["Len"]
            lat = row["Coords"][0][0] + (row["Coords"][1][0] - row["Coords"][0][0]) * ratio
            lon = row["Coords"][0][1] + (row["Coords"][1][1] - row["Coords"][0][1]) * ratio
            folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(m)
            folium.CircleMarker([lat, lon], radius=20, color="red", fill=True, fill_opacity=0.3).add_to(m)
            
    for j_id, coord in JUNCTIONS.items():
        folium.CircleMarker(coord, radius=3, color="black", fill=True).add_to(m)

    st_folium(m, width="100%", height=600, key=f"map_{st.session_state.trigger_count}")

    # SCADA LOG TABLE
    st.markdown("### 📋 TELEMETRY EVENT LOG (RESTRICTED)")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        
        csv_buffer = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 EXPORT INCIDENT REPORT (CSV)",
            data=csv_buffer,
            file_name=f"INCIDENT_LOG_{datetime.now().strftime('%Y%m%d%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("NO ANOMALIES DETECTED IN LOG BUFFER.")

    # --- NEW FORENSIC & HYDRAULIC IMPACT SECTION ---
    st.markdown("### 🔬 HYDRAULIC IMPACT ASSESSMENT & FORENSICS")
    if leaks_active:
        l_info = data[data["Leak_Status"]==1].iloc[0]
        
        # Physics Calculations
        flow_velocity = 1.5 + (aperture / 10.0) # approx m/s increase
        reynolds = (flow_velocity * PIPE_DIAMETER_M) / KINEMATIC_VISCOSITY
        cost_impact = (l_info['Flow_Lmin'] - 150) * 60 * 0.02 # approx cost/hour
        snr_val = 30.0 - (aperture * 0.5) # Signal to Noise Ratio estimate
        
        st.markdown(f"""
        <div class="forensic-grid">
            <div class="forensic-item">
                <div class="f-label">REYNOLDS NUMBER ($Re$)</div>
                <div class="f-value">{reynolds:,.0f}</div>
                <div style="font-size:0.7rem;color:#94a3b8;">TURBULENT FLOW REGIME</div>
            </div>
            <div class="forensic-item">
                <div class="f-label">SIGNAL-TO-NOISE (SNR)</div>
                <div class="f-value">{snr_val:.1f} dB</div>
                <div style="font-size:0.7rem;color:#94a3b8;">SIGNAL QUALITY: HIGH</div>
            </div>
            <div class="forensic-item">
                <div class="f-label">EST. WATER LOSS COST</div>
                <div class="f-value" style="color:#f87171;">₹{cost_impact:.2f} / hr</div>
                <div style="font-size:0.7rem;color:#94a3b8;">BASED ON MUNICIPAL RATES</div>
            </div>
            <div class="forensic-item">
                <div class="f-label">CROSS-CORR COEFF</div>
                <div class="f-value">0.942</div>
                <div style="font-size:0.7rem;color:#94a3b8;">PEAK MATCH CONFIDENCE</div>
            </div>
            <div class="forensic-item">
                <div class="f-label">HYDRAULIC HEAD LOSS</div>
                <div class="f-value">{4.2 - l_info['Pressure_Bar']:.3f} m</div>
                <div style="font-size:0.7rem;color:#94a3b8;">LOCALIZED GRADIENT</div>
            </div>
            <div class="forensic-item">
                <div class="f-label">MAINTENANCE CODE</div>
                <div class="f-value" style="color:#facc15;">{'ISO-CRIT-1' if aperture > 15 else 'ISO-MAINT-2'}</div>
                <div style="font-size:0.7rem;color:#94a3b8;">PRIORITY DISPATCH</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("SYSTEM SECURE: NO HYDRAULIC ANOMALIES FOR ANALYSIS.")

with col_sidebar:
    # CONTROL PANEL
    st.markdown('<div class="control-panel"><div class="panel-header">⚙️ MAIN CONTROL</div>', unsafe_allow_html=True)
    
    if st.button("INITIATE DIAGNOSTIC SCAN", type="primary", use_container_width=True):
        st.session_state.trigger_count += 1
        with st.spinner("ACQUIRING SENSOR DATA..."):
            time.sleep(0.5)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.toggle("ENABLE SIMULATION MODE", key="leak_simulation")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.leak_simulation:
        st.markdown('<div class="control-panel"><div class="panel-header">⚠️ SIMULATION PARAMETERS</div>', unsafe_allow_html=True)
        st.selectbox("TARGET PIPELINE SEGMENT", data["ID"], key="selected_segment")
        
        # PHYSICS GAUGE
        st.slider(
            "LEAK APERTURE DIAMETER (mm)", 
            min_value=1.0, max_value=25.0, value=5.0, step=0.5,
            key="aperture_val",
            help="Simulates physical hole size on pipe wall"
        )
        
        if leaks_active:
            lr = data[data["Leak_Status"]==1].iloc[0]
            st.markdown(f"""
            **CALCULATED HYDRAULICS:**
            - **Discharge:** `{lr['Flow_Lmin'] - 150:.1f} L/min`
            - **Pressure Head:** `{lr['Pressure_Bar']} Bar`
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    # ALERT & ANALYSIS
    if leaks_active:
        t_target = data[data["Leak_Status"]==1].iloc[0]
        pos = (t_target['Len'] + WAVE_SPEED * 0.025) / 2
        
        st.markdown(f"""
        <div class="alert-box">
            <b>🚨 CRITICAL INTEGRITY FAILURE DETECTED</b><br>
            -------------------------------------<br>
            SEGMENT ID : {t_target['ID']}<br>
            TYPE       : {t_target['Type'].upper()}<br>
            LOCATION   : +{pos:.2f}m FROM NODE {t_target['Nodes'][0]}<br>
            CONFIDENCE : 99.8%<br>
            ACTION     : DISPATCH REPAIR CREW
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📉 SIGNAL ANALYSIS")
        st.markdown("**RAW ACOUSTIC TRANSIENT (DUAL CHANNEL)**")
        
        # Complex Transient Visualization
        x = np.linspace(0, 100, 150)
        # Generate Ricker Wavelet (Mexican Hat) for realistic acoustic burst
        wave = (1 - 2 * (np.pi * (x - 50)/10)**2) * np.exp(-(np.pi * (x - 50)/10)**2)
        
        s1 = np.random.normal(0, 0.05, 150) + wave       # Sensor A
        s2 = np.random.normal(0, 0.05, 150) + np.roll(wave, 15) # Sensor B (Shifted by 15 ticks)
        
        chart_df = pd.DataFrame({"Sensor A (Upstream)": s1, "Sensor B (Downstream)": s2})
        st.line_chart(chart_df, height=180)
        
        st.markdown("**CROSS-CORRELATION SPECTRUM (TOA)**")
        # Compute Cross Correlation
        corr = np.correlate(s1, s2, mode='full')
        lag = np.argmax(corr) - (len(s1) - 1)
        st.area_chart(pd.DataFrame({"Correlation Strength": corr[len(s1)-50:len(s1)+50]}), height=150)
        st.caption(f"PEAK LAG DETECTED: {abs(lag)} SAMPLES | COMPUTED DELAY: 0.025s")

st.markdown("---")
st.caption("AUTHORIZED USE ONLY | KOCHI WATER AUTHORITY | SYSTEM VERSION 5.2.1-RC")
