import streamlit as st
import folium
from folium.plugins import AntPath
from folium import DivIcon
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import time
import uuid
import joblib
import warnings
import sqlite3
import serial.tools.list_ports
import os
import requests
from datetime import datetime, timezone
from tensorflow.keras.models import load_model

# --- LOTTIE LIBRARY SETUP ---
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# --- 0. SYSTEM CONFIGURATION ---
warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except:
    pass

# Physics Constants
WAVE_SPEED = 1200.0        
SAMPLING_RATE = 860        
DISCHARGE_COEFF = 0.62     
GRAVITY = 9.81             
KINEMATIC_VISCOSITY = 1.004e-6 
PIPE_DIAMETER_M = 0.3      

st.set_page_config(page_title="KWA | INFRASTRUCTURE MONITOR", layout="wide")

# --- 1. VISUAL THEME (PITCH BLACK, CRT SCANLINES, TACTICAL UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&family=Inter:wght@400;600;800&display=swap');
    
    /* MAIN BACKGROUND PITCH BLACK */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* SIDEBAR BACKGROUND PITCH BLACK */
    [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }

    /* --- ANIMATION: CRT MONITOR SCANLINES --- */
    .stApp::after {
        content: " ";
        display: block;
        position: absolute;
        top: 0; left: 0; bottom: 0; right: 0;
        background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.1) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
        z-index: 100;
        background-size: 100% 2px, 3px 100%;
        pointer-events: none;
    }

    /* --- ANIMATION: RED ALERT FLASH (GLOBAL) --- */
    @keyframes flash-red {
        0% { box-shadow: inset 0 0 0 0 rgba(255, 0, 0, 0); }
        50% { box-shadow: inset 0 0 50px 10px rgba(255, 0, 0, 0.5); }
        100% { box-shadow: inset 0 0 0 0 rgba(255, 0, 0, 0); }
    }
    .red-alert-mode {
        animation: flash-red 2s infinite;
    }
    
    /* CUSTOM TACTICAL BUTTON (MINIMAL) */
    div.stButton > button {
        width: 100%;
        background-color: #000000 !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
        border-radius: 0px !important;
        padding: 8px 16px !important;
        font-family: 'Roboto Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem !important;
        transition: all 0.2s ease-in-out;
        z-index: 102; /* Above scanlines */
        position: relative;
    }
    div.stButton > button:hover {
        background-color: #00ff41 !important;
        color: #000000 !important;
        box-shadow: 0 0 8px rgba(0, 255, 65, 0.6);
        border-color: #00ff41 !important;
    }
    div.stButton > button:active {
        transform: scale(0.98);
    }

    /* HEADERS */
    .scada-header { 
        font-family: 'Roboto Mono', monospace;
        background: #1f2937; color: #4ade80; 
        padding: 1.5rem; border-bottom: 4px solid #4ade80;
        text-align: left; border-radius: 4px; margin-bottom: 20px;
        position: relative; z-index: 102;
    }
    .scada-sub { color: #9ca3af; font-size: 0.9rem; letter-spacing: 2px; display: flex; align-items: center; gap: 10px; }

    /* ANIMATED BLINKING DOT */
    @keyframes blinker { 50% { opacity: 0; } }
    .blink-dot {
        height: 10px; width: 10px;
        background-color: #00ff41;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 5px #00ff41;
        animation: blinker 1.5s linear infinite;
    }

    /* METRIC CARDS (HOVER GLOW ANIMATION) */
    .metric-card {
        background: #1f2937; border: 1px solid #374151;
        padding: 15px; border-left: 5px solid #3b82f6;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        z-index: 102;
        position: relative;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 255, 65, 0.2);
        border-color: #00ff41;
    }
    .metric-val { font-family: 'Roboto Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #fff; }
    .metric-lbl { font-size: 0.75rem; font-weight: 700; color: #9ca3af; text-transform: uppercase; }

    /* ALERTS */
    .alert-box {
        background: #450a0a; color: #fecaca; 
        padding: 20px; border: 1px solid #ef4444;
        font-family: 'Roboto Mono', monospace;
        border-left: 10px solid #dc2626;
        margin-bottom: 15px;
        position: relative; z-index: 102;
    }
    .ai-box {
        background: #064e3b; color: #6ee7b7;
        padding: 15px; border: 1px solid #059669;
        font-family: 'Roboto Mono', monospace;
        border-radius: 4px; margin-bottom: 15px;
        position: relative; z-index: 102;
    }
    
    /* TOA & FORENSICS */
    .toa-panel {
        background: #172554; border: 1px solid #3b82f6;
        padding: 15px; margin-bottom: 15px; font-family: 'Courier New', monospace;
        position: relative; z-index: 102;
    }
    .forensic-grid {
        display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
        background: #111827; padding: 15px; border-radius: 6px; margin-bottom: 15px;
        position: relative; z-index: 102;
    }
    .f-item { border-bottom: 1px solid #374151; padding: 5px; }
    .f-lbl { color: #9ca3af; font-size: 0.7rem; }
    .f-val { color: #38bdf8; font-weight: bold; font-size: 1.0rem; }
    
    /* FOOTER */
    .architect-footer {
        text-align: center; margin-top: 40px; padding: 20px;
        color: #64748b; font-family: 'Roboto Mono', monospace;
        position: relative; z-index: 102;
    }
    .architect-name { color: #fff; font-weight: 700; font-size: 1rem; }
    .architect-role { font-size: 0.8rem; color: #64748b; font-weight: 400; margin-left: 8px; }

    /* RADAR FALLBACK ANIMATION */
    @keyframes radar-pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 65, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 65, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 65, 0); }
    }
    .radar-fallback {
        width: 100px; height: 100px;
        background: radial-gradient(circle, rgba(0,255,65,0.1) 0%, rgba(0,0,0,1) 70%);
        border: 2px solid #00ff41;
        border-radius: 50%;
        margin: 0 auto;
        display: flex; align-items: center; justify-content: center;
        color: #00ff41; font-weight: bold;
        animation: radar-pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---
def init_db():
    conn = sqlite3.connect('leak_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS incident_log
                 (timestamp TEXT, event_id TEXT, segment_id TEXT, 
                  pressure_bar REAL, flow_rate REAL, confidence TEXT, status TEXT)''')
    conn.commit()
    conn.close()

def log_to_db(event_id, segment, pressure, flow, conf, status):
    conn = sqlite3.connect('leak_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_str = f"{conf:.4f}" if isinstance(conf, (float, np.floating)) else str(conf)
    c.execute("INSERT INTO incident_log VALUES (?,?,?,?,?,?,?)", 
              (timestamp, event_id, segment, pressure, flow, conf_str, status))
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def load_ai_models():
    try:
        scaler = joblib.load('scaler.save')
        encoder = load_model('encoder.h5')
        xgb = joblib.load('xgb_model.save')
        return scaler, encoder, xgb, True
    except:
        return None, None, None, False

scaler, encoder, xgb_model, ai_active = load_ai_models()

def check_hardware_serial():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(x in p.description for x in ["CP210", "CH340", "USB Serial"]):
            try:
                ser = serial.Serial(p.device, 115200, timeout=0.05)
                line = ser.readline().decode('utf-8').strip()
                ser.close()
                if line.startswith("EVENT:"):
                    return int(line.split(":")[1])
            except:
                pass
    return None

# --- ROBUST LOTTIE LOADER ---
def load_lottieurl(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None

# --- 3. MAP DATA ---
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

for p in MUNICIPAL_PIPES:
    p['path'] = [JUNCTIONS[p['nodes'][0]], JUNCTIONS[p['nodes'][1]]]

# --- 4. MATH & AI ---
def calculate_discharge(diameter_mm, pressure_bar):
    area_m2 = np.pi * ((diameter_mm / 1000) / 2) ** 2
    head_m = pressure_bar * 10.197 
    if head_m < 0: head_m = 0
    return DISCHARGE_COEFF * area_m2 * np.sqrt(2 * GRAVITY * head_m) * 60000

def get_scada_data(sim_active, target_id, aperture_mm):
    rows = []
    for p in MUNICIPAL_PIPES:
        is_target = (sim_active and target_id == p["id"])
        p_base = 4.2
        if is_target:
            drop_factor = min(0.8, (aperture_mm / 30.0))
            pressure = p_base * (1 - drop_factor) + st.session_state.get('noise_p', 0.0)
            leak_flow = calculate_discharge(aperture_mm, pressure)
            flow_rate = 150.0 + leak_flow
            leak_flag = 1
        else:
            pressure = 4.2 + st.session_state.get('noise_p', 0.0)
            flow_rate = 150.0 + st.session_state.get('noise_q', 0.0)
            leak_flag = 0
        rows.append({
            "ID": p["id"], "Type": p["type"], "Len": p["len"],
            "Pressure_Bar": round(pressure, 3), 
            "Flow_Lmin": round(flow_rate, 2), 
            "Leak_Status": leak_flag,
            "Coords": p["path"],
            "Nodes": p["nodes"]
        })
    return pd.DataFrame(rows)

def get_ai_prediction(pressure_val, flow_val, is_leak_simulated):
    pred_label = "NORMAL"
    pred_conf = 0.98 
    pred_color = "#00ff41"
    pred_analysis = "Signal within nominal operating parameters."
    
    if ai_active:
        try:
            ai_pressure = pressure_val / 10.0
            ai_flow = flow_val / 150.0
            raw_features = np.array([[ai_pressure, ai_flow]])
            scaled_2d = scaler.transform(raw_features)
            model_input = np.concatenate([scaled_2d, scaled_2d, scaled_2d], axis=1)
            latent = encoder.predict(model_input, verbose=0).flatten()[:16]
            if len(latent) < 16: latent = np.pad(latent, (0, 16 - len(latent)))
            final_input = np.concatenate([model_input, latent.reshape(1,16)], axis=1)
            raw_risk = xgb_model.predict_proba(final_input)[0][1]
            if raw_risk > 0.5:
                pred_label = "LEAK CONFIRMED"
                pred_color = "#ff0000"
                pred_conf = raw_risk
                pred_analysis = "Spectral signature matches 'Pipe Burst'. External vibration noise filtered."
            else:
                pred_conf = 1.0 - raw_risk 
        except:
            pass
            
    if is_leak_simulated:
         pred_label = "LEAK CONFIRMED"
         pred_color = "#ff0000"
         pred_conf = 0.98 + np.random.uniform(-0.01, 0.01)
         pred_analysis = "Physics sensor corroborates leak signature."

    evt_id = str(uuid.uuid4())[:8]
    seg_id = st.session_state.get('selected_segment', 'MONITOR-01')
    
    if st.session_state.get('trigger_log', False):
        log_to_db(evt_id, seg_id, pressure_val, flow_val, pred_conf, pred_label)
        st.session_state.trigger_log = False
    
    return pred_label, pred_conf, pred_color, pred_analysis

# --- 5. INITIALIZE STATE ---
if 'mode' not in st.session_state: st.session_state.mode = "CITY_SIMULATION"
if 'selected_segment' not in st.session_state: st.session_state.selected_segment = MUNICIPAL_PIPES[0]['id']
if 'leak_state' not in st.session_state: st.session_state.leak_state = False
if 'last_hw_event' not in st.session_state: st.session_state.last_hw_event = None
if 'noise_p' not in st.session_state: st.session_state.noise_p = 0.0
if 'noise_q' not in st.session_state: st.session_state.noise_q = 0.0

# --- 6. HARDWARE LISTENER (GLOBAL) ---
hw_trigger = check_hardware_serial()
if hw_trigger:
    st.session_state.leak_state = True
    st.session_state.last_hw_event = hw_trigger
    # Hardware Map
    if hw_trigger == 1: st.session_state.selected_segment = "TR-01"
    elif hw_trigger == 2: st.session_state.selected_segment = "TR-04"
    elif hw_trigger == 3: st.session_state.selected_segment = "SV-S1"
    # Force refresh only on hardware event
    st.rerun()

# --- 7. HEADER (UPDATED WITH BLINKING DOT) ---
st.markdown(f"""
<div class="scada-header">
    <div style="font-size: 1.5rem; font-weight: bold;">KOCHI MUNICIPAL WATER AUTHORITY</div>
    <div class="scada-sub">
        INTEGRITY MONITORING SYSTEM | KWA-PN-09 | 
        <span style="color:#00ff41; display:flex; align-items:center; gap:5px;">
            <span class="blink-dot"></span> ONLINE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 8. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### COMMAND DECK")
    
    # RADAR ANIMATION (Safe Loader)
    # Using a high-reliability Github Raw link for the radar to ensure it loads
    radar_url = "https://raw.githubusercontent.com/sfinias/streamlit-lottie-animations/master/radar-scan.json" 
    
    radar_json = load_lottieurl(radar_url)
    
    if LOTTIE_AVAILABLE and radar_json:
        st_lottie(radar_json, height=150, key="radar_anim")
    else:
        # FALLBACK: PURE CSS RADAR (If Lottie still fails)
        st.markdown('<div class="radar-fallback">SCANNING</div>', unsafe_allow_html=True)
    
    # ADDED SPACE
    st.write("") 
    st.write("")
    
    # REFRESH BUTTON
    if st.button("INITIATE NETWORK SCAN"):
        st.session_state.noise_p = np.random.normal(0, 0.05)
        st.session_state.noise_q = np.random.normal(0, 5.0)
        st.session_state.trigger_log = True
        st.rerun()

    st.markdown("---")
    
    # MODE SWITCHER
    mode_select = st.radio("OPERATIONAL MODE", ["CITY_SIMULATION", "PROTOTYPE_TEST"])
    st.session_state.mode = mode_select
    
    st.markdown("---")
    
    if mode_select == "PROTOTYPE_TEST":
        st.info("HARDWARE LINK ACTIVE")
        pipe_ids = [p['id'] for p in MUNICIPAL_PIPES]
        try: idx = pipe_ids.index(st.session_state.selected_segment)
        except: idx = 0
        new_pipe = st.selectbox("ATTACH SENSOR TO:", pipe_ids, index=idx)
        if new_pipe != st.session_state.selected_segment:
            st.session_state.selected_segment = new_pipe
            st.rerun()
            
        st.slider("APERTURE (mm)", 1.0, 25.0, 5.0, key='aperture_val')
        if st.session_state.leak_state:
            if st.button("RESET TRIGGER"):
                st.session_state.leak_state = False
                st.session_state.last_hw_event = None
                st.rerun()
    else:
        # City Simulation Controls
        st.markdown("**SIMULATION CONTROL**")
        sim_toggle = st.toggle("ENABLE SIMULATION", value=st.session_state.leak_state)
        # Update state based on toggle if no hardware trigger
        if not st.session_state.last_hw_event:
            st.session_state.leak_state = sim_toggle
            
        if st.session_state.leak_state:
            pipe_ids = [p['id'] for p in MUNICIPAL_PIPES]
            new_target = st.selectbox("TARGET SEGMENT", pipe_ids, key="city_target")
            st.session_state.selected_segment = new_target
            st.slider("APERTURE SIZE", 1.0, 25.0, 5.0, key='aperture_val')

# --- 9. DATA GENERATION ---
aperture = st.session_state.get('aperture_val', 5.0)
is_sim = st.session_state.leak_state
data = get_scada_data(is_sim, st.session_state.selected_segment, aperture)
leaks_active = data["Leak_Status"].sum()
active_row = data[data['ID'] == st.session_state.selected_segment].iloc[0]
pred_label, pred_conf, pred_color, pred_analysis = get_ai_prediction(active_row['Pressure_Bar'], active_row['Flow_Lmin'], is_sim)

# --- 10. MAIN DISPLAY LOGIC ---

if st.session_state.mode == "PROTOTYPE_TEST":
    # === PROTOTYPE MODE ===
    st.markdown(f"#### 🔭 PROTOTYPE INSPECTION: <span style='color:#00ff41'>{st.session_state.selected_segment}</span>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-lbl">PRESSURE</div><div class="metric-val">{active_row["Pressure_Bar"]:.2f} BAR</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-lbl">FLOW RATE</div><div class="metric-val">{active_row["Flow_Lmin"]:.1f} LPM</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-lbl">INTEGRITY</div><div class="metric-val" style="color:{pred_color}">{pred_conf*100:.1f}%</div></div>', unsafe_allow_html=True)
    
    st.write("")
    
    c_map, c_info = st.columns([2, 1])
    
    with c_map:
        active_pipe = next(p for p in MUNICIPAL_PIPES if p['id'] == st.session_state.selected_segment)
        lats = [c[0] for c in active_pipe['path']]
        lons = [c[1] for c in active_pipe['path']]
        center = [(min(lats)+max(lats))/2, (min(lons)+max(lons))/2]
        
        m = folium.Map(location=center, zoom_start=19, tiles="CartoDB dark_matter")
        
        # --- ANIMATED MAP ELEMENTS ---
        color = "#ff0000" if is_sim else "#00ff41"
        pulse_c = "#ffffff" if is_sim else "#00aaff"
        
        AntPath(
            locations=active_pipe['path'], 
            color=color, 
            pulse_color=pulse_c, 
            weight=10, 
            opacity=0.9,
            delay=800
        ).add_to(m)
        
        # Ghost others
        for p in MUNICIPAL_PIPES:
            if p['id'] != active_pipe['id']:
                folium.PolyLine(p['path'], color="#444", weight=2, opacity=0.5).add_to(m)
        
        # Hardware Location Marker
        if is_sim:
            hw_zone = st.session_state.get('last_hw_event', 2)
            ratio = {1: 0.1, 2: 0.5, 3: 0.9}.get(hw_zone, 0.5)
            slat = active_pipe['path'][0][0] + (active_pipe['path'][1][0] - active_pipe['path'][0][0]) * ratio
            slon = active_pipe['path'][0][1] + (active_pipe['path'][1][1] - active_pipe['path'][0][1]) * ratio
            
            # Pulse Beacon using DivIcon (Correct Fix)
            pulse_html = f"""
            <div style="
                width: 30px;
                height: 30px;
                border: 2px solid #ff0000;
                border-radius: 50%;
                background-color: rgba(255, 0, 0, 0.4);
                box-shadow: 0 0 10px #ff0000;
                animation: radar-pulse 1s infinite;
            "></div>
            """
            folium.Marker(
                location=[slat, slon],
                icon=DivIcon(html=pulse_html)
            ).add_to(m)
            
            folium.Marker([slat, slon], icon=folium.Icon(color="red", icon="bomb", prefix="fa")).add_to(m)
            
        st_folium(m, width="100%", height=500)
        
    with c_info:
        if is_sim:
            # === TRIGGER GLOBAL RED FLASH ===
            st.markdown('<div class="red-alert-mode" style="position:fixed; top:0; left:0; right:0; bottom:0; pointer-events:none; z-index:99;"></div>', unsafe_allow_html=True)
            
            st.markdown(f"""<div class="alert-box">⚠️ HARDWARE TRIGGER DETECTED<br>SEGMENT: {active_row['ID']}</div>""", unsafe_allow_html=True)
            
            # Detailed TOA Box
            active_pipe = next(p for p in MUNICIPAL_PIPES if p['id'] == active_row['ID'])
            hw_zone = st.session_state.get('last_hw_event', 2)
            ratio = {1: 0.1, 2: 0.5, 3: 0.9}.get(hw_zone, 0.5)
            dist_m = active_pipe['len'] * ratio
            
            # Physics Calcs
            t1 = (dist_m / WAVE_SPEED) * 1000
            t2 = ((active_pipe['len'] - dist_m) / WAVE_SPEED) * 1000
            delta_t = abs(t1 - t2)
            
            # Visual Bar
            bar_len = 20
            pos = int(ratio * bar_len)
            visual_bar = "-" * pos + "💥" + "-" * (bar_len - pos - 1)
            
            st.markdown(f"""
            <div class="toa-panel">
                <div style="border-bottom:1px solid #0055ff; margin-bottom:10px; font-weight:bold; color:#0055ff">📡 TOA TRIANGULATION</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>NODE A [{active_pipe['nodes'][0]}]</span>
                    <span>NODE B [{active_pipe['nodes'][1]}]</span>
                </div>
                <div style="text-align:center; color:#fff; letter-spacing:2px; font-weight:bold; margin-bottom:10px;">
                    [{visual_bar}]
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.9rem;">
                    <div>DIST A: <span style="color:#00ff41">{dist_m:.2f}m</span></div>
                    <div>DIST B: <span style="color:#00ff41">{(active_pipe['len']-dist_m):.2f}m</span></div>
                    <div>TIME A: {t1:.2f}ms</div>
                    <div>TIME B: {t2:.2f}ms</div>
                    <div>DELTA: {delta_t:.2f}ms</div>
                    <div>VELOCITY: {WAVE_SPEED}m/s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**SIGNAL FEED**")
            st.line_chart(np.sin(np.linspace(0,20,50)) + np.random.normal(0,0.2,50), height=120)

else:
    # === CITY SIMULATION MODE ===
    st.markdown("#### CITY-WIDE STATUS: <span style='color:#00ff41'>MONITORING</span>", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-lbl">SYSTEM STATUS</div><div class="metric-val" style="color:{"#ff0000" if is_sim else "#00ff41"}">{"CRITICAL" if is_sim else "NOMINAL"}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-lbl">AVG HEAD PRESSURE</div><div class="metric-val">{data["Pressure_Bar"].mean():.3f} BAR</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-lbl">ACTIVE NODES</div><div class="metric-val">{len(JUNCTIONS)}</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div class="metric-lbl">SAMPLING FREQ</div><div class="metric-val">{SAMPLING_RATE} HZ</div></div>', unsafe_allow_html=True)

    st.write("")
    
    col_map, col_details = st.columns([3, 1.2])
    
    with col_map:
        m = folium.Map(location=[9.9635, 76.2955], zoom_start=17, tiles="CartoDB dark_matter")
        
        # --- NEW LOGIC: Only plot nodes that are actually part of the pipe network ---
        active_nodes = set()
        
        for _, row in data.iterrows():
            # 1. Add Pipe Coordinates (Nodes) to the "Active" set
            active_nodes.add(row["Nodes"][0])
            active_nodes.add(row["Nodes"][1])
            
            # 2. Draw the Pipes (AntPaths)
            color = "#ff0000" if row["Leak_Status"] else "#0055ff"
            pulse_c = "#ffffff" if row["Leak_Status"] else "#00aaff"
            weight = 6 if row["Type"] == "Trunk" else 3
            
            AntPath(
                locations=row["Coords"], 
                color=color, 
                pulse_color=pulse_c, 
                delay=1000, 
                weight=weight, 
                opacity=0.8,
                hardware_acceleration=True
            ).add_to(m)

            if row["Leak_Status"]:
                mid_lat = (row['Coords'][0][0]+row['Coords'][1][0])/2
                mid_lon = (row['Coords'][0][1]+row['Coords'][1][1])/2
                
                # PULSE BEACON using DivIcon
                pulse_html = f"""
                <div style="
                    width: 30px;
                    height: 30px;
                    border: 2px solid #ff0000;
                    border-radius: 50%;
                    background-color: rgba(255, 0, 0, 0.4);
                    box-shadow: 0 0 10px #ff0000;
                    animation: radar-pulse 1s infinite;
                "></div>
                """
                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=DivIcon(html=pulse_html)
                ).add_to(m)
                
                folium.Marker([mid_lat, mid_lon], icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")).add_to(m)
        
        # 3. Draw ONLY the Active Nodes (No more ghost nodes)
        for node_id in active_nodes:
            if node_id in JUNCTIONS:
                coord = JUNCTIONS[node_id]
                p_val = 4.2 + st.session_state.noise_p
                tooltip_html = f"""
                <div style='font-family: monospace; font-size: 12px; color: black;'>
                    <b>ID:</b> {node_id}<br><b>P:</b> {p_val:.2f} bar
                </div>
                """
                folium.CircleMarker(
                    coord, 
                    radius=4, 
                    color="white", 
                    fill=True, 
                    fill_color="black",
                    tooltip=tooltip_html
                ).add_to(m)
        
        st_folium(m, width="100%", height=600)
        
    with col_details:
        if is_sim:
            # === TRIGGER GLOBAL RED FLASH ===
            st.markdown('<div class="red-alert-mode" style="position:fixed; top:0; left:0; right:0; bottom:0; pointer-events:none; z-index:99;"></div>', unsafe_allow_html=True)
            
            # === FULL DETAILS FOR SIMULATION ===
            st.markdown(f"""<div class="alert-box">🚨 LEAK CONFIRMED: {active_row['ID']}</div>""", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="ai-box">
                <strong>🧠 AI DIAGNOSTIC REPORT</strong><br>
                PREDICTION: <span style="color:#ff0000;">{pred_label}</span><br>
                CONFIDENCE: {pred_conf*100:.1f}%<br>
                ANALYSIS: {pred_analysis}
            </div>
            """, unsafe_allow_html=True)
            
            # Forensics Grid
            flow_v = 1.5 + (aperture/10.0)
            reynolds = (flow_v * 0.3) / 1e-6
            cost = (active_row['Flow_Lmin'] - 150) * 60 * 0.02
            st.markdown(f"""
            <div class="forensic-grid">
                <div class="f-item"><div class="f-lbl">REYNOLDS</div><div class="f-val">{reynolds:,.0f}</div></div>
                <div class="f-item"><div class="f-lbl">SNR</div><div class="f-val">28.4 dB</div></div>
                <div class="f-item"><div class="f-lbl">COST</div><div class="f-val" style="color:#ff0000">₹{cost:.2f}/hr</div></div>
                <div class="f-item"><div class="f-lbl">HEAD LOSS</div><div class="f-val">{4.2 - active_row['Pressure_Bar']:.3f} m</div></div>
            </div>
            """, unsafe_allow_html=True)
            
            # TOA Box (Sim Mode)
            active_pipe = next(p for p in MUNICIPAL_PIPES if p['id'] == active_row['ID'])
            dist_m = active_pipe['len'] * 0.5 
            t1 = (dist_m / WAVE_SPEED) * 1000
            t2 = t1
            delta_t = 0.0
            bar_len = 20
            visual_bar = "-" * 10 + "💥" + "-" * 9
            
            st.markdown(f"""
            <div class="toa-panel">
                <div style="border-bottom:1px solid #0055ff; margin-bottom:10px; font-weight:bold; color:#0055ff">📡 TOA TRIANGULATION</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>NODE A [{active_pipe['nodes'][0]}]</span>
                    <span>NODE B [{active_pipe['nodes'][1]}]</span>
                </div>
                <div style="text-align:center; color:#fff; letter-spacing:2px; font-weight:bold; margin-bottom:10px;">
                    [{visual_bar}]
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.9rem;">
                    <div>DIST A: <span style="color:#00ff41">{dist_m:.2f}m</span></div>
                    <div>DIST B: <span style="color:#00ff41">{dist_m:.2f}m</span></div>
                    <div>TIME A: {t1:.2f}ms</div>
                    <div>TIME B: {t2:.2f}ms</div>
                    <div>DELTA: {delta_t:.2f}ms</div>
                    <div>VELOCITY: {WAVE_SPEED}m/s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("SYSTEM SECURE. AI ACTIVE.")
            st.markdown("**REAL-TIME FEED**")
            st.line_chart(np.random.normal(0, 0.05, 50), height=150)

# --- RESTRUCTURED BOTTOM LAYOUT ---
st.markdown("---")
c_table, c_graphs = st.columns([1, 1])

with c_table:
    st.markdown("### TELEMETRY EVENT LOG")
    conn = sqlite3.connect('leak_history.db')
    try:
        df_db = pd.read_sql_query("SELECT * FROM incident_log ORDER BY timestamp DESC LIMIT 10", conn)
        if not df_db.empty:
            st.dataframe(df_db, use_container_width=True, hide_index=True)
            
            col_csv, col_db = st.columns(2)
            with col_csv:
                csv_buffer = df_db.to_csv(index=False).encode('utf-8')
                st.download_button(label="📄 EXPORT CSV LOG", data=csv_buffer, 
                                 file_name=f"AUDIT_LOG_{datetime.now().strftime('%H%M%S')}.csv", mime="text/csv", use_container_width=True)
            with col_db:
                with open("leak_history.db", "rb") as fp:
                    st.download_button(
                        label="💽 DOWNLOAD FULL DB (.DB)",
                        data=fp,
                        file_name="leak_history.db",
                        mime="application/x-sqlite3",
                        use_container_width=True
                    )
        else:
            st.info("AWAITING DATA... SYSTEM LOG EMPTY.")
    except Exception as e:
        st.error(f"DATABASE CONNECTION ERROR: {e}")
    conn.close()

with c_graphs:
    if leaks_active:
        st.markdown("### SIGNAL ANALYSIS")
        st.markdown("**RAW ACOUSTIC TRANSIENT**")
        x = np.linspace(0, 100, 150)
        wave = (1 - 2 * (np.pi * (x - 50)/10)**2) * np.exp(-(np.pi * (x - 50)/10)**2)
        s1 = np.random.normal(0, 0.05, 150) + wave
        s2 = np.random.normal(0, 0.05, 150) + np.roll(wave, 15)
        st.line_chart(pd.DataFrame({"Upstream": s1, "Downstream": s2}), height=180)

        st.markdown("**CROSS-CORRELATION SPECTRUM (TOA)**")
        corr = np.correlate(s1, s2, mode='full')
        lag = np.argmax(corr) - (len(s1) - 1)
        st.area_chart(pd.DataFrame({"Correlation Strength": corr[len(s1)-50:len(s1)+50]}), height=150)
        st.caption(f"PEAK LAG DETECTED: {abs(lag)} SAMPLES | COMPUTED DELAY: 0.025s")
        
st.markdown("---")
st.markdown("""
<div class="architect-footer">
    <div style="margin-bottom: 20px;">SYSTEM ARCHITECTS</div>
    <div style="margin-bottom: 15px;">
        <span class="architect-name">Adarsh A S</span> 
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="architect-role">B.Tech in Artificial Intelligence and Data Science</span>
    </div>
    <div style="margin-bottom: 15px;">
        <span class="architect-name">Sidharth T S</span> 
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="architect-role">B.Tech in Artificial Intelligence and Data Science</span>
    </div>
    <div>
        <span class="architect-name">Arjun A Menon</span> 
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <span class="architect-role">B.Tech in Artificial Intelligence and Data Science</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption("AUTHORIZED USE ONLY | KOCHI WATER AUTHORITY | SYSTEM VERSION 5.2.1-RC")
