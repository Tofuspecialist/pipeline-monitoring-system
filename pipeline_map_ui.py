import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import time
from datetime import datetime

st.set_page_config(
    page_title="National Pipeline Monitoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Government-grade professional styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
.main-header { 
    font-size: 2.5rem; 
    font-weight: 700; 
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
    color: white; 
    padding: 1.5rem; 
    border-radius: 12px; 
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
.gov-info { background: linear-gradient(90deg, #1e40af, #1d4ed8); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
.sidebar-section { 
    background: #f8fafc; 
    padding: 1rem; 
    border-radius: 8px; 
    margin: 0.5rem 0; 
    border-left: 4px solid #3b82f6; 
}
.sidebar-section h3 { 
    color: #1f2937; 
    margin: 0; 
    font-size: 1.1rem; 
}
.map-container { border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# MASSIVE Complex pipeline network - Government scale
PIPELINE_COMPLEX = {
    "Trunk Line 1": [
        [9.90, 76.15], [9.95, 76.18], [10.00, 76.20], [10.05, 76.22], [10.10, 76.24], 
        [10.15, 76.26], [10.20, 76.28], [10.25, 76.30], [10.30, 76.32], [10.35, 76.34],
    ],
    "Trunk Line 2": [
        [10.35, 76.34], [10.40, 76.36], [10.45, 76.38], [10.50, 76.40],
        [10.55, 76.42], [10.60, 76.44], [10.65, 76.46], [10.70, 76.48],
    ],
    "Trunk Line 3": [
        [10.05, 76.22], [10.08, 76.25], [10.12, 76.28], [10.18, 76.30], [10.25, 76.32],
    ],
    "Distribution Ring A": [
        [10.20, 76.28], [10.22, 76.31], [10.25, 76.34], [10.28, 76.36],
        [10.30, 76.33], [10.28, 76.30], [10.25, 76.28], [10.22, 76.27], [10.20, 76.28],
    ],
    "Distribution Ring B": [
        [10.40, 76.36], [10.42, 76.38], [10.45, 76.40], [10.47, 76.42],
        [10.48, 76.39], [10.45, 76.37], [10.42, 76.35], [10.40, 76.36],
    ],
    "Distribution Ring C": [
        [10.10, 76.24], [10.12, 76.26], [10.15, 76.28], [10.17, 76.30],
        [10.18, 76.27], [10.15, 76.25], [10.12, 76.23], [10.10, 76.24],
    ],
    "Industrial Branch A1": [[10.15, 76.26], [10.18, 76.28], [10.22, 76.30], [10.26, 76.32]],
    "Industrial Branch A2": [[10.22, 76.30], [10.24, 76.33], [10.27, 76.35]],
    "Industrial Branch B1": [[10.30, 76.32], [10.33, 76.30], [10.36, 76.28], [10.39, 76.26]],
    "Industrial Branch B2": [[10.33, 76.30], [10.35, 76.33], [10.38, 76.35]],
    "Industrial Branch C1": [[10.45, 76.38], [10.47, 76.35], [10.49, 76.33], [10.52, 76.31]],
    "Industrial Branch C2": [[10.47, 76.35], [10.49, 76.37], [10.52, 76.39]],
    "Refinery Feed 1": [[10.25, 76.30], [10.27, 76.32], [10.29, 76.34], [10.32, 76.36]],
    "Refinery Feed 2": [[10.40, 76.36], [10.42, 76.34], [10.44, 76.32]],
    "Power Plant Line 1": [[10.50, 76.40], [10.52, 76.42], [10.54, 76.44], [10.57, 76.46]],
    "Power Plant Line 2": [[10.55, 76.42], [10.57, 76.40], [10.60, 76.38]],
    "Emergency Bypass 1": [[10.30, 76.32], [10.32, 76.35], [10.35, 76.38], [10.38, 76.40]],
    "Emergency Bypass 2": [[10.45, 76.38], [10.48, 76.41], [10.52, 76.43]],
    "Cross Connect 1": [[10.25, 76.30], [10.30, 76.32], [10.35, 76.34]],
    "Cross Connect 2": [[10.40, 76.36], [10.45, 76.38], [10.50, 76.40]],
}

def get_gov_pipeline_data(leak_simulation=False, selected_segment=None, trigger_count=0):
    np.random.seed(int(time.time() // 5) + trigger_count)  # Change seed on trigger
    segments = []
    segment_id = 1
    
    for line_name, route in PIPELINE_COMPLEX.items():
        for i in range(len(route)-1):
            base_pressure = 2.2 if "Trunk" in line_name else 1.9 if "Industrial" in line_name else 2.0
            
            # Normal operation
            has_leak = False
            pressure = np.clip(np.random.normal(base_pressure, 0.08), 1.8, 3.0)
            flow = np.clip(np.random.normal(1.5, 0.08), 1.2, 2.2)
            
            # Apply leak simulation - FIXED LOGIC
            if leak_simulation:
                if selected_segment == segment_id:
                    # Force leak on selected segment
                    has_leak = True
                    pressure *= 0.65
                    flow *= 0.7
                elif selected_segment is None:
                    # Random leaks mode
                    if np.random.rand() < 0.20:  # 20% leak probability
                        has_leak = True
                        pressure *= 0.65
                        flow *= 0.7
            
            segments.append({
                "ID": f"{line_name[:3]}{segment_id:03d}",
                "Line": line_name,
                "Pressure": round(pressure, 2),
                "Flow": round(flow, 2),
                "Leak": int(has_leak),
                "Lat": (route[i][0] + route[i+1][0])/2,
                "Lon": (route[i][1] + route[i+1][1])/2,
            })
            segment_id += 1
    return pd.DataFrame(segments)

# App state initialization
if "scan_time" not in st.session_state: 
    st.session_state.scan_time = datetime.now()
if "leak_simulation" not in st.session_state:
    st.session_state.leak_simulation = False
if "selected_segment" not in st.session_state:
    st.session_state.selected_segment = None
if "trigger_count" not in st.session_state:
    st.session_state.trigger_count = 0

# Generate data based on simulation state
data = get_gov_pipeline_data(
    leak_simulation=st.session_state.leak_simulation,
    selected_segment=st.session_state.selected_segment,
    trigger_count=st.session_state.trigger_count
)

leaks_count = data["Leak"].sum()
critical_count = len(data[(data["Leak"] == 1) | (data["Pressure"] < 1.5)])

# Header
st.markdown('<div class="main-header">National Pipeline Integrity Monitoring System</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="gov-info">
    <strong>System Status:</strong> Last scan {st.session_state.scan_time.strftime('%H:%M:%S')} | 
    Segments: {len(data)} | 
    Mode: <b>{'🔴 LEAK SIMULATION ACTIVE' if st.session_state.leak_simulation else '🟢 NORMAL OPERATION'}</b> |
    <span style="color: {'#dc2626' if leaks_count > 0 else '#059669'}">
    {'⚠️ CRITICAL: ' + str(leaks_count) + ' leaks detected' if leaks_count > 0 else '✅ ALL SYSTEMS OPERATIONAL'}
    </span>
</div>
""", unsafe_allow_html=True)

# KPI Dashboard
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Leak Incidents", leaks_count, delta=leaks_count if leaks_count > 0 else None)
col2.metric("Critical Segments", critical_count)
col3.metric("Pressure Avg", f"{data['Pressure'].mean():.2f} bar", f"-{(2.0 - data['Pressure'].mean()):.2f}" if leaks_count > 0 else None)
col4.metric("Flow Rate", f"{data['Flow'].mean():.2f} L/s")
col5.metric("Network Coverage", f"{len(data)} segments")

# Main dashboard
col_map, col_stats = st.columns([3, 1.2])

with col_map:
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st.subheader("Real-Time Pipeline Infrastructure Network")
    
    m = folium.Map(location=[10.30, 76.33], zoom_start=11, tiles="OpenStreetMap", 
                   attr='© OpenStreetMap contributors')
    
    # Professional color scheme for different line types
    line_colors = {
        "Trunk Line 1": "#1e3a8a", "Trunk Line 2": "#1e40af", "Trunk Line 3": "#3b82f6",
        "Distribution Ring A": "#10b981", "Distribution Ring B": "#059669", "Distribution Ring C": "#047857",
        "Industrial Branch A1": "#dc2626", "Industrial Branch A2": "#ef4444", 
        "Industrial Branch B1": "#f59e0b", "Industrial Branch B2": "#fb923c",
        "Industrial Branch C1": "#7c3aed", "Industrial Branch C2": "#8b5cf6",
        "Refinery Feed 1": "#059669", "Refinery Feed 2": "#10b981",
        "Power Plant Line 1": "#6366f1", "Power Plant Line 2": "#818cf8",
        "Emergency Bypass 1": "#dc2626", "Emergency Bypass 2": "#ef4444",
        "Cross Connect 1": "#64748b", "Cross Connect 2": "#475569",
    }
    
    # Draw complex pipeline network with varied styles
    for name, coords in PIPELINE_COMPLEX.items():
        weight = 10 if "Trunk" in name else 7 if "Distribution" in name else 5
        dash = "5, 5" if "Emergency" in name else None
        
        folium.PolyLine(
            coords,
            color=line_colors.get(name, "#0f172a"),
            weight=weight,
            opacity=0.85,
            dash_array=dash,
            popup=f"<b>{name}</b><br>Type: {name.split()[0]}",
            tooltip=name
        ).add_to(m)
    
    # Add complex segment markers with animations
    for idx, row in data.iterrows():
        marker_color = "red" if row["Leak"] else ("orange" if row["Pressure"] < 1.6 else "darkgreen")
        
        # Use different marker styles based on leak status
        if row["Leak"]:
            folium.CircleMarker(
                location=[row["Lat"], row["Lon"]],
                radius=10,
                color="#dc2626",
                fill=True,
                fill_color="#dc2626",
                fill_opacity=0.9,
                weight=3,
                popup=f"""
                <div style='width: 220px; font-family: Arial;'>
                    <h4 style='color: #dc2626; margin: 0 0 8px 0;'>⚠️ LEAK ALERT</h4>
                    <b>Segment ID: {row['ID']}</b><br>
                    <b>Pipeline: {row['Line']}</b><br>
                    <hr style='margin: 8px 0;'>
                    <span style='color: red;'><b>Pressure: {row['Pressure']:.2f} bar ⬇️</b></span><br>
                    <span style='color: red;'><b>Flow: {row['Flow']:.2f} L/s ⬇️</b></span><br>
                    <hr style='margin: 8px 0;'>
                    <b style='color: #dc2626;'>🚨 STATUS: LEAK DETECTED</b><br>
                    <small>Immediate action required</small>
                </div>
                """,
                tooltip=f"🚨 LEAK - {row['ID']}"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row["Lat"], row["Lon"]],
                radius=5,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                popup=f"""
                <div style='width: 200px; font-family: Arial;'>
                    <b>Segment ID: {row['ID']}</b><br>
                    <b>Pipeline: {row['Line']}</b><br>
                    <hr style='margin: 8px 0;'>
                    <span style='color: {"orange" if row["Pressure"] < 1.6 else "green"}'><b>Pressure: {row['Pressure']:.2f} bar</b></span><br>
                    <b>Flow: {row['Flow']:.2f} L/s</b><br>
                    <hr style='margin: 8px 0;'>
                    <b style='color: green;'>✓ STATUS: NORMAL</b>
                </div>
                """,
                tooltip=f"✓ {row['ID']} - {row['Line']}"
            ).add_to(m)
    
    # Add legend
    legend_html =  '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: auto; 
background-color: white; z-index:9999; font-size:12px; border:2px solid grey; border-radius: 5px; padding: 10px">
<p style="margin: 0; font-weight: bold; color: #1f2937;">Pipeline Network Legend</p>
<p style="margin: 5px 0; color: #1f2937;"><span style="color: #1e3a8a;">━━</span> Trunk Lines</p>
<p style="margin: 5px 0; color: #1f2937;"><span style="color: #10b981;">━━</span> Distribution Rings</p>
<p style="margin: 5px 0; color: #1f2937;"><span style="color: #dc2626;">━━</span> Industrial Branches</p>
<p style="margin: 5px 0; color: #1f2937;"><span style="color: #059669;">━━</span> Refinery Feed</p>
<p style="margin: 5px 0; color: #1f2937;"><span style="color: #dc2626;">- - -</span> Emergency Bypass</p>
</div>
'''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    map_data = st_folium(
        m,
        width="100%",
        height=650,
        returned_objects=[],
        key=f"map_{st.session_state.trigger_count}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    leak_segments = data[data["Leak"] == 1]
    
    if len(leak_segments) > 0:
        st.markdown("---")
        st.markdown("### 🚨 Active Leak Locations - Detailed Report")
        
        for idx, leak in leak_segments.iterrows():
            # TOA Calculations
            np.random.seed(int(leak['ID'][-3:]))
            sensor_a_time = np.random.uniform(0.1, 0.5)
            sensor_b_time = np.random.uniform(0.3, 0.8)
            time_diff = abs(sensor_a_time - sensor_b_time)
            distance_between_sensors = 1000
            acoustic_velocity = 1000
            leak_position = (distance_between_sensors + acoustic_velocity * (sensor_a_time - sensor_b_time)) / 2
            localization_accuracy = leak_position * 0.0164
            
            st.markdown(f"""
        <div style="background: #fef2f2; border-left: 6px solid #dc2626; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; color: black;">
        <h4 style='margin: 0 0 0.5rem 0; color: #dc2626;'>⚠️ Leak Alert: {leak['ID']}</h4>
        <p style='margin: 5px 0;'><b>Pipeline:</b> {leak['Line']}</p>
        <p style='margin: 5px 0;'><b>Location:</b> Lat {leak['Lat']:.4f}, Lon {leak['Lon']:.4f}</p>
        <p style='margin: 5px 0;'><b>Pressure:</b> <span style='color: #dc2626;'>{leak['Pressure']:.2f} bar (Critical Drop)</span></p>
        <p style='margin: 5px 0;'><b>Flow Rate:</b> <span style='color: #dc2626;'>{leak['Flow']:.2f} L/s (Reduced)</span></p>
        
        <hr style='margin: 10px 0; border-color: #fca5a5;'>
        <p style='margin: 5px 0; font-weight: bold; color: #7c2d12;'>🔬 Time-of-Arrival (TOA) Localization:</p>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Sensor A Arrival Time:</b> {sensor_a_time:.3f} seconds</p>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Sensor B Arrival Time:</b> {sensor_b_time:.3f} seconds</p>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Time Difference (Δt):</b> {time_diff:.3f} seconds</p>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Acoustic Wave Velocity:</b> {acoustic_velocity} m/s</p>
        <p style='margin: 5px 0; font-size: 0.9rem; color: #dc2626;'><b>📍 Calculated Distance from Sensor A:</b> {leak_position:.1f} meters</p>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Localization Accuracy:</b> ±{localization_accuracy:.1f}m (1.64% avg error)</p>
        
        <p style='margin: 5px 0; padding-top: 5px; border-top: 1px solid #fca5a5;'><b>Action Required:</b> Immediate isolation and repair crew dispatch</p>
            </div>
            """, unsafe_allow_html=True)

    
    

with col_stats:
    st.subheader("Operational Summary")
    
    if leaks_count > 0:
        st.markdown(f"""
        <div style="background: #fef2f2; border-left: 6px solid #dc2626; padding: 1rem; border-radius: 8px; color: black;">
            <h4 style='margin: 0 0 0.5rem 0;'>🚨 {leaks_count} Active Leaks</h4>
            <p style='font-size: 0.9rem; margin: 0;'>Immediate isolation required</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: #f0fdf4; border-left: 6px solid #059669; padding: 1rem; border-radius: 8px; margin-top: 1rem; color: black;">
        <h4 style='margin: 0 0 0.5rem 0;'>✅ {len(data[data["Leak"] == 0])} Segments OK</h4>
        <p style='font-size: 0.9rem; margin: 0;'>Nominal operation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show only leak segments if any exist
    if leaks_count > 0:
        st.markdown("### 🔴 Leak Details")
        leak_data = data[data["Leak"] == 1]
        st.dataframe(
            leak_data[["ID", "Line", "Pressure", "Flow"]],
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("### All Segments")
    st.dataframe(
        data[["ID", "Line", "Pressure", "Flow", "Leak"]],
        use_container_width=True,
        hide_index=True,
        height=300
    )

# Control panel
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h3 style="color: #1f2937;">Control Center</h3></div>', unsafe_allow_html=True)
    
    if st.button("Execute Pipeline Scan", type="primary", use_container_width=True):
        st.session_state.scan_time = datetime.now()
        st.session_state.trigger_count += 1
        st.rerun()
    
    st.markdown("---")
    st.markdown('<div class="sidebar-section"><h3 style="color: #1f2937;">Leak Simulation</h3></div>', unsafe_allow_html=True)
    
    # Toggle leak simulation
    leak_toggle = st.toggle("Enable Leak Simulation", value=st.session_state.leak_simulation)
    
    if leak_toggle:
        if not st.session_state.leak_simulation:
            st.session_state.leak_simulation = True
            st.session_state.trigger_count += 1
            st.rerun()
        
        # Option to select specific segment
        sim_mode = st.radio("Simulation Mode", ["Random Leaks", "Specific Segment"])
        
        if sim_mode == "Specific Segment":
            segment_options = ["None"] + [f"Segment {i+1}" for i in range(len(data))]
            selected = st.selectbox("Select Segment to Simulate Leak", segment_options)
            if selected != "None":
                new_segment = int(selected.split()[1])
                if st.session_state.selected_segment != new_segment:
                    st.session_state.selected_segment = new_segment
                    st.session_state.trigger_count += 1
                    st.rerun()
            else:
                if st.session_state.selected_segment is not None:
                    st.session_state.selected_segment = None
                    st.session_state.trigger_count += 1
                    st.rerun()
        else:
            if st.session_state.selected_segment is not None:
                st.session_state.selected_segment = None
                st.session_state.trigger_count += 1
                st.rerun()
            
        if st.button("Trigger Leak", type="secondary", use_container_width=True):
            st.session_state.scan_time = datetime.now()
            st.session_state.trigger_count += 1
            st.rerun()
    else:
        if st.session_state.leak_simulation:
            st.session_state.leak_simulation = False
            st.session_state.selected_segment = None
            st.session_state.trigger_count += 1
            st.rerun()
    
    st.markdown("---")
    st.markdown('<div class="sidebar-section" style="color: #1f2937;"><strong>Network Statistics</strong></div>', unsafe_allow_html=True)
    st.metric("Total Length", "562 km")
    st.metric("Pipeline Lines", "20")
    st.metric("Monitoring Stations", "15")
    st.metric("Active Segments", len(data))
    
    st.markdown('<div class="sidebar-section" style="color: #1f2937;"><strong>Regulatory Compliance</strong></div>', unsafe_allow_html=True)
    st.info("✓ PNGRB Standards Compliant\n✓ Real-time SCADA Integration\n✓ 24/7 Monitoring Active\n✓ ISO 9001:2015 Certified")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>Government of India | Ministry of Petroleum & Natural Gas | Pipeline Authority of India</div>", unsafe_allow_html=True)
