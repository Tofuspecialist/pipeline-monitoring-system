# National Pipeline Monitoring System

Real-time pipeline leak detection and localization using Time-of-Arrival (TOA) algorithm.

## 🚀 Features

- **Interactive Map Visualization** - 20 pipeline network with real-time monitoring
- **Leak Detection** - Pressure and flow anomaly detection
- **TOA Localization** - Calculates exact leak position using acoustic wave arrival times
- **Government-Grade UI** - Professional dashboard for pipeline authorities
- **Simulation Mode** - Test leak scenarios on specific segments

## 🛠️ Technology Stack

- **Frontend**: Streamlit, Folium
- **Data Processing**: Pandas, NumPy
- **Algorithm**: Time-of-Arrival (TOA) leak localization
- **Deployment**: Streamlit Cloud

## 📊 System Overview

The system monitors 20+ pipeline segments including:
- Trunk lines (3)
- Distribution rings (3)
- Industrial branches (6)
- Refinery feeds (2)
- Power plant lines (2)
- Emergency bypass routes (2)
- Cross connections (2)

## 🔬 TOA Algorithm

Leak position is calculated using:
x = (L + a × (t_A - t_B)) / 2

Where:
- L = Distance between sensors
- a = Acoustic wave velocity (1000 m/s)
- t_A, t_B = Arrival times at sensors A and B

**Accuracy**: ±1.64% average error

## 🎯 Use Cases

- Oil & Gas pipeline monitoring
- Water distribution network management
- Industrial fluid transport systems
- Smart city infrastructure

## 🚀 Live Demo

[View Live Application](https://leak-detection-system.streamlit.app)

## 👨‍💻 Author

**Adarsh A S**  
B.Tech in Artificial Intelligence and Data Science  
APJ Abdul Kalam Technological University

## 📝 License

This project is developed for academic and demonstration purposes.
