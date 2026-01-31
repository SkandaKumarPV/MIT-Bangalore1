"""
Page 1: Overview
Problem statement and key parameters for Nokia 5G Fronthaul Network Optimization
"""

import streamlit as st

st.set_page_config(page_title="Overview", page_icon="📋", layout="wide")

# Initialize theme from session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar theme toggle
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle_overview")
    st.session_state.dark_mode = theme_toggle
    st.markdown("---")

# Enhanced CSS with dark mode
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); }
        .main-header {
            font-size: 2.8rem;
            background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #4ECDC4;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #4ECDC4;
            padding-bottom: 0.5rem;
        }
        .info-box {
            background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 5px solid #4ECDC4;
            margin: 1rem 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            transition: transform 0.3s ease;
        }
        .info-box:hover { transform: translateY(-5px); box-shadow: 0 12px 24px rgba(78, 205, 196, 0.3); }
        .param-box {
            background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            margin: 0.5rem 0;
            border: 1px solid #444;
            transition: all 0.3s ease;
        }
        .param-box:hover { transform: translateY(-5px); border-color: #4ECDC4; }
        .param-box h4 { color: #4ECDC4; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        .main-header {
            font-size: 2.8rem;
            background: linear-gradient(135deg, #124191 0%, #1e5fa8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #124191;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #124191;
            padding-bottom: 0.5rem;
        }
        .info-box {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 5px solid #124191;
            margin: 1rem 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .info-box:hover { transform: translateY(-5px); box-shadow: 0 12px 24px rgba(18, 65, 145, 0.2); }
        .param-box {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        .param-box:hover { transform: translateY(-5px); border-color: #124191; }
        .param-box h4 { color: #124191; }
        </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">📋 Overview</p>', unsafe_allow_html=True)
st.markdown("**Nokia 5G Fronthaul Network Optimization – Base Solution**")

st.markdown("---")

# Problem Statement
st.markdown('<p class="sub-header">📌 Problem Statement</p>', unsafe_allow_html=True)

st.markdown("""
In a 5G Open Radio Access Network (O-RAN), the **fronthaul** connects Distributed Units (DUs) 
to Radio Units (RUs) serving multiple cells. Understanding the fronthaul topology and accurately 
estimating link capacities are critical for network planning and optimization.

**Given:**
- Slot-level traffic measurements (throughput and packet loss) for 24 5G cells
- Time resolution: 1 ms (1 slot)
- Unknown topology: Which cells share which fronthaul links?
- Unknown link capacities

**Goal:**
Develop a solution that addresses two key challenges:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Challenge 1</h3>
        <h4>Fronthaul Topology Identification</h4>
        <p><strong>Objective:</strong> Determine which cells are connected to which fronthaul links.</p>
        <p><strong>Approach:</strong> Analyze correlated packet loss patterns. Cells sharing the same 
        fronthaul link experience simultaneous packet loss during congestion.</p>
        <p><strong>Output:</strong> Cell-to-link mapping</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Challenge 2</h3>
        <h4>Link Capacity Estimation</h4>
        <p><strong>Objective:</strong> Estimate the capacity of each fronthaul link.</p>
        <p><strong>Approach:</strong> Aggregate slot-level traffic for cells on each link and analyze 
        traffic distribution, considering buffer capacity.</p>
        <p><strong>Output:</strong> Link capacity values (with and without buffer)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Key Parameters
st.markdown('<p class="sub-header">🔧 Key Parameters</p>', unsafe_allow_html=True)

st.markdown("""
The solution operates under the following system parameters:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="param-box">
        <h4 style="color: #124191;">Network Structure</h4>
        <ul>
            <li><strong>Total Cells:</strong> 24</li>
            <li><strong>Total Links:</strong> 3</li>
            <li><strong>Architecture:</strong> DU → Links → Cells</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="param-box">
        <h4 style="color: #124191;">Timing Parameters</h4>
        <ul>
            <li><strong>Slot Duration:</strong> 1 ms</li>
            <li><strong>Symbol Duration:</strong> 35.7 μs</li>
            <li><strong>Buffer Size:</strong> 4 symbols</li>
            <li><strong>Buffer Time:</strong> 142.8 μs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="param-box">
        <h4 style="color: #124191;">Performance Constraints</h4>
        <ul>
            <li><strong>Max Packet Loss:</strong> 1%</li>
            <li><strong>Target Overload:</strong> < 1% of slots</li>
            <li><strong>QoS Requirement:</strong> Ultra-reliable low-latency</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Expected Outcomes
st.markdown('<p class="sub-header">✅ Expected Outcomes</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h4>Nokia Challenge 1: Topology Identification</h4>
    <ul>
        <li><strong>Figure 1:</strong> Packet loss pattern visualization showing correlated loss events</li>
        <li><strong>Figure 2:</strong> Network topology diagram showing DU, links, and cell groupings</li>
        <li><strong>Mapping Table:</strong> Cell ID → Link ID assignments</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h4>Nokia Challenge 2: Capacity Estimation</h4>
    <ul>
        <li><strong>Figure 3:</strong> Slot-level aggregated traffic per link over time</li>
        <li><strong>Capacity Values:</strong> Estimated capacity with and without buffer</li>
        <li><strong>Validation:</strong> Demonstration that packet loss ≤ 1% constraint is satisfied</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Methodology Summary
st.markdown('<p class="sub-header">🔬 Solution Methodology</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Topology Identification Process:**
    
    1. **Data Loading**: Import slot-level packet loss data for all 24 cells
    2. **Correlation Analysis**: Compute pairwise correlation of packet loss patterns
    3. **Clustering**: Group cells with high correlation (shared link congestion)
    4. **Validation**: Verify groupings using anchor cells and correlation thresholds
    5. **Visualization**: Generate topology diagrams and pattern plots
    """)

with col2:
    st.markdown("""
    **Capacity Estimation Process:**
    
    1. **Traffic Aggregation**: Sum throughput per slot for cells on each link
    2. **Distribution Analysis**: Analyze traffic distribution and identify peak loads
    3. **Buffer Modeling**: Account for 142.8 μs buffer absorption capability
    4. **Capacity Calculation**: Determine minimum capacity to satisfy packet loss constraint
    5. **Validation**: Verify overload percentage < 1% for all links
    """)

st.markdown("---")

st.success("""
**🎯 This solution demonstrates full compliance with both Nokia challenges and expected outcomes.**
Navigate to subsequent pages to explore detailed results and visualizations.
""")

st.markdown("---")
st.caption("MIT-Bangalore Team | Nokia 5G Fronthaul Network Optimization Challenge")
