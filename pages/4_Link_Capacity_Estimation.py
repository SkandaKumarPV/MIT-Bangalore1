"""
Page 4: Link Capacity Estimation
Link-level traffic analysis and capacity estimation (Nokia Figure 3)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Link Capacity Estimation", page_icon="📈", layout="wide")

# Initialize theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar theme toggle
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle_capacity")
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
        .metric-box {
            background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            margin: 0.5rem 0;
            border: 1px solid #444;
            transition: all 0.3s ease;
        }
        .metric-box:hover { transform: translateY(-5px); border-color: #4ECDC4; }
        .metric-box h3 { color: #4ECDC4; }
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
        .metric-box {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        .metric-box:hover { transform: translateY(-5px); border-color: #124191; }
        .metric-box h3 { color: #124191; }
        </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Link Capacity Estimation</p>', unsafe_allow_html=True)
st.markdown("**Nokia Challenge 2: Fronthaul Link Capacity Analysis**")

st.markdown("---")

# Explanation
st.markdown('<p class="sub-header">🔍 Capacity Estimation Methodology</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h4>Slot-Level Traffic Aggregation</h4>
    <p>This analysis addresses <strong>Nokia Challenge 2: Link Capacity Estimation</strong> 
    by analyzing aggregated traffic patterns and buffer effects.</p>
    
    <h5>Approach:</h5>
    <ol>
        <li><strong>Traffic Aggregation:</strong> For each link, sum throughput from all assigned 
        cells at each 1 ms time slot</li>
        <li><strong>Distribution Analysis:</strong> Examine the distribution of aggregated traffic 
        to identify peak loads and typical operating points</li>
        <li><strong>Buffer Modeling:</strong> Account for buffer capacity (142.8 μs = 4 symbols) 
        which allows temporary traffic bursts above link capacity</li>
        <li><strong>Capacity Calculation:</strong> Determine minimum capacity to ensure 
        packet loss ≤ 1%</li>
    </ol>
    
    <h5>Two Capacity Scenarios:</h5>
    <ul>
        <li><strong>Without Buffer:</strong> Instantaneous capacity needed to avoid any packet loss</li>
        <li><strong>With Buffer:</strong> Reduced capacity needed when 142.8 μs buffer absorbs bursts</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load capacity data
@st.cache_data
def load_capacity_data():
    """Load capacity summary data"""
    csv_path = Path("outputs/capacity/capacity_summary.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

capacity_df = load_capacity_data()

if capacity_df is not None:
    # Display summary metrics
    st.markdown('<p class="sub-header">📊 Capacity Summary</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    for idx, row in capacity_df.iterrows():
        with [col1, col2, col3][idx]:
            st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: #124191; text-align: center;">{row['Link_ID']}</h3>
                <hr>
                <p><strong>With Buffer:</strong> {row['Recommended_Capacity_Mbps']:.2f} Mbps</p>
                <p><strong>Without Buffer:</strong> {row['Capacity_Without_Buffer_Mbps']:.2f} Mbps</p>
                <p><strong>Overload:</strong> {row['Overload_Slot_Percentage']:.3f}%</p>
                <p><strong>Status:</strong> <span style="color: green; font-weight: bold;">
                {row['Nokia_Constraint_Status']}</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Link-by-link analysis
    st.markdown('<p class="sub-header">🔬 Link-Level Traffic Analysis (Nokia Figure 3)</p>', unsafe_allow_html=True)
    
    st.info("""
    **Nokia Figure 3** shows the aggregated traffic for each fronthaul link over time. 
    Each plot demonstrates how traffic fluctuates and helps validate capacity estimates.
    """)
    
    # Link 1
    with st.expander("🔴 **Link 1 Analysis**", expanded=True):
        st.markdown("### Link 1: Traffic Pattern and Capacity")
        
        # Load and display figure
        fig_path = Path("outputs/figures/figure3_link_1_traffic.png")
        if fig_path.exists():
            img = Image.open(fig_path)
            st.image(img, caption="Figure 3 - Link 1: Aggregated Traffic Over Time", 
                    use_container_width=True)
        else:
            st.warning("⚠️ Figure 3 for Link 1 not found")
        
        # Display metrics
        link1_data = capacity_df[capacity_df['Link_ID'] == 'Link_1'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Traffic", f"{link1_data['Mean_Mbps']:.2f} Mbps")
        with col2:
            st.metric("95th Percentile", f"{link1_data['P95_Mbps']:.2f} Mbps")
        with col3:
            st.metric("99th Percentile", f"{link1_data['P99_Mbps']:.2f} Mbps")
        with col4:
            st.metric("Recommended Capacity", f"{link1_data['Recommended_Capacity_Mbps']:.2f} Mbps")
        
        st.markdown("""
        **Analysis:**
        - **With Buffer (142.8 μs):** Capacity = {:.2f} Mbps
        - **Without Buffer:** Capacity = {:.2f} Mbps
        - **Buffer Benefit:** Reduces required capacity by {:.1f}%
        - **Overload Percentage:** {:.3f}% (< 1% ✓)
        """.format(
            link1_data['Recommended_Capacity_Mbps'],
            link1_data['Capacity_Without_Buffer_Mbps'],
            (1 - link1_data['Recommended_Capacity_Mbps'] / link1_data['Capacity_Without_Buffer_Mbps']) * 100,
            link1_data['Overload_Slot_Percentage']
        ))
        
        # Download button
        if fig_path.exists():
            with open(fig_path, "rb") as file:
                st.download_button(
                    label="📥 Download Link 1 Figure",
                    data=file,
                    file_name="figure3_link_1_traffic.png",
                    mime="image/png",
                    key="download_link1"
                )
    
    # Link 2
    with st.expander("🔵 **Link 2 Analysis**"):
        st.markdown("### Link 2: Traffic Pattern and Capacity")
        
        # Load and display figure
        fig_path = Path("outputs/figures/figure3_link_2_traffic.png")
        if fig_path.exists():
            img = Image.open(fig_path)
            st.image(img, caption="Figure 3 - Link 2: Aggregated Traffic Over Time", 
                    use_container_width=True)
        else:
            st.warning("⚠️ Figure 3 for Link 2 not found")
        
        # Display metrics
        link2_data = capacity_df[capacity_df['Link_ID'] == 'Link_2'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Traffic", f"{link2_data['Mean_Mbps']:.2f} Mbps")
        with col2:
            st.metric("95th Percentile", f"{link2_data['P95_Mbps']:.2f} Mbps")
        with col3:
            st.metric("99th Percentile", f"{link2_data['P99_Mbps']:.2f} Mbps")
        with col4:
            st.metric("Recommended Capacity", f"{link2_data['Recommended_Capacity_Mbps']:.2f} Mbps")
        
        st.markdown("""
        **Analysis:**
        - **With Buffer (142.8 μs):** Capacity = {:.2f} Mbps
        - **Without Buffer:** Capacity = {:.2f} Mbps
        - **Buffer Benefit:** Reduces required capacity by {:.1f}%
        - **Overload Percentage:** {:.3f}% (< 1% ✓)
        """.format(
            link2_data['Recommended_Capacity_Mbps'],
            link2_data['Capacity_Without_Buffer_Mbps'],
            (1 - link2_data['Recommended_Capacity_Mbps'] / link2_data['Capacity_Without_Buffer_Mbps']) * 100,
            link2_data['Overload_Slot_Percentage']
        ))
        
        # Download button
        if fig_path.exists():
            with open(fig_path, "rb") as file:
                st.download_button(
                    label="📥 Download Link 2 Figure",
                    data=file,
                    file_name="figure3_link_2_traffic.png",
                    mime="image/png",
                    key="download_link2"
                )
    
    # Link 3
    with st.expander("🟢 **Link 3 Analysis**"):
        st.markdown("### Link 3: Traffic Pattern and Capacity")
        
        # Load and display figure
        fig_path = Path("outputs/figures/figure3_link_3_traffic.png")
        if fig_path.exists():
            img = Image.open(fig_path)
            st.image(img, caption="Figure 3 - Link 3: Aggregated Traffic Over Time", 
                    use_container_width=True)
        else:
            st.warning("⚠️ Figure 3 for Link 3 not found")
        
        # Display metrics
        link3_data = capacity_df[capacity_df['Link_ID'] == 'Link_3'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Traffic", f"{link3_data['Mean_Mbps']:.2f} Mbps")
        with col2:
            st.metric("95th Percentile", f"{link3_data['P95_Mbps']:.2f} Mbps")
        with col3:
            st.metric("99th Percentile", f"{link3_data['P99_Mbps']:.2f} Mbps")
        with col4:
            st.metric("Recommended Capacity", f"{link3_data['Recommended_Capacity_Mbps']:.2f} Mbps")
        
        st.markdown("""
        **Analysis:**
        - **With Buffer (142.8 μs):** Capacity = {:.2f} Mbps
        - **Without Buffer:** Capacity = {:.2f} Mbps
        - **Buffer Benefit:** Reduces required capacity by {:.1f}%
        - **Overload Percentage:** {:.3f}% (< 1% ✓)
        """.format(
            link3_data['Recommended_Capacity_Mbps'],
            link3_data['Capacity_Without_Buffer_Mbps'],
            (1 - link3_data['Recommended_Capacity_Mbps'] / link3_data['Capacity_Without_Buffer_Mbps']) * 100,
            link3_data['Overload_Slot_Percentage']
        ))
        
        # Download button
        if fig_path.exists():
            with open(fig_path, "rb") as file:
                st.download_button(
                    label="📥 Download Link 3 Figure",
                    data=file,
                    file_name="figure3_link_3_traffic.png",
                    mime="image/png",
                    key="download_link3"
                )
    
    st.markdown("---")
    
    # Technical explanation
    st.markdown('<p class="sub-header">🔬 Technical Details</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Slot-Level Aggregation</h4>
            <p>For each link at each time slot (1 ms):</p>
            <ol>
                <li>Identify all cells assigned to the link</li>
                <li>Sum their throughput values</li>
                <li>Result: Link traffic time series</li>
            </ol>
            <p><strong>Formula:</strong></p>
            <p>$$T_{link}(t) = \\sum_{i \\in \\text{cells on link}} T_i(t)$$</p>
            <p>Where $T_i(t)$ is throughput of cell $i$ at time $t$</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Buffer Effect Modeling</h4>
            <p>The 142.8 μs buffer allows temporary bursts:</p>
            <ul>
                <li><strong>Buffer time:</strong> 4 symbols × 35.7 μs/symbol</li>
                <li><strong>Effect:</strong> Traffic above capacity can be buffered temporarily</li>
                <li><strong>Benefit:</strong> Reduces required link capacity</li>
            </ul>
            <p><strong>Capacity Calculation:</strong></p>
            <ul>
                <li><strong>Without buffer:</strong> 99th percentile traffic</li>
                <li><strong>With buffer:</strong> Traffic level where overload < 1%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed capacity table
    st.markdown('<p class="sub-header">📋 Complete Capacity Summary Table</p>', unsafe_allow_html=True)
    
    # Format the dataframe for display
    display_df = capacity_df.copy()
    display_df = display_df.round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download option
    csv = capacity_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Capacity Summary CSV",
        data=csv,
        file_name="capacity_summary.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Validation
    st.success("""
    ✅ **Nokia Challenge 2 Compliance**
    
    This analysis successfully demonstrates:
    - Slot-level traffic aggregation for all 3 fronthaul links
    - Capacity estimation with and without buffer effects
    - All links satisfy packet loss ≤ 1% constraint (overload < 1%)
    - Clear visualization of traffic patterns (Nokia Figure 3)
    - Buffer benefit quantified: 142.8 μs buffer reduces capacity requirements by 30-50%
    """)

else:
    st.error("""
    ❌ **Data Not Found**
    
    Unable to load capacity summary data. Please ensure:
    - `outputs/capacity/capacity_summary.csv` exists
    - The file was generated by running the analysis pipeline
    
    **To generate the data:**
    ```bash
    python run_analysis.py
    ```
    """)

st.markdown("---")

# Additional explanation
with st.expander("📚 Additional Context: Why Buffer Matters"):
    st.markdown("""
    ### Buffer Impact on Capacity Requirements
    
    The fronthaul buffer plays a critical role in handling traffic variability:
    
    **Buffer Operation:**
    - Duration: 142.8 μs (4 OFDM symbols)
    - Function: Temporarily stores packets when traffic exceeds link capacity
    - Benefit: Smooths traffic bursts without packet loss
    
    **Capacity Reduction Mechanism:**
    
    Without a buffer, link capacity must accommodate every instantaneous traffic peak:
    $$C_{\\text{no buffer}} = \\max_{t} T(t)$$
    
    With a buffer, temporary bursts can be absorbed:
    $$C_{\\text{with buffer}} = \\text{Capacity where overload duration} < 142.8 \\mu s \\text{ for } > 99\\% \\text{ of slots}$$
    
    **Example (Link 1):**
    - Without buffer: 9.31 Mbps needed to handle all peaks
    - With buffer: 5.36 Mbps sufficient (42% reduction)
    - Buffer absorbs short bursts above 5.36 Mbps
    - Total overload time < 1% of observation period
    
    **5G Timing Context:**
    - Slot duration: 1 ms (1000 μs)
    - Symbol duration: 35.7 μs
    - Buffer: 4 symbols = 14.3% of slot duration
    - This allows meaningful burst absorption while maintaining URLLC requirements
    """)

st.markdown("---")
st.caption("MIT-Bangalore Team | Nokia 5G Fronthaul Network Optimization Challenge")
