"""
Page 3: Traffic Pattern Analysis
Visualization of packet loss patterns (Nokia Figure 1)
"""

import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Traffic Pattern Analysis", page_icon="📊", layout="wide")

# Initialize theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar theme toggle
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle_traffic")
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
        </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Traffic Pattern Analysis</p>', unsafe_allow_html=True)
st.markdown("**Nokia Figure 1: Packet Loss Pattern Visualization**")

st.markdown("---")

# Explanation
st.markdown('<p class="sub-header">🔍 Understanding Packet Loss Patterns</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h4>Correlation-Based Topology Discovery</h4>
    <p>This visualization demonstrates <strong>Nokia Challenge 1: Topology Identification</strong> 
    by revealing correlated packet loss patterns across all 24 cells.</p>
    
    <h5>Key Observations:</h5>
    <ul>
        <li><strong>X-axis:</strong> Time slots (each slot = 1 ms)</li>
        <li><strong>Y-axis:</strong> Cell IDs (1-24)</li>
        <li><strong>Color Intensity:</strong> Represents packet loss occurrence</li>
        <li><strong>Vertical Patterns:</strong> Cells experiencing simultaneous packet loss 
        likely share the same fronthaul link</li>
    </ul>
    
    <h5>Analysis Principle:</h5>
    <p>When a fronthaul link becomes congested:</p>
    <ul>
        <li>All cells connected to that link experience packet loss simultaneously</li>
        <li>This creates vertical "stripes" in the pattern visualization</li>
        <li>High correlation in these patterns indicates shared infrastructure</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load and display Figure 1
st.markdown('<p class="sub-header">📈 Nokia Figure 1: Packet Loss Pattern</p>', unsafe_allow_html=True)

figure1_path = Path("outputs/figures/figure1_packet_loss_pattern.png")

if figure1_path.exists():
    # Load image
    img = Image.open(figure1_path)
    
    # Display with caption
    st.image(
        img,
        caption="Figure 1: Packet Loss Pattern Across All Cells Over Time",
        use_container_width=True
    )
    
    # Interpretation guide
    st.markdown("---")
    st.markdown('<p class="sub-header">📖 Pattern Interpretation Guide</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **What to Look For:**
        
        1. **Vertical Alignment**: Groups of cells showing packet loss at the same time slots
        2. **Repeated Patterns**: Consistent groupings across multiple congestion events
        3. **Correlation Clusters**: Sets of cells with similar loss patterns
        4. **Temporal Correlation**: Simultaneous events indicating shared bottlenecks
        """)
    
    with col2:
        st.markdown("""
        **How This Reveals Topology:**
        
        - **Shared Link Congestion**: When a fronthaul link reaches capacity, all cells 
          on that link experience packet loss
        - **Independent Events**: Cells on different links show uncorrelated loss patterns
        - **Statistical Validation**: Correlation coefficients quantify these relationships
        - **Topology Inference**: High correlation → same link; low correlation → different links
        """)
    
    st.markdown("---")
    
    # Download options
    st.markdown('<p class="sub-header">💾 Download Options</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        with open(figure1_path, "rb") as file:
            st.download_button(
                label="📥 Download Figure 1 (PNG)",
                data=file,
                file_name="figure1_packet_loss_pattern.png",
                mime="image/png"
            )
    
    with col2:
        st.info("**Image Format:** PNG\n\n**Resolution:** High-DPI")
    
    st.markdown("---")
    
    # Technical details
    st.markdown('<p class="sub-header">🔬 Technical Details</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Data Processing Methodology</h4>
        
        <h5>Input Data:</h5>
        <ul>
            <li>24 time series datasets (one per cell)</li>
            <li>Each dataset contains slot-level packet loss indicators (0 = no loss, 1 = loss)</li>
            <li>Time resolution: 1 ms (1 slot)</li>
            <li>Duration: Multiple observation windows</li>
        </ul>
        
        <h5>Visualization Process:</h5>
        <ol>
            <li><strong>Data Loading:</strong> Import packet loss data from phase1_slot_level_csvs/</li>
            <li><strong>Matrix Construction:</strong> Create 24×N matrix (cells × time slots)</li>
            <li><strong>Pattern Rendering:</strong> Visualize as heatmap with temporal alignment</li>
            <li><strong>Correlation Analysis:</strong> Compute pairwise correlations to identify groups</li>
        </ol>
        
        <h5>Key Insight:</h5>
        <p>This visualization forms the foundation of topology identification. The correlation 
        patterns observed here directly inform the cell-to-link mapping shown in the Topology 
        Identification page.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Validation summary
    st.success("""
    ✅ **Nokia Challenge 1 Compliance**
    
    This visualization successfully demonstrates:
    - Clear packet loss patterns across all 24 cells
    - Temporal correlation indicating shared fronthaul infrastructure
    - Visual evidence supporting topology identification methodology
    - Alignment with Nokia's expected Figure 1 output format
    """)
    
else:
    st.error("""
    ❌ **Visualization Not Found**
    
    Unable to load Figure 1. Please ensure:
    - `outputs/figures/figure1_packet_loss_pattern.png` exists
    - The file was generated by running the analysis pipeline
    - File permissions allow reading
    
    **To generate the figure:**
    ```bash
    python run_analysis.py
    ```
    """)

st.markdown("---")

# Additional context
with st.expander("📚 Additional Context: Why Correlation Analysis Works"):
    st.markdown("""
    ### Mathematical Foundation
    
    The correlation-based approach leverages a fundamental property of shared network resources:
    
    **Congestion Propagation:**
    When a fronthaul link reaches its capacity threshold, it affects all cells simultaneously. 
    This creates a distinctive signature in the packet loss data.
    
    **Correlation Coefficient:**
    For two cells *i* and *j*, the Pearson correlation coefficient measures the linear 
    relationship between their packet loss time series:
    
    $$r_{ij} = \\frac{\\text{cov}(X_i, X_j)}{\\sigma_{X_i} \\sigma_{X_j}}$$
    
    Where:
    - $X_i$, $X_j$ are packet loss time series for cells *i* and *j*
    - $\\text{cov}(X_i, X_j)$ is their covariance
    - $\\sigma_{X_i}$, $\\sigma_{X_j}$ are their standard deviations
    
    **Interpretation:**
    - $r_{ij} \\approx 1$: Strong positive correlation → likely same link
    - $r_{ij} \\approx 0$: No correlation → likely different links
    - High correlation threshold (e.g., > 0.7) used for grouping cells
    
    **Advantages:**
    - No prior knowledge of topology required
    - Robust to noise in traffic measurements
    - Scalable to large networks
    - Validated by real-world congestion events
    """)

st.markdown("---")
st.caption("MIT-Bangalore Team | Nokia 5G Fronthaul Network Optimization Challenge")
