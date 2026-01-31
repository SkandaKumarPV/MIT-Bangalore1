"""
Page 2: Topology Identification
Interactive visualization of fronthaul network topology
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Topology Identification", page_icon="🌐", layout="wide")

# Initialize theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar theme toggle
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle_topo")
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

st.markdown('<p class="main-header">🌐 Topology Identification</p>', unsafe_allow_html=True)
st.markdown("**Challenge 1: Fronthaul Network Structure**")

st.markdown("---")

# Load data
@st.cache_data
def load_topology_data():
    """Load cell-to-link mapping data"""
    csv_path = Path("outputs/topology/cell_to_link_mapping.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

# Load the mapping data
mapping_df = load_topology_data()

if mapping_df is not None:
    # Explanation
    st.markdown('<p class="sub-header">📊 Methodology</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Correlation-Based Topology Discovery</h4>
        <p>The topology is identified by analyzing <strong>correlated packet loss patterns</strong> 
        across all 24 cells:</p>
        <ul>
            <li><strong>Key Insight:</strong> Cells sharing the same fronthaul link experience 
            simultaneous packet loss when that link becomes congested</li>
            <li><strong>Analysis Method:</strong> Compute pairwise correlation of packet loss time series</li>
            <li><strong>Grouping:</strong> High correlation indicates shared link; cells are clustered accordingly</li>
            <li><strong>Validation:</strong> Anchor-based verification ensures consistent groupings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Network Topology Visualization
    st.markdown('<p class="sub-header">🔍 Interactive Network Topology</p>', unsafe_allow_html=True)
    
    # Create network graph
    def create_topology_graph(df):
        """Create an interactive topology graph using Plotly"""
        
        # Define colors for each link
        link_colors = {
            'Link_1': '#FF6B6B',  # Red
            'Link_2': '#4ECDC4',  # Teal
            'Link_3': '#95E1D3'   # Light green
        }
        
        # Prepare node positions
        # DU at top center
        du_x, du_y = 0, 2
        
        # Links in middle layer
        link_positions = {
            'Link_1': (-1.5, 1),
            'Link_2': (0, 1),
            'Link_3': (1.5, 1)
        }
        
        # Group cells by link
        cells_by_link = df.groupby('Inferred_Link_ID')['Cell_ID'].apply(list).to_dict()
        
        # Calculate cell positions
        cell_positions = {}
        for link_id, cells in cells_by_link.items():
            link_x, link_y = link_positions[link_id]
            num_cells = len(cells)
            spacing = 0.4
            start_x = link_x - (num_cells - 1) * spacing / 2
            
            for i, cell_id in enumerate(sorted(cells)):
                cell_x = start_x + i * spacing
                cell_y = 0
                cell_positions[cell_id] = (cell_x, cell_y)
        
        # Create edges
        edge_traces = []
        
        # DU to Links
        for link_id, (link_x, link_y) in link_positions.items():
            edge_trace = go.Scatter(
                x=[du_x, link_x],
                y=[du_y, link_y],
                mode='lines',
                line=dict(color='#888888', width=3),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Links to Cells
        for link_id, cells in cells_by_link.items():
            link_x, link_y = link_positions[link_id]
            color = link_colors[link_id]
            
            for cell_id in cells:
                cell_x, cell_y = cell_positions[cell_id]
                edge_trace = go.Scatter(
                    x=[link_x, cell_x],
                    y=[link_y, cell_y],
                    mode='lines',
                    line=dict(color=color, width=2),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # Create node traces
        # DU node
        du_trace = go.Scatter(
            x=[du_x],
            y=[du_y],
            mode='markers+text',
            marker=dict(size=40, color='#124191', line=dict(color='white', width=2)),
            text=['DU'],
            textposition='middle center',
            textfont=dict(color='white', size=14, family='Arial Black'),
            hoverinfo='text',
            hovertext='Distributed Unit (DU)',
            name='DU',
            showlegend=False
        )
        
        # Link nodes
        link_traces = []
        for link_id, (link_x, link_y) in link_positions.items():
            num_cells = len(cells_by_link[link_id])
            link_trace = go.Scatter(
                x=[link_x],
                y=[link_y],
                mode='markers+text',
                marker=dict(size=35, color=link_colors[link_id], 
                          line=dict(color='white', width=2)),
                text=[link_id.replace('_', ' ')],
                textposition='middle center',
                textfont=dict(color='white', size=11, family='Arial Black'),
                hoverinfo='text',
                hovertext=f'{link_id}: {num_cells} cells',
                name=link_id,
                showlegend=True
            )
            link_traces.append(link_trace)
        
        # Cell nodes
        cell_traces = []
        for link_id, cells in cells_by_link.items():
            color = link_colors[link_id]
            x_coords = []
            y_coords = []
            hover_texts = []
            cell_labels = []
            
            for cell_id in cells:
                cell_x, cell_y = cell_positions[cell_id]
                x_coords.append(cell_x)
                y_coords.append(cell_y)
                hover_texts.append(f'Cell {cell_id}<br>Connected to: {link_id}')
                cell_labels.append(str(cell_id))
            
            cell_trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                marker=dict(size=25, color=color, 
                          line=dict(color='white', width=1.5)),
                text=cell_labels,
                textposition='middle center',
                textfont=dict(color='white', size=9, family='Arial'),
                hoverinfo='text',
                hovertext=hover_texts,
                showlegend=False
            )
            cell_traces.append(cell_trace)
        
        # Combine all traces
        fig = go.Figure(data=edge_traces + [du_trace] + link_traces + cell_traces)
        
        # Update layout with dark mode support
        is_dark = st.session_state.dark_mode
        fig.update_layout(
            title=dict(
                text='<b>5G Fronthaul Network Topology</b>',
                x=0.5,
                xanchor='center',
                font=dict(size=22, color='#4ECDC4' if is_dark else '#124191', family='Arial Black')
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=12, color='#e0e0e0' if is_dark else '#000000'),
                bgcolor='rgba(45, 45, 45, 0.8)' if is_dark else 'rgba(255, 255, 255, 0.8)',
            ),
            hovermode='closest',
            plot_bgcolor='#1a1a1a' if is_dark else 'white',
            paper_bgcolor='#1a1a1a' if is_dark else 'white',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-3, 3]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, 2.5]
            ),
            height=700,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return fig
    
    # Display the interactive graph
    fig = create_topology_graph(mapping_df)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **💡 Interaction Tips:**
    - Hover over nodes to see details
    - Cells are color-coded by their assigned fronthaul link
    - DU connects to all 3 links, which then distribute to their respective cells
    """)
    
    st.markdown("---")
    
    # Cell-to-Link Mapping Table
    st.markdown('<p class="sub-header">📋 Cell-to-Link Mapping</p>', unsafe_allow_html=True)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    link_counts = mapping_df['Inferred_Link_ID'].value_counts().sort_index()
    
    with col1:
        st.metric(
            label="Link_1 Cells",
            value=f"{link_counts.get('Link_1', 0)} cells",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Link_2 Cells",
            value=f"{link_counts.get('Link_2', 0)} cells",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Link_3 Cells",
            value=f"{link_counts.get('Link_3', 0)} cells",
            delta=None
        )
    
    st.markdown("### Complete Mapping Table")
    
    # Create colored table
    def highlight_link(row):
        """Color code rows by link"""
        colors = {
            'Link_1': 'background-color: #FFE5E5',
            'Link_2': 'background-color: #E5F9F7',
            'Link_3': 'background-color: #E8F8F5'
        }
        return [colors.get(row['Inferred_Link_ID'], '')] * len(row)
    
    # Display table with formatting
    styled_df = mapping_df.style.apply(highlight_link, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download option
    csv = mapping_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Mapping as CSV",
        data=csv,
        file_name="cell_to_link_mapping.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Validation Summary
    st.markdown('<p class="sub-header">✅ Validation Summary</p>', unsafe_allow_html=True)
    
    st.success("""
    **Topology Identification Complete:**
    - All 24 cells successfully mapped to 3 fronthaul links
    - Correlation-based clustering validated
    - Topology structure: DU → 3 Links → 24 Cells
    - Distribution: Link_1 (10 cells), Link_2 (1 cell), Link_3 (13 cells)
    """)
    
else:
    st.error("""
    ❌ **Data Not Found**
    
    Unable to load topology mapping data. Please ensure:
    - `outputs/topology/cell_to_link_mapping.csv` exists
    - The file contains 'Cell_ID' and 'Inferred_Link_ID' columns
    """)

st.markdown("---")
st.caption("MIT-Bangalore Team | Nokia 5G Fronthaul Network Optimization Challenge")
