"""
Sliding Window Feature Extraction
Replays historical fronthaul data and generates ML-ready features for congestion analysis.

Window Configuration:
- WINDOW_SIZE = 50 slots (1 slot = 1 ms)
- STEP_SIZE = 1 slot (sliding window)

Features Computed:
- Throughput statistics (mean, max, std, trend)
- Packet loss metrics (count, rate, time since last, burst length)
- Utilization metrics (average, peak)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# Configuration
WINDOW_SIZE = 50  # slots
STEP_SIZE = 1     # slots
OUTPUT_DIR = Path("data")
DEFAULT_LINK_CAPACITY_MBPS = 10000.0  # 10 Gbps - typical fronthaul capacity


def load_slot_level_data() -> Dict[str, pd.DataFrame]:
    """
    Load slot-level throughput and packet loss data for all cells.
    
    Returns:
        Dict mapping cell_id to DataFrame with columns: [slot, throughput_mbps, packet_loss]
    """
    print("📂 Loading slot-level data...")
    
    data_dir = Path("phase1_slot_level_csvs")
    cell_data = {}
    
    for cell_id in range(1, 25):  # 24 cells
        throughput_file = data_dir / f"cell_{cell_id}_throughput_per_slot.csv"
        loss_file = data_dir / f"cell_{cell_id}_packet_loss_per_slot.csv"
        
        if throughput_file.exists() and loss_file.exists():
            try:
                # Load throughput (cell, slot, throughput)
                throughput_df = pd.read_csv(throughput_file)
                throughput_df = throughput_df[['slot', 'throughput']].rename(columns={'throughput': 'throughput_mbps'})
                
                # Load packet loss (cell, slot, slotStart, txPackets, rxPackets, ..., packet_loss, loss_flag)
                # Use slotStart as the slot index to match with throughput
                # Aggregate packet loss per slot (max or mean)
                loss_df = pd.read_csv(loss_file)
                loss_df = loss_df[['slotStart', 'packet_loss']].rename(columns={'slotStart': 'slot'})
                # Take max packet loss per slot (any loss = loss)
                loss_df = loss_df.groupby('slot')['packet_loss'].max().reset_index()
                
                # Merge
                cell_df = pd.merge(throughput_df, loss_df, on='slot', how='left')
                cell_df['packet_loss'] = cell_df['packet_loss'].fillna(0)  # Fill missing with 0
                cell_data[cell_id] = cell_df
                
            except Exception as e:
                print(f"  ⚠️  Error loading cell {cell_id}: {e}")
        else:
            print(f"  ⚠️  Files not found for cell {cell_id}")
            
    print(f"✅ Loaded data for {len(cell_data)} cells")
    return cell_data


def load_topology_mapping() -> Dict[int, str]:
    """
    Load cell-to-link mapping.
    
    Returns:
        Dict mapping cell_id to link_id (e.g., {1: 'Link_2', 2: 'Link_3', ...})
    """
    print("📂 Loading topology mapping...")
    
    topology_file = Path("outputs/topology/cell_to_link_mapping.csv")
    topology_df = pd.read_csv(topology_file)
    
    mapping = dict(zip(topology_df['Cell_ID'], topology_df['Inferred_Link_ID']))
    
    print(f"✅ Loaded mapping for {len(mapping)} cells")
    return mapping


def aggregate_link_data(cell_data: Dict[int, pd.DataFrame], 
                        topology: Dict[int, str]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate cell-level data to link-level data.
    
    Args:
        cell_data: Dict of cell_id -> DataFrame
        topology: Dict of cell_id -> link_id
        
    Returns:
        Dict mapping link_id to DataFrame with columns: [slot, throughput_mbps, packet_loss_flag]
    """
    print("🔄 Aggregating data per link...")
    
    # Group cells by link
    links = {}
    for cell_id, link_id in topology.items():
        if link_id not in links:
            links[link_id] = []
        links[link_id].append(cell_id)
    
    link_data = {}
    
    for link_id, cell_ids in links.items():
        # Collect all cell dataframes for this link
        cell_dfs = []
        for cell_id in cell_ids:
            if cell_id in cell_data:
                df = cell_data[cell_id].copy()
                df['cell_id'] = cell_id  # Track source cell
                cell_dfs.append(df)
        
        if not cell_dfs:
            continue
        
        # Concatenate all cells
        combined = pd.concat(cell_dfs, ignore_index=True)
        
        # Aggregate per slot: sum throughput, max packet_loss
        aggregated = combined.groupby('slot').agg({
            'throughput_mbps': 'sum',
            'packet_loss': 'max'
        }).reset_index()
        
        aggregated = aggregated.rename(columns={'packet_loss': 'packet_loss_flag'})
        aggregated['packet_loss_flag'] = aggregated['packet_loss_flag'].astype(int)
        
        link_data[link_id] = aggregated.sort_values('slot').reset_index(drop=True)
    
    print(f"✅ Aggregated data for {len(link_data)} links")
    return link_data


def compute_throughput_trend(values: np.ndarray) -> float:
    """
    Compute linear trend (slope) of throughput values.
    
    Args:
        values: Array of throughput values
        
    Returns:
        Slope coefficient (Mbps/slot)
    """
    if len(values) < 2:
        return 0.0
    
    x = np.arange(len(values))
    
    # Linear regression: y = mx + b
    # Solve using least squares
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, values, rcond=None)[0]
    
    return float(m)


def compute_max_burst_length(loss_flags: np.ndarray) -> int:
    """
    Compute maximum consecutive packet loss burst length.
    
    Args:
        loss_flags: Binary array (1=loss, 0=no loss)
        
    Returns:
        Maximum consecutive 1's
    """
    if len(loss_flags) == 0 or np.sum(loss_flags) == 0:
        return 0
    
    max_burst = 0
    current_burst = 0
    
    for flag in loss_flags:
        if flag == 1:
            current_burst += 1
            max_burst = max(max_burst, current_burst)
        else:
            current_burst = 0
    
    return max_burst


def compute_time_since_last_loss(loss_flags: np.ndarray) -> int:
    """
    Compute slots since last packet loss (from end of window).
    
    Args:
        loss_flags: Binary array (1=loss, 0=no loss)
        
    Returns:
        Number of slots since last loss (window size if no loss in window)
    """
    if len(loss_flags) == 0:
        return WINDOW_SIZE
    
    # Find last occurrence of 1 (loss)
    loss_indices = np.where(loss_flags == 1)[0]
    
    if len(loss_indices) == 0:
        return WINDOW_SIZE
    
    last_loss_idx = loss_indices[-1]
    slots_since = len(loss_flags) - 1 - last_loss_idx
    
    return slots_since


def extract_window_features(window_df: pd.DataFrame, 
                            link_id: str,
                            window_start_slot: int,
                            capacity_mbps: float = 1000.0) -> Dict:
    """
    Extract features from a single sliding window.
    
    Args:
        window_df: DataFrame with columns [slot, throughput_mbps, packet_loss_flag]
        link_id: Link identifier
        window_start_slot: Starting slot of this window
        capacity_mbps: Link capacity for utilization calculation
        
    Returns:
        Dictionary of computed features
    """
    throughput = window_df['throughput_mbps'].values
    loss_flags = window_df['packet_loss_flag'].values
    
    # Throughput statistics
    mean_throughput = float(np.mean(throughput))
    max_throughput = float(np.max(throughput))
    std_throughput = float(np.std(throughput))
    throughput_trend = compute_throughput_trend(throughput)
    
    # Packet loss metrics
    loss_count = int(np.sum(loss_flags))
    loss_rate = float(loss_count / len(loss_flags))
    time_since_last_loss = compute_time_since_last_loss(loss_flags)
    max_burst_length = compute_max_burst_length(loss_flags)
    
    # Utilization metrics
    avg_utilization = mean_throughput / capacity_mbps
    peak_utilization = max_throughput / capacity_mbps
    
    return {
        'link_id': link_id,
        'window_start_slot': window_start_slot,
        'window_end_slot': window_start_slot + WINDOW_SIZE - 1,
        'mean_throughput': mean_throughput,
        'max_throughput': max_throughput,
        'std_throughput': std_throughput,
        'throughput_trend': throughput_trend,
        'loss_count': loss_count,
        'loss_rate': loss_rate,
        'time_since_last_loss': time_since_last_loss,
        'max_burst_length': max_burst_length,
        'avg_utilization': avg_utilization,
        'peak_utilization': peak_utilization
    }


def load_link_capacities() -> Dict[str, float]:
    """
    Load link capacity estimates from the capacity summary.
    
    Returns:
        Dict mapping link_id to recommended capacity in Mbps
    """
    capacity_file = Path("outputs/capacity/capacity_summary.csv")
    
    if not capacity_file.exists():
        print("  ⚠️  Capacity file not found, using default 1000 Mbps")
        return {}
    
    df = pd.read_csv(capacity_file)
    capacities = {}
    
    for _, row in df.iterrows():
        link_id = row['Link_ID']
        # Use recommended capacity with buffer
        capacity = row['Recommended_Capacity_Mbps']
        capacities[link_id] = capacity
    
    return capacities


def generate_sliding_window_features(link_data: Dict[str, pd.DataFrame],
                                     link_capacities: Dict[str, float] = None) -> pd.DataFrame:
    """
    Generate sliding window features for all links.
    
    Args:
        link_data: Dict of link_id -> DataFrame
        link_capacities: Dict of link_id -> capacity in Mbps (optional)
        
    Returns:
        DataFrame with all computed features
    """
    print(f"🔄 Generating sliding window features (window={WINDOW_SIZE}, step={STEP_SIZE})...")
    
    if link_capacities is None:
        link_capacities = {}
    
    all_features = []
    
    for link_id, df in link_data.items():
        print(f"\n  Processing {link_id}...")
        
        # Get capacity for this link (use default fronthaul capacity)
        capacity_mbps = link_capacities.get(link_id, DEFAULT_LINK_CAPACITY_MBPS)
        
        # For utilization, use a realistic fronthaul link capacity, not the estimated requirement
        # Typical fronthaul links are 1-10 Gbps
        utilization_capacity = DEFAULT_LINK_CAPACITY_MBPS
        
        # Sort by slot to ensure chronological order
        df = df.sort_values('slot').reset_index(drop=True)
        
        num_slots = len(df)
        num_windows = (num_slots - WINDOW_SIZE) // STEP_SIZE + 1
        
        print(f"    Total slots: {num_slots}")
        print(f"    Windows to generate: {num_windows}")
        
        for i in range(0, num_slots - WINDOW_SIZE + 1, STEP_SIZE):
            window_df = df.iloc[i:i + WINDOW_SIZE]
            window_start_slot = df.iloc[i]['slot']
            
            features = extract_window_features(
                window_df, 
                link_id, 
                window_start_slot,
                utilization_capacity  # Use realistic capacity for utilization
            )
            
            all_features.append(features)
        
        print(f"    ✅ Generated {len([f for f in all_features if f['link_id'] == link_id])} windows")
    
    features_df = pd.DataFrame(all_features)
    
    print(f"\n✅ Total features generated: {len(features_df)} rows")
    return features_df


def save_features(features_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR):
    """
    Save features to CSV file.
    
    Args:
        features_df: DataFrame with computed features
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "sliding_window_features.csv"
    features_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Features saved to: {output_file}")
    print(f"   Shape: {features_df.shape}")
    print(f"   Columns: {list(features_df.columns)}")


def print_feature_summary(features_df: pd.DataFrame):
    """
    Print statistical summary of generated features.
    
    Args:
        features_df: DataFrame with computed features
    """
    print("\n" + "="*70)
    print("📊 FEATURE SUMMARY")
    print("="*70)
    
    for link_id in features_df['link_id'].unique():
        link_features = features_df[features_df['link_id'] == link_id]
        
        print(f"\n{link_id}:")
        print(f"  Windows: {len(link_features)}")
        print(f"  Mean Throughput: {link_features['mean_throughput'].mean():.2f} Mbps")
        print(f"  Max Throughput: {link_features['max_throughput'].max():.2f} Mbps")
        print(f"  Avg Loss Rate: {link_features['loss_rate'].mean():.4f}")
        print(f"  Avg Utilization: {link_features['avg_utilization'].mean():.4%}")
        print(f"  Peak Utilization: {link_features['peak_utilization'].max():.4%}")
    
    print("\n" + "="*70)
    print("Feature Statistics:")
    print("="*70)
    print(features_df.describe().round(4))


def main():
    """
    Main execution function.
    """
    print("="*70)
    print("🚀 SLIDING WINDOW FEATURE EXTRACTION")
    print("="*70)
    print(f"Configuration:")
    print(f"  Window Size: {WINDOW_SIZE} slots (50 ms)")
    print(f"  Step Size: {STEP_SIZE} slot (1 ms)")
    print(f"  Output: {OUTPUT_DIR}/sliding_window_features.csv")
    print("="*70)
    
    # Step 1: Load cell-level data
    cell_data = load_slot_level_data()
    
    # Step 2: Load topology mapping
    topology = load_topology_mapping()
    
    # Step 3: Load link capacities
    link_capacities = load_link_capacities()
    if link_capacities:
        print(f"✅ Loaded capacities: {link_capacities}")
    
    # Step 4: Aggregate to link-level
    link_data = aggregate_link_data(cell_data, topology)
    
    # Step 5: Generate sliding window features
    features_df = generate_sliding_window_features(link_data, link_capacities)
    
    # Step 6: Save features
    save_features(features_df)
    
    # Step 7: Print summary
    print_feature_summary(features_df)
    
    print("\n" + "="*70)
    print("✅ FEATURE EXTRACTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
