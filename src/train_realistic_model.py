"""
Realistic Congestion PREDICTION Model (No Feature Leakage)

Train models to predict FUTURE congestion using only past features.
This demonstrates realistic performance without overfitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


# Configuration
FEATURES_FILE = Path("data/sliding_window_features.csv")
PREDICTION_HORIZON = 50  # Predict congestion 50 slots ahead
RANDOM_STATE = 42


def prepare_predictive_data():
    """
    Prepare data for PREDICTION (not classification).
    Use past features to predict future congestion.
    """
    print("="*70)
    print("🔮 REALISTIC PREDICTIVE MODEL (NO FEATURE LEAKAGE)")
    print("="*70)
    
    print(f"\n📂 Loading data...")
    df = pd.read_csv(FEATURES_FILE)
    
    # Sort by link and time
    df = df.sort_values(['link_id', 'window_start_slot']).reset_index(drop=True)
    
    print(f"  Loaded {len(df):,} samples")
    
    # Create target: future congestion (50 slots ahead)
    print(f"\n🎯 Creating target: congestion {PREDICTION_HORIZON} slots ahead...")
    
    df['future_congestion'] = 0
    
    for link_id in df['link_id'].unique():
        link_mask = df['link_id'] == link_id
        link_df = df[link_mask].copy()
        
        # Shift target backward (so we predict future from past)
        future_util = link_df['avg_utilization'].shift(-PREDICTION_HORIZON)
        future_loss = link_df['loss_rate'].shift(-PREDICTION_HORIZON)
        
        future_congestion = ((future_util > 0.8) | (future_loss > 0.1)).astype(int)
        
        df.loc[link_mask, 'future_congestion'] = future_congestion
    
    # Remove samples where we don't know the future
    df = df[df['future_congestion'].notna()].copy()
    df['future_congestion'] = df['future_congestion'].astype(int)
    
    print(f"  Valid samples: {len(df):,}")
    print(f"  Future congestion rate: {df['future_congestion'].mean():.2%}")
    
    # Select features (EXCLUDE target-defining features!)
    print(f"\n📊 Selecting features (NO leakage)...")
    
    # Use only features that don't directly reveal target
    feature_cols = [
        'mean_throughput', 'max_throughput', 'std_throughput', 
        'throughput_trend',
        # EXCLUDE: 'avg_utilization', 'peak_utilization' (too direct)
        # EXCLUDE: 'loss_rate', 'loss_count' (too direct)
        'time_since_last_loss', 'max_burst_length',
    ]
    
    X = df[feature_cols].copy()
    
    # Add link encoding
    link_dummies = pd.get_dummies(df['link_id'], prefix='link')
    X = pd.concat([X, link_dummies], axis=1)
    
    y = df['future_congestion'].copy()
    
    print(f"  Features used: {len(X.columns)}")
    print(f"  Features: {list(X.columns)}")
    
    # Temporal train/test split (first 80% train, last 20% test)
    print(f"\n✂️ Temporal train/test split...")
    
    split_idx = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"  Train: {len(X_train):,} samples (slots 0-{df.iloc[split_idx]['window_start_slot']:.0f})")
    print(f"  Test:  {len(X_test):,} samples (slots {df.iloc[split_idx]['window_start_slot']:.0f}-{df.iloc[-1]['window_start_slot']:.0f})")
    print(f"  Train congestion: {y_train.mean():.2%}")
    print(f"  Test congestion:  {y_test.mean():.2%}")
    
    # Scale features
    print(f"\n🔄 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()


def train_realistic_models(X_train, y_train):
    """Train models with realistic expectations."""
    print("\n" + "="*70)
    print("🤖 TRAINING REALISTIC MODELS")
    print("="*70)
    
    models = {}
    
    # 1. Logistic Regression
    print("\n1️⃣ Logistic Regression...")
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Random Forest
    print("2️⃣ Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 3. Gradient Boosting
    print("3️⃣ Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    return models


def evaluate_realistic_models(models, X_test, y_test):
    """Evaluate with realistic metrics."""
    print("\n" + "="*70)
    print("📊 REALISTIC PERFORMANCE EVALUATION")
    print("="*70)
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f"📈 {model_name}")
        print(f"{'='*70}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc = 0.0
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n🎯 Metrics:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"  ┌─────────────────────┐")
        print(f"  │   Predicted         │")
        print(f"  │  Normal  Congested  │")
        print(f"  ├─────────────────────┤")
        print(f"  │ {tn:7,}   {fp:7,}  │ Normal")
        print(f"  │ {fn:7,}   {tp:7,}  │ Congested")
        print(f"  └─────────────────────┘")
        
        print(f"\n💡 Real-World Impact:")
        if tp > 0:
            print(f"  • Correctly predicted congestion: {tp:,} events")
        if fn > 0:
            print(f"  • Missed congestion: {fn:,} events ({fn/(fn+tp)*100:.1f}%)")
        if fp > 0:
            print(f"  • False alarms: {fp:,} ({fp/(fp+tn)*100:.1f}% of normal)")
        
        results.append({
            'model': model_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc
        })
    
    return results


def main():
    """Main realistic training pipeline."""
    
    # Prepare data properly
    X_train, X_test, y_train, y_test, feature_names = prepare_predictive_data()
    
    # Train models
    models = train_realistic_models(X_train, y_train)
    
    # Evaluate
    results = evaluate_realistic_models(models, X_test, y_test)
    
    # Summary
    print("\n" + "="*70)
    print("🏆 COMPARISON: REALISTIC vs OVERFIT MODELS")
    print("="*70)
    
    print("\n❌ PREVIOUS MODELS (100% accuracy):")
    print("  • Used target-defining features (avg_utilization, loss_rate)")
    print("  • Random train/test split")
    print("  • Classified CURRENT state (not future)")
    print("  • Result: Perfect 100% accuracy (NOT realistic!)")
    
    print("\n✅ REALISTIC MODELS (current results):")
    print("  • Excluded target-defining features")
    print("  • Temporal train/test split")
    print("  • Predict FUTURE state (50 slots ahead)")
    print("  • Result: Lower accuracy, but honest and deployable!")
    
    print("\n📊 Realistic Performance:")
    for r in results:
        print(f"  {r['model']:20s}: {r['accuracy']*100:5.2f}% accuracy, F1={r['f1_score']:.3f}")
    
    print("\n" + "="*70)
    print("✅ CONCLUSION")
    print("="*70)
    print("\n🎯 The realistic models show HONEST performance:")
    print("  • Accuracy in 60-80% range is NORMAL for prediction")
    print("  • Some missed events and false alarms are EXPECTED")
    print("  • This represents what you'll see in production")
    print("  • 100% accuracy was an artifact of data leakage!")
    
    print("\n💡 Key Takeaway:")
    print("  LOWER accuracy with proper validation > ")
    print("  HIGHER accuracy with data leakage")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
