import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CONFIG
# ============================================
BASE_PATH = Path("D:/OST/SignNet/SignNet+")
SPLITS = ['landmarks_train', 'landmarks_dev', 'landmarks_test']

# Landmark Groups (Standard MediaPipe Holistic = 543 + 10 custom)
LANDMARK_GROUPS = {
    'left_hand': list(range(0, 21)),  # Landmarks 0-20
    'right_hand': list(range(21, 42)),  # Landmarks 21-41
    'pose': list(range(42, 75)),  # Landmarks 42-74 (33 pose)
    'face': list(range(75, 543)),  # Landmarks 75-542 (468 face)
    'custom': list(range(543, 553))  # Landmarks 543-552 (10 custom)
}

# Quality Thresholds
Z_NEAR_ZERO_THRESHOLD = 0.05  # |z| < 0.05 → "low quality"
XY_VALID_RANGE = (0.0, 1.0)  # Valid x,y range
TEMPORAL_JUMP_THRESHOLD = 0.3  # Max allowed jump between frames


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_xyz(landmarks, landmark_indices):
    """
    Extract x,y,z for specific landmarks.

    Args:
        landmarks: [T, Features] array
        landmark_indices: List of landmark indices (e.g., [0, 1, 2, ...])

    Returns:
        x, y, z: Each [T, N] arrays
    """
    x_indices = [idx * 3 for idx in landmark_indices]
    y_indices = [idx * 3 + 1 for idx in landmark_indices]
    z_indices = [idx * 3 + 2 for idx in landmark_indices]

    x = landmarks[:, x_indices]
    y = landmarks[:, y_indices]
    z = landmarks[:, z_indices]

    return x, y, z


def compute_quality_metrics(landmarks, group_indices):
    """
    Compute quality metrics for a landmark group.

    Returns:
        dict with quality metrics
    """
    x, y, z = extract_xyz(landmarks, group_indices)

    num_frames, num_landmarks = x.shape

    # 1. Near-zero Z-values (proxy for low quality)
    near_zero_z = np.abs(z) < Z_NEAR_ZERO_THRESHOLD
    pct_near_zero = near_zero_z.sum() / near_zero_z.size * 100

    # 2. Out-of-range X,Y (invalid landmarks)
    x_invalid = (x < XY_VALID_RANGE[0]) | (x > XY_VALID_RANGE[1])
    y_invalid = (y < XY_VALID_RANGE[0]) | (y > XY_VALID_RANGE[1])
    pct_invalid_xy = ((x_invalid | y_invalid).sum() / x_invalid.size) * 100

    # 3. Temporal jumps (tracking failures)
    if num_frames > 1:
        x_diff = np.abs(np.diff(x, axis=0))
        y_diff = np.abs(np.diff(y, axis=0))

        temporal_jumps = (x_diff > TEMPORAL_JUMP_THRESHOLD) | (y_diff > TEMPORAL_JUMP_THRESHOLD)
        pct_jumps = temporal_jumps.sum() / temporal_jumps.size * 100
    else:
        pct_jumps = 0.0

    # 4. Constant landmarks (never move)
    x_variance = x.var(axis=0)
    y_variance = y.var(axis=0)
    constant_landmarks = (x_variance < 1e-6) & (y_variance < 1e-6)
    pct_constant = constant_landmarks.sum() / num_landmarks * 100

    return {
        'mean_z': z.mean(),
        'std_z': z.std(),
        'pct_near_zero_z': pct_near_zero,
        'pct_invalid_xy': pct_invalid_xy,
        'pct_temporal_jumps': pct_jumps,
        'pct_constant_landmarks': pct_constant,
        'mean_x': x.mean(),
        'mean_y': y.mean(),
    }


def analyze_sample(npz_path):
    """
    Analyze a single sample.

    Returns:
        dict with statistics
    """
    try:
        sample = np.load(npz_path, allow_pickle=True)
        landmarks = sample['landmarks']  # Shape: [T, Features]
        glosses = sample['glosses']

        num_frames = landmarks.shape[0]

        # Analyze each landmark group
        group_metrics = {}
        for group_name, indices in LANDMARK_GROUPS.items():
            if len(indices) > 0:  # Skip empty groups
                metrics = compute_quality_metrics(landmarks, indices)
                group_metrics[group_name] = metrics

        # Overall quality score (weighted average)
        overall_issues = 0
        for group_name, metrics in group_metrics.items():
            # Weight by number of landmarks in group
            weight = len(LANDMARK_GROUPS[group_name]) / 553
            group_score = (
                    metrics['pct_near_zero_z'] * 0.3 +
                    metrics['pct_invalid_xy'] * 0.4 +
                    metrics['pct_temporal_jumps'] * 0.3
            )
            overall_issues += group_score * weight

        # Quality decision
        quality_label = 'good' if overall_issues < 20 else ('medium' if overall_issues < 40 else 'poor')

        return {
            'sample_id': npz_path.stem,
            'glosses': ' '.join(glosses) if isinstance(glosses, np.ndarray) else str(glosses),
            'num_frames': num_frames,
            'num_glosses': len(glosses) if isinstance(glosses, np.ndarray) else 1,
            **{f'{group}_near_zero_z': metrics['pct_near_zero_z']
               for group, metrics in group_metrics.items()},
            **{f'{group}_invalid_xy': metrics['pct_invalid_xy']
               for group, metrics in group_metrics.items()},
            **{f'{group}_temporal_jumps': metrics['pct_temporal_jumps']
               for group, metrics in group_metrics.items()},
            'overall_quality_score': overall_issues,
            'quality_label': quality_label,
        }
    except Exception as e:
        print(f"Error processing {npz_path.name}: {e}")
        return None


# ============================================
# MAIN ANALYSIS
# ============================================

def run_data_quality_analysis():
    """
    Run data quality analysis on all samples.
    """
    all_results = []

    for split in SPLITS:
        print(f"\n{'=' * 60}")
        print(f"Analyzing {split}...")
        print(f"{'=' * 60}")

        split_path = BASE_PATH / split
        npz_files = list(split_path.glob("*.npz"))

        print(f"Found {len(npz_files)} samples")

        for npz_path in tqdm(npz_files, desc=f"Processing {split}"):
            result = analyze_sample(npz_path)
            if result is not None:
                result['split'] = split
                all_results.append(result)

    df = pd.DataFrame(all_results)
    return df


# ============================================
# REPORTING
# ============================================

def create_report(df, output_dir):
    """
    Create visualizations and reports.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"  - Train: {len(df[df['split'] == 'landmarks_train'])}")
    print(f"  - Dev: {len(df[df['split'] == 'landmarks_dev'])}")
    print(f"  - Test: {len(df[df['split'] == 'landmarks_test'])}")

    print("\n--- Quality Distribution ---")
    quality_counts = df['quality_label'].value_counts()
    for label in ['good', 'medium', 'poor']:
        count = quality_counts.get(label, 0)
        pct = count / len(df) * 100
        print(f"{label.capitalize():8s}: {count:5d} ({pct:5.1f}%)")

    print("\n--- Quality Score Statistics ---")
    print(f"Mean quality score: {df['overall_quality_score'].mean():.2f}")
    print(f"Median: {df['overall_quality_score'].median():.2f}")
    print(f"Std: {df['overall_quality_score'].std():.2f}")

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Quality score distribution
    axes[0, 0].hist(df['overall_quality_score'], bins=50, edgecolor='black')
    axes[0, 0].axvline(20, color='green', linestyle='--', label='Good threshold')
    axes[0, 0].axvline(40, color='orange', linestyle='--', label='Medium threshold')
    axes[0, 0].set_xlabel('Overall Quality Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Quality Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Quality labels
    quality_counts.plot(kind='bar', ax=axes[0, 1], color=['green', 'orange', 'red'])
    axes[0, 1].set_xlabel('Quality Label')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Sample Quality Distribution')
    axes[0, 1].grid(alpha=0.3, axis='y')

    # Plot 3: Near-zero Z percentage per group
    group_cols = [col for col in df.columns if 'near_zero_z' in col]
    df[group_cols].mean().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Landmark Group')
    axes[1, 0].set_ylabel('% Near-Zero Z')
    axes[1, 0].set_title('Low Quality Z-Values per Group')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(alpha=0.3, axis='y')

    # Plot 4: Invalid XY percentage per group
    invalid_cols = [col for col in df.columns if 'invalid_xy' in col]
    df[invalid_cols].mean().plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Landmark Group')
    axes[1, 1].set_ylabel('% Invalid X,Y')
    axes[1, 1].set_title('Invalid Coordinates per Group')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'data_quality_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: data_quality_analysis.png")

    # Save CSV
    csv_path = output_dir / 'data_quality_detailed.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: data_quality_detailed.csv")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    good_count = quality_counts.get('good', 0)
    good_pct = good_count / len(df) * 100

    if good_pct >= 70:
        print(f"\n✅ FILTER OUT 'poor' quality samples")
        print(f"   Keep 'good' + 'medium': {len(df[df['quality_label'] != 'poor'])} samples")
    elif good_pct >= 50:
        print(f"\n⚠️  KEEP ONLY 'good' quality samples")
        print(f"   Good samples: {good_count} ({good_pct:.1f}%)")
    else:
        print(f"\n❌ WARNING: Low overall data quality!")
        print(f"   Only {good_pct:.1f}% good samples")
        print(f"   Consider: Re-extract features or use all data with caution")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    print("Starting Data Quality Analysis (without confidence scores)...")

    df_results = run_data_quality_analysis()
    create_report(df_results, output_dir=BASE_PATH / "data_quality_analysis_results")

    print("\n✅ Analysis complete!")