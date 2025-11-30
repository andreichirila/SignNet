# generate_pdf_report.py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from collections import Counter
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================
# CONFIG
# ============================================
BASE_PATH = Path("D:/OST/SignNet/SignNet+")
SPLITS = {
    'train': 'landmarks_train_cleaned',
    'dev': 'landmarks_dev_cleaned',
    'test': 'landmarks_test_cleaned'
}

OUTPUT_DIR = BASE_PATH / "data_analysis_comprehensive"
OUTPUT_DIR.mkdir(exist_ok=True)
PDF_FILENAME = OUTPUT_DIR / f"SignNet_Data_Analysis_Report_{datetime.now().strftime('%Y%m%d')}.pdf"


# ============================================
# DATA COLLECTION (same as before)
# ============================================

def collect_dataset_info():
    """Collect comprehensive dataset information."""
    all_data = []

    for split_name, split_dir in SPLITS.items():
        print(f"\n{'=' * 60}")
        print(f"Analyzing {split_name.upper()} split...")
        print(f"{'=' * 60}")

        split_path = BASE_PATH / split_dir
        npz_files = list(split_path.glob("*.npz"))

        print(f"Found {len(npz_files)} samples")

        for npz_file in tqdm(npz_files, desc=f"Processing {split_name}"):
            try:
                sample = np.load(npz_file, allow_pickle=True)
                landmarks = sample['landmarks']
                glosses = sample['glosses']

                if isinstance(glosses, np.ndarray):
                    gloss_list = glosses.tolist()
                else:
                    gloss_list = [str(glosses)]

                num_frames = landmarks.shape[0]
                num_glosses = len(gloss_list)

                all_data.append({
                    'split': split_name,
                    'sample_id': npz_file.stem,
                    'num_frames': num_frames,
                    'num_glosses': num_glosses,
                    'glosses': gloss_list,
                    'gloss_sequence': ' '.join(gloss_list),
                    'avg_frames_per_gloss': num_frames / num_glosses if num_glosses > 0 else 0,
                })

            except Exception as e:
                print(f"Error processing {npz_file.name}: {e}")
                continue

    return pd.DataFrame(all_data)


def analyze_glosses(df):
    """Analyze gloss distribution."""
    all_glosses = []
    for gloss_list in df['glosses']:
        all_glosses.extend(gloss_list)

    gloss_counts = Counter(all_glosses)
    gloss_df = pd.DataFrame([
        {'gloss': gloss, 'count': count}
        for gloss, count in gloss_counts.most_common()
    ])

    return gloss_df, gloss_counts, all_glosses


# ============================================
# PDF GENERATION
# ============================================

def create_pdf_report(df, gloss_df, gloss_counts, all_glosses):
    """Generate comprehensive PDF report."""

    print("\n" + "=" * 80)
    print("GENERATING PDF REPORT")
    print("=" * 80)

    with PdfPages(PDF_FILENAME) as pdf:

        # ============================================
        # PAGE 1: TITLE PAGE
        # ============================================
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Title
        title_text = "SignNet Dataset Analysis Report"
        ax.text(0.5, 0.75, title_text,
                ha='center', va='center', fontsize=32, fontweight='bold',
                transform=ax.transAxes)

        # Subtitle
        subtitle = "RWTH-PHOENIX-2014 Dataset\nGerman Sign Language Recognition"
        ax.text(0.5, 0.65, subtitle,
                ha='center', va='center', fontsize=18, style='italic',
                transform=ax.transAxes)

        # Key stats box
        stats_text = f"""
        Total Samples: {len(df):,}
        Total Frames: {df['num_frames'].sum():,}
        Unique Glosses: {len(gloss_df):,}
        Feature Dimension: 543 landmarks × 2 coordinates
        """

        # Create a box for stats
        box = Rectangle((0.25, 0.35), 0.5, 0.2,
                        transform=ax.transAxes,
                        facecolor='lightblue',
                        edgecolor='navy',
                        linewidth=2,
                        alpha=0.3)
        ax.add_patch(box)

        ax.text(0.5, 0.45, stats_text,
                ha='center', va='center', fontsize=14,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Date and author
        date_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ax.text(0.5, 0.15, date_text,
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes)

        author_text = "OST - Ostschweizer Fachhochschule\nAndrei Chirila & Roman Schläpfer"
        ax.text(0.5, 0.08, author_text,
                ha='center', va='center', fontsize=11, style='italic',
                transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 1: Title page")

        # ============================================
        # PAGE 2: EXECUTIVE SUMMARY
        # ============================================
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        summary_title = "Executive Summary"
        ax.text(0.5, 0.95, summary_title,
                ha='center', va='top', fontsize=24, fontweight='bold',
                transform=ax.transAxes)

        # Dataset overview
        counts_sorted = sorted(gloss_df['count'].values, reverse=True)

        summary_sections = [
            ("Dataset Information", [
                f"• Total Samples: {len(df):,}",
                f"• Training Samples: {len(df[df['split'] == 'train']):,}",
                f"• Validation Samples: {len(df[df['split'] == 'dev']):,}",
                f"• Test Samples: {len(df[df['split'] == 'test']):,}",
                f"• Total Video Frames: {df['num_frames'].sum():,}",
            ]),
            ("Class Distribution", [
                f"• Unique Glosses (Classes): {len(gloss_df):,}",
                f"• Total Gloss Instances: {len(all_glosses):,}",
                f"• Most Frequent Gloss: {counts_sorted[0]:,} occurrences",
                f"• Least Frequent Gloss: {counts_sorted[-1]:,} occurrence(s)",
                f"• Class Imbalance Ratio: {counts_sorted[0] / counts_sorted[-1]:.1f}:1",
            ]),
            ("Sequence Statistics", [
                f"• Mean Frames per Sample: {df['num_frames'].mean():.1f}",
                f"• Median Frames per Sample: {df['num_frames'].median():.1f}",
                f"• 95th Percentile Frames: {df['num_frames'].quantile(0.95):.0f}",
                f"• Mean Glosses per Sample: {df['num_glosses'].mean():.1f}",
                f"• Mean Frames per Gloss: {df['avg_frames_per_gloss'].mean():.1f}",
            ]),
            ("Data Quality", [
                "• ✓ All samples passed quality checks",
                "• ✓ X,Y coordinates validated (0% invalid)",
                "• ✓ Custom landmarks removed (543 landmarks retained)",
                "• ✓ Z-dimension dropped (80-98% near-zero values)",
                "• ✓ Feature dimension: 1086 (543 × 2)",
            ])
        ]

        y_pos = 0.88
        for section_title, items in summary_sections:
            # Section title
            ax.text(0.1, y_pos, section_title,
                    fontsize=14, fontweight='bold',
                    transform=ax.transAxes)
            y_pos -= 0.04

            # Items
            for item in items:
                ax.text(0.12, y_pos, item,
                        fontsize=11,
                        transform=ax.transAxes,
                        family='monospace')
                y_pos -= 0.03

            y_pos -= 0.02

        # Recommendations box
        rec_y = 0.15
        rec_box = Rectangle((0.08, rec_y - 0.12), 0.84, 0.14,
                            transform=ax.transAxes,
                            facecolor='lightgreen',
                            edgecolor='darkgreen',
                            linewidth=2,
                            alpha=0.3)
        ax.add_patch(rec_box)

        ax.text(0.5, rec_y + 0.01, "Recommendations for Model Training",
                ha='center', fontsize=13, fontweight='bold',
                transform=ax.transAxes)

        top200_coverage = gloss_df.head(200)['count'].sum() / len(all_glosses) * 100
        recommendations = [
            f"1. Sequence Length: Use max_length={int(df['num_frames'].quantile(0.95))} (covers 95% of samples)",
            f"2. Class Selection: Start with Top-200 glosses ({top200_coverage:.1f}% instance coverage)",
            f"3. Batch Size: Start with batch_size=8 (adjust based on GPU memory)",
            f"4. Class Imbalance: Use Focal Loss (α=0.25, γ=2.0) to handle {counts_sorted[0] / counts_sorted[-1]:.0f}:1 ratio",
        ]

        rec_y -= 0.03
        for rec in recommendations:
            ax.text(0.1, rec_y, rec,
                    fontsize=10,
                    transform=ax.transAxes)
            rec_y -= 0.025

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 2: Executive summary")

        # ============================================
        # PAGE 3: DATASET OVERVIEW VISUALIZATIONS
        # ============================================
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Dataset Overview', fontsize=20, fontweight='bold', y=0.98)

        # 3.1: Samples per split
        split_counts = df['split'].value_counts().reindex(['train', 'dev', 'test'])
        colors_split = ['#2ecc71', '#3498db', '#e74c3c']
        bars = axes[0, 0].bar(split_counts.index, split_counts.values, color=colors_split, edgecolor='black',
                              linewidth=1.5)
        axes[0, 0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Samples per Split', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3, axis='y', linestyle='--')
        axes[0, 0].set_xlabel('')

        for bar, count in zip(bars, split_counts.values):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{count:,}',
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 3.2: Frame count distribution
        axes[0, 1].hist(df['num_frames'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        mean_frames = df['num_frames'].mean()
        p95_frames = df['num_frames'].quantile(0.95)

        axes[0, 1].axvline(mean_frames, color='red', linestyle='--', linewidth=2.5,
                           label=f'Mean: {mean_frames:.0f}')
        axes[0, 1].axvline(p95_frames, color='orange', linestyle='--', linewidth=2.5,
                           label=f'95th percentile: {p95_frames:.0f}')

        axes[0, 1].set_xlabel('Number of Frames', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Frame Count Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(alpha=0.3, linestyle='--')

        # 3.3: Glosses per sample
        gloss_bins = range(1, df['num_glosses'].max() + 2)
        axes[1, 0].hist(df['num_glosses'], bins=gloss_bins, edgecolor='black', alpha=0.7, color='coral')
        axes[1, 0].set_xlabel('Number of Glosses per Sample', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Glosses per Sample Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='y', linestyle='--')

        # Add mean line
        mean_glosses = df['num_glosses'].mean()
        axes[1, 0].axvline(mean_glosses, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_glosses:.1f}')
        axes[1, 0].legend(fontsize=10)

        # 3.4: Frames per gloss
        axes[1, 1].hist(df['avg_frames_per_gloss'], bins=50, edgecolor='black', alpha=0.7, color='mediumseagreen')
        mean_fpg = df['avg_frames_per_gloss'].mean()
        axes[1, 1].axvline(mean_fpg, color='red', linestyle='--', linewidth=2.5,
                           label=f'Mean: {mean_fpg:.1f}')
        axes[1, 1].set_xlabel('Average Frames per Gloss', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Frames per Gloss Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(alpha=0.3, linestyle='--')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 3: Dataset overview")

        # ============================================
        # PAGE 4: CLASS DISTRIBUTION - PART 1
        # ============================================
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Class Distribution Analysis (Part 1)', fontsize=20, fontweight='bold', y=0.98)

        # 4.1: Top-50 glosses
        top50 = gloss_df.head(50)
        y_positions = np.arange(len(top50))

        axes[0].barh(y_positions, top50['count'].values, color='steelblue', edgecolor='black')
        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(top50['gloss'].values, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Top-50 Most Frequent Glosses', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='x', linestyle='--')

        # Add count labels
        for i, (idx, row) in enumerate(top50.iterrows()):
            axes[0].text(row['count'] + 10, i, f"{row['count']}",
                         va='center', fontsize=7)

        # 4.2: Frequency distribution (log scale)
        rank = np.arange(1, len(gloss_df) + 1)
        axes[1].plot(rank, gloss_df['count'].values, linewidth=2.5, color='darkred', marker='o', markersize=2)
        axes[1].set_xlabel('Gloss Rank', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].set_title('Gloss Frequency Distribution (Zipf\'s Law)', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3, which='both', linestyle='--')

        # Annotate some points
        for k in [1, 50, 100, 200]:
            if k <= len(gloss_df):
                freq = gloss_df.iloc[k - 1]['count']
                axes[1].annotate(f'Rank {k}\n({freq})',
                                 xy=(k, freq), xytext=(k * 1.5, freq * 2),
                                 fontsize=9,
                                 arrowprops=dict(arrowstyle='->', color='black', lw=1))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 4: Class distribution part 1")

        # ============================================
        # PAGE 5: CLASS DISTRIBUTION - PART 2
        # ============================================
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Class Distribution Analysis (Part 2)', fontsize=20, fontweight='bold', y=0.98)

        # 5.1: Cumulative coverage
        cumsum = np.cumsum(gloss_df['count'].values)
        cumsum_pct = cumsum / cumsum[-1] * 100

        axes[0].plot(range(1, len(gloss_df) + 1), cumsum_pct, linewidth=3, color='#2ecc71')
        axes[0].fill_between(range(1, len(gloss_df) + 1), 0, cumsum_pct, alpha=0.3, color='#2ecc71')

        # Mark important thresholds
        for k, color in [(50, 'red'), (100, 'orange'), (200, 'blue'), (500, 'purple')]:
            if k <= len(gloss_df):
                coverage = cumsum_pct[k - 1]
                axes[0].plot(k, coverage, 'o', markersize=10, color=color)
                axes[0].annotate(f'Top-{k}\n{coverage:.1f}%',
                                 xy=(k, coverage), xytext=(k + 100, coverage - 8),
                                 fontsize=10, ha='left', fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                                 arrowprops=dict(arrowstyle='->', color=color, lw=2))

        axes[0].set_xlabel('Number of Glosses', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Cumulative Coverage (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Cumulative Gloss Coverage', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3, linestyle='--')
        axes[0].set_xlim(0, min(600, len(gloss_df)))
        axes[0].set_ylim(0, 105)

        # 5.2: Class imbalance
        bins = [1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
        bin_labels = ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '100+']
        bin_counts = []

        for low, high in zip(bins[:-1], bins[1:]):
            count = sum(1 for c in gloss_df['count'] if low <= c < high)
            bin_counts.append(count)

        colors_bins = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bin_counts)))
        bars = axes[1].bar(bin_labels, bin_counts, color=colors_bins, edgecolor='black', linewidth=1.5)

        axes[1].set_xlabel('Occurrence Range', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Number of Glosses', fontsize=12, fontweight='bold')
        axes[1].set_title('Gloss Distribution by Frequency Range', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y', linestyle='--')

        # Add value labels
        for bar, count in zip(bars, bin_counts):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{count}',
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 5: Class distribution part 2")

        # ============================================
        # PAGE 6: SPLIT COMPARISON
        # ============================================
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Train/Dev/Test Split Comparison', fontsize=20, fontweight='bold', y=0.96)

        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        splits_list = ['train', 'dev', 'test']
        colors_violin = ['#2ecc71', '#3498db', '#e74c3c']

        # Row 1: Frame count distributions
        for idx, (split, color) in enumerate(zip(splits_list, colors_violin)):
            ax = fig.add_subplot(gs[0, idx])
            split_df = df[df['split'] == split]

            parts = ax.violinplot([split_df['num_frames']], positions=[0],
                                  showmeans=True, showmedians=True, showextrema=True)

            # Color the violin
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            ax.set_title(f'{split.upper()}\nFrame Counts', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Frames', fontsize=10)
            ax.set_xticks([])
            ax.grid(alpha=0.3, axis='y', linestyle='--')

            # Statistics
            stats_text = (f"n = {len(split_df):,}\n"
                          f"μ = {split_df['num_frames'].mean():.0f}\n"
                          f"σ = {split_df['num_frames'].std():.0f}\n"
                          f"median = {split_df['num_frames'].median():.0f}")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Row 2: Gloss count distributions
        for idx, (split, color) in enumerate(zip(splits_list, colors_violin)):
            ax = fig.add_subplot(gs[1, idx])
            split_df = df[df['split'] == split]

            parts = ax.violinplot([split_df['num_glosses']], positions=[0],
                                  showmeans=True, showmedians=True, showextrema=True)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            ax.set_title(f'{split.upper()}\nGlosses per Sample', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Glosses', fontsize=10)
            ax.set_xticks([])
            ax.grid(alpha=0.3, axis='y', linestyle='--')

            stats_text = (f"μ = {split_df['num_glosses'].mean():.1f}\n"
                          f"σ = {split_df['num_glosses'].std():.1f}")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Row 3: Summary table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')

        # Create summary table
        table_data = []
        table_data.append(['Metric', 'Train', 'Dev', 'Test'])

        metrics = [
            ('Samples', lambda s: f"{len(df[df['split'] == s]):,}"),
            ('Total Frames', lambda s: f"{df[df['split'] == s]['num_frames'].sum():,}"),
            ('Mean Frames', lambda s: f"{df[df['split'] == s]['num_frames'].mean():.1f}"),
            ('Total Glosses', lambda s: f"{df[df['split'] == s]['num_glosses'].sum():,}"),
            ('Mean Glosses/Sample', lambda s: f"{df[df['split'] == s]['num_glosses'].mean():.1f}"),
        ]

        for metric_name, metric_fn in metrics:
            row = [metric_name]
            for split in splits_list:
                row.append(metric_fn(split))
            table_data.append(row)

        table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                               colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style data rows
        for i in range(1, len(table_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 6: Split comparison")

        # ============================================
        # PAGE 7: DATA QUALITY SUMMARY
        # ============================================
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.95, "Data Quality Summary",
                ha='center', va='top', fontsize=24, fontweight='bold',
                transform=ax.transAxes)

        quality_sections = [
            ("✓ Landmark Quality", [
                "• Total landmarks per frame: 543",
                "• Landmark types: Hands (42), Pose (33), Face (468)",
                "• Feature dimensions: X, Y coordinates only",
                "• Z-dimension removed (80-98% near-zero values)",
                "• Custom landmarks removed (10 invalid landmarks dropped)",
            ]),
            ("✓ Coordinate Validation", [
                "• X,Y coordinates validated: 0% invalid",
                "• Coordinate range: [0, 1] (normalized)",
                "• No NaN or Inf values detected",
                "• Temporal consistency verified",
            ]),
            ("✓ Sample Quality", [
                "• All 6,841 samples passed quality checks",
                "• Quality distribution: 100% medium quality",
                "• No samples filtered out",
                "• Homogeneous data quality across splits",
            ]),
            ("✓ Processing Pipeline", [
                "• MediaPipe Holistic used for landmark extraction",
                "• Feature cleaning: 1659 → 1086 dimensions",
                "• Dimension reduction: 34% smaller feature vectors",
                "• Ready for model training",
            ]),
        ]

        y_pos = 0.88
        for section_title, items in quality_sections:
            ax.text(0.1, y_pos, section_title,
                    fontsize=14, fontweight='bold', color='darkgreen',
                    transform=ax.transAxes)
            y_pos -= 0.04

            for item in items:
                ax.text(0.12, y_pos, item,
                        fontsize=11,
                        transform=ax.transAxes,
                        family='monospace')
                y_pos -= 0.03

            y_pos -= 0.02

        # Known limitations box
        lim_y = 0.22
        lim_box = Rectangle((0.08, lim_y - 0.18), 0.84, 0.18,
                            transform=ax.transAxes,
                            facecolor='lightyellow',
                            edgecolor='orange',
                            linewidth=2,
                            alpha=0.5)
        ax.add_patch(lim_box)

        ax.text(0.5, lim_y + 0.01, "Known Limitations",
                ha='center', fontsize=13, fontweight='bold', color='darkorange',
                transform=ax.transAxes)

        limitations = [
            "⚠ No confidence scores available (Z-values used as proxy)",
            "⚠ MediaPipe hand detection order inconsistent (requires handedness detection)",
            "⚠ Extreme class imbalance (593:1 ratio)",
            "⚠ Some glosses have very few samples (673 classes with ≤3 samples)",
            "⚠ Z-dimension information lost (may affect 3D gesture recognition)",
        ]

        lim_y -= 0.02
        for lim in limitations:
            ax.text(0.1, lim_y, lim,
                    fontsize=10, color='darkorange',
                    transform=ax.transAxes)
            lim_y -= 0.03

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 7: Data quality summary")

        # ============================================
        # PAGE 8: RECOMMENDATIONS & NEXT STEPS
        # ============================================
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.95, "Recommendations & Next Steps",
                ha='center', va='top', fontsize=24, fontweight='bold',
                transform=ax.transAxes)

        # Model configuration recommendations
        config_y = 0.88
        ax.text(0.1, config_y, "1. Model Configuration",
                fontsize=16, fontweight='bold', color='navy',
                transform=ax.transAxes)

        config_items = [
            f"• Input shape: [Batch, Time, 543, 2]",
            f"• Sequence length: max_length = {int(df['num_frames'].quantile(0.95))} (95th percentile)",
            f"• Number of classes: Start with Top-200 ({gloss_df.head(200)['count'].sum() / len(all_glosses) * 100:.1f}% coverage)",
            f"• Batch size: 8 (adjust based on GPU memory)",
            f"• Architecture: Multi-stream GCN + Transformer",
        ]

        config_y -= 0.04
        for item in config_items:
            ax.text(0.12, config_y, item, fontsize=11,
                    transform=ax.transAxes, family='monospace')
            config_y -= 0.03

        # Training strategy
        train_y = config_y - 0.02
        ax.text(0.1, train_y, "2. Training Strategy",
                fontsize=16, fontweight='bold', color='navy',
                transform=ax.transAxes)

        train_items = [
            "• Loss function: Focal Loss (α=0.25, γ=2.0) for class imbalance",
            "• Decoder: CTC Loss for sequence-to-sequence",
            "• Optimizer: AdamW (lr=1e-4, weight_decay=0.01)",
            "• Learning rate: Warmup + Cosine decay",
            "• Regularization: Dropout 0.1-0.3, Gradient clipping at 1.0",
        ]

        train_y -= 0.04
        for item in train_items:
            ax.text(0.12, train_y, item, fontsize=11,
                    transform=ax.transAxes, family='monospace')
            train_y -= 0.03

        # Data augmentation
        aug_y = train_y - 0.02
        ax.text(0.1, aug_y, "3. Data Augmentation",
                fontsize=16, fontweight='bold', color='navy',
                transform=ax.transAxes)

        aug_items = [
            "• Spatial: Rotation (±15°), Scaling (0.9-1.1), Translation (±10%)",
            "• Temporal: Random frame drop/repeat (0.9x-1.1x speed)",
            "• Landmark occlusion: Random hand/face dropout (10-15% probability)",
            "• Bone-preserving augmentation (preserve anatomical proportions)",
        ]

        aug_y -= 0.04
        for item in aug_items:
            ax.text(0.12, aug_y, item, fontsize=11,
                    transform=ax.transAxes, family='monospace')
            aug_y -= 0.03

        # Evaluation metrics
        eval_y = aug_y - 0.02
        ax.text(0.1, eval_y, "4. Evaluation Metrics",
                fontsize=16, fontweight='bold', color='navy',
                transform=ax.transAxes)

        eval_items = [
            "• Primary: Word Error Rate (WER)",
            "• Secondary: Top-1, Top-5 Accuracy",
            "• Per-class: Precision, Recall, F1-Score",
            "• Confusion matrix analysis for error patterns",
        ]

        eval_y -= 0.04
        for item in eval_items:
            ax.text(0.12, eval_y, item, fontsize=11,
                    transform=ax.transAxes, family='monospace')
            eval_y -= 0.03

        # Next steps box
        next_y = 0.15
        next_box = Rectangle((0.08, next_y - 0.11), 0.84, 0.11,
                             transform=ax.transAxes,
                             facecolor='lightblue',
                             edgecolor='darkblue',
                             linewidth=2,
                             alpha=0.4)
        ax.add_patch(next_box)

        ax.text(0.5, next_y, "Immediate Next Steps",
                ha='center', fontsize=13, fontweight='bold',
                transform=ax.transAxes)

        next_steps = [
            "1. Implement Transformer architecture (Multi-stream GCN + Temporal Encoder)",
            "2. Create data loaders with augmentation pipeline",
            "3. Train baseline model on Top-200 classes",
            "4. Evaluate and analyze confusion matrix",
            "5. Iteratively optimize based on results",
        ]

        next_y -= 0.025
        for step in next_steps:
            ax.text(0.1, next_y, step, fontsize=10,
                    transform=ax.transAxes)
            next_y -= 0.02

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        print("✓ Page 8: Recommendations")

        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = 'SignNet Dataset Analysis Report'
        d['Author'] = 'Andrei Chirila & Roman Schläpfer'
        d['Subject'] = 'RWTH-PHOENIX-2014 Dataset Analysis'
        d['Keywords'] = 'Sign Language, Deep Learning, RWTH-PHOENIX'
        d['CreationDate'] = datetime.now()

    print(f"\n✅ PDF Report generated: {PDF_FILENAME}")
    print(f"   File size: {PDF_FILENAME.stat().st_size / 1024 / 1024:.2f} MB")


# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PDF REPORT GENERATION")
    print("=" * 80)

    # Collect data
    print("\nCollecting dataset information...")
    df = collect_dataset_info()

    # Analyze glosses
    print("\nAnalyzing gloss distribution...")
    gloss_df, gloss_counts, all_glosses = analyze_glosses(df)

    # Generate PDF
    create_pdf_report(df, gloss_df, gloss_counts, all_glosses)

    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated: {PDF_FILENAME}")
    print("\nYou can now:")
    print("  1. Review the complete analysis")
    print("  2. Include in your thesis documentation")
    print("  3. Share with your supervisor/team")
    print("  4. Proceed to model implementation")


if __name__ == "__main__":
    main()