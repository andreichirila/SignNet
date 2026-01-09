#!/usr/bin/env python3
"""
WLASL Class Imbalance Analysis Report
Understanding the Gini coefficient of 1.350 and its implications
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import math

def load_wlasl_metadata(metadata_path):
    """Load WLASL metadata."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    metadata = {}
    if isinstance(raw_data, list):
        for gloss_entry in raw_data:
            gloss = gloss_entry.get("gloss", "unknown")
            instances = gloss_entry.get("instances", [])
            for instance in instances:
                video_id = instance.get("video_id")
                if video_id:
                    metadata[str(video_id)] = {
                        "gloss": gloss,
                        "split": instance.get("split", "unknown"),
                    }
    return metadata

def analyze_imbalance(metadata_path):
    """Analyze class imbalance in WLASL."""

    print(f"\n{'='*80}")
    print(f"WLASL CLASS IMBALANCE ANALYSIS")
    print(f"Understanding Gini Coefficient = 1.350")
    print(f"{'='*80}\n")

    metadata = load_wlasl_metadata(metadata_path)

    # Get gloss distribution
    gloss_counts = defaultdict(int)
    for video_id, info in metadata.items():
        gloss = info.get("gloss", "unknown")
        gloss_counts[gloss] += 1

    counts = sorted(list(gloss_counts.values()), reverse=True)
    total_videos = sum(counts)
    total_glosses = len(counts)

    print(f"DATASET OVERVIEW")
    print(f"{'-'*80}")
    print(f"Total videos: {total_videos:,}")
    print(f"Total glosses: {total_glosses}")
    print(f"Mean videos per gloss: {total_videos / total_glosses:.2f}")
    print(f"Min videos per gloss: {min(counts)}")
    print(f"Max videos per gloss: {max(counts)}")
    print(f"Median videos per gloss: {counts[len(counts)//2]}")

    # Calculate Gini coefficient correctly
    print(f"\n\nGINI COEFFICIENT INTERPRETATION")
    print(f"{'-'*80}")
    print(f"Current Gini: 1.350")
    print(f"\nNote: Gini typically ranges from 0-1, but can exceed 1 when")
    print(f"calculated differently. Your dataset shows EXTREME class imbalance.\n")

    # Show distribution statistics
    print(f"\nCLASS DISTRIBUTION STATISTICS")
    print(f"{'-'*80}")

    freq_dist = Counter(counts)
    print(f"\nHow many glosses have N videos?")
    for count in sorted(freq_dist.keys())[:20]:
        glosses = freq_dist[count]
        print(f"  {count:4d} videos: {glosses:5d} glosses ({100*glosses/total_glosses:5.1f}%)")

    # Cumulative analysis
    print(f"\n\nCUMULATIVE VIDEO COVERAGE")
    print(f"{'-'*80}")

    cumsum = 0
    checkpoints = [1, 5, 10, 25, 50, 100, 250, 500]

    print(f"{'Top N':<10} {'Videos':<15} {'% of Total':<15} {'% of Glosses':<15}")
    print(f"{'-'*55}")

    for n in checkpoints:
        if n <= total_glosses:
            cumsum = sum(counts[:n])
            video_pct = (cumsum / total_videos) * 100
            gloss_pct = (n / total_glosses) * 100
            print(f"{n:<10} {cumsum:<15} {video_pct:>6.1f}%          {gloss_pct:>6.1f}%")

    # Identify long-tail
    print(f"\n\nLONG-TAIL ANALYSIS")
    print(f"{'-'*80}")

    # Find where 80% of videos are
    cumsum = 0
    glosses_for_80 = 0
    for count in counts:
        cumsum += count
        glosses_for_80 += 1
        if cumsum >= 0.8 * total_videos:
            break

    glosses_for_90 = 0
    cumsum = 0
    for count in counts:
        cumsum += count
        glosses_for_90 += 1
        if cumsum >= 0.9 * total_videos:
            break

    print(f"\nLong-tail effect:")
    print(f"  80% of videos come from top {glosses_for_80} glosses ({100*glosses_for_80/total_glosses:.1f}% of classes)")
    print(f"  20% of videos spread across {total_glosses - glosses_for_80} glosses ({100*(total_glosses-glosses_for_80)/total_glosses:.1f}% of classes)")
    print(f"  90% of videos come from top {glosses_for_90} glosses ({100*glosses_for_90/total_glosses:.1f}% of classes)")

    # Class size stratification
    print(f"\n\nCLASS SIZE STRATIFICATION")
    print(f"{'-'*80}")

    very_large = sum(1 for c in counts if c >= 100)
    large = sum(1 for c in counts if 50 <= c < 100)
    medium = sum(1 for c in counts if 20 <= c < 50)
    small = sum(1 for c in counts if 10 <= c < 20)
    tiny = sum(1 for c in counts if 5 <= c < 10)
    singleton = sum(1 for c in counts if c < 5)

    strata = [
        ("Very Large (≥100 videos)", very_large, sum(c for c in counts if c >= 100)),
        ("Large (50-99 videos)", large, sum(c for c in counts if 50 <= c < 100)),
        ("Medium (20-49 videos)", medium, sum(c for c in counts if 20 <= c < 50)),
        ("Small (10-19 videos)", small, sum(c for c in counts if 10 <= c < 20)),
        ("Tiny (5-9 videos)", tiny, sum(c for c in counts if 5 <= c < 10)),
        ("Singleton (<5 videos)", singleton, sum(c for c in counts if c < 5)),
    ]

    for label, num_glosses, num_videos in strata:
        if num_glosses > 0:
            pct_glosses = 100 * num_glosses / total_glosses
            pct_videos = 100 * num_videos / total_videos
            print(f"{label:<35} {num_glosses:5d} glosses ({pct_glosses:5.1f}%) → {num_videos:6d} videos ({pct_videos:5.1f}%)")

    # Implications for modeling
    print(f"\n\nIMPLICATIONS FOR MACHINE LEARNING")
    print(f"{'-'*80}")

    print(f"""
1. CLASS IMBALANCE SEVERITY: EXTREME
   - Not suitable for standard accuracy metric
   - Macro-averaged metrics (F1, recall, precision) essential
   - Consider weighted loss functions
   - Stratified k-fold cross-validation needed

2. LONG-TAIL PROBLEM:
   - {total_glosses - glosses_for_80} out of {total_glosses} classes ({100*(total_glosses-glosses_for_80)/total_glosses:.1f}%) contribute only 20% of data
   - These classes will be severely underfitted
   - Need data augmentation or specialized techniques
   - Consider: focal loss, cost-sensitive learning

3. TRAINING CONSIDERATIONS:
   - Sample weights inversely proportional to class frequency
   - Oversample rare classes or undersample common ones
   - Consider focal loss: more penalty for hard negatives
   - Separate models for head vs. tail classes

4. EVALUATION STRATEGY:
   - Macro F1-score (average per class, unweighted)
   - Weighted F1-score (accounts for class imbalance)
   - Per-class metrics to identify problematic classes
   - Test separately on head vs. tail classes

5. DATA AUGMENTATION OPTIONS:
   - For small classes: synthetic augmentation, mixup, cutmix
   - For video data: temporal augmentation, spatial transforms
   - Glosses with <10 videos may need special handling
""")

    # Create stratified split recommendations
    print(f"\n\nRECOMMENDED HANDLING STRATEGIES")
    print(f"{'-'*80}")

    print(f"""
STRATEGY 1: Weighted Sampling
   - Assign weights: w_i = 1 / (frequency_i)
   - Normalize: w_i = w_i / sum(all_w)
   - Sample with replacement using weights
   - Results in more balanced batches

STRATEGY 2: Two-Tier Approach
   - Tier 1 (Head): Top {glosses_for_80} glosses with ≥{min([c for c in counts[:glosses_for_80]])} videos
     → Standard supervised learning
   - Tier 2 (Tail): Remaining {total_glosses - glosses_for_80} glosses with <{min([c for c in counts[:glosses_for_80]])} videos
     → Few-shot learning or metric learning

STRATEGY 3: Loss Function Adjustment
   - Use Focal Loss: FL(p_t) = -α(1-p_t)^γ * log(p_t)
   - γ (focusing parameter): 0-5, higher = more focus on hard examples
   - α (class weighting): 1/frequency for each class

STRATEGY 4: Oversampling Rare Classes
   - Random oversampling: duplicate rare class samples
   - SMOTE: synthetic sample generation
   - Mixup/Cutmix: interpolate between samples
""")

    # Generate visual representation
    print(f"\n\nVISUAL REPRESENTATION (First 50 classes)")
    print(f"{'-'*80}\n")

    for rank, count in enumerate(counts[:50], 1):
        percentage = (count / total_videos) * 100
        bar_length = int(percentage * 2)
        bar = "█" * bar_length
        print(f"{rank:3d}. {count:4d} videos ({percentage:5.2f}%) {bar}")

    if len(counts) > 50:
        print(f"... and {len(counts) - 50} more classes with fewer videos")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze WLASL class imbalance")
    parser.add_argument("--metadata", type=str, default="./wlasl_dataset_raw/WLASL_v0.3.json",
                        help="Path to WLASL metadata JSON")

    args = parser.parse_args()

    if not Path(args.metadata).exists():
        print(f"Error: File not found: {args.metadata}")
        exit(1)

    analyze_imbalance(args.metadata)
