#!/usr/bin/env python3
"""
WLASL Dataset JSON Analysis (No external dependencies)
Comprehensive statistics on glosses, splits, and distribution.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

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
                        **{k: v for k, v in instance.items() if k not in ["video_id", "gloss"]}
                    }

    return metadata


def analyze_wlasl(metadata_path):
    """Analyze WLASL dataset."""

    print(f"\n{'='*80}")
    print(f"WLASL DATASET ANALYSIS")
    print(f"{'='*80}\n")

    metadata = load_wlasl_metadata(metadata_path)

    # Basic statistics
    total_videos = len(metadata)
    print(f"Total videos: {total_videos:,}\n")

    # Split analysis
    print(f"{'='*80}")
    print(f"SPLIT DISTRIBUTION")
    print(f"{'='*80}\n")

    split_counts = Counter()
    for video_id, info in metadata.items():
        split = info.get("split", "unknown")
        split_counts[split] += 1

    total_by_split = sum(split_counts.values())
    for split in sorted(split_counts.keys()):
        count = split_counts[split]
        percentage = (count / total_by_split) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {split:15s}: {count:5d} videos ({percentage:6.2f}%) {bar}")

    # Gloss analysis
    print(f"\n{'='*80}")
    print(f"GLOSS STATISTICS")
    print(f"{'='*80}\n")

    gloss_videos = defaultdict(list)
    for video_id, info in metadata.items():
        gloss = info.get("gloss", "unknown")
        gloss_videos[gloss].append(video_id)

    total_glosses = len(gloss_videos)
    print(f"Total unique glosses: {total_glosses:,}\n")

    # Distribution of videos per gloss
    gloss_counts = {g: len(vids) for g, vids in gloss_videos.items()}
    counts = list(gloss_counts.values())

    print(f"Videos per gloss statistics:")
    print(f"  Min: {min(counts)}")
    print(f"  Max: {max(counts)}")
    print(f"  Mean: {sum(counts) / len(counts):.2f}")
    print(f"  Median: {sorted(counts)[len(counts)//2]}")
    print(f"  Mode: {Counter(counts).most_common(1)[0][0]}")

    # Frequency distribution
    freq_dist = Counter(counts)
    print(f"\nGlosses by video count:")
    for count in sorted(freq_dist.keys())[:15]:  # Top 15 frequencies
        glosses = freq_dist[count]
        percentage = (glosses / total_glosses) * 100
        print(f"  {count:3d} videos: {glosses:5d} glosses ({percentage:6.2f}%)")

    if len(freq_dist) > 15:
        print(f"  ... and {len(freq_dist) - 15} more frequency values")

    # Split-Gloss cross-tabulation
    print(f"\n{'='*80}")
    print(f"SPLIT-GLOSS DISTRIBUTION")
    print(f"{'='*80}\n")

    split_gloss_counts = defaultdict(lambda: defaultdict(int))
    for video_id, info in metadata.items():
        split = info.get("split", "unknown")
        gloss = info.get("gloss", "unknown")
        split_gloss_counts[split][gloss] += 1

    for split in sorted(split_gloss_counts.keys()):
        gloss_dict = split_gloss_counts[split]
        total_in_split = sum(gloss_dict.values())
        unique_glosses = len(gloss_dict)
        print(f"\n{split.upper()}:")
        print(f"  Total videos: {total_in_split}")
        print(f"  Unique glosses: {unique_glosses}")

        counts_split = list(gloss_dict.values())
        print(f"  Videos per gloss (stats):")
        print(f"    Min: {min(counts_split)}, Max: {max(counts_split)}, Mean: {sum(counts_split) / len(counts_split):.2f}")

    # Top glosses by video count
    print(f"\n{'='*80}")
    print(f"TOP 30 GLOSSES BY VIDEO COUNT")
    print(f"{'='*80}\n")

    sorted_glosses = sorted(gloss_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"{'Rank':<6} {'Gloss':<35} {'Videos':<8} {'Percentage':<12} {'Bar':<40}")
    print(f"{'-'*6} {'-'*35} {'-'*8} {'-'*12} {'-'*40}")

    for rank, (gloss, count) in enumerate(sorted_glosses[:30], 1):
        percentage = (count / total_videos) * 100
        bar = "█" * int(percentage / 0.25)  # Scale bar
        print(f"{rank:<6} {gloss:<35} {count:<8} {percentage:>6.2f}%     {bar}")

    # Bottom glosses (rare glosses)
    print(f"\n{'='*80}")
    print(f"RARE GLOSSES (Bottom 20 by video count)")
    print(f"{'='*80}\n")

    print(f"{'Rank':<6} {'Gloss':<35} {'Videos':<8} {'Percentage':<12}")
    print(f"{'-'*6} {'-'*35} {'-'*8} {'-'*12}")

    bottom_glosses = sorted(sorted_glosses, key=lambda x: x[1])[:20]
    for rank, (gloss, count) in enumerate(bottom_glosses, 1):
        percentage = (count / total_videos) * 100
        print(f"{rank:<6} {gloss:<35} {count:<8} {percentage:>6.2f}%")

    # Gloss coverage
    print(f"\n{'='*80}")
    print(f"GLOSS COVERAGE ANALYSIS")
    print(f"{'='*80}\n")

    cumsum = 0
    thresholds = [25, 50, 75, 90, 95, 99]

    print(f"{'Coverage':<15} {'# Glosses':<15} {'% of Glosses':<15}")
    print(f"{'-'*15} {'-'*15} {'-'*15}")

    for idx, (gloss, count) in enumerate(sorted_glosses):
        cumsum += count
        percentage = (cumsum / total_videos) * 100

        for t in thresholds:
            if percentage >= t:
                glosses_needed = idx + 1
                glosses_pct = (glosses_needed / total_glosses) * 100
                print(f"{t}% of videos    {glosses_needed:<15} {glosses_pct:>6.2f}%")
                thresholds.remove(t)

    # Distribution shape
    print(f"\n{'='*80}")
    print(f"DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")

    # Calculate Gini coefficient (measure of inequality)
    total = sum(counts)
    cumsum_gini = 0
    gini = 0
    n = len(counts)

    for i, c in enumerate(sorted(counts)):
        cumsum_gini += c
        gini += (2 * i + 1) * c

    gini = (2 * gini) / (n * total) - (n + 1) / n

    print(f"Distribution inequality (Gini coefficient): {gini:.3f}")
    print(f"  (0 = equal distribution, 1 = highly skewed)")

    # Count distribution by range
    ranges = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 500), (501, 10000)]
    print(f"\nGlosses by video count ranges:")

    for min_v, max_v in ranges:
        count_in_range = sum(1 for v in counts if min_v <= v <= max_v)
        if count_in_range > 0:
            percentage = (count_in_range / total_glosses) * 100
            total_vids_in_range = sum(v for v in counts if min_v <= v <= max_v)
            print(f"  {min_v:4d}-{max_v:4d} videos: {count_in_range:5d} glosses ({percentage:6.2f}%) - {total_vids_in_range:6d} total videos")

    # Save detailed report
    print(f"\n{'='*80}")
    print(f"SAVING DETAILED REPORTS")
    print(f"{'='*80}\n")

    # Glosses ranked by count
    with open("wlasl_glosses_ranked.txt", "w", encoding='utf-8') as f:
        f.write("WLASL Glosses Ranked by Video Count\n")
        f.write(f"Total glosses: {total_glosses}\n")
        f.write(f"Total videos: {total_videos}\n\n")

        f.write(f"{'Rank':<8} {'Gloss':<40} {'Videos':<12} {'Percentage':<15}\n")
        f.write(f"{'-'*75}\n")

        for rank, (gloss, count) in enumerate(sorted_glosses, 1):
            percentage = (count / total_videos) * 100
            f.write(f"{rank:<8} {gloss:<40} {count:<12} {percentage:>6.2f}%\n")

    print("✓ Saved: wlasl_glosses_ranked.txt")

    # Split-gloss summary
    with open("wlasl_split_gloss_summary.txt", "w", encoding='utf-8') as f:
        f.write("WLASL Split-Gloss Summary\n\n")

        for split in sorted(split_gloss_counts.keys()):
            gloss_dict = split_gloss_counts[split]
            f.write(f"\n{'='*60}\n")
            f.write(f"{split.upper()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total videos: {sum(gloss_dict.values())}\n")
            f.write(f"Unique glosses: {len(gloss_dict)}\n\n")

            f.write(f"{'Gloss':<40} {'Videos':<12}\n")
            f.write(f"{'-'*52}\n")

            for gloss, count in sorted(gloss_dict.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{gloss:<40} {count:<12}\n")

    print("✓ Saved: wlasl_split_gloss_summary.txt")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {total_videos:,}")
    print(f"Total glosses: {total_glosses:,}")
    print(f"Splits: {', '.join(sorted(split_counts.keys()))}")
    print(f"Videos per gloss range: {min(counts)} - {max(counts)}")
    print(f"Gini coefficient: {gini:.3f} (inequality measure)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze WLASL dataset JSON (no dependencies)")
    parser.add_argument("--metadata", type=str, default="./wlasl_dataset_raw/WLASL_v0.3.json",
                        help="Path to WLASL metadata JSON")

    args = parser.parse_args()

    if not Path(args.metadata).exists():
        print(f"Error: File not found: {args.metadata}")
        exit(1)

    analyze_wlasl(args.metadata)
