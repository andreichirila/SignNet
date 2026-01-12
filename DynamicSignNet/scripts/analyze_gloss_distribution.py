#!/usr/bin/env python3
"""
Gloss Distribution Analyzer for Sign Language Dataset
Analyzes .npz files and counts samples per gloss based on filename patterns.
Includes Gini Coefficient calculation to measure dataset imbalance.
"""

import os
import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np  # Neu für die Gini-Berechnung
import argparse

def calculate_gini(array):
    """
    Berechnet den Gini-Koeffizienten eines Arrays (Maß für die Ungleichverteilung).
    0 = Perfekte Gleichverteilung
    1 = Maximale Ungleichverteilung (Long-Tail)
    """
    # Sicherstellen, dass es ein flaches Numpy-Array ist und sortieren
    array = np.sort(array).astype(float)
    n = len(array)
    if n <= 1:
        return 0.0
    
    # Index-Vektor (1 bis n)
    index = np.arange(1, n + 1)
    
    # Gini-Formel für diskrete Werte
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def analyze_gloss_distribution(directory):
    """
    Analyze gloss distribution from .npz filenames
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist!")
        sys.exit(1)

    file_list = list(directory_path.glob("*.npz"))

    if len(file_list) == 0:
        print(f"Error: No .npz files found in '{directory}'")
        sys.exit(1)

    print(f"Found {len(file_list)} .npz files in {directory}")

    glosses = []
    invalid_files = []

    for file_path in file_list:
        filename = file_path.name
        name = filename.replace(".npz", "")
        parts = name.split("_")

        if len(parts) >= 2:
            gloss = "_".join(parts[:-1])
            glosses.append(gloss)
        else:
            invalid_files.append(filename)

    if invalid_files:
        print(f"\nWarning: {len(invalid_files)} files with unexpected format:")
        for f in invalid_files[:10]:
            print(f"  - {f}")

    gloss_counts = Counter(glosses)
    df = pd.DataFrame.from_dict(gloss_counts, orient='index', columns=['count'])
    df.index.name = 'gloss'
    df = df.sort_values('count', ascending=False)
    df = df.reset_index()

    return df, gloss_counts, len(file_list)

def print_statistics(df, total_files):
    """Print detailed statistics about gloss distribution"""

    # Gini berechnen
    gini_index = calculate_gini(df['count'].values)

    print(f"\n{'='*70}")
    print("GLOSS DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")

    print(f"\nTotal files: {total_files}")
    print(f"Unique glosses: {len(df)}")

    print(f"\n{'='*70}")
    print("TOP 20 MOST FREQUENT GLOSSES")
    print(f"{'='*70}")
    print(df.head(20).to_string(index=False))

    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Mean samples per gloss:   {df['count'].mean():.2f}")
    print(f"Median samples per gloss: {df['count'].median():.2f}")
    print(f"Std dev:                  {df['count'].std():.2f}")
    print(f"Min samples:              {df['count'].min()}")
    print(f"Max samples:              {df['count'].max()}")
    print(f"\nGINI INDEX:               {gini_index:.4f}")
    
    # Interpretation des Gini-Werts
    if gini_index > 0.7:
        status = "Extreme Imbalance (Massive Long-Tail)"
    elif gini_index > 0.4:
        status = "High Imbalance"
    else:
        status = "Moderate to Low Imbalance"
    print(f"Interpretation:           {status}")

    print(f"\n{'='*70}")
    print("DISTRIBUTION BY SAMPLE COUNT")
    print(f"{'='*70}")

    bins = [1, 5, 10, 20, 50, 100, float('inf')]
    labels = ['1-4', '5-9', '10-19', '20-49', '50-99', '100+']

    for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
        if high == float('inf'):
            count = len(df[df['count'] >= low])
        else:
            count = len(df[(df['count'] >= low) & (df['count'] < high)])

        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"{label:>10} samples: {count:>4} glosses ({pct:>5.1f}%)")

def save_results(df, output_file):
    """Save results to CSV file"""
    df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze gloss distribution in sign language dataset'
    )

    parser.add_argument(
        'directory',
        type=str,
        default='..\\word_landmarks_extracted/',
        help='Path to directory containing .npz files'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='gloss_distribution.csv',
        help='Output CSV file name (default: gloss_distribution.csv)'
    )

    args = parser.parse_args()
    df, counts, total_files = analyze_gloss_distribution(args.directory)
    print_statistics(df, total_files)
    save_results(df, args.output)

if __name__ == "__main__":
    main()