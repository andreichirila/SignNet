#!/usr/bin/env python3
"""
Dataset Validation and Anomaly Detection Script
For Sign Language Translation Dataset

This script performs comprehensive validation checks on your preprocessed
landmark dataset to ensure data quality and detect processing errors.

Usage:
    python validate_dataset.py --landmarks_dir ./landmarks_train \
                               --vocab_file vocab.json \
                               --output validation_report.json
"""

import os
import glob
import numpy as np
import json
from collections import Counter, defaultdict
from tqdm import tqdm


class DatasetValidator:
    """Comprehensive validator for sign language landmark dataset"""

    def __init__(self, landmarks_dir, vocab_file=None):
        self.landmarks_dir = landmarks_dir
        self.vocab_file = vocab_file
        self.samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
        self.vocab = self._load_vocab() if vocab_file else None

        # Results storage
        self.anomalies = {
            'critical': [],
            'warning': [],
            'info': []
        }

        self.stats = {
            'total_samples': len(self.samples),
            'total_frames': 0,
            'total_glosses': 0,
            'sequence_lengths': [],
            'gloss_counts': [],
            'landmark_ranges': [],
            'gloss_distribution': Counter(),
            'sequence_gloss_ratio': [],
        }

    def _load_vocab(self):
        """Load vocabulary from JSON file"""
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r') as f:
                vocab = json.load(f)
            print(f"✓ Loaded vocabulary: {len(vocab)} entries")
            return vocab
        return None

    def validate_all(self):
        """Run all validation checks"""
        print("\n" + "="*70)
        print("DATASET VALIDATION AND ANOMALY DETECTION")
        print("="*70)
        print(f"Dataset: {self.landmarks_dir}")
        print(f"Total samples: {len(self.samples)}")
        print("="*70)

        # Run checks
        self._check_file_integrity()
        self._check_data_structure()
        self._check_landmark_validity()
        self._check_gloss_sequence_alignment()
        self._check_vocabulary_coverage()
        self._check_statistical_outliers()
        self._check_duplicate_sequences()

        # Print summary
        self._print_summary()

        return self.anomalies, self.stats

    def _check_file_integrity(self):
        """Check if all NPZ files can be loaded"""
        print("\n[1/7] Checking file integrity...")

        corrupted_files = []
        for sample_path in tqdm(self.samples, desc="Loading files"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                # Check required keys
                if 'landmarks' not in data or 'glosses' not in data:
                    self.anomalies['critical'].append({
                        'type': 'missing_keys',
                        'file': os.path.basename(sample_path),
                        'message': 'Missing required keys (landmarks or glosses)'
                    })
            except Exception as e:
                corrupted_files.append((sample_path, str(e)))
                self.anomalies['critical'].append({
                    'type': 'corrupted_file',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        if corrupted_files:
            print(f"  ⚠️  Found {len(corrupted_files)} corrupted files")
        else:
            print("  ✓ All files are valid and loadable")

    def _check_data_structure(self):
        """Verify data structure consistency"""
        print("\n[2/7] Checking data structure...")

        expected_feature_dim = 1659  # 553 landmarks * 3 coordinates

        for sample_path in tqdm(self.samples, desc="Validating structure"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']

                # Check landmark dimensions
                if landmarks.ndim != 2:
                    self.anomalies['critical'].append({
                        'type': 'invalid_landmark_dims',
                        'file': os.path.basename(sample_path),
                        'expected': 2,
                        'actual': landmarks.ndim,
                        'shape': landmarks.shape
                    })

                # Check feature dimension
                if landmarks.shape[1] != expected_feature_dim:
                    self.anomalies['warning'].append({
                        'type': 'unexpected_feature_dim',
                        'file': os.path.basename(sample_path),
                        'expected': expected_feature_dim,
                        'actual': landmarks.shape[1]
                    })

                # Check gloss structure
                if not isinstance(glosses, (list, np.ndarray)):
                    self.anomalies['critical'].append({
                        'type': 'invalid_gloss_type',
                        'file': os.path.basename(sample_path),
                        'type_found': type(glosses).__name__
                    })

                # Store stats
                self.stats['total_frames'] += landmarks.shape[0]
                self.stats['total_glosses'] += len(glosses)
                self.stats['sequence_lengths'].append(landmarks.shape[0])
                self.stats['gloss_counts'].append(len(glosses))

            except Exception as e:
                self.anomalies['critical'].append({
                    'type': 'structure_check_failed',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        print(f"  ✓ Processed {len(self.samples)} samples")
        print(f"  ✓ Total frames: {self.stats['total_frames']:,}")
        print(f"  ✓ Total glosses: {self.stats['total_glosses']:,}")

    def _check_landmark_validity(self):
        """Check for invalid landmark values"""
        print("\n[3/7] Checking landmark validity...")

        invalid_count = 0

        for sample_path in tqdm(self.samples, desc="Validating landmarks"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks'].astype(np.float32)

                # Check for NaN
                if np.isnan(landmarks).any():
                    nan_count = np.isnan(landmarks).sum()
                    self.anomalies['critical'].append({
                        'type': 'nan_values',
                        'file': os.path.basename(sample_path),
                        'nan_count': int(nan_count),
                        'total_values': landmarks.size
                    })
                    invalid_count += 1

                # Check for Inf
                if np.isinf(landmarks).any():
                    inf_count = np.isinf(landmarks).sum()
                    self.anomalies['critical'].append({
                        'type': 'inf_values',
                        'file': os.path.basename(sample_path),
                        'inf_count': int(inf_count),
                        'total_values': landmarks.size
                    })
                    invalid_count += 1

                # Check for extreme values
                min_val, max_val = landmarks.min(), landmarks.max()
                self.stats['landmark_ranges'].append((min_val, max_val))

                if min_val < -100 or max_val > 100:
                    self.anomalies['warning'].append({
                        'type': 'extreme_landmark_values',
                        'file': os.path.basename(sample_path),
                        'min': float(min_val),
                        'max': float(max_val)
                    })

                # Check for all-zero frames
                zero_frames = np.all(landmarks == 0, axis=1)
                if zero_frames.any():
                    zero_count = zero_frames.sum()
                    if zero_count > landmarks.shape[0] * 0.5:
                        self.anomalies['warning'].append({
                            'type': 'excessive_zero_frames',
                            'file': os.path.basename(sample_path),
                            'zero_frames': int(zero_count),
                            'total_frames': landmarks.shape[0]
                        })

            except Exception as e:
                self.anomalies['critical'].append({
                    'type': 'landmark_validation_failed',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        if invalid_count == 0:
            print("  ✓ No invalid values (NaN/Inf) detected")
        else:
            print(f"  ⚠️  Found {invalid_count} files with invalid values")

    def _check_gloss_sequence_alignment(self):
        """Check if glosses align properly with landmark sequences"""
        print("\n[4/7] Checking gloss-sequence alignment...")

        misaligned_count = 0

        for sample_path in tqdm(self.samples, desc="Checking alignment"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']

                num_frames = landmarks.shape[0]
                num_glosses = len(glosses)

                # Calculate ratio
                ratio = num_frames / num_glosses if num_glosses > 0 else 0
                self.stats['sequence_gloss_ratio'].append(ratio)

                # Check for empty sequences
                if num_frames == 0:
                    self.anomalies['critical'].append({
                        'type': 'empty_landmark_sequence',
                        'file': os.path.basename(sample_path)
                    })
                    misaligned_count += 1

                if num_glosses == 0:
                    self.anomalies['critical'].append({
                        'type': 'empty_gloss_sequence',
                        'file': os.path.basename(sample_path)
                    })
                    misaligned_count += 1

                # Check for suspicious ratios
                if ratio > 0:
                    if ratio < 5:
                        self.anomalies['warning'].append({
                            'type': 'low_frames_per_gloss',
                            'file': os.path.basename(sample_path),
                            'frames': num_frames,
                            'glosses': num_glosses,
                            'ratio': round(ratio, 2)
                        })
                        misaligned_count += 1
                    elif ratio > 200:
                        self.anomalies['warning'].append({
                            'type': 'high_frames_per_gloss',
                            'file': os.path.basename(sample_path),
                            'frames': num_frames,
                            'glosses': num_glosses,
                            'ratio': round(ratio, 2)
                        })
                        misaligned_count += 1

            except Exception as e:
                self.anomalies['critical'].append({
                    'type': 'alignment_check_failed',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        avg_ratio = np.mean(self.stats['sequence_gloss_ratio']) if self.stats['sequence_gloss_ratio'] else 0
        print(f"  ✓ Average frames per gloss: {avg_ratio:.2f}")

        if misaligned_count == 0:
            print("  ✓ All sequences properly aligned")
        else:
            print(f"  ⚠️  Found {misaligned_count} potentially misaligned sequences")

    def _check_vocabulary_coverage(self):
        """Check if all glosses are in vocabulary"""
        print("\n[5/7] Checking vocabulary coverage...")

        if not self.vocab:
            print("  ⚠️  No vocabulary file provided, skipping check")
            return

        unknown_glosses = set()
        out_of_range_glosses = []

        vocab_size = len(self.vocab)

        for sample_path in tqdm(self.samples, desc="Checking vocabulary"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                glosses = data['glosses']

                for gloss in glosses:
                    gloss_str = str(gloss)

                    # Update distribution
                    self.stats['gloss_distribution'][gloss_str] += 1

                    # Check if gloss exists in vocabulary
                    if gloss_str not in self.vocab:
                        unknown_glosses.add(gloss_str)
                        self.anomalies['warning'].append({
                            'type': 'unknown_gloss',
                            'file': os.path.basename(sample_path),
                            'gloss': gloss_str
                        })
                    else:
                        # Check if gloss ID is within valid range
                        gloss_id = self.vocab[gloss_str]
                        if gloss_id < 0 or gloss_id >= vocab_size:
                            out_of_range_glosses.append((gloss_str, gloss_id))
                            self.anomalies['critical'].append({
                                'type': 'gloss_id_out_of_range',
                                'file': os.path.basename(sample_path),
                                'gloss': gloss_str,
                                'id': gloss_id,
                                'vocab_size': vocab_size
                            })

            except Exception as e:
                self.anomalies['critical'].append({
                    'type': 'vocabulary_check_failed',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        print(f"  ✓ Unique glosses found: {len(self.stats['gloss_distribution'])}")
        print(f"  ✓ Vocabulary size: {vocab_size}")

        if unknown_glosses:
            print(f"  ⚠️  Found {len(unknown_glosses)} unknown glosses")
            print(f"      Examples: {list(unknown_glosses)[:5]}")
        else:
            print("  ✓ All glosses are in vocabulary")

        if out_of_range_glosses:
            print(f"  ⚠️  Found {len(out_of_range_glosses)} glosses with invalid IDs")

    def _check_statistical_outliers(self):
        """Detect statistical outliers"""
        print("\n[6/7] Detecting statistical outliers...")

        seq_lengths = np.array(self.stats['sequence_lengths'])
        gloss_counts = np.array(self.stats['gloss_counts'])

        seq_mean, seq_std = seq_lengths.mean(), seq_lengths.std()
        gloss_mean, gloss_std = gloss_counts.mean(), gloss_counts.std()

        print(f"  Sequence lengths: mean={seq_mean:.1f}, std={seq_std:.1f}")
        print(f"  Gloss counts: mean={gloss_mean:.1f}, std={gloss_std:.1f}")

        # Find outliers (> 3 std)
        seq_outliers = np.abs(seq_lengths - seq_mean) > 3 * seq_std
        gloss_outliers = np.abs(gloss_counts - gloss_mean) > 3 * gloss_std

        for i, (is_seq, is_gloss) in enumerate(zip(seq_outliers, gloss_outliers)):
            if is_seq or is_gloss:
                self.anomalies['info'].append({
                    'type': 'statistical_outlier',
                    'file': os.path.basename(self.samples[i]),
                    'sequence_length': int(seq_lengths[i]),
                    'gloss_count': int(gloss_counts[i])
                })

        outlier_count = seq_outliers.sum() + gloss_outliers.sum()
        if outlier_count > 0:
            print(f"  ℹ️  Found {outlier_count} statistical outliers")
        else:
            print("  ✓ No significant outliers detected")

    def _check_duplicate_sequences(self):
        """Check for duplicate sequences"""
        print("\n[7/7] Checking for duplicates...")

        hashes = {}
        duplicates = []

        for sample_path in tqdm(self.samples, desc="Computing hashes"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']

                signature = (
                    landmarks.shape,
                    tuple(landmarks[0].tolist()[:10]),
                    tuple(landmarks[-1].tolist()[:10])
                )

                if signature in hashes:
                    duplicates.append((os.path.basename(sample_path), hashes[signature]))
                    self.anomalies['info'].append({
                        'type': 'potential_duplicate',
                        'file': os.path.basename(sample_path),
                        'similar_to': hashes[signature]
                    })
                else:
                    hashes[signature] = os.path.basename(sample_path)

            except:
                pass

        if duplicates:
            print(f"  ℹ️  Found {len(duplicates)} potential duplicates")
        else:
            print("  ✓ No duplicates detected")

    def _print_summary(self):
        """Print summary"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        critical = len(self.anomalies['critical'])
        warning = len(self.anomalies['warning'])
        info = len(self.anomalies['info'])

        print(f"\n{'Status':<20} {'Count':<10}")
        print("-" * 30)
        print(f"{'Critical Issues:':<20} {critical:<10} {'🔴' if critical > 0 else '✓'}")
        print(f"{'Warnings:':<20} {warning:<10} {'⚠️' if warning > 0 else '✓'}")
        print(f"{'Info:':<20} {info:<10}")

        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)

        if self.stats['sequence_lengths']:
            print(f"Total samples: {self.stats['total_samples']:,}")
            print(f"Total frames: {self.stats['total_frames']:,}")
            print(f"Total glosses: {self.stats['total_glosses']:,}")
            print(f"\nSequence lengths:")
            print(f"  Min: {min(self.stats['sequence_lengths']):,}")
            print(f"  Max: {max(self.stats['sequence_lengths']):,}")
            print(f"  Mean: {np.mean(self.stats['sequence_lengths']):.1f}")
            print(f"  Median: {np.median(self.stats['sequence_lengths']):.1f}")
            print(f"\nGloss counts:")
            print(f"  Min: {min(self.stats['gloss_counts']):,}")
            print(f"  Max: {max(self.stats['gloss_counts']):,}")
            print(f"  Mean: {np.mean(self.stats['gloss_counts']):.1f}")
            print(f"  Median: {np.median(self.stats['gloss_counts']):.1f}")

            if self.stats['sequence_gloss_ratio']:
                avg = np.mean(self.stats['sequence_gloss_ratio'])
                print(f"\nAvg frames per gloss: {avg:.2f}")

        if self.stats['gloss_distribution']:
            print("\nTop 10 glosses:")
            for i, (g, c) in enumerate(self.stats['gloss_distribution'].most_common(10), 1):
                print(f"  {i:2d}. {g:<20} {c:>6,}")

        print("\n" + "="*70)

    def save_report(self, output_file="validation_report.json"):
        """Save report to JSON"""
        report = {
            'anomalies': self.anomalies,
            'statistics': {
                'total_samples': self.stats['total_samples'],
                'total_frames': self.stats['total_frames'],
                'total_glosses': self.stats['total_glosses'],
                'sequence_length_stats': {
                    'min': int(min(self.stats['sequence_lengths'])) if self.stats['sequence_lengths'] else 0,
                    'max': int(max(self.stats['sequence_lengths'])) if self.stats['sequence_lengths'] else 0,
                    'mean': float(np.mean(self.stats['sequence_lengths'])) if self.stats['sequence_lengths'] else 0,
                    'median': float(np.median(self.stats['sequence_lengths'])) if self.stats['sequence_lengths'] else 0,
                },
                'gloss_count_stats': {
                    'min': int(min(self.stats['gloss_counts'])) if self.stats['gloss_counts'] else 0,
                    'max': int(max(self.stats['gloss_counts'])) if self.stats['gloss_counts'] else 0,
                    'mean': float(np.mean(self.stats['gloss_counts'])) if self.stats['gloss_counts'] else 0,
                    'median': float(np.median(self.stats['gloss_counts'])) if self.stats['gloss_counts'] else 0,
                },
                'top_glosses': dict(self.stats['gloss_distribution'].most_common(50))
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate dataset')
    parser.add_argument('--landmarks_dir', required=True, help='Landmarks directory')
    parser.add_argument('--vocab_file', default=None, help='Vocabulary file')
    parser.add_argument('--output', default='validation_report.json', help='Output file')

    args = parser.parse_args()

    validator = DatasetValidator(args.landmarks_dir, args.vocab_file)
    validator.validate_all()
    validator.save_report(args.output)

    if validator.anomalies['critical']:
        print("\n⚠️  CRITICAL ISSUES FOUND!")
        return 1
    elif validator.anomalies['warning']:
        print("\n⚠️  Warnings found")
        return 0
    else:
        print("\n✓ Validation successful!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
