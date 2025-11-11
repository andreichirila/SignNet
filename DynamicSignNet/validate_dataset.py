#!/usr/bin/env python3
"""
Dataset Validation and Anomaly Detection Script
For Sign Language Translation Dataset

Enhanced with comprehensive distribution plotting.
"""

import os
import glob
import numpy as np
import json
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


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
            'gloss_frame_stats': defaultdict(list),
            'frames_per_gloss_all': [],
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

        self._check_file_integrity()
        self._check_data_structure()
        self._check_landmark_validity()
        self._check_gloss_sequence_alignment()
        self._check_vocabulary_coverage()
        self._check_statistical_outliers()
        self._check_duplicate_sequences()

        self._print_summary()

        return self.anomalies, self.stats

    def _check_file_integrity(self):
        """Check if all NPZ files can be loaded"""
        print("\n[1/7] Checking file integrity...")

        corrupted_files = []
        for sample_path in tqdm(self.samples, desc="Loading files"):
            try:
                data = np.load(sample_path, allow_pickle=True)
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

        expected_feature_dim = 1659

        for sample_path in tqdm(self.samples, desc="Validating structure"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']

                if landmarks.ndim != 2:
                    self.anomalies['critical'].append({
                        'type': 'invalid_landmark_dims',
                        'file': os.path.basename(sample_path),
                        'expected': 2,
                        'actual': landmarks.ndim,
                        'shape': landmarks.shape
                    })

                if landmarks.shape[1] != expected_feature_dim:
                    self.anomalies['warning'].append({
                        'type': 'unexpected_feature_dim',
                        'file': os.path.basename(sample_path),
                        'expected': expected_feature_dim,
                        'actual': landmarks.shape[1]
                    })

                if not isinstance(glosses, (list, np.ndarray)):
                    self.anomalies['critical'].append({
                        'type': 'invalid_gloss_type',
                        'file': os.path.basename(sample_path),
                        'type_found': type(glosses).__name__
                    })

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

                if np.isnan(landmarks).any():
                    nan_count = np.isnan(landmarks).sum()
                    self.anomalies['critical'].append({
                        'type': 'nan_values',
                        'file': os.path.basename(sample_path),
                        'nan_count': int(nan_count),
                        'total_values': landmarks.size
                    })
                    invalid_count += 1

                if np.isinf(landmarks).any():
                    inf_count = np.isinf(landmarks).sum()
                    self.anomalies['critical'].append({
                        'type': 'inf_values',
                        'file': os.path.basename(sample_path),
                        'inf_count': int(inf_count),
                        'total_values': landmarks.size
                    })
                    invalid_count += 1

                min_val, max_val = landmarks.min(), landmarks.max()
                self.stats['landmark_ranges'].append((min_val, max_val))

                if min_val < -100 or max_val > 100:
                    self.anomalies['warning'].append({
                        'type': 'extreme_landmark_values',
                        'file': os.path.basename(sample_path),
                        'min': float(min_val),
                        'max': float(max_val)
                    })

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
        print("\n[4/7] Checking gloss-sequence alignment and collecting per-gloss stats...")

        misaligned_count = 0

        for sample_path in tqdm(self.samples, desc="Checking alignment"):
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']

                num_frames = landmarks.shape[0]
                num_glosses = len(glosses)

                ratio = num_frames / num_glosses if num_glosses > 0 else 0
                self.stats['sequence_gloss_ratio'].append(ratio)

                if num_glosses > 0:
                    avg_frames_per_gloss = num_frames / num_glosses
                    self.stats['frames_per_gloss_all'].append(avg_frames_per_gloss)

                    for gloss in glosses:
                        gloss_str = str(gloss)
                        self.stats['gloss_frame_stats'][gloss_str].append(avg_frames_per_gloss)

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
                    self.stats['gloss_distribution'][gloss_str] += 1

                    if gloss_str not in self.vocab:
                        unknown_glosses.add(gloss_str)
                        self.anomalies['warning'].append({
                            'type': 'unknown_gloss',
                            'file': os.path.basename(sample_path),
                            'gloss': gloss_str
                        })
                    else:
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

    def plot_distributions(self, output_dir="./plots"):
        """Generate comprehensive distribution plots"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*70)
        print("GENERATING DISTRIBUTION PLOTS")
        print("="*70)
        print(f"Output directory: {output_dir}/\n")

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. Top 30 gloss frequency bar chart
        plt.figure(figsize=(16, 8))
        top_glosses = dict(self.stats['gloss_distribution'].most_common(30))
        bars = plt.bar(range(len(top_glosses)), list(top_glosses.values()),
                       color='steelblue', edgecolor='black', linewidth=0.7)
        plt.xticks(range(len(top_glosses)), list(top_glosses.keys()),
                   rotation=45, ha='right', fontsize=10)
        plt.xlabel('Gloss', fontsize=13, fontweight='bold')
        plt.ylabel('Frequency', fontsize=13, fontweight='bold')
        plt.title('Top 30 Most Frequent Glosses', fontsize=15, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gloss_frequency_top30.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: gloss_frequency_top30.png")

        # 2. Full gloss frequency (log scale)
        plt.figure(figsize=(14, 7))
        frequencies = sorted(self.stats['gloss_distribution'].values(), reverse=True)
        plt.plot(range(len(frequencies)), frequencies, linewidth=2.5,
                color='darkblue', marker='o', markersize=3, alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Gloss Rank', fontsize=13, fontweight='bold')
        plt.ylabel('Frequency (log scale)', fontsize=13, fontweight='bold')
        plt.title('Gloss Frequency Distribution - Zipf\'s Law',
                 fontsize=15, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/gloss_frequency_all_log.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: gloss_frequency_all_log.png")

        # 3. Frames per gloss histogram
        if self.stats['frames_per_gloss_all']:
            plt.figure(figsize=(14, 7))
            fpg = np.array(self.stats['frames_per_gloss_all'])
            n, bins, patches = plt.hist(fpg, bins=60, color='forestgreen',
                                       alpha=0.75, edgecolor='black', linewidth=0.8)
            plt.axvline(fpg.mean(), color='red', linestyle='--', linewidth=2.5,
                       label=f'Mean: {fpg.mean():.2f}', alpha=0.8)
            plt.axvline(np.median(fpg), color='orange', linestyle='--', linewidth=2.5,
                       label=f'Median: {np.median(fpg):.2f}', alpha=0.8)
            plt.xlabel('Frames per Gloss', fontsize=13, fontweight='bold')
            plt.ylabel('Frequency', fontsize=13, fontweight='bold')
            plt.title('Distribution of Frames per Gloss', fontsize=15, fontweight='bold', pad=20)
            plt.legend(fontsize=12, loc='upper right')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/frames_per_gloss_histogram.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: frames_per_gloss_histogram.png")

        # 4. Average frames per top 30 glosses
        plt.figure(figsize=(16, 8))
        top_glosses_list = [g for g, c in self.stats['gloss_distribution'].most_common(30)]
        avg_frames = [np.mean(self.stats['gloss_frame_stats'][g])
                     for g in top_glosses_list if g in self.stats['gloss_frame_stats']]

        if avg_frames:
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(avg_frames)))
            bars = plt.bar(range(len(avg_frames)), avg_frames, color=colors,
                          edgecolor='black', linewidth=0.7)
            plt.xticks(range(len(top_glosses_list[:len(avg_frames)])),
                      top_glosses_list[:len(avg_frames)], rotation=45, ha='right', fontsize=10)
            plt.xlabel('Gloss', fontsize=13, fontweight='bold')
            plt.ylabel('Average Frames', fontsize=13, fontweight='bold')
            plt.title('Average Frame Duration per Gloss (Top 30)',
                     fontsize=15, fontweight='bold', pad=20)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/avg_frames_per_gloss_top30.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: avg_frames_per_gloss_top30.png")

        # 5. Sequence length distribution
        if self.stats['sequence_lengths']:
            plt.figure(figsize=(14, 7))
            n, bins, patches = plt.hist(self.stats['sequence_lengths'], bins=60,
                                       color='coral', alpha=0.75, edgecolor='black', linewidth=0.8)
            mean_len = np.mean(self.stats['sequence_lengths'])
            median_len = np.median(self.stats['sequence_lengths'])
            plt.axvline(mean_len, color='red', linestyle='--', linewidth=2.5,
                       label=f'Mean: {mean_len:.1f}', alpha=0.8)
            plt.axvline(median_len, color='orange', linestyle='--', linewidth=2.5,
                       label=f'Median: {median_len:.1f}', alpha=0.8)
            plt.xlabel('Sequence Length (frames)', fontsize=13, fontweight='bold')
            plt.ylabel('Frequency', fontsize=13, fontweight='bold')
            plt.title('Distribution of Sequence Lengths', fontsize=15, fontweight='bold', pad=20)
            plt.legend(fontsize=12, loc='upper right')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sequence_length_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: sequence_length_distribution.png")

        # 6. Gloss count per sequence
        if self.stats['gloss_counts']:
            plt.figure(figsize=(14, 7))
            n, bins, patches = plt.hist(self.stats['gloss_counts'], bins=40,
                                       color='mediumpurple', alpha=0.75,
                                       edgecolor='black', linewidth=0.8)
            mean_gc = np.mean(self.stats['gloss_counts'])
            median_gc = np.median(self.stats['gloss_counts'])
            plt.axvline(mean_gc, color='red', linestyle='--', linewidth=2.5,
                       label=f'Mean: {mean_gc:.1f}', alpha=0.8)
            plt.axvline(median_gc, color='orange', linestyle='--', linewidth=2.5,
                       label=f'Median: {median_gc:.1f}', alpha=0.8)
            plt.xlabel('Glosses per Sequence', fontsize=13, fontweight='bold')
            plt.ylabel('Frequency', fontsize=13, fontweight='bold')
            plt.title('Distribution of Gloss Count per Sequence',
                     fontsize=15, fontweight='bold', pad=20)
            plt.legend(fontsize=12, loc='upper right')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/gloss_count_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: gloss_count_distribution.png")

        # 7. Frequency vs Duration scatter
        gloss_freq_vs_frames = []
        for gloss in self.stats['gloss_frame_stats'].keys():
            freq = self.stats['gloss_distribution'][gloss]
            avg_frames = np.mean(self.stats['gloss_frame_stats'][gloss])
            gloss_freq_vs_frames.append((freq, avg_frames))

        if gloss_freq_vs_frames:
            plt.figure(figsize=(14, 8))
            freqs, avg_frames_list = zip(*gloss_freq_vs_frames)
            scatter = plt.scatter(freqs, avg_frames_list, alpha=0.6, s=60,
                                c=np.log1p(freqs), cmap='viridis',
                                edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, label='log(Frequency + 1)')
            plt.xscale('log')
            plt.xlabel('Gloss Frequency (log scale)', fontsize=13, fontweight='bold')
            plt.ylabel('Average Frames per Gloss', fontsize=13, fontweight='bold')
            plt.title('Gloss Frequency vs Average Duration',
                     fontsize=15, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/frequency_vs_duration_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: frequency_vs_duration_scatter.png")

        print(f"\n✓ All {7} plots saved to {output_dir}/")
        print("="*70)

    def _print_summary(self):
        """Print validation summary"""
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

            if self.stats['frames_per_gloss_all']:
                fpg = np.array(self.stats['frames_per_gloss_all'])
                print(f"\nFrames per gloss (global):")
                print(f"  Min: {fpg.min():.2f}")
                print(f"  Max: {fpg.max():.2f}")
                print(f"  Mean: {fpg.mean():.2f}")
                print(f"  Median: {np.median(fpg):.2f}")
                print(f"  Std: {fpg.std():.2f}")

        if self.stats['gloss_distribution']:
            print("\n" + "="*70)
            print("GLOSS STATISTICS")
            print("="*70)
            print(f"Unique glosses: {len(self.stats['gloss_distribution'])}")

            print("\nTop 10 most frequent glosses:")
            print(f"{'Rank':<6} {'Gloss':<30} {'Count':<10} {'Avg Frames':<12}")
            print("-" * 68)
            for i, (gloss, count) in enumerate(self.stats['gloss_distribution'].most_common(10), 1):
                if gloss in self.stats['gloss_frame_stats']:
                    avg_frames = np.mean(self.stats['gloss_frame_stats'][gloss])
                    print(f"{i:<6} {gloss:<30} {count:>6,}     {avg_frames:>8.2f}")
                else:
                    print(f"{i:<6} {gloss:<30} {count:>6,}     {'N/A':>8}")

            print("\nGlosses with unusual average frame counts:")
            gloss_avg_frames = {}
            for gloss, frame_list in self.stats['gloss_frame_stats'].items():
                if len(frame_list) >= 3:
                    gloss_avg_frames[gloss] = np.mean(frame_list)

            if gloss_avg_frames:
                sorted_by_frames = sorted(gloss_avg_frames.items(), key=lambda x: x[1])

                print("\n  Shortest duration (top 5):")
                for gloss, avg_frames in sorted_by_frames[:5]:
                    count = self.stats['gloss_distribution'][gloss]
                    print(f"    {gloss:<30} {avg_frames:>8.2f} frames (n={count})")

                print("\n  Longest duration (top 5):")
                for gloss, avg_frames in sorted_by_frames[-5:]:
                    count = self.stats['gloss_distribution'][gloss]
                    print(f"    {gloss:<30} {avg_frames:>8.2f} frames (n={count})")

        print("\n" + "="*70)

    def save_report(self, output_file="validation_report.json"):
        """Save validation report to JSON"""

        gloss_stats = {}
        for gloss, frame_list in self.stats['gloss_frame_stats'].items():
            if frame_list:
                gloss_stats[gloss] = {
                    'count': self.stats['gloss_distribution'][gloss],
                    'avg_frames': float(np.mean(frame_list)),
                    'min_frames': float(np.min(frame_list)),
                    'max_frames': float(np.max(frame_list)),
                    'std_frames': float(np.std(frame_list))
                }

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
                'frames_per_gloss_stats': {
                    'min': float(np.min(self.stats['frames_per_gloss_all'])) if self.stats['frames_per_gloss_all'] else 0,
                    'max': float(np.max(self.stats['frames_per_gloss_all'])) if self.stats['frames_per_gloss_all'] else 0,
                    'mean': float(np.mean(self.stats['frames_per_gloss_all'])) if self.stats['frames_per_gloss_all'] else 0,
                    'median': float(np.median(self.stats['frames_per_gloss_all'])) if self.stats['frames_per_gloss_all'] else 0,
                    'std': float(np.std(self.stats['frames_per_gloss_all'])) if self.stats['frames_per_gloss_all'] else 0,
                },
                'unique_glosses': len(self.stats['gloss_distribution']),
                'top_glosses': dict(self.stats['gloss_distribution'].most_common(50)),
                'per_gloss_stats': gloss_stats
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate dataset and generate plots')
    parser.add_argument('--landmarks_dir', required=True, help='Landmarks directory')
    parser.add_argument('--vocab_file', default=None, help='Vocabulary file')
    parser.add_argument('--output', default='validation_report.json', help='Output JSON file')
    parser.add_argument('--plots', default='./plots', help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')

    args = parser.parse_args()

    validator = DatasetValidator(args.landmarks_dir, args.vocab_file)
    validator.validate_all()
    validator.save_report(args.output)

    # Generate plots unless disabled
    if not args.no_plots:
        validator.plot_distributions(args.plots)

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
