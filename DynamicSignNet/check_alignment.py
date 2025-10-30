#!/usr/bin/env python3
"""
Gloss-to-Sequence Alignment Checker
Specialized tool to verify glosses belong to correct video sequences

This script focuses specifically on alignment issues between landmark
sequences and their corresponding gloss annotations.

Usage:
    python check_alignment.py --landmarks_dir ./landmarks_train \
                              --vocab_file vocab.json \
                              --output alignment_report.json
"""

import os
import glob
import numpy as np
import json
from collections import defaultdict
import re


class GlossAlignmentChecker:
    """Check gloss-to-sequence alignment correctness"""

    def __init__(self, landmarks_dir, vocab_file=None):
        self.landmarks_dir = landmarks_dir
        self.vocab_file = vocab_file
        self.samples = sorted(glob.glob(os.path.join(landmarks_dir, "*.npz")))
        self.vocab = self._load_vocab() if vocab_file else None
        self.issues = []

    def _load_vocab(self):
        """Load vocabulary"""
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'r') as f:
                return json.load(f)
        return None

    def check_filename_consistency(self):
        """Check filename-gloss consistency"""
        print("\n[1/5] Checking filename-gloss consistency...")

        inconsistencies = []

        for sample_path in self.samples[:100]:
            try:
                data = np.load(sample_path, allow_pickle=True)
                glosses = data['glosses']

                filename = os.path.basename(sample_path)
                filename_parts = re.split(r'[_\-\.]', filename.lower())

                gloss_strs = [str(g).lower() for g in glosses]
                found = any(part in gloss_strs for part in filename_parts)

                if not found and len(glosses) < 20:
                    inconsistencies.append({
                        'file': filename,
                        'glosses': list(glosses[:5]),
                        'message': 'Glosses not in filename'
                    })

            except Exception as e:
                self.issues.append({
                    'check': 'filename_consistency',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        if inconsistencies:
            print(f"  ℹ️  {len(inconsistencies)} files where glosses don't match filename")
        else:
            print("  ✓ Filename patterns consistent")

    def check_temporal_consistency(self):
        """Check temporal consistency"""
        print("\n[2/5] Checking temporal consistency...")

        temporal_issues = []

        for sample_path in self.samples:
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']

                num_frames = landmarks.shape[0]
                num_glosses = len(glosses)

                if num_glosses == 0:
                    temporal_issues.append({
                        'file': os.path.basename(sample_path),
                        'issue': 'empty_gloss_sequence',
                        'frames': num_frames
                    })
                    continue

                frames_per_gloss = num_frames / num_glosses

                if frames_per_gloss < 3:
                    temporal_issues.append({
                        'file': os.path.basename(sample_path),
                        'issue': 'too_few_frames_per_gloss',
                        'frames': num_frames,
                        'glosses': num_glosses,
                        'ratio': round(frames_per_gloss, 2)
                    })
                elif frames_per_gloss > 300:
                    temporal_issues.append({
                        'file': os.path.basename(sample_path),
                        'issue': 'too_many_frames_per_gloss',
                        'frames': num_frames,
                        'glosses': num_glosses,
                        'ratio': round(frames_per_gloss, 2)
                    })

            except Exception as e:
                self.issues.append({
                    'check': 'temporal_consistency',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        if temporal_issues:
            print(f"  ⚠️  Found {len(temporal_issues)} temporal issues")
            for issue in temporal_issues[:3]:
                print(f"      • {issue['file']}: {issue['issue']}")
        else:
            print("  ✓ Temporal consistency OK")

        return temporal_issues

    def check_vocabulary_mapping(self):
        """Verify vocabulary mapping"""
        print("\n[3/5] Checking vocabulary mapping...")

        if not self.vocab:
            print("  ⚠️  No vocabulary, skipping")
            return []

        mapping_issues = []
        vocab_size = len(self.vocab)
        idx_to_gloss = {v: k for k, v in self.vocab.items()}

        for sample_path in self.samples:
            try:
                data = np.load(sample_path, allow_pickle=True)
                glosses = data['glosses']

                for i, gloss in enumerate(glosses):
                    gloss_str = str(gloss)

                    if gloss_str not in self.vocab:
                        mapping_issues.append({
                            'file': os.path.basename(sample_path),
                            'position': i,
                            'gloss': gloss_str,
                            'issue': 'not_in_vocabulary'
                        })
                    else:
                        gloss_id = self.vocab[gloss_str]

                        if gloss_id < 0:
                            mapping_issues.append({
                                'file': os.path.basename(sample_path),
                                'position': i,
                                'gloss': gloss_str,
                                'id': gloss_id,
                                'issue': 'negative_id'
                            })
                        elif gloss_id >= vocab_size:
                            mapping_issues.append({
                                'file': os.path.basename(sample_path),
                                'position': i,
                                'gloss': gloss_str,
                                'id': gloss_id,
                                'issue': 'id_exceeds_vocab_size',
                                'vocab_size': vocab_size
                            })

                        if gloss_id in idx_to_gloss and idx_to_gloss[gloss_id] != gloss_str:
                            mapping_issues.append({
                                'file': os.path.basename(sample_path),
                                'position': i,
                                'gloss': gloss_str,
                                'id': gloss_id,
                                'issue': 'reverse_mapping_mismatch',
                                'mapped_to': idx_to_gloss[gloss_id]
                            })

            except Exception as e:
                self.issues.append({
                    'check': 'vocabulary_mapping',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        if mapping_issues:
            print(f"  🔴 CRITICAL: {len(mapping_issues)} mapping issues!")
            issue_types = defaultdict(int)
            for issue in mapping_issues:
                issue_types[issue['issue']] += 1
            for issue_type, count in issue_types.items():
                print(f"      • {issue_type}: {count}")
        else:
            print("  ✓ Vocabulary mapping correct")

        return mapping_issues

    def check_sequence_ordering(self):
        """Check gloss ordering"""
        print("\n[4/5] Checking sequence ordering...")

        ordering_issues = []
        special_tokens = ['<pad>', '<blank>', '<sos>', '<eos>', '<unk>']

        for sample_path in self.samples:
            try:
                data = np.load(sample_path, allow_pickle=True)
                glosses = data['glosses']
                gloss_list = [str(g) for g in glosses]

                for i, gloss in enumerate(gloss_list):
                    if gloss in special_tokens:
                        if gloss == '<sos>' and i != 0:
                            ordering_issues.append({
                                'file': os.path.basename(sample_path),
                                'issue': 'sos_not_at_start',
                                'position': i
                            })
                        elif gloss == '<eos>' and i != len(gloss_list) - 1:
                            ordering_issues.append({
                                'file': os.path.basename(sample_path),
                                'issue': 'eos_not_at_end',
                                'position': i
                            })
                        elif gloss == '<pad>' and i < len(gloss_list) - 1:
                            ordering_issues.append({
                                'file': os.path.basename(sample_path),
                                'issue': 'pad_in_middle',
                                'position': i
                            })

                # Check repetitions
                for i in range(len(gloss_list) - 2):
                    if gloss_list[i] == gloss_list[i+1] == gloss_list[i+2]:
                        if gloss_list[i] not in special_tokens:
                            ordering_issues.append({
                                'file': os.path.basename(sample_path),
                                'issue': 'triple_repetition',
                                'gloss': gloss_list[i],
                                'position': i
                            })
                            break

            except Exception as e:
                self.issues.append({
                    'check': 'sequence_ordering',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        if ordering_issues:
            print(f"  ⚠️  {len(ordering_issues)} ordering issues")
            issue_types = defaultdict(int)
            for issue in ordering_issues:
                issue_types[issue['issue']] += 1
            for issue_type, count in issue_types.items():
                print(f"      • {issue_type}: {count}")
        else:
            print("  ✓ Sequence ordering correct")

        return ordering_issues

    def check_landmark_correlation(self):
        """Check landmark-gloss correlation"""
        print("\n[5/5] Checking landmark correlation...")

        correlation_issues = []

        for sample_path in self.samples[:50]:
            try:
                data = np.load(sample_path, allow_pickle=True)
                landmarks = data['landmarks']
                glosses = data['glosses']

                if len(glosses) == 0:
                    continue

                # Frame-to-frame variation
                frame_diffs = np.diff(landmarks, axis=0)
                frame_variations = np.linalg.norm(frame_diffs, axis=1)

                # Estimate boundaries
                num_frames = landmarks.shape[0]
                num_glosses = len(glosses)
                frames_per_gloss = num_frames / num_glosses

                expected_boundaries = [int(i * frames_per_gloss) for i in range(1, num_glosses)]
                boundary_window = max(2, int(frames_per_gloss * 0.1))

                low_motion_boundaries = 0
                for boundary in expected_boundaries:
                    if boundary < len(frame_variations):
                        start = max(0, boundary - boundary_window)
                        end = min(len(frame_variations), boundary + boundary_window)

                        window_motion = frame_variations[start:end]
                        avg_motion = frame_variations.mean()

                        if window_motion.mean() < avg_motion * 0.3:
                            low_motion_boundaries += 1

                if low_motion_boundaries > len(expected_boundaries) * 0.6:
                    correlation_issues.append({
                        'file': os.path.basename(sample_path),
                        'issue': 'low_motion_at_boundaries',
                        'low_motion_count': low_motion_boundaries,
                        'total_boundaries': len(expected_boundaries)
                    })

            except Exception as e:
                self.issues.append({
                    'check': 'landmark_correlation',
                    'file': os.path.basename(sample_path),
                    'error': str(e)
                })

        if correlation_issues:
            print(f"  ⚠️  {len(correlation_issues)} correlation issues")
        else:
            print("  ✓ Landmark correlation OK")

        return correlation_issues

    def run_all_checks(self):
        """Run all checks"""
        print("="*70)
        print("GLOSS-TO-SEQUENCE ALIGNMENT VERIFICATION")
        print("="*70)
        print(f"Dataset: {self.landmarks_dir}")
        print(f"Samples: {len(self.samples)}")
        print("="*70)

        results = {}

        self.check_filename_consistency()
        results['temporal'] = self.check_temporal_consistency()
        results['vocabulary'] = self.check_vocabulary_mapping()
        results['ordering'] = self.check_sequence_ordering()
        results['correlation'] = self.check_landmark_correlation()

        # Summary
        print("\n" + "="*70)
        print("ALIGNMENT CHECK SUMMARY")
        print("="*70)

        total_issues = sum(len(issues) for issues in results.values())

        if total_issues == 0:
            print("\n✓ All alignment checks passed!")
            print("  Glosses correctly aligned with sequences.")
        else:
            print(f"\n⚠️  Found {total_issues} potential issues:")
            for check, issues in results.items():
                if issues:
                    print(f"  • {check}: {len(issues)} issues")

        print("="*70)

        return results

    def save_report(self, output_file="alignment_report.json"):
        """Save report"""
        results = self.run_all_checks()

        report = {
            'dataset': self.landmarks_dir,
            'total_samples': len(self.samples),
            'alignment_checks': results,
            'processing_issues': self.issues
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Check alignment')
    parser.add_argument('--landmarks_dir', required=True, help='Landmarks dir')
    parser.add_argument('--vocab_file', default=None, help='Vocab file')
    parser.add_argument('--output', default='alignment_report.json', help='Output')

    args = parser.parse_args()

    checker = GlossAlignmentChecker(args.landmarks_dir, args.vocab_file)
    checker.save_report(args.output)


if __name__ == "__main__":
    main()
