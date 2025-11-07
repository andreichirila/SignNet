#!/usr/bin/env python3
"""
Phoenix Landmark NPZ File Validator
Checks NPZ file contents and validates data integrity.
"""

import os
import numpy as np
import sys
from pathlib import Path
import argparse


class NPZValidator:
    """Validate Phoenix landmark NPZ files."""

    def __init__(self, npz_file):
        """
        Initialize validator.

        Args:
            npz_file: Path to NPZ file
        """
        self.npz_file = Path(npz_file)
        self.data = None
        self.status = {
            "file_exists": False,
            "file_readable": False,
            "has_landmarks": False,
            "has_glosses": False,
            "has_handedness": False,
            "landmarks_shape_valid": False,
            "glosses_valid": False,
            "handedness_valid": False,
            "no_nan_values": False,
            "no_inf_values": False,
            "hand_positions_valid": False,
            "face_positions_valid": False,
            "pose_positions_valid": False,
        }
        self.issues = []
        self.warnings = []

    def validate(self):
        """Run all validation checks."""
        print(f"\n{'='*80}")
        print(f"Validating: {self.npz_file.name}")
        print(f"{'='*80}")

        # Check 1: File exists
        if not self.npz_file.exists():
            self.issues.append(f"File not found: {self.npz_file}")
            return False
        self.status["file_exists"] = True
        print(f"✓ File exists: {self.npz_file.stat().st_size / (1024**2):.2f} MB")

        # Check 2: File is readable
        try:
            self.data = np.load(self.npz_file)
            self.status["file_readable"] = True
            print(f"✓ File is readable")
        except Exception as e:
            self.issues.append(f"Cannot read NPZ file: {e}")
            return False

        # Check 3: Required keys
        required_keys = ['landmarks', 'glosses']
        optional_keys = ['handedness']

        available_keys = list(self.data.keys())
        print(f"\nAvailable keys: {available_keys}")

        for key in required_keys:
            if key not in available_keys:
                self.issues.append(f"Missing required key: {key}")
            else:
                self.status[f"has_{key}"] = True
                print(f"  ✓ {key}")

        for key in optional_keys:
            if key in available_keys:
                self.status[f"has_{key}"] = True
                print(f"  ✓ {key} (optional)")
            else:
                self.warnings.append(f"Missing optional key: {key}")

        if self.issues:
            return False

        # Check 4: Data shapes
        print(f"\nData shapes:")
        self._check_landmarks_shape()
        self._check_glosses()
        self._check_handedness()

        # Check 5: Data quality
        print(f"\nData quality:")
        self._check_nan_inf()
        self._check_landmark_ranges()

        return len(self.issues) == 0

    def _check_landmarks_shape(self):
        """Check landmarks shape and content."""
        if 'landmarks' not in self.data:
            return

        landmarks = self.data['landmarks']
        print(f"  landmarks: {landmarks.shape} (dtype: {landmarks.dtype})")

        # Check shape
        if len(landmarks.shape) != 2:
            self.issues.append(
                f"Landmarks must be 2D array, got {len(landmarks.shape)}D"
            )
            return

        num_frames, num_features = landmarks.shape

        # Expected: 126 (hands) + 1434 (face) + 99 (pose) = 1659
        expected_features = 1659
        if num_features != expected_features:
            self.issues.append(
                f"Expected {expected_features} features, got {num_features}"
            )
            return

        if num_frames == 0:
            self.issues.append("Landmarks have 0 frames")
            return

        self.status["landmarks_shape_valid"] = True
        print(f"    ✓ Shape valid: {num_frames} frames × {num_features} features")
        print(f"    ✓ Expected: 1659 (126 hands + 1434 face + 99 pose)")

    def _check_glosses(self):
        """Check glosses (labels)."""
        if 'glosses' not in self.data:
            return

        glosses = self.data['glosses']
        print(f"  glosses: {glosses.shape} (dtype: {glosses.dtype})")
        print(f"    Labels: {list(glosses)}")

        self.status["glosses_valid"] = True
        print(f"    ✓ Glosses valid")

    def _check_handedness(self):
        """Check handedness data."""
        if 'handedness' not in self.data:
            self.warnings.append("No handedness data")
            return

        handedness = self.data['handedness']
        print(f"  handedness: {handedness.shape} (dtype: {handedness.dtype})")

        landmarks = self.data.get('landmarks')
        if landmarks is not None:
            num_frames = landmarks.shape[0]

            if handedness.shape[0] != num_frames:
                self.issues.append(
                    f"Handedness frames {handedness.shape[0]} != "
                    f"landmarks frames {num_frames}"
                )
                return

            if handedness.shape[1] != 2:
                self.issues.append(
                    f"Handedness should have 2 hands, got {handedness.shape[1]}"
                )
                return

        self.status["handedness_valid"] = True

        # Show sample handedness
        if handedness.shape[0] > 0:
            print(f"    Sample handedness:")
            print(f"      Frame 0: {handedness[0]}")
            if handedness.shape[0] > 1:
                print(f"      Frame 1: {handedness[1]}")
        print(f"    ✓ Handedness valid")

    def _check_nan_inf(self):
        """Check for NaN and Inf values."""
        if 'landmarks' not in self.data:
            return

        landmarks = self.data['landmarks'].astype(np.float32)

        nan_count = np.isnan(landmarks).sum()
        inf_count = np.isinf(landmarks).sum()

        if nan_count > 0:
            self.warnings.append(f"Found {nan_count} NaN values in landmarks")
        else:
            self.status["no_nan_values"] = True
            print(f"  ✓ No NaN values")

        if inf_count > 0:
            self.warnings.append(f"Found {inf_count} Inf values in landmarks")
        else:
            self.status["no_inf_values"] = True
            print(f"  ✓ No Inf values")

    def _check_landmark_ranges(self):
        """Check if landmark coordinates are in valid ranges."""
        if 'landmarks' not in self.data:
            return

        landmarks = self.data['landmarks'].astype(np.float32)

        # Split into components
        hand_lms = landmarks[:, :126]
        face_lms = landmarks[:, 126:1560]
        pose_lms = landmarks[:, 1560:]

        issues = []

        # Check hand landmarks (0-1 normalized coordinates)
        hand_valid = self._check_coordinate_range(hand_lms, "hand", 0, 1)
        if hand_valid:
            self.status["hand_positions_valid"] = True
        else:
            issues.append("Hand landmark coordinates out of range")

        # Check face landmarks (0-1 normalized coordinates)
        face_valid = self._check_coordinate_range(face_lms, "face", 0, 1)
        if face_valid:
            self.status["face_positions_valid"] = True
        else:
            issues.append("Face landmark coordinates out of range")

        # Check pose landmarks (0-1 normalized coordinates)
        pose_valid = self._check_coordinate_range(pose_lms, "pose", 0, 1)
        if pose_valid:
            self.status["pose_positions_valid"] = True
        else:
            issues.append("Pose landmark coordinates out of range")

        if not issues:
            print(f"  ✓ Hand landmarks in valid range [0, 1]")
            print(f"  ✓ Face landmarks in valid range [0, 1]")
            print(f"  ✓ Pose landmarks in valid range [0, 1]")
        else:
            for issue in issues:
                self.warnings.append(issue)

    def _check_coordinate_range(self, lms, name, min_val, max_val):
        """Check if coordinates are within expected range."""
        # Filter out zero coordinates (undetected landmarks)
        non_zero_mask = (lms != 0).any(axis=1)
        if non_zero_mask.sum() == 0:
            self.warnings.append(f"All {name} landmarks are zero (undetected)")
            return True

        non_zero_lms = lms[non_zero_mask]

        out_of_range = (non_zero_lms < min_val) | (non_zero_lms > max_val)
        if out_of_range.any():
            out_count = out_of_range.sum()
            return False

        return True

    def print_summary(self):
        """Print validation summary."""
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")

        passed = sum(1 for v in self.status.values() if v)
        total = len(self.status)

        print(f"\nChecks passed: {passed}/{total}")

        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.issues and not self.warnings:
            print(f"\n✅ File is VALID and ready for use!")
        elif not self.issues:
            print(f"\n⚠️  File is VALID but has warnings")
        else:
            print(f"\n❌ File has ISSUES and needs attention")

        print(f"\n{'='*80}\n")

    def get_detailed_info(self):
        """Get detailed file information."""
        if self.data is None:
            return

        print(f"\n{'='*80}")
        print("DETAILED INFORMATION")
        print(f"{'='*80}")

        if 'landmarks' in self.data:
            landmarks = self.data['landmarks'].astype(np.float32)
            print(f"\nLandmarks Statistics:")
            print(f"  Shape: {landmarks.shape}")
            print(f"  Min: {np.nanmin(landmarks):.6f}")
            print(f"  Max: {np.nanmax(landmarks):.6f}")
            print(f"  Mean: {np.nanmean(landmarks):.6f}")
            print(f"  Std: {np.nanstd(landmarks):.6f}")
            print(f"  Dtype: {landmarks.dtype}")

            # Component breakdown
            hand_lms = landmarks[:, :126]
            face_lms = landmarks[:, 126:1560]
            pose_lms = landmarks[:, 1560:]

            print(f"\n  Hand landmarks (126):")
            print(f"    Non-zero: {(hand_lms != 0).sum()} / {hand_lms.size}")

            print(f"\n  Face landmarks (1434):")
            print(f"    Non-zero: {(face_lms != 0).sum()} / {face_lms.size}")

            print(f"\n  Pose landmarks (99):")
            print(f"    Non-zero: {(pose_lms != 0).sum()} / {pose_lms.size}")

            handedness = self.data['handedness']  # (T, 2)

            print(f'Handedness labels (first 5 frames): {handedness[0:5]}')

            # Left hand: indices 0-62
            left_hand = landmarks[:, 0:63]
            left_frames = (left_hand != 0).any(axis=1).sum()

            # Right hand: indices 63-125
            right_hand = landmarks[:, 63:126]
            right_frames = (right_hand != 0).any(axis=1).sum()

            print(f'Frames with left hand: {left_frames} / {len(landmarks)}')
            print(f'Frames with right hand: {right_frames} / {len(landmarks)}')

        print(f"\n{'='*80}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Validate Phoenix landmark NPZ files'
    )
    parser.add_argument(
        'npz_file',
        type=str,
        help='Path to NPZ file to validate'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed statistics'
    )

    args = parser.parse_args()

    validator = NPZValidator(args.npz_file)
    is_valid = validator.validate()
    validator.print_summary()

    if args.detailed:
        validator.get_detailed_info()

    return 0 if is_valid else 1


if __name__ == '__main__':
    exit(main())
