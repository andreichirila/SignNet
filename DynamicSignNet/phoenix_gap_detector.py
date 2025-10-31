#!/usr/bin/env python3
"""
Phoenix Dataset Gap Detector and Reorganizer
Detects gaps in frame numbering AND splits by different video filenames.
Creates new date folders for different video segments.
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrameGapDetector:
    """Detect and reorganize frame sequences with gaps or different filenames."""

    def __init__(self, root_dir: str, gap_threshold: int = 5):
        """
        Initialize gap detector.

        Args:
            root_dir: Root directory containing class folders with date subfolders
            gap_threshold: Minimum frame number difference to consider as gap
        """
        self.root_dir = Path(root_dir)
        self.gap_threshold = gap_threshold

    def extract_frame_number(self, filename: str) -> int:
        """
        Extract frame number from filename.
        Expected format: ...fn000000-0.png
        """
        match = re.search(r'fn(\d+)', filename)
        if match:
            return int(match.group(1))
        return -1

    def extract_video_id(self, filename: str) -> str:
        """
        Extract video identifier from filename.
        Expected format: 02December_2010_Thursday_heute.avi_pid0_fn000040-0.png
        Returns: "02December_2010_Thursday_heute.avi_pid0"
        """
        # Extract everything before the frame number pattern (fn...)
        match = re.search(r'^(.+?)_fn\d+', filename)
        if match:
            return match.group(1)
        return filename

    def detect_gaps_in_folder(self, folder_path: Path) -> List[Tuple[int, int, int]]:
        """
        Detect frame numbering gaps in a folder.

        Returns:
            List of tuples: (gap_start_idx, gap_end_idx, gap_size)
        """
        png_files = sorted([f for f in folder_path.glob('*.png')])

        if len(png_files) < 2:
            return []

        frame_numbers = []

        for png_file in png_files:
            frame_num = self.extract_frame_number(png_file.name)
            if frame_num != -1:
                frame_numbers.append(frame_num)

        if len(frame_numbers) < 2:
            return []

        gaps = []
        sorted_frames = sorted(frame_numbers)

        for i in range(len(sorted_frames) - 1):
            current_frame = sorted_frames[i]
            next_frame = sorted_frames[i + 1]
            gap_size = next_frame - current_frame

            if gap_size > self.gap_threshold:
                gaps.append((i, i + 1, gap_size))
                logger.info(
                    f"Gap detected in {folder_path.name}: "
                    f"frame {current_frame} → {next_frame} (gap size: {gap_size})"
                )

        return gaps

    def detect_filename_changes_in_folder(self, folder_path: Path) -> List[int]:
        """
        Detect where video filenames change in a folder.

        Returns:
            List of indices where filename changes occur
        """
        png_files = sorted([f for f in folder_path.glob('*.png')])

        if len(png_files) < 2:
            return []

        # Extract video IDs
        video_ids = [self.extract_video_id(f.name) for f in png_files]

        # Find indices where video ID changes
        change_indices = []

        for i in range(len(video_ids) - 1):
            if video_ids[i] != video_ids[i + 1]:
                change_indices.append(i + 1)
                logger.info(
                    f"Filename change detected in {folder_path.name} at index {i + 1}: "
                    f"'{video_ids[i]}' → '{video_ids[i + 1]}'"
                )

        return change_indices

    def get_next_available_index(self, class_dir: Path) -> int:
        """
        Get the next available numerical index for a date folder.
        """
        existing_indices = []

        for item in class_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                existing_indices.append(int(item.name))

        if not existing_indices:
            return 0

        max_idx = max(existing_indices)
        next_idx = max_idx + 1

        logger.info(
            f"Found existing date folders in {class_dir.name}: {sorted(existing_indices)}. "
            f"Next available: {next_idx}"
        )

        return next_idx

    def reorganize_frames_by_filename(self, class_dir: Path, date_dir: Path, change_indices: List[int]) -> None:
        """
        Reorganize frames by splitting at filename changes.

        First segment stays in original date_dir.
        Subsequent segments go to new date folders.

        Args:
            class_dir: Parent class directory
            date_dir: Date directory containing frames
            change_indices: List of indices where filename changes
        """
        if not change_indices:
            logger.debug(f"No filename changes in {date_dir.name}")
            return

        png_files = sorted([f for f in date_dir.glob('*.png')])

        if len(png_files) == 0:
            logger.warning(f"No PNG files found in {date_dir}")
            return

        # Create segments based on filename changes
        segments = []
        start_idx = 0

        for change_idx in change_indices:
            segments.append(png_files[start_idx:change_idx])
            start_idx = change_idx

        # Add remaining frames
        segments.append(png_files[start_idx:])

        # Get starting index for new date folders
        start_folder_idx = self.get_next_available_index(class_dir)

        # First segment stays in original date_dir
        # Subsequent segments go to new date folders
        for seg_offset, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            if seg_offset == 0:
                # Keep first segment in original date_dir
                logger.info(f"Keeping first segment ({len(segment)} frames) in {date_dir.name}")
            else:
                # Create new date folder for subsequent segments
                seg_idx = start_folder_idx + seg_offset - 1
                new_date_dir = class_dir / f"{seg_idx:04d}"

                # If folder exists, find next available index
                attempt = 0
                while new_date_dir.exists() and attempt < 100:
                    seg_idx += 1
                    new_date_dir = class_dir / f"{seg_idx:04d}"
                    attempt += 1

                if attempt >= 100:
                    logger.error(
                        f"Could not find available folder index for {class_dir.name}"
                    )
                    continue

                new_date_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"Creating new date folder {seg_idx:04d} with {len(segment)} frames "
                    f"(video: {self.extract_video_id(segment[0].name)})"
                )

                for png_file in segment:
                    dest = new_date_dir / png_file.name
                    try:
                        shutil.move(str(png_file), str(dest))
                    except Exception as e:
                        logger.error(f"Failed to move {png_file.name}: {e}")

    def reorganize_frames_at_gaps(self, class_dir: Path, date_dir: Path, gaps: List[Tuple[int, int, int]]) -> None:
        """
        Reorganize frames at gaps by creating new date folders.

        First segment stays in original date_dir.
        Subsequent segments go to new date folders.

        Args:
            class_dir: Parent class directory
            date_dir: Date directory containing frames with gaps
            gaps: List of gap tuples
        """
        if not gaps:
            logger.debug(f"No gaps in {date_dir.name}")
            return

        png_files = sorted([f for f in date_dir.glob('*.png')])

        if len(png_files) == 0:
            logger.warning(f"No PNG files found in {date_dir}")
            return

        # Extract frame numbers
        frame_data = []
        for png_file in png_files:
            frame_num = self.extract_frame_number(png_file.name)
            if frame_num != -1:
                frame_data.append((frame_num, png_file))

        if len(frame_data) == 0:
            logger.warning(f"No frames with valid frame numbers in {date_dir}")
            return

        frame_data.sort(key=lambda x: x[0])

        # Create segments based on gaps
        segments = []
        current_segment = []

        for frame_num, png_file in frame_data:
            current_segment.append(png_file)

            # Check if we need to split at a gap
            for gap_start_idx, gap_end_idx, gap_size in gaps:
                if len(current_segment) == gap_start_idx + 1:
                    segments.append(current_segment)
                    current_segment = []
                    break

        # Add remaining frames
        if current_segment:
            segments.append(current_segment)

        # Get starting index for new date folders
        start_folder_idx = self.get_next_available_index(class_dir)

        # First segment stays in original date_dir
        # Subsequent segments go to new date folders
        for seg_offset, segment in enumerate(segments):
            if len(segment) == 0:
                continue

            if seg_offset == 0:
                logger.info(f"Keeping first segment ({len(segment)} frames) in {date_dir.name}")
            else:
                # Create new date folder for subsequent segments
                seg_idx = start_folder_idx + seg_offset - 1
                new_date_dir = class_dir / f"{seg_idx:04d}"

                # If folder exists, find next available index
                attempt = 0
                while new_date_dir.exists() and attempt < 100:
                    seg_idx += 1
                    new_date_dir = class_dir / f"{seg_idx:04d}"
                    attempt += 1

                if attempt >= 100:
                    logger.error(
                        f"Could not find available folder index for {class_dir.name}"
                    )
                    continue

                new_date_dir.mkdir(parents=True, exist_ok=True)

                logger.info(
                    f"Creating new date folder {seg_idx:04d} with {len(segment)} frames"
                )

                for png_file in segment:
                    dest = new_date_dir / png_file.name
                    try:
                        shutil.move(str(png_file), str(dest))
                    except Exception as e:
                        logger.error(f"Failed to move {png_file.name}: {e}")

    def scan_all_folders_by_filename(self, move_files: bool = False) -> Dict[str, List]:
        """
        Scan all date subfolders and detect filename changes.

        Args:
            move_files: If True, reorganize files; if False, only report

        Returns:
            Dictionary mapping folder paths to filename change indices
        """
        results = {}

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name == "dataset_summary.txt":
                continue

            class_name = class_dir.name
            logger.info(f"\n=== Scanning class: {class_name} (by filename) ===")

            for date_dir in sorted(class_dir.iterdir()):
                if not date_dir.is_dir():
                    continue

                # Check if already fully reorganized
                has_png_files = any(f.suffix == '.png' for f in date_dir.iterdir())

                if not has_png_files:
                    logger.info(f"Skipping {date_dir.name} (already fully reorganized)")
                    continue

                changes = self.detect_filename_changes_in_folder(date_dir)

                if changes:
                    logger.warning(
                        f"{date_dir.name}: Found {len(changes)} filename change(s)"
                    )
                    results[str(date_dir)] = changes

                    if move_files:
                        logger.info(f"Reorganizing {date_dir.name}...")
                        self.reorganize_frames_by_filename(class_dir, date_dir, changes)
                else:
                    logger.info(f"{date_dir.name}: No filename changes detected ✓")

        return results

    def scan_all_folders_by_gaps(self, move_files: bool = False) -> Dict[str, List]:
        """
        Scan all date subfolders and detect gaps.

        Args:
            move_files: If True, reorganize files; if False, only report

        Returns:
            Dictionary mapping folder paths to detected gaps
        """
        results = {}

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name == "dataset_summary.txt":
                continue

            class_name = class_dir.name
            logger.info(f"\n=== Scanning class: {class_name} (by gaps) ===")

            for date_dir in sorted(class_dir.iterdir()):
                if not date_dir.is_dir():
                    continue

                has_png_files = any(f.suffix == '.png' for f in date_dir.iterdir())

                if not has_png_files:
                    logger.info(f"Skipping {date_dir.name} (already fully reorganized)")
                    continue

                gaps = self.detect_gaps_in_folder(date_dir)

                if gaps:
                    logger.warning(
                        f"{date_dir.name}: Found {len(gaps)} gap(s)"
                    )
                    results[str(date_dir)] = gaps

                    if move_files:
                        logger.info(f"Reorganizing {date_dir.name}...")
                        self.reorganize_frames_at_gaps(class_dir, date_dir, gaps)
                else:
                    logger.info(f"{date_dir.name}: No gaps detected ✓")

        return results

    def generate_report(self, results: Dict[str, List], report_file: str, report_type: str) -> None:
        """
        Generate a report.

        Args:
            results: Results dictionary
            report_file: Output report filename
            report_type: "filename" or "gaps"
        """
        with open(report_file, 'w') as f:
            title = f"Phoenix Dataset {report_type.upper()} Detection Report"
            f.write(title + "\n")
            f.write("=" * 80 + "\n\n")

            total_issues = sum(len(v) for v in results.values())
            f.write(f"Total folders with issues: {len(results)}\n")
            f.write(f"Total issues detected: {total_issues}\n\n")

            f.write("Details:\n")
            f.write("-" * 80 + "\n")

            for folder_path, issues in sorted(results.items()):
                f.write(f"\n{folder_path}\n")
                if report_type == "filename":
                    for idx in issues:
                        f.write(f"  Filename change at index {idx}\n")
                else:
                    for gap_start_idx, gap_end_idx, gap_size in issues:
                        f.write(f"  Gap at index {gap_start_idx}: size {gap_size}\n")

        logger.info(f"Report saved to {report_file}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect and reorganize frame sequences by filename changes or gaps'
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        required=True,
        help='Root directory containing class folders with date subfolders'
    )
    parser.add_argument(
        '--gap-threshold',
        type=int,
        default=5,
        help='Minimum frame gap size to consider as discontinuity (default: 5)'
    )
    parser.add_argument(
        '--split-by',
        type=str,
        choices=['filename', 'gaps', 'both'],
        default='filename',
        help='What to split by: filename, gaps, or both (default: filename)'
    )
    parser.add_argument(
        '--reorganize',
        action='store_true',
        help='Actually move files; without this, only reports'
    )

    args = parser.parse_args()

    try:
        detector = FrameGapDetector(args.root_dir, gap_threshold=args.gap_threshold)

        logger.info(f"Scanning {args.root_dir}...")

        if args.split_by in ['filename', 'both']:
            logger.info("\n" + "="*80)
            logger.info("PHASE 1: FILENAME-BASED SPLITTING")
            logger.info("="*80)
            results_filename = detector.scan_all_folders_by_filename(move_files=args.reorganize)
            detector.generate_report(results_filename, 'filename_report.txt', 'filename')

        if args.split_by in ['gaps', 'both']:
            logger.info("\n" + "="*80)
            logger.info("PHASE 2: GAP-BASED SPLITTING")
            logger.info("="*80)
            results_gaps = detector.scan_all_folders_by_gaps(move_files=args.reorganize)
            detector.generate_report(results_gaps, 'gap_report.txt', 'gaps')

        if args.reorganize:
            logger.info(f"\nReorganization complete!")
        else:
            logger.info(f"\nDetection complete. Run with --reorganize to apply changes.")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
