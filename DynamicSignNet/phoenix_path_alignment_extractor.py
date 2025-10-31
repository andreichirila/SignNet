#!/usr/bin/env python3
"""
Phoenix Weather Dataset Frame Extractor (PNG path-based)
Extracts frames organized by training classes from alignment files.
Alignment file format: path_to_png label_id
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhoenixPathAlignmentExtractor:
    """Extract and organize PNG frames from Phoenix dataset using path-based alignments."""

    def __init__(self, dataset_root: str, output_dir: str):
        """
        Initialize the extractor.

        Args:
            dataset_root: Path to Phoenix dataset root directory
            output_dir: Output directory for organized frames
        """
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset root: {self.dataset_root}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_training_classes(self, classes_file: str) -> Dict[int, str]:
        """
        Load training classes from trainingclasses.txt.
        Maps label IDs to class names (glosses).
        Format: GLOSS_NAME LABEL_ID

        Args:
            classes_file: Path to trainingclasses.txt

        Returns:
            Dictionary mapping label ID (int) to gloss name (str)
        """
        classes_path = self.dataset_root / classes_file

        if not classes_path.exists():
            raise FileNotFoundError(f"Training classes file not found: {classes_path}")

        classes = {}

        with open(classes_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    gloss = parts[0]
                    try:
                        label_id = int(parts[-1])  # Take last part as ID
                        gloss_name = re.sub(r'\d+$', '', gloss.split()[0])  # "OKTOBER2" → "OKTOBER"
                        classes[label_id] = gloss_name
                    except ValueError:
                        logger.warning(f"Could not parse label ID from line: {line}")
                        continue

        logger.info(f"Loaded {len(classes)} training classes")
        return classes



    def parse_alignment_file(self, alignment_file: str) -> List[Dict]:
        """
        Parse alignment file to extract PNG paths and corresponding labels.

        Expected format (one per line):
        path/to/png/file.png label_id
        Example: features/fullFrame-210x260px/train/01April_2010_Thursday_heute_default-8/1/01April_2010_Thursday_heute.avi_pid0_fn000110-0.png 3366

        Args:
            alignment_file: Path to alignment file

        Returns:
            List of dictionaries with keys:
            - 'png_path': relative path to PNG file
            - 'label_id': integer label ID
        """
        alignment_path = self.dataset_root / alignment_file

        if not alignment_path.exists():
            raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

        alignments = []

        with open(alignment_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    # Split from right to separate label from path
                    parts = line.rsplit(maxsplit=1)

                    if len(parts) != 2:
                        logger.warning(
                            f"Skipping malformed line {line_num}: {line}"
                        )
                        continue

                    png_path = parts[0]
                    label_id = int(parts[1])

                    alignments.append({
                        'png_path': png_path,
                        'label_id': label_id
                    })

                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Error parsing line {line_num}: {line} - {e}"
                    )
                    continue

        logger.info(f"Parsed {len(alignments)} alignments from {alignment_file}")
        return alignments

    def group_frames_by_class(
        self,
        alignments: List[Dict],
        label_to_gloss: Dict[int, str]
    ) -> Dict[str, List[str]]:
        """
        Group PNG paths by training class (gloss).

        Args:
            alignments: List of alignment dictionaries
            label_to_gloss: Dictionary mapping label ID to gloss name

        Returns:
            Dictionary mapping class name to list of PNG paths
        """
        class_frames = {}

        for alignment in alignments:
            label_id = alignment['label_id']
            png_path = alignment['png_path']

            # Get gloss name from label ID
            if label_id not in label_to_gloss:
                logger.warning(
                    f"Label ID {label_id} not found in training classes"
                )
                continue

            gloss = label_to_gloss[label_id]

            if gloss not in class_frames:
                class_frames[gloss] = []

            class_frames[gloss].append(png_path)

        # Log statistics
        logger.info(
            f"Grouped into {len(class_frames)} classes with data"
        )

        for gloss in sorted(class_frames.keys()):
            logger.debug(f"  {gloss}: {len(class_frames[gloss])} frames")

        return class_frames

    def verify_png_files(self, class_frames: Dict[str, List[str]]) -> Tuple[int, int]:
        """
        Verify that all PNG files in alignments exist.

        Args:
            class_frames: Dictionary mapping class to PNG paths

        Returns:
            Tuple of (total_files, missing_files)
        """
        total_files = 0
        missing_files = 0

        for gloss, png_paths in class_frames.items():
            for png_path in png_paths:
                full_path = self.dataset_root / png_path
                total_files += 1

                if not full_path.exists():
                    missing_files += 1
                    if missing_files <= 5:  # Log only first 5
                        logger.warning(f"Missing file: {png_path}")

        if missing_files > 5:
            logger.warning(f"... and {missing_files - 5} more missing files")

        if missing_files > 0:
            logger.info(
                f"Verification: {total_files - missing_files}/{total_files} "
                f"files found ({100 * (1 - missing_files/total_files):.1f}%)"
            )
        else:
            logger.info(f"Verification: All {total_files} files found ✓")

        return total_files, missing_files

    def extract_date_from_filename(self, png_path: str) -> str:
        """
        Extract date from PNG filename.
        Expected format: DD{Month}_{YEAR}_...
        Example: 01April_2010_Thursday_heute.avi_pid0_fn000006-0.png
        Returns: "01April_2010"
        
        Args:
            png_path: Path to PNG file
            
        Returns:
            Date string (DD{Month}_{YEAR}) or empty string if not found
        """
        filename = Path(png_path).stem
        parts = filename.split('_')
        
        if len(parts) >= 2:
            # First two parts should be date and month, third should be year
            date_part = parts[0]  # "01April"
            year_part = parts[1]  # "2010"
            return f"{date_part}_{year_part}"
        
        return ""

    def organize_frames(
        self,
        class_frames: Dict[str, List[str]],
        copy_mode: bool = True,
        single_class: str = None
    ) -> int:
        """
        Copy or link PNG frames to class directories based on alignments.
        Organizes frames into subfolders by date.

        Args:
            class_frames: Dictionary mapping class to PNG paths
            copy_mode: If True, copy files; if False, create symlinks
            single_class: If specified, only process this class

        Returns:
            Number of successfully processed frames
        """
        operation = "Copying" if copy_mode else "Linking"

        total_frames_processed = 0
        failed_frames = 0

        for gloss, png_paths in class_frames.items():
            # Skip if filtering by single class
            if single_class and gloss != single_class:
                continue

            # Create class directory
            class_dir = self.output_dir / gloss
            class_dir.mkdir(parents=True, exist_ok=True)

            # Track unique dates and assign folder indices
            date_to_folder_idx = {}
            folder_counter = 0

            for frame_idx, png_path in enumerate(png_paths):
                source_path = self.dataset_root / png_path
                
                # Extract date from filename
                date_str = self.extract_date_from_filename(png_path)
                
                # Assign folder index to date if not seen before
                if date_str not in date_to_folder_idx:
                    date_to_folder_idx[date_str] = folder_counter
                    folder_counter += 1
                
                folder_idx = date_to_folder_idx[date_str]
                
                # Create subfolder based on date
                date_dir = class_dir / f"{folder_idx:04d}"
                date_dir.mkdir(parents=True, exist_ok=True)
                
                output_filename = Path(png_path).name
                output_path = date_dir / output_filename

                try:
                    if not source_path.exists():
                        logger.warning(f"File not found: {png_path}")
                        failed_frames += 1
                        continue

                    if copy_mode:
                        shutil.copy2(source_path, output_path)
                    else:
                        if output_path.exists() or output_path.is_symlink():
                            output_path.unlink()
                        output_path.symlink_to(source_path.absolute())

                    total_frames_processed += 1

                except Exception as e:
                    logger.error(
                        f"Error {operation.lower()} {png_path}: {e}"
                    )
                    failed_frames += 1

                if total_frames_processed % 1000 == 0:
                    logger.info(
                        f"{operation} {total_frames_processed} frames... to folder {output_path}"
                    )

        if failed_frames > 0:
            logger.warning(f"Failed to process {failed_frames} frames")

        logger.info(
            f"Frame organization complete! "
            f"Processed: {total_frames_processed}, Failed: {failed_frames}"
        )

        return total_frames_processed



    def generate_dataset_summary(
        self,
        class_frames: Dict[str, List[str]],
        label_to_gloss: Dict[int, str],
        total_processed: int
    ) -> None:
        """
        Generate a summary of the extracted dataset.

        Args:
            class_frames: Dictionary mapping class to PNG paths
            label_to_gloss: Dictionary mapping label ID to gloss name
            total_processed: Number of successfully processed frames
        """
        summary_file = self.output_dir / "dataset_summary.txt"

        total_alignments = sum(len(paths) for paths in class_frames.values())
        classes_with_data = len(class_frames)

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Phoenix PNG Dataset Extraction Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total training classes: {len(label_to_gloss)}\n")
            f.write(f"Classes with data: {classes_with_data}\n")
            f.write(f"Total alignments: {total_alignments}\n")
            f.write(f"Successfully processed: {total_processed}\n")
            f.write(f"Failed: {total_alignments - total_processed}\n\n")
            f.write("Class Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Class':<40} {'Frames':<10}\n")
            f.write("-" * 60 + "\n")

            for gloss in sorted(class_frames.keys()):
                frame_count = len(class_frames[gloss])
                f.write(f"{gloss:<40} {frame_count:<10}\n")

        logger.info(f"Summary saved to {summary_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract and organize PNG frames from Phoenix dataset using path-based alignments'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Path to Phoenix dataset root directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for organized frames'
    )
    parser.add_argument(
        '--classes-file',
        type=str,
        default='trainingclasses.txt',
        help='Path to training classes file (relative to dataset root)'
    )
    parser.add_argument(
        '--alignment-file',
        type=str,
        default='train.alignment',
        help='Path to alignment file with format: png_path label_id'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symlinks instead of copying frames (saves disk space)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify file existence, do not copy/link'
    )
    
    parser.add_argument(
        '--single-class',
        type=str,
        default=None,
        help='Only process frames for this specific class (e.g., OKTOBER)'
    )

    args = parser.parse_args()

    try:
        # Initialize extractor
        extractor = PhoenixPathAlignmentExtractor(args.dataset_root, args.output_dir)

        # Load training classes (maps label_id -> gloss name)
        label_to_gloss = extractor.load_training_classes(args.classes_file)

        # Parse alignment file
        alignments = extractor.parse_alignment_file(args.alignment_file)

        # Group PNG paths by class
        class_frames = extractor.group_frames_by_class(alignments, label_to_gloss)

        # Verify files
        total_files, missing_files = extractor.verify_png_files(class_frames)

        if args.verify_only:
            logger.info("Verification mode: exiting without copying/linking files")
            return 0

        # Organize frames by class
        copy_mode = not args.symlink
        total_processed = extractor.organize_frames(class_frames, copy_mode=copy_mode, single_class=args.single_class)

        # Generate summary
        extractor.generate_dataset_summary(class_frames, label_to_gloss, total_processed)

        logger.info("Dataset organization completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during organization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
