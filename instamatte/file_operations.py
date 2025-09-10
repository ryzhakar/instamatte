"""File system operations for image processing workflows."""

import re
import textwrap
from pathlib import Path
from typing import List, Set

from rich.console import Console

console = Console()


def expand_brace_pattern_into_individual_patterns(pattern: str) -> List[str]:
    """Expand brace pattern like '*.{jpg,png}' into ['*.jpg', '*.png']."""
    brace_match = re.match(r"^(.*?)\{(.*?)\}(.*)$", pattern)
    if not brace_match:
        return [pattern]

    prefix, extensions_list, suffix = brace_match.groups()
    individual_extensions = [ext.strip() for ext in extensions_list.split(",")]

    return [
        f"{prefix}{extension}{suffix}" for extension in individual_extensions
    ]


def expand_glob_patterns_with_braces(pattern: str) -> List[str]:
    """Expand glob patterns containing braces into multiple search patterns."""
    if "{" not in pattern:
        return [pattern]

    try:
        return expand_brace_pattern_into_individual_patterns(pattern)
    except (re.error, IndexError) as pattern_error:
        console.print(
            textwrap.dedent(f"""
            [yellow]Warning: Pattern parsing error,
            using original pattern: {pattern_error}
        """).strip()
        )
        return [pattern]


def discover_images_matching_pattern(
    work_directory: Path, pattern_specification: str
) -> Set[Path]:
    """Find all images matching pattern specification in work directory."""
    expanded_patterns = expand_glob_patterns_with_braces(pattern_specification)
    discovered_image_files = set()

    for individual_glob_pattern in expanded_patterns:
        matching_files = work_directory.glob(individual_glob_pattern)
        discovered_image_files.update(matching_files)

    return discovered_image_files


def generate_unique_output_filename(
    output_directory: Path, original_image_path: Path
) -> Path:
    """Generate unique filename for output to prevent file collisions."""
    preferred_output_path = output_directory / original_image_path.name

    if not preferred_output_path.exists():
        return preferred_output_path

    original_stem = original_image_path.stem
    original_suffix = original_image_path.suffix
    collision_counter = 1

    while True:
        incremented_filename = (
            f"{original_stem}_{collision_counter}{original_suffix}"
        )
        candidate_output_path = output_directory / incremented_filename

        if not candidate_output_path.exists():
            return candidate_output_path

        collision_counter += 1


def ensure_directory_exists_and_accessible(directory_path: Path) -> None:
    """Ensure directory exists and is accessible for writing."""
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as filesystem_error:
        console.print(
            f"[red]Error creating directory {directory_path}: {filesystem_error}"
        )
        raise


def validate_work_directory_exists(work_directory_path: Path) -> None:
    """Validate that work directory exists and is accessible."""
    if not work_directory_path.exists():
        console.print(f"[red]Directory not found: {work_directory_path}")
        raise FileNotFoundError(
            f"Work directory does not exist: {work_directory_path}"
        )

    if not work_directory_path.is_dir():
        console.print(f"[red]Path is not a directory: {work_directory_path}")
        raise NotADirectoryError(
            f"Path is not a directory: {work_directory_path}"
        )
