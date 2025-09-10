"""Command-line interface for instamatte image processing."""

import functools
import textwrap
from pathlib import Path
from typing import Annotated, Callable, TypeVar

import typer
from rich.console import Console
from rich.progress import track

from .configuration import (
    SocialFormat,
    discover_configuration_file,
    load_social_format_configurations,
    prepare_format_output_directory,
)
from .file_operations import (
    discover_images_matching_pattern,
    validate_work_directory_exists,
)
from .image_processing import (
    ImageProcessingError,
    process_image_to_format,
    stitch_images_into_horizontal_panorama,
    split_image_into_seamless_carousel_slices,
)

console = Console()

T = TypeVar("T")


def handle_application_errors(
    command_function: Callable[..., T],
) -> Callable[..., T]:
    """Handle errors in CLI commands gracefully."""

    @functools.wraps(command_function)
    def error_handling_wrapper(*args: object, **kwargs: object) -> T:
        try:
            return command_function(*args, **kwargs)
        except typer.Exit:
            # Pass through typer.Exit for correct CLI exit codes
            raise
        except Exception as unexpected_error:
            console.print(f"[red]Error: {unexpected_error}")
            raise typer.Exit(1) from unexpected_error

    return error_handling_wrapper


def process_single_image_with_error_capture(
    image_path: Path, format_specification: SocialFormat
) -> tuple[bool, str]:
    """Process single image and return success status with error details."""
    try:
        process_image_to_format(image_path, format_specification)
        return True, ""
    except ImageProcessingError as processing_error:
        return False, str(processing_error)
    except ValueError as validation_error:
        return False, f"Value error: {validation_error}"
    except OSError as filesystem_error:
        return False, f"File system error: {filesystem_error}"
    except Exception as unexpected_error:
        error_description = f"Unexpected error: {unexpected_error.__class__.__name__}: {unexpected_error}"
        console.print(f"[red]Critical: {error_description}")
        raise RuntimeError(
            f"Unhandled error processing {image_path}"
        ) from unexpected_error


def process_images_for_social_format(
    format_specification: SocialFormat, work_directory: Path
) -> None:
    """Process all images matching pattern for specific social media format."""
    discovered_images = sorted(
        discover_images_matching_pattern(
            work_directory, format_specification.image_pattern
        )
    )

    if not discovered_images:
        console.print(
            textwrap.dedent(f"""
            [yellow]No images found for {format_specification.output_dir}
        """).strip()
        )
        return

    console.print(
        textwrap.dedent(f"""
        [blue]Processing {len(discovered_images)} images for {format_specification.output_dir}
    """).strip()
    )

    processing_errors = []
    for image_path in track(
        discovered_images,
        description=f"Processing {format_specification.output_dir}...",
    ):
        processing_success, error_message = (
            process_single_image_with_error_capture(
                image_path, format_specification
            )
        )
        if not processing_success:
            processing_errors.append((image_path, error_message))

    for failed_image_path, error_description in processing_errors:
        console.print(
            f"[red]Error processing {failed_image_path}: {error_description}"
        )


def execute_batch_image_processing_workflow(work_directory_path: str) -> None:
    """Execute complete batch processing workflow for all configured formats."""
    work_directory = Path(work_directory_path).resolve()
    validate_work_directory_exists(work_directory)

    configuration_file = discover_configuration_file(work_directory)
    format_specifications = load_social_format_configurations(
        configuration_file
    )

    for format_specification in format_specifications:
        prepared_format = prepare_format_output_directory(
            format_specification, work_directory
        )
        process_images_for_social_format(prepared_format, work_directory)


def execute_panorama_stitching_workflow(
    work_directory_path: str, output_filename: str | None = None
) -> None:
    """Execute panorama stitching workflow for all images in directory."""
    work_directory = Path(work_directory_path).resolve()
    validate_work_directory_exists(work_directory)

    # Use standard image patterns for discovery
    DEFAULT_IMAGE_PATTERN = "*.{jpg,jpeg,png,gif,bmp}"
    discovered_images = list(
        discover_images_matching_pattern(work_directory, DEFAULT_IMAGE_PATTERN)
    )

    if not discovered_images:
        console.print(
            "[yellow]No images found in directory for panorama stitching"
        )
        return

    if len(discovered_images) < 2:
        console.print(
            "[yellow]At least 2 images required for panorama stitching"
        )
        return

    # Generate output filename if not provided
    if output_filename is None:
        output_filename = "panorama.png"

    output_path = work_directory / output_filename

    try:
        stitch_images_into_horizontal_panorama(discovered_images, output_path)
        console.print(f"[green]Panorama created successfully: {output_path}")
    except ImageProcessingError as stitching_error:
        console.print(f"[red]Error creating panorama: {stitching_error}")
        raise typer.Exit(1) from stitching_error


def execute_carousel_splitting_workflow(
    source_image_path: str,
    target_slice_width: int = 1350,
    target_slice_height: int = 1080,
    output_directory_path: str | None = None,
) -> None:
    """Execute carousel splitting workflow for seamless Instagram carousels."""
    source_image_file = Path(source_image_path).resolve()

    if not source_image_file.exists():
        console.print(f"[red]Image file not found: {source_image_file}")
        raise typer.Exit(1)

    if not source_image_file.is_file():
        console.print(f"[red]Path is not a file: {source_image_file}")
        raise typer.Exit(1)

    output_directory = (
        Path(output_directory_path).resolve()
        if output_directory_path
        else source_image_file.parent / "carousel_slices"
    )

    try:
        generated_slice_paths = split_image_into_seamless_carousel_slices(
            source_image_file,
            target_slice_width,
            target_slice_height,
            output_directory,
        )

        console.print(
            f"[green]Created {len(generated_slice_paths)} seamless carousel slices:"
        )
        for slice_path in generated_slice_paths:
            console.print(f"  • {slice_path.name}")
        console.print(f"[green]Output directory: {output_directory}")

    except ImageProcessingError as splitting_error:
        console.print(f"[red]Error splitting image: {splitting_error}")
        raise typer.Exit(1) from splitting_error


app = typer.Typer(
    add_completion=False,
    help="Multi-format image processor for social media with elegant matting.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.command("matte")
@handle_application_errors
def process_images_command(
    input_directory: Annotated[
        str,
        typer.Argument(
            help="Directory containing images and instamatte.yaml configuration"
        ),
    ],
) -> None:
    """Process images to multiple social media formats with elegant matting."""
    execute_batch_image_processing_workflow(input_directory)


@app.command("stitch")
@handle_application_errors
def stitch_panorama_command(
    input_directory: Annotated[
        str,
        typer.Argument(
            help="Directory containing images to stitch into panorama"
        ),
    ],
    output_filename: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output filename for panorama (default: panorama.png)",
        ),
    ] = None,
) -> None:
    """Stitch images horizontally into a panorama with height normalization."""
    execute_panorama_stitching_workflow(input_directory, output_filename)


@app.command("split")
@handle_application_errors
def split_carousel_command(
    image_file: Annotated[
        str,
        typer.Argument(
            help="Image file to split into seamless carousel slices"
        ),
    ],
    width: Annotated[
        int,
        typer.Option(
            "--width",
            "-w",
            help="Target width for carousel slices (default: 1350)",
        ),
    ] = 1080,
    height: Annotated[
        int,
        typer.Option(
            "--height",
            "-h",
            help="Target height for carousel slices (default: 1080)",
        ),
    ] = 1350,
    output_directory: Annotated[
        str | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for carousel slices (default: ./carousel_slices)",
        ),
    ] = None,
) -> None:
    """Split image into seamless carousel slices with automatic slice detection."""
    execute_carousel_splitting_workflow(
        image_file, width, height, output_directory
    )


@app.callback(no_args_is_help=True)
def main_command() -> None:
    """Multi-format image processor for social media with elegant matting."""


# Create the main application instance
cli_application = app
