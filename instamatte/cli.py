"""Command-line interface for instamatte image processing."""

import functools
import textwrap
from pathlib import Path
from typing import Annotated, Callable, List, TypeVar

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
from .image_processing import ImageProcessingError, process_image_to_format

console = Console()

T = TypeVar("T")


def handle_application_errors(command_function: Callable[..., T]) -> Callable[..., T]:
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
        raise RuntimeError(f"Unhandled error processing {image_path}") from unexpected_error


def process_images_for_social_format(
    format_specification: SocialFormat, work_directory: Path
) -> None:
    """Process all images matching pattern for specific social media format."""
    discovered_images = sorted(
        discover_images_matching_pattern(work_directory, format_specification.image_pattern)
    )

    if not discovered_images:
        console.print(textwrap.dedent(f"""
            [yellow]No images found for {format_specification.output_dir}
        """).strip())
        return

    console.print(textwrap.dedent(f"""
        [blue]Processing {len(discovered_images)} images for {format_specification.output_dir}
    """).strip())

    processing_errors = []
    for image_path in track(
        discovered_images, 
        description=f"Processing {format_specification.output_dir}..."
    ):
        processing_success, error_message = process_single_image_with_error_capture(
            image_path, format_specification
        )
        if not processing_success:
            processing_errors.append((image_path, error_message))

    for failed_image_path, error_description in processing_errors:
        console.print(f"[red]Error processing {failed_image_path}: {error_description}")


def execute_batch_image_processing_workflow(work_directory_path: str) -> None:
    """Execute complete batch processing workflow for all configured formats."""
    work_directory = Path(work_directory_path).resolve()
    validate_work_directory_exists(work_directory)

    configuration_file = discover_configuration_file(work_directory)
    format_specifications = load_social_format_configurations(configuration_file)

    for format_specification in format_specifications:
        prepared_format = prepare_format_output_directory(format_specification, work_directory)
        process_images_for_social_format(prepared_format, work_directory)


app = typer.Typer(
    add_completion=False,
    help="Multi-format image processor for social media with elegant matting.",
    no_args_is_help=True,
)

@app.command("matte")
@handle_application_errors
def process_images_command(
    input_directory: Annotated[
        str, 
        typer.Argument(
            help="Directory containing images and instamatte.yaml configuration"
        )
    ],
) -> None:
    """Process images to multiple social media formats with elegant matting."""
    execute_batch_image_processing_workflow(input_directory)

@app.callback(invoke_without_command=True)
@handle_application_errors
def main_command(
    ctx: typer.Context,
    input_directory: Annotated[
        str | None,
        typer.Argument(help="Directory with images and config.yaml"),
    ] = None,
) -> None:
    """Process images to multiple social media formats with elegant matting.
    
    When no subcommand is specified, defaults to processing images.
    """
    if ctx.invoked_subcommand is None:
        if input_directory is None:
            console.print("[red]Error: Input directory is required")
            raise typer.Exit(1)
        execute_batch_image_processing_workflow(input_directory)


# Create the main application instance
cli_application = app
