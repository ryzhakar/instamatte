"""Multi-format image processor for social media."""

import functools
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, TypeVar

import typer
import yaml
from PIL import Image, ImageColor, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
from rich.progress import track

__version__ = "0.2.2"

console = Console()
app = typer.Typer(add_completion=False)

CONFIG_FILENAME = "instamatte.yaml"

# Constants
EXTREME_ASPECT_RATIO = 5.0
MAX_CONSERVATIVE_FILL = 70.0
MAX_MAT_THICKNESS_PERCENT = 50.0


class SocialFormat(BaseModel):
    """Social media format specification with dimensions and styling."""

    output_dir: str = "instamatte-processed"
    canvas_width: int = Field(default=1080, ge=1)
    canvas_height: int = Field(default=1920, ge=1)
    fill_percentage: float = Field(default=80.0, gt=0, le=100)
    mat_thickness: float = Field(
        default=5.0, ge=0, le=MAX_MAT_THICKNESS_PERCENT,
    )
    image_pattern: str = Field(default="*.{jpg,jpeg,png,gif,bmp}")
    mat_color: str = Field(default="WHITE")

    @field_validator("mat_thickness")
    def mat_must_be_reasonable(cls, v: float) -> float:
        """Validate that mat thickness is reasonable."""
        if v >= MAX_MAT_THICKNESS_PERCENT:
            error_msg = "Mat thickness percentage must be less than 50%"
            raise ValueError(error_msg)
        return v

    @field_validator("mat_color")
    def color_must_be_valid(cls, v: str) -> str:
        """Validate that color string can be parsed by PIL."""
        try:
            ImageColor.getrgb(v)
        except ValueError as e:
            error_msg = textwrap.dedent(f"""
                Invalid color '{v}'. Use color name (e.g., 'BLACK')
                or hex (e.g., '#000000')
            """).strip()
            raise ValueError(error_msg) from e
        else:
            return v


class ImageMattingError(Exception):
    """Exception for image matting operations."""


def ensure_positive_dimensions(width: int, height: int) -> None:
    """Verify image dimensions are positive."""
    if width <= 0:
        msg = "Image width must be positive"
        raise ImageMattingError(msg)
    if height <= 0:
        msg = "Image height must be positive"
        raise ImageMattingError(msg)


def calculate_mat_pixels(
    canvas_width: int, canvas_height: int, mat_percentage: float,
) -> int:
    """Calculate mat thickness in pixels based on percentage."""
    smallest_dimension = min(canvas_width, canvas_height)
    requested_mat = int(smallest_dimension * (mat_percentage / 100))

    max_allowable_mat = smallest_dimension // 2 - 1
    if requested_mat >= max_allowable_mat:
        console.print(textwrap.dedent("""
            [yellow]Warning: Mat thickness too large,
            reducing to maximum possible value
        """).strip())
        return max_allowable_mat - 1

    return requested_mat


def determine_mountable_area(
    canvas_width: int, canvas_height: int, mat_pixels: int,
) -> tuple[int, int]:
    """Calculate the available area for mounting images after mat."""
    return (
        max(1, canvas_width - 2 * mat_pixels),
        max(1, canvas_height - 2 * mat_pixels),
    )


def adjust_fill_for_panoramic_images(
    image_ratio: float,
    canvas_area: int,
    fill_percentage: float,
) -> float:
    """Reduce fill percentage for extreme aspect ratios."""
    aspect_extremity = max(image_ratio, 1/image_ratio)
    if aspect_extremity <= EXTREME_ASPECT_RATIO:
        return canvas_area * (fill_percentage / 100)

    conservative_fill = min(fill_percentage, MAX_CONSERVATIVE_FILL)
    return canvas_area * (conservative_fill / 100)


def calculate_ideal_dimensions(
    target_area: float, image_ratio: float,
) -> tuple[float, float]:
    """Calculate ideal dimensions to achieve target area."""
    ideal_width = (target_area * image_ratio) ** 0.5
    ideal_height = ideal_width / image_ratio
    return ideal_width, ideal_height


def fit_image_within_mat(
    ideal_width: float,
    ideal_height: float,
    mountable_width: int,
    mountable_height: int,
    image_ratio: float,
) -> tuple[int, int]:
    """Scale image to fit within mat while maintaining aspect ratio."""
    if ideal_width <= mountable_width and ideal_height <= mountable_height:
        return round(ideal_width), round(ideal_height)

    if ideal_width / mountable_width > ideal_height / mountable_height:
        return mountable_width, round(mountable_width / image_ratio)
    return round(mountable_height * image_ratio), mountable_height


@dataclass
class SizingParameters:
    """Parameters for determining optimal image size."""

    image_width: int
    image_height: int
    canvas_width: int
    canvas_height: int
    fill_percentage: float
    mat_percentage: float


def determine_optimal_size(params: SizingParameters) -> tuple[int, int]:
    """Calculate optimal image size based on format specifications."""
    ensure_positive_dimensions(params.image_width, params.image_height)

    canvas_area = params.canvas_width * params.canvas_height
    image_ratio = params.image_width / params.image_height

    mat_pixels = calculate_mat_pixels(
        params.canvas_width, params.canvas_height, params.mat_percentage,
    )
    mountable_width, mountable_height = determine_mountable_area(
        params.canvas_width, params.canvas_height, mat_pixels,
    )

    target_area = adjust_fill_for_panoramic_images(
        image_ratio, canvas_area, params.fill_percentage,
    )

    ideal_width, ideal_height = calculate_ideal_dimensions(
        target_area, image_ratio,
    )

    return fit_image_within_mat(
        ideal_width,
        ideal_height,
        mountable_width,
        mountable_height,
        image_ratio,
    )


def create_unique_filename(output_dir: Path, original_path: Path) -> Path:
    """Generate unique filename for output to avoid collisions."""
    base_output = output_dir / original_path.name

    if not base_output.exists():
        return base_output

    name_parts = original_path.stem, original_path.suffix
    counter = 1
    while True:
        new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
        new_path = output_dir / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def apply_background_to_transparent_image(
    img: Image.Image, mat_color: str,
) -> Image.Image:
    """Replace transparent parts of image with mat color."""
    background = Image.new("RGB", img.size, mat_color)

    # Extract alpha channel using a ternary operator
    alpha_channel = img.split()[3] if img.mode == "RGBA" else img.split()[1]

    background.paste(img, mask=alpha_channel)
    return background


def prepare_image_for_mounting(img: Image.Image, mat_color: str) -> Image.Image:
    """Prepare image for mounting by handling transparency and color spaces."""
    if img.mode in ("RGBA", "LA"):
        return apply_background_to_transparent_image(img, mat_color)
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def resize_for_mounting(
    img: Image.Image, format_config: SocialFormat,
) -> Image.Image:
    """Resize image according to format specifications."""
    params = SizingParameters(
        image_width=img.width,
        image_height=img.height,
        canvas_width=format_config.canvas_width,
        canvas_height=format_config.canvas_height,
        fill_percentage=format_config.fill_percentage,
        mat_percentage=format_config.mat_thickness,
    )

    new_size = determine_optimal_size(params)

    if new_size[0] <= 0 or new_size[1] <= 0:
        error_msg = f"Invalid calculated dimensions: {new_size}"
        raise ImageMattingError(error_msg)

    return img.resize(new_size, Image.Resampling.LANCZOS)


def create_mat(width: int, height: int, mat_color: str) -> Image.Image:
    """Create mat image with specified dimensions and color."""
    try:
        return Image.new("RGB", (width, height), mat_color)
    except ValueError as e:
        error_msg = f"Invalid mat color '{mat_color}': {e}"
        raise ValueError(error_msg) from e


def center_image_on_mat(mat: Image.Image, img: Image.Image) -> Image.Image:
    """Place image in center of mat."""
    x = (mat.width - img.width) // 2
    y = (mat.height - img.height) // 2
    mat.paste(img, (x, y))
    return mat


def mount_image(img_path: Path, format_config: SocialFormat) -> None:
    """Process a single image according to format specifications."""
    try:
        with Image.open(img_path) as img:
            prepared_img = prepare_image_for_mounting(
                img, format_config.mat_color,
            )
            resized_img = resize_for_mounting(prepared_img, format_config)

            mat = create_mat(
                format_config.canvas_width,
                format_config.canvas_height,
                format_config.mat_color,
            )
            mounted_img = center_image_on_mat(mat, resized_img)

            output_dir = Path(format_config.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            out_path = create_unique_filename(output_dir, img_path)

            mounted_img.save(out_path, quality=95)

    except UnidentifiedImageError:
        # Image format issues - can't be parsed
        raise ImageMattingError(f"Could not identify image format for {img_path}")
    except (OSError, FileNotFoundError) as e:
        # File system issues
        raise ImageMattingError(f"File error: {e}") from e
    # ImageMattingError and ValueError pass through for specific handling


def expand_brace_pattern(pattern: str) -> list[str]:
    """Expand brace pattern like '*.{jpg,png}' into ['*.jpg', '*.png']."""
    match = re.match(r"^(.*?)\{(.*?)\}(.*)$", pattern)
    if not match:
        return [pattern]

    prefix, extensions, suffix = match.groups()
    return [f"{prefix}{ext.strip()}{suffix}" for ext in extensions.split(",")]


def expand_glob_patterns(pattern: str) -> list[str]:
    """Expand glob patterns with braces into multiple patterns."""
    if "{" not in pattern:
        return [pattern]

    try:
        return expand_brace_pattern(pattern)
    except (re.error, IndexError) as e:
        console.print(textwrap.dedent(f"""
            [yellow]Warning: Pattern parsing error,
            using original pattern: {e}
        """).strip())
        return [pattern]


def find_images_to_process(work_dir: Path, pattern_str: str) -> set[Path]:
    """Find all images matching pattern in work directory."""
    patterns = expand_glob_patterns(pattern_str)
    matching_files = set()

    for glob_pattern in patterns:
        matching_files.update(work_dir.glob(glob_pattern))

    return matching_files


def create_sample_config(config_path: Path) -> None:
    """Create sample configuration file."""
    try:
        with config_path.open("w") as f:
            yaml.dump([SocialFormat().model_dump()], f, sort_keys=False)
        console.print(f"[green]Created default config at {config_path}")
    except (PermissionError, OSError) as e:
        console.print(f"[red]Error creating config file: {e}")
        raise


def load_social_formats(config_path: Path) -> list[SocialFormat]:
    """Load social media format configurations from YAML file."""
    try:
        with config_path.open() as f:
            return [
                SocialFormat.model_validate(fmt)
                for fmt in yaml.safe_load(f)
            ]
    except ValidationError as e:
        error_message = textwrap.dedent(f"""
            [red]Invalid configuration in {config_path}:
            [red]{e}
        """).strip()
        console.print(error_message)
        raise typer.Exit(1) from e
    except (FileNotFoundError, PermissionError) as e:
        console.print(f"[red]Error accessing config file: {e}")
        raise typer.Exit(1) from e
    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML in {config_path}: {e}")
        raise typer.Exit(1) from e


def prepare_output_directory(
    format_config: SocialFormat, work_dir: Path,
) -> SocialFormat:
    """Prepare output directory for processing images."""
    format_config.output_dir = str(work_dir / format_config.output_dir)
    try:
        Path(format_config.output_dir).mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        console.print(textwrap.dedent(f"""
            [red]Error creating output directory
            {format_config.output_dir}: {e}
        """).strip())
        raise
    return format_config


def process_single_image(
    img_path: Path, format_config: SocialFormat,
) -> tuple[bool, str]:
    """Process a single image and return success status with any error."""
    try:
        mount_image(img_path, format_config)
    except ImageMattingError as e:
        # Known application-specific errors
        return False, str(e)
    except ValueError as e:
        # Input validation errors
        return False, f"Value error: {e}"
    except OSError as e:
        # Catch any file system errors that might have escaped mount_image
        return False, f"File system error: {e}"
    except Exception as e:
        # Unexpected errors - log and re-raise for debugging
        error_msg = f"Unexpected error: {e.__class__.__name__}: {e}"
        console.print(f"[red]Critical: {error_msg}")
        raise RuntimeError(f"Unhandled error processing {img_path}") from e
    else:
        return True, ""


def process_social_format(format_config: SocialFormat, work_dir: Path) -> None:
    """Process all images for a single social media format."""
    images = sorted(
        find_images_to_process(work_dir, format_config.image_pattern),
    )

    if not images:
        console.print(textwrap.dedent(f"""
            [yellow]No images found for {format_config.output_dir}
        """).strip())
        return

    console.print(textwrap.dedent(f"""
        [blue]Processing {len(images)} images for {format_config.output_dir}
    """).strip())

    # Process all images and collect errors
    errors = []
    for img_path in track(
        images, description=f"Processing {format_config.output_dir}...",
    ):
        success, error_msg = process_single_image(img_path, format_config)
        if not success:
            errors.append((img_path, error_msg))

    # Report errors after processing
    for img_path, error_msg in errors:
        console.print(f"[red]Error processing {img_path}: {error_msg}")


T = TypeVar("T")


def handle_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Handle errors in the main application flow."""
    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> T:
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            # Pass through typer.Exit exceptions for correct CLI exit codes
            raise
        except Exception as e:
            console.print(f"[red]Error: {e}")
            raise typer.Exit(1) from e
    return wrapper


@app.command()
@handle_errors
def main(
    input_dir: Annotated[
        str, typer.Argument(help="Directory with images and config.yaml"),
    ],
) -> None:
    """Process images to multiple social media formats with elegant matting."""
    work_dir = Path(input_dir).resolve()

    if not work_dir.exists():
        console.print(f"[red]Directory not found: {work_dir}")
        raise typer.Exit(1)

    config_path = work_dir / CONFIG_FILENAME

    if not config_path.exists():
        create_sample_config(config_path)

    formats = load_social_formats(config_path)

    for format_config in formats:
        prepared_format = prepare_output_directory(format_config, work_dir)
        process_social_format(prepared_format, work_dir)


if __name__ == "__main__":
    app()
