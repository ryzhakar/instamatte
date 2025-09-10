"""Core image processing algorithms for social media formatting."""

import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from rich.console import Console

from .configuration import SocialFormat
from .file_operations import generate_unique_output_filename

console = Console()

EXTREME_ASPECT_RATIO_THRESHOLD = 5.0
CONSERVATIVE_FILL_PERCENTAGE_FOR_PANORAMIC = 70.0


class ImageProcessingError(Exception):
    """Exception for image processing operations."""


@dataclass
class ImageSizingParameters:
    """Parameters for calculating optimal image dimensions within format constraints."""

    source_image_width: int
    source_image_height: int
    target_canvas_width: int
    target_canvas_height: int
    desired_fill_percentage: float
    mat_thickness_percentage: float


def validate_image_dimensions_are_positive(width: int, height: int) -> None:
    """Verify image dimensions are positive integers."""
    if width <= 0:
        raise ImageProcessingError("Image width must be positive")
    if height <= 0:
        raise ImageProcessingError("Image height must be positive")


def calculate_mat_thickness_in_pixels(
    canvas_width: int, canvas_height: int, mat_thickness_percentage: float
) -> int:
    """Calculate mat thickness in pixels from percentage of smallest canvas dimension."""
    smallest_canvas_dimension = min(canvas_width, canvas_height)
    requested_mat_thickness = int(smallest_canvas_dimension * (mat_thickness_percentage / 100))

    maximum_allowable_mat_thickness = smallest_canvas_dimension // 2 - 1
    if requested_mat_thickness >= maximum_allowable_mat_thickness:
        console.print(textwrap.dedent("""
            [yellow]Warning: Mat thickness too large,
            reducing to maximum possible value
        """).strip())
        return maximum_allowable_mat_thickness - 1

    return requested_mat_thickness


def calculate_available_mounting_area(
    canvas_width: int, canvas_height: int, mat_thickness_pixels: int
) -> tuple[int, int]:
    """Calculate available area for image placement after mat thickness."""
    available_width = max(1, canvas_width - 2 * mat_thickness_pixels)
    available_height = max(1, canvas_height - 2 * mat_thickness_pixels)
    return available_width, available_height


def adjust_fill_percentage_for_extreme_aspect_ratios(
    image_aspect_ratio: float,
    canvas_total_area: int,
    requested_fill_percentage: float,
) -> float:
    """Reduce fill percentage for panoramic or very tall images."""
    aspect_ratio_extremity = max(image_aspect_ratio, 1 / image_aspect_ratio)
    
    if aspect_ratio_extremity <= EXTREME_ASPECT_RATIO_THRESHOLD:
        return canvas_total_area * (requested_fill_percentage / 100)

    conservative_fill_percentage = min(requested_fill_percentage, CONSERVATIVE_FILL_PERCENTAGE_FOR_PANORAMIC)
    return canvas_total_area * (conservative_fill_percentage / 100)


def calculate_dimensions_for_target_area(
    target_surface_area: float, image_aspect_ratio: float
) -> tuple[float, float]:
    """Calculate width and height to achieve target surface area."""
    optimal_width = (target_surface_area * image_aspect_ratio) ** 0.5
    optimal_height = optimal_width / image_aspect_ratio
    return optimal_width, optimal_height


def constrain_image_to_available_mounting_area(
    ideal_width: float,
    ideal_height: float,
    available_mounting_width: int,
    available_mounting_height: int,
    image_aspect_ratio: float,
) -> tuple[int, int]:
    """Scale image to fit within available mounting area while preserving aspect ratio."""
    if ideal_width <= available_mounting_width and ideal_height <= available_mounting_height:
        return round(ideal_width), round(ideal_height)

    width_constraint_factor = ideal_width / available_mounting_width
    height_constraint_factor = ideal_height / available_mounting_height
    
    if width_constraint_factor > height_constraint_factor:
        constrained_width = available_mounting_width
        constrained_height = round(available_mounting_width / image_aspect_ratio)
    else:
        constrained_width = round(available_mounting_height * image_aspect_ratio)
        constrained_height = available_mounting_height
    
    return constrained_width, constrained_height


def determine_optimal_image_dimensions(sizing_parameters: ImageSizingParameters) -> tuple[int, int]:
    """Calculate optimal image dimensions based on format specifications and constraints."""
    validate_image_dimensions_are_positive(
        sizing_parameters.source_image_width, 
        sizing_parameters.source_image_height
    )

    canvas_total_area = sizing_parameters.target_canvas_width * sizing_parameters.target_canvas_height
    image_aspect_ratio = sizing_parameters.source_image_width / sizing_parameters.source_image_height

    mat_thickness_pixels = calculate_mat_thickness_in_pixels(
        sizing_parameters.target_canvas_width,
        sizing_parameters.target_canvas_height,
        sizing_parameters.mat_thickness_percentage,
    )
    
    available_width, available_height = calculate_available_mounting_area(
        sizing_parameters.target_canvas_width,
        sizing_parameters.target_canvas_height,
        mat_thickness_pixels,
    )

    target_surface_area = adjust_fill_percentage_for_extreme_aspect_ratios(
        image_aspect_ratio,
        canvas_total_area,
        sizing_parameters.desired_fill_percentage,
    )

    ideal_width, ideal_height = calculate_dimensions_for_target_area(
        target_surface_area, image_aspect_ratio
    )

    final_width, final_height = constrain_image_to_available_mounting_area(
        ideal_width,
        ideal_height,
        available_width,
        available_height,
        image_aspect_ratio,
    )

    return final_width, final_height


def replace_transparency_with_mat_color(source_image: Image.Image, mat_color: str) -> Image.Image:
    """Replace transparent pixels with mat color for clean backgrounds."""
    mat_background = Image.new("RGB", source_image.size, mat_color)

    transparency_mask = (
        source_image.split()[3] if source_image.mode == "RGBA" 
        else source_image.split()[1]
    )

    mat_background.paste(source_image, mask=transparency_mask)
    return mat_background


def prepare_image_for_format_processing(source_image: Image.Image, mat_color: str) -> Image.Image:
    """Prepare image for processing by handling transparency and color modes."""
    if source_image.mode in ("RGBA", "LA"):
        return replace_transparency_with_mat_color(source_image, mat_color)
    if source_image.mode != "RGB":
        return source_image.convert("RGB")
    return source_image


def resize_image_for_format_specifications(
    source_image: Image.Image, format_specification: SocialFormat
) -> Image.Image:
    """Resize image according to social media format specifications."""
    sizing_parameters = ImageSizingParameters(
        source_image_width=source_image.width,
        source_image_height=source_image.height,
        target_canvas_width=format_specification.canvas_width,
        target_canvas_height=format_specification.canvas_height,
        desired_fill_percentage=format_specification.fill_percentage,
        mat_thickness_percentage=format_specification.mat_thickness,
    )

    optimal_width, optimal_height = determine_optimal_image_dimensions(sizing_parameters)

    if optimal_width <= 0 or optimal_height <= 0:
        raise ImageProcessingError(f"Invalid calculated dimensions: {optimal_width}x{optimal_height}")

    return source_image.resize((optimal_width, optimal_height), Image.Resampling.LANCZOS)


def create_canvas_with_mat_color(canvas_width: int, canvas_height: int, mat_color: str) -> Image.Image:
    """Create canvas background with specified dimensions and mat color."""
    try:
        return Image.new("RGB", (canvas_width, canvas_height), mat_color)
    except ValueError as color_error:
        raise ValueError(f"Invalid mat color '{mat_color}': {color_error}") from color_error


def mount_image_on_canvas_center(canvas: Image.Image, processed_image: Image.Image) -> Image.Image:
    """Mount processed image in center of canvas."""
    horizontal_offset = (canvas.width - processed_image.width) // 2
    vertical_offset = (canvas.height - processed_image.height) // 2
    canvas.paste(processed_image, (horizontal_offset, vertical_offset))
    return canvas


def apply_color_preservation_settings(icc_profile: bytes | None) -> dict:
    """Configure save parameters for optimal color preservation."""
    save_parameters = {"quality": 95}
    
    if icc_profile:
        save_parameters["icc_profile"] = icc_profile
    
    return save_parameters


def apply_jpeg_optimization_settings(output_path: Path, save_parameters: dict) -> dict:
    """Apply JPEG-specific optimization settings for high quality output."""
    if output_path.suffix.lower() in (".jpg", ".jpeg"):
        save_parameters.update({
            "subsampling": 0,      # Disable chroma subsampling for quality
            "optimize": True,      # Enable optimization
            "progressive": True    # Progressive encoding for web compatibility
        })
    
    return save_parameters


def process_image_to_format(source_image_path: Path, format_specification: SocialFormat) -> None:
    """Process single image according to social media format specifications."""
    try:
        with Image.open(source_image_path) as source_image:
            preserved_icc_profile = source_image.info.get("icc_profile")
            
            color_corrected_image = prepare_image_for_format_processing(
                source_image, format_specification.mat_color
            )
            
            resized_image = resize_image_for_format_specifications(
                color_corrected_image, format_specification
            )

            canvas = create_canvas_with_mat_color(
                format_specification.canvas_width,
                format_specification.canvas_height,
                format_specification.mat_color,
            )
            
            final_image = mount_image_on_canvas_center(canvas, resized_image)

            output_directory = Path(format_specification.output_dir)
            output_file_path = generate_unique_output_filename(output_directory, source_image_path)

            save_parameters = apply_color_preservation_settings(preserved_icc_profile)
            save_parameters = apply_jpeg_optimization_settings(output_file_path, save_parameters)

            final_image.save(output_file_path, **save_parameters)

    except UnidentifiedImageError:
        raise ImageProcessingError(f"Could not identify image format for {source_image_path}")
    except (OSError, FileNotFoundError) as filesystem_error:
        raise ImageProcessingError(f"File error: {filesystem_error}") from filesystem_error
