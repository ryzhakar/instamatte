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
    requested_mat_thickness = int(
        smallest_canvas_dimension * (mat_thickness_percentage / 100)
    )

    maximum_allowable_mat_thickness = smallest_canvas_dimension // 2 - 1
    if requested_mat_thickness >= maximum_allowable_mat_thickness:
        console.print(
            textwrap.dedent("""
            [yellow]Warning: Mat thickness too large,
            reducing to maximum possible value
        """).strip()
        )
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

    conservative_fill_percentage = min(
        requested_fill_percentage, CONSERVATIVE_FILL_PERCENTAGE_FOR_PANORAMIC
    )
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
    if (
        ideal_width <= available_mounting_width
        and ideal_height <= available_mounting_height
    ):
        return round(ideal_width), round(ideal_height)

    width_constraint_factor = ideal_width / available_mounting_width
    height_constraint_factor = ideal_height / available_mounting_height

    if width_constraint_factor > height_constraint_factor:
        constrained_width = available_mounting_width
        constrained_height = round(
            available_mounting_width / image_aspect_ratio
        )
    else:
        constrained_width = round(
            available_mounting_height * image_aspect_ratio
        )
        constrained_height = available_mounting_height

    return constrained_width, constrained_height


def determine_optimal_image_dimensions(
    sizing_parameters: ImageSizingParameters,
) -> tuple[int, int]:
    """Calculate optimal image dimensions based on format specifications and constraints."""
    validate_image_dimensions_are_positive(
        sizing_parameters.source_image_width,
        sizing_parameters.source_image_height,
    )

    canvas_total_area = (
        sizing_parameters.target_canvas_width
        * sizing_parameters.target_canvas_height
    )
    image_aspect_ratio = (
        sizing_parameters.source_image_width
        / sizing_parameters.source_image_height
    )

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


def replace_transparency_with_mat_color(
    source_image: Image.Image, mat_color: str
) -> Image.Image:
    """Replace transparent pixels with mat color for clean backgrounds."""
    mat_background = Image.new("RGB", source_image.size, mat_color)

    transparency_mask = (
        source_image.split()[3]
        if source_image.mode == "RGBA"
        else source_image.split()[1]
    )

    mat_background.paste(source_image, mask=transparency_mask)
    return mat_background


def prepare_image_for_format_processing(
    source_image: Image.Image, mat_color: str
) -> Image.Image:
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

    optimal_width, optimal_height = determine_optimal_image_dimensions(
        sizing_parameters
    )

    if optimal_width <= 0 or optimal_height <= 0:
        raise ImageProcessingError(
            f"Invalid calculated dimensions: {optimal_width}x{optimal_height}"
        )

    return source_image.resize(
        (optimal_width, optimal_height), Image.Resampling.LANCZOS
    )


def create_canvas_with_mat_color(
    canvas_width: int, canvas_height: int, mat_color: str
) -> Image.Image:
    """Create canvas background with specified dimensions and mat color."""
    try:
        return Image.new("RGB", (canvas_width, canvas_height), mat_color)
    except ValueError as color_error:
        raise ValueError(
            f"Invalid mat color '{mat_color}': {color_error}"
        ) from color_error


def mount_image_on_canvas_center(
    canvas: Image.Image, processed_image: Image.Image
) -> Image.Image:
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


def apply_jpeg_optimization_settings(
    output_path: Path, save_parameters: dict
) -> dict:
    """Apply JPEG-specific optimization settings for high quality output."""
    if output_path.suffix.lower() in (".jpg", ".jpeg"):
        save_parameters.update(
            {
                "subsampling": 0,  # Disable chroma subsampling for quality
                "optimize": True,  # Enable optimization
                "progressive": True,  # Progressive encoding for web compatibility
            }
        )

    return save_parameters


def discover_maximum_height_among_images(image_paths: list[Path]) -> int:
    """Find the tallest image height among all images for panorama scaling."""
    maximum_height = 0

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                maximum_height = max(maximum_height, image.height)
        except (UnidentifiedImageError, OSError):
            console.print(
                f"[yellow]Warning: Skipping unreadable image {image_path}"
            )
            continue

    if maximum_height == 0:
        raise ImageProcessingError(
            "No valid images found for panorama stitching"
        )

    return maximum_height


def scale_image_to_target_height_preserving_aspect_ratio(
    source_image: Image.Image, target_height: int
) -> Image.Image:
    """Scale image to target height while maintaining original aspect ratio."""
    if source_image.height == target_height:
        return source_image

    aspect_ratio = source_image.width / source_image.height
    target_width = round(target_height * aspect_ratio)

    return source_image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )


def calculate_total_panorama_width(
    image_paths: list[Path], target_height: int
) -> int:
    """Calculate total width needed for panorama after scaling all images to target height."""
    total_width = 0

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                aspect_ratio = image.width / image.height
                scaled_width = round(target_height * aspect_ratio)
                total_width += scaled_width
        except (UnidentifiedImageError, OSError):
            continue

    return total_width


def create_panorama_canvas_with_optimal_settings(
    total_width: int, height: int
) -> Image.Image:
    """Create high-quality canvas for panorama stitching."""
    if total_width <= 0 or height <= 0:
        raise ImageProcessingError(
            f"Invalid panorama dimensions: {total_width}x{height}"
        )

    return Image.new("RGB", (total_width, height), "WHITE")


def collect_and_preserve_color_profiles_from_images(
    image_paths: list[Path],
) -> bytes | None:
    """Collect ICC color profile from first available image for panorama preservation."""
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                icc_profile = image.info.get("icc_profile")
                if icc_profile:
                    return icc_profile
        except (UnidentifiedImageError, OSError):
            continue

    return None


def stitch_images_into_horizontal_panorama(
    image_paths: list[Path], output_path: Path
) -> None:
    """Stitch multiple images horizontally into a panorama with height normalization."""
    if not image_paths:
        raise ImageProcessingError("No images provided for panorama stitching")

    lexicographically_sorted_paths = sorted(image_paths)
    console.print(
        f"[blue]Stitching {len(lexicographically_sorted_paths)} images into panorama"
    )

    target_height = discover_maximum_height_among_images(
        lexicographically_sorted_paths
    )
    total_width = calculate_total_panorama_width(
        lexicographically_sorted_paths, target_height
    )

    panorama_canvas = create_panorama_canvas_with_optimal_settings(
        total_width, target_height
    )
    preserved_color_profile = collect_and_preserve_color_profiles_from_images(
        lexicographically_sorted_paths
    )

    current_horizontal_position = 0

    for image_path in lexicographically_sorted_paths:
        try:
            with Image.open(image_path) as source_image:
                color_corrected_image = prepare_image_for_format_processing(
                    source_image, "WHITE"
                )
                height_normalized_image = (
                    scale_image_to_target_height_preserving_aspect_ratio(
                        color_corrected_image, target_height
                    )
                )

                panorama_canvas.paste(
                    height_normalized_image, (current_horizontal_position, 0)
                )
                current_horizontal_position += height_normalized_image.width

        except (UnidentifiedImageError, OSError) as processing_error:
            console.print(
                f"[yellow]Warning: Skipping {image_path}: {processing_error}"
            )
            continue

    save_parameters = {"optimize": True}
    if preserved_color_profile:
        save_parameters["icc_profile"] = preserved_color_profile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panorama_canvas.save(output_path, "PNG", **save_parameters)


def resize_image_to_target_height_for_carousel_processing(
    source_image: Image.Image, target_height: int
) -> Image.Image:
    """Resize image to exact target height while preserving aspect ratio."""
    if source_image.height == target_height:
        return source_image

    aspect_ratio = source_image.width / source_image.height
    target_width = round(target_height * aspect_ratio)

    return source_image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )


def calculate_complete_carousel_slices_count(
    resized_width: int, target_slice_width: int
) -> int:
    """Calculate how many complete carousel slices fit in resized image width."""
    complete_slices = resized_width // target_slice_width
    MAXIMUM_INSTAGRAM_CAROUSEL_SLICES = 20

    return min(complete_slices, MAXIMUM_INSTAGRAM_CAROUSEL_SLICES)


def extract_carousel_slice_at_position(
    resized_image: Image.Image,
    slice_index: int,
    target_slice_width: int,
    target_slice_height: int,
) -> Image.Image:
    """Extract carousel slice at specific position with exact target dimensions."""
    left_position = slice_index * target_slice_width
    right_position = left_position + target_slice_width

    return resized_image.crop(
        (left_position, 0, right_position, target_slice_height)
    )


def split_image_into_seamless_carousel_slices(
    source_image_path: Path,
    target_slice_width: int = 1350,
    target_slice_height: int = 1080,
    output_directory: Path | None = None,
) -> list[Path]:
    """Split image into seamless carousel slices with automatic slice detection.

    Algorithm:
    1. Resize source to target height preserving aspect ratio
    2. Calculate complete slices that fit in width
    3. Extract each slice sequentially from left to right
    4. Discard any remaining width on the right
    5. Save as high-quality JPEG without borders
    """
    if output_directory is None:
        output_directory = source_image_path.parent / "carousel_slices"

    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(source_image_path) as source_image:
            preserved_color_profile = source_image.info.get("icc_profile")

            processed_source = prepare_image_for_format_processing(
                source_image, "WHITE"
            )

            # Step 1: Resize to target height, preserving aspect ratio
            height_normalized_image = (
                resize_image_to_target_height_for_carousel_processing(
                    processed_source, target_slice_height
                )
            )

            # Step 2: Calculate complete slices that fit
            complete_slices_count = calculate_complete_carousel_slices_count(
                height_normalized_image.width, target_slice_width
            )

            if complete_slices_count == 0:
                raise ImageProcessingError(
                    f"Image too narrow for carousel slicing. "
                    f"Need at least {target_slice_width}px width, got {height_normalized_image.width}px"
                )

            console.print(
                f"[blue]Creating {complete_slices_count} seamless carousel slices "
                f"({target_slice_width}×{target_slice_height}px each)"
            )

            # Step 3: Extract each slice sequentially
            output_file_paths = []

            for slice_index in range(complete_slices_count):
                carousel_slice = extract_carousel_slice_at_position(
                    height_normalized_image,
                    slice_index,
                    target_slice_width,
                    target_slice_height,
                )

                slice_filename = (
                    f"{source_image_path.stem}_slice_{slice_index + 1:02d}.jpg"
                )
                output_file_path = output_directory / slice_filename

                # Step 5: Save as high-quality JPEG
                save_parameters = {
                    "quality": 95,
                    "optimize": True,
                    "subsampling": 0,  # Disable chroma subsampling for quality
                }
                if preserved_color_profile:
                    save_parameters["icc_profile"] = preserved_color_profile

                carousel_slice.save(output_file_path, "JPEG", **save_parameters)
                output_file_paths.append(output_file_path)

            # Log discarded width information
            total_used_width = complete_slices_count * target_slice_width
            discarded_width = height_normalized_image.width - total_used_width

            if discarded_width > 0:
                console.print(
                    f"[yellow]Discarded {discarded_width}px of excess width from right edge"
                )

            return output_file_paths

    except UnidentifiedImageError:
        raise ImageProcessingError(
            f"Could not identify image format for {source_image_path}"
        )
    except (OSError, FileNotFoundError) as filesystem_error:
        raise ImageProcessingError(
            f"File error: {filesystem_error}"
        ) from filesystem_error


def process_image_to_format(
    source_image_path: Path, format_specification: SocialFormat
) -> None:
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
            output_file_path = generate_unique_output_filename(
                output_directory, source_image_path
            )

            save_parameters = apply_color_preservation_settings(
                preserved_icc_profile
            )
            save_parameters = apply_jpeg_optimization_settings(
                output_file_path, save_parameters
            )

            final_image.save(output_file_path, **save_parameters)

    except UnidentifiedImageError:
        raise ImageProcessingError(
            f"Could not identify image format for {source_image_path}"
        )
    except (OSError, FileNotFoundError) as filesystem_error:
        raise ImageProcessingError(
            f"File error: {filesystem_error}"
        ) from filesystem_error
