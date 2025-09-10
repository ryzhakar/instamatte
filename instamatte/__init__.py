"""Multi-format image processor for social media."""

__version__ = "0.3.0"

from .configuration import SocialFormat
from .image_processing import process_image_to_format, stitch_images_into_horizontal_panorama

__all__ = ["SocialFormat", "process_image_to_format", "stitch_images_into_horizontal_panorama"]
