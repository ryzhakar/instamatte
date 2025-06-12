"""Multi-format image processor for social media."""

from pathlib import Path
import os
from typing import Annotated, Tuple, Optional, Dict, Any
import re

import typer
import yaml
from PIL import Image, ImageColor
from pydantic import BaseModel, Field, field_validator, ValidationError
from rich.console import Console
from rich.progress import track

__version__ = "0.2.2"

console = Console()
app = typer.Typer(add_completion=False)

CONFIG_FILENAME = "instamatte.yaml"


class FormatConfig(BaseModel):
    """Configuration for a single output format."""

    output_dir: str = "instamatte-processed"
    frame_width: int = Field(default=1080, ge=1)
    frame_height: int = Field(default=1920, ge=1)
    target_surface_pct: float = Field(default=80.0, gt=0, le=100)
    margin_pct: float = Field(default=5.0, ge=0, le=50)
    pattern: str = Field(default="*.{jpg,jpeg,png,gif,bmp}")
    bg_color: str = Field(default="WHITE")

    @field_validator("margin_pct")
    def margin_must_be_valid(cls, v: float) -> float:
        if v >= 50:
            raise ValueError("Margin percentage must be less than 50%")
        return v
        
    @field_validator("bg_color")
    def color_must_be_valid(cls, v: str) -> str:
        try:
            # Verify color is valid by attempting to convert it
            ImageColor.getrgb(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid color '{v}'. Use color name (e.g., 'BLACK') or hex (e.g., '#000000')")


def validate_image_dimensions(img_width: int, img_height: int) -> None:
    """Validate image dimensions are positive."""
    if img_width <= 0:
        raise ValueError("Image width must be positive")
    if img_height <= 0:
        raise ValueError("Image height must be positive")


def calculate_dimensions(
    img_width: int,
    img_height: int,
    frame_width: int,
    frame_height: int,
    target_surface_pct: float,
    margin_pct: float,
) -> Tuple[int, int]:
    """Calculate dimensions maximizing surface area while respecting minimum margin."""
    # Validate inputs to prevent division by zero
    validate_image_dimensions(img_width, img_height)
    
    frame_area = frame_width * frame_height
    min_frame_dim = min(frame_width, frame_height)
    min_margin_px = int(min_frame_dim * (margin_pct / 100))
    
    # Ensure margins aren't too large for the frame
    max_margin = min(frame_width, frame_height) // 2 - 1
    if min_margin_px >= max_margin:
        console.print(f"[yellow]Warning: Margin too large, reducing to maximum possible value")
        min_margin_px = max_margin - 1
        
    # Calculate available space accounting for margins
    max_width = max(1, frame_width - 2 * min_margin_px)
    max_height = max(1, frame_height - 2 * min_margin_px)
    
    # Determine aspect ratio safely
    img_ratio = img_width / img_height
    
    # Calculate target area and dimensions
    target_area = frame_area * (target_surface_pct / 100)
    
    # Handle extreme aspect ratios by capping the target area
    aspect_ratio_factor = max(img_ratio, 1/img_ratio)
    if aspect_ratio_factor > 5:  # If extreme aspect ratio (very wide or very tall)
        target_surface_pct = min(target_surface_pct, 70)  # Cap at 70% to avoid strange results
        target_area = frame_area * (target_surface_pct / 100)
    
    trial_width = (target_area * img_ratio) ** 0.5
    trial_height = trial_width / img_ratio

    # Apply proper rounding to avoid dimension inconsistencies
    if trial_width <= max_width and trial_height <= max_height:
        return round(trial_width), round(trial_height)

    if trial_width / max_width > trial_height / max_height:
        return max_width, round(max_width / img_ratio)
    else:
        return round(max_height * img_ratio), max_height


def get_safe_output_path(output_dir: Path, img_path: Path) -> Path:
    """Create a unique output path to avoid filename collisions."""
    base_output = output_dir / img_path.name
    
    # If file doesn't exist yet, return the original path
    if not base_output.exists():
        return base_output
        
    # If file exists, add a suffix to make it unique
    name_parts = img_path.stem, img_path.suffix
    counter = 1
    while True:
        new_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
        new_path = output_dir / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def process_image(img_path: Path, cfg: FormatConfig) -> None:
    """Process a single image for one output format."""
    try:
        with Image.open(img_path) as img:
            # Handle transparency - convert to RGB with background color
            if img.mode in ("RGBA", "LA"):
                # Create a background with specified color
                bg = Image.new("RGB", img.size, cfg.bg_color)
                # Handle different alpha channel positions safely
                if img.mode == "RGBA":
                    alpha_channel = img.split()[3]  # RGBA - alpha is channel 3
                else:  # LA mode
                    alpha_channel = img.split()[1]  # LA - alpha is channel 1
                bg.paste(img, mask=alpha_channel)
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            new_size = calculate_dimensions(
                img.width,
                img.height,
                cfg.frame_width,
                cfg.frame_height,
                cfg.target_surface_pct,
                cfg.margin_pct,
            )

            # Safety check for dimensions
            if new_size[0] <= 0 or new_size[1] <= 0:
                raise ValueError(f"Invalid calculated dimensions: {new_size}")

            # Use LANCZOS for high-quality downsampling
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            try:
                background = Image.new(
                    "RGB", (cfg.frame_width, cfg.frame_height), cfg.bg_color
                )
            except ValueError as e:
                raise ValueError(f"Invalid background color '{cfg.bg_color}': {e}")

            x = (cfg.frame_width - new_size[0]) // 2
            y = (cfg.frame_height - new_size[1]) // 2

            background.paste(resized, (x, y))

            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            out_path = get_safe_output_path(output_dir, img_path)
            
            background.save(out_path, quality=95)
            
    except Exception as e:
        console.print(f"[red]Error processing {img_path}: {e}")
        raise


def parse_glob_pattern(pattern: str) -> list[str]:
    """Parse glob pattern safely to handle various formats."""
    patterns = []
    
    # Handle standard glob patterns without braces
    if "{" not in pattern:
        patterns = [pattern]
    else:
        try:
            # Extract parts before and after the brace pattern
            match = re.match(r"^(.*?)\{(.*?)\}(.*)$", pattern)
            if match:
                prefix, extensions, suffix = match.groups()
                for ext in extensions.split(","):
                    patterns.append(f"{prefix}{ext.strip()}{suffix}")
            else:
                # If regex doesn't match but has braces, use original pattern
                patterns = [pattern]
        except Exception:
            # Fallback to original pattern if parsing fails
            patterns = [pattern]
            
    return patterns


@app.command()
def main(
    input_dir: Annotated[
        str, typer.Argument(help="Directory with images and config.yaml")
    ],
) -> None:
    """Process images to multiple output formats defined in config.yaml."""
    try:
        # Resolve relative path to absolute
        work_dir = Path(input_dir).resolve()
        
        if not work_dir.exists():
            console.print(f"[red]Directory not found: {work_dir}")
            raise typer.Exit(1)
            
        config_path = work_dir / CONFIG_FILENAME
        
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump([FormatConfig().model_dump()], f, sort_keys=False)
            console.print(f"[green]Created default config at {config_path}")

        with open(config_path) as f:
            try:
                formats = [
                    FormatConfig.model_validate(fmt) for fmt in yaml.safe_load(f)
                ]
            except ValidationError as e:
                console.print(f"[red]Invalid configuration in {config_path}:")
                console.print(f"[red]{e}")
                raise typer.Exit(1)

        format_images = {}

        for fmt in formats:
            # Make output directory path absolute relative to the working directory
            fmt.output_dir = str(work_dir / fmt.output_dir)
            Path(fmt.output_dir).mkdir(parents=True, exist_ok=True)
            format_images[fmt.output_dir] = set()
            
            patterns = parse_glob_pattern(fmt.pattern)
            
            for pattern in patterns:
                format_images[fmt.output_dir].update(work_dir.glob(pattern))

            images = sorted(format_images[fmt.output_dir])
            if not images:
                console.print(f"[yellow]No images found for {fmt.output_dir}")
                continue

            console.print(f"[blue]Processing {len(images)} images for {fmt.output_dir}")
            for img_path in track(images, description=f"Processing {fmt.output_dir}..."):
                try:
                    process_image(img_path, fmt)
                except Exception as e:
                    console.print(f"[red]Error processing {img_path}: {e}")

    except Exception as e:
        console.print(f"[red]Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
