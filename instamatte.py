"""Multi-format image processor for social media."""

import glob
from pathlib import Path
from typing import Annotated, Tuple

import typer
import yaml
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.progress import track

__version__ = "0.1.0"

console = Console()
app = typer.Typer(add_completion=False)


class FormatConfig(BaseModel):
    """Configuration for a single output format."""

    output_dir: Path
    frame_width: int = Field(default=1080, ge=1)
    frame_height: int = Field(default=1920, ge=1)
    target_surface_pct: float = Field(default=85.0, gt=0, le=100)
    margin_pct: float = Field(default=5.0, ge=0, le=50)
    pattern: str = Field(default="*.{jpg,jpeg,png,gif,bmp}")
    bg_color: str = Field(default="WHITE")

    @field_validator("margin_pct")
    def margin_must_be_valid(cls, v: float) -> float:
        if v >= 50:
            raise ValueError("Margin percentage must be less than 50%")
        return v


def calculate_dimensions(
    img_width: int,
    img_height: int,
    frame_width: int,
    frame_height: int,
    target_surface_pct: float,
    margin_pct: float,
) -> Tuple[int, int]:
    """Calculate dimensions maximizing surface area while respecting minimum margin."""
    frame_area = frame_width * frame_height
    min_frame_dim = min(frame_width, frame_height)
    min_margin_px = int(min_frame_dim * (margin_pct / 100))
    img_ratio = img_width / img_height

    max_width = frame_width - 2 * min_margin_px
    max_height = frame_height - 2 * min_margin_px

    target_area = frame_area * (target_surface_pct / 100)
    trial_width = (target_area * img_ratio) ** 0.5
    trial_height = trial_width / img_ratio

    if trial_width <= max_width and trial_height <= max_height:
        return int(trial_width), int(trial_height)

    if trial_width / max_width > trial_height / max_height:
        return max_width, int(max_width / img_ratio)
    else:
        return int(max_height * img_ratio), max_height


def process_image(img_path: Path, cfg: FormatConfig) -> None:
    """Process a single image for one output format."""
    with Image.open(img_path) as img:
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, cfg.bg_color)
            bg.paste(img, mask=img.split()[-1])
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

        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        background = Image.new(
            "RGB", (cfg.frame_width, cfg.frame_height), cfg.bg_color
        )

        x = (cfg.frame_width - new_size[0]) // 2
        y = (cfg.frame_height - new_size[1]) // 2

        background.paste(resized, (x, y))

        actual_surface_pct = (
            (new_size[0] * new_size[1])
            / (cfg.frame_width * cfg.frame_height)
            * 100
        )
        min_margin_pct = min(
            x / min(cfg.frame_width, cfg.frame_height) * 100,
            y / min(cfg.frame_width, cfg.frame_height) * 100,
        )

        console.print(
            f"[blue]{cfg.output_dir}/{img_path.name}:[/] "
            f"surface: {actual_surface_pct:.1f}% "
            f"(target: {cfg.target_surface_pct}%), "
            f"margin: {min_margin_pct:.1f}% "
            f"(min: {cfg.margin_pct}%)"
        )

        out_path = cfg.output_dir / img_path.name
        background.save(out_path, quality=95)


@app.command()
def main(
    input_dir: Annotated[
        str, typer.Argument(help="Directory with images and config.yaml")
    ],
) -> None:
    """Process images to multiple output formats defined in config.yaml."""
    try:
        work_dir = Path(input_dir)
        config_path = work_dir / "config.yaml"

        if not config_path.exists():
            raise typer.Exit(f"No config.yaml found in {work_dir}")

        # Load formats from config
        with open(config_path) as f:
            formats = [
                FormatConfig.model_validate(fmt) for fmt in yaml.safe_load(f)
            ]

        # Find all unique images
        all_images = set()
        for fmt in formats:
            fmt.output_dir.mkdir(parents=True, exist_ok=True)

            base_pattern = fmt.pattern
            if "{" in base_pattern:
                prefix, ext_list = base_pattern.split("{")
                extensions = ext_list.rstrip("}").split(",")
                for ext in extensions:
                    pattern = f"{prefix}{ext.strip()}"
                    search_path = str(work_dir / pattern)
                    all_images.update(Path(p) for p in glob.glob(search_path))
            else:
                search_path = str(work_dir / base_pattern)
                all_images.update(Path(p) for p in glob.glob(search_path))

        images = sorted(all_images)

        if not images:
            console.print("[yellow]No matching images found")
            raise typer.Exit(1)

        # Process each image in each format
        for img_path in track(images, description="Processing..."):
            try:
                for fmt in formats:
                    process_image(img_path, fmt)
            except Exception as e:
                console.print(f"[red]Error processing {img_path}: {e}")

    except Exception as e:
        console.print(f"[red]Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
