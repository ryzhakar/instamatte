"""Configuration models and loading for social media formats."""

import textwrap
from pathlib import Path
from typing import List

import yaml
from PIL import ImageColor
from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
import typer

console = Console()

CONFIG_FILENAME = "instamatte.yaml"
MAX_MAT_THICKNESS_PERCENT = 50.0


class SocialFormat(BaseModel):
    """Social media format specification with dimensions and styling."""

    output_dir: str = "instamatte-processed"
    canvas_width: int = Field(default=1080, ge=1)
    canvas_height: int = Field(default=1920, ge=1)
    fill_percentage: float = Field(default=80.0, gt=0, le=100)
    mat_thickness: float = Field(
        default=5.0,
        ge=0,
        le=MAX_MAT_THICKNESS_PERCENT,
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


def create_sample_configuration_file(config_path: Path) -> None:
    """Create sample configuration file with default social format."""
    try:
        with config_path.open("w") as f:
            yaml.dump([SocialFormat().model_dump()], f, sort_keys=False)
        console.print(f"[green]Created default config at {config_path}")
    except (PermissionError, OSError) as e:
        console.print(f"[red]Error creating config file: {e}")
        raise


def load_social_format_configurations(config_path: Path) -> List[SocialFormat]:
    """Load social media format configurations from YAML file."""
    try:
        with config_path.open() as f:
            raw_configurations = yaml.safe_load(f)
            return [
                SocialFormat.model_validate(format_config)
                for format_config in raw_configurations
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


def discover_configuration_file(work_directory: Path) -> Path:
    """Discover configuration file in work directory, creating if needed."""
    config_path = work_directory / CONFIG_FILENAME

    if not config_path.exists():
        create_sample_configuration_file(config_path)

    return config_path


def prepare_format_output_directory(
    format_config: SocialFormat, work_directory: Path
) -> SocialFormat:
    """Prepare output directory for format processing."""
    absolute_output_path = work_directory / format_config.output_dir
    format_config.output_dir = str(absolute_output_path)

    try:
        absolute_output_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        console.print(
            textwrap.dedent(f"""
            [red]Error creating output directory
            {format_config.output_dir}: {e}
        """).strip()
        )
        raise

    return format_config
