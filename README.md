# Instamatte

Batch process images to multiple social media formats with precise sizing and elegant whitespace.

## Core Concept

Instamatte helps content creators by solving two common issues:
- Converting the same image into multiple formats (stories, carousels, posts)
- Achieving aesthetically pleasing whitespace proportions automatically

### Key Features

- Surface area targeting: optimize image size while maintaining aspect ratio
- Minimum margin enforcement: ensure elegant whitespace around images
- Multiple output formats from single source
- Automatic format-specific directories
- Standard colors and dimensions for social platforms
- Clear progress reporting with achieved metrics

## Installation

```bash
pip install instamatte
```

## Quick Start

1. Create a `config.yaml` in your images directory:
```yaml
- output_dir: stories
  frame_width: 1080
  frame_height: 1920
  target_surface_pct: 80
  margin_pct: 5
```

2. Run:
```bash
instamatte /path/to/images
```

## Configuration

Each format is configured with these parameters:

### Required

- `output_dir`: Directory for processed images
  - Example: `stories`, `carousels`
  - Created automatically if doesn't exist

### Optional

- `frame_width`: Output frame width in pixels
  - Default: 1080
  - Common values: 1080 (standard), 1200 (high-res)

- `frame_height`: Output frame height in pixels
  - Default: 1920
  - Common values: 1920 (stories), 1350 (carousel), 1080 (square)

- `target_surface_pct`: Target image area as frame percentage
  - Default: 85.0
  - Range: 0-100
  - Higher values = larger image
  - Actual may be lower if constrained by margin

- `margin_pct`: Minimum margin as percentage of smallest frame dimension
  - Default: 5.0
  - Range: 0-50
  - Example: 5.0 in 1080x1920 = minimum 54px margin
  - Takes precedence over target_surface_pct

- `pattern`: Glob pattern for input files
  - Default: "*.{jpg,jpeg,png,gif,bmp}"
  - Supports multiple extensions: "*.{jpg,png}"
  - Can be format-specific: "posts/*.jpg"

- `bg_color`: Background color name or hex
  - Default: "WHITE"
  - Examples: "BLACK", "#FFFFFF"

## Example Configurations

### Instagram Story and Carousel
```yaml
- output_dir: stories
  frame_width: 1080
  frame_height: 1920
  target_surface_pct: 80
  margin_pct: 5

- output_dir: carousels
  frame_width: 1080
  frame_height: 1350
  target_surface_pct: 90
  margin_pct: 3
```

### High-res Posts with Black Background
```yaml
- output_dir: posts
  frame_width: 1200
  frame_height: 1200
  target_surface_pct: 95
  margin_pct: 2.5
  bg_color: "#000000"
```

## Python API

For programmatic use or custom workflows:

```python
from pathlib import Path
from instamatte import FormatConfig, process_image

config = FormatConfig(
    output_dir="output",
    frame_width=1080,
    frame_height=1920,
    target_surface_pct=85,
    margin_pct=5,
)

process_image(Path("input.jpg"), config)
```

## How It Works

1. **Surface Targeting**:
   - Calculate maximum possible image dimensions
   - Try to achieve target surface percentage
   - Maintain original aspect ratio

2. **Margin Enforcement**:
   - If target surface would violate minimum margin
   - Reduce image size to respect margin
   - Equal margins on all sides

3. **Output Generation**:
   - Create white (or specified color) background
   - Center and paste processed image
   - Save with high quality (95%)
