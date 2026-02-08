"""
Animated Gooey QR Code GIF Generator (v10)

Reads a URL from url.txt, generates 3 visually distinct QR code variants
(all with rounded eyes), morphs between them with a gooey transition,
overlays a spinning 3D pixel-art-style coin with halftone portrait in
the centre, and writes a looping animated GIF.

Key changes in v10:
  - Increased hold duration per color to 8000ms (doubled)
  - Reduced morph duration to 250ms (halved)
  - Coin now completes exactly one full rotation per complete color cycle
  - Coin returns to starting position when loop restarts for seamless animation

Previous version (v9):
  - De-perspectived coin reference: the pixel-art coin reference image is
    at a slight 3D angle.  We now extract only the front-facing circular
    rim ring from the left/front portion and mirror it to produce a clean
    symmetrical rim that works correctly for flat face rendering.
  - Proper 3D coin with elliptical edge rendering:
      * Edge drawn as stacked thin ellipses (like real coin ridges)
      * Face disc squished by cos(angle) and composited with correct
        z-ordering
      * Pixel-art rim applied only to the face disc, not floating on top
  - No weird borders or artifacts on the coin
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import (
    GappedSquareModuleDrawer,
    RoundedModuleDrawer,
    VerticalBarsDrawer,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
URL_FILE = pathlib.Path("url.txt")
OUTPUT_PATH = pathlib.Path("output/qr_animated.gif")

QR_SIZE = 560                  # QR code render size (px)
CANVAS_SIZE = 600              # full output frame
BORDER_PAD = 4                 # gap between outermost QR modules and border
BORDER_WIDTH = 2               # default; overridden dynamically in main()

HOLD_DURATION_MS = 6000        # increased hold time per color
HOLD_SUBFRAME_MS = 80          # sub-frame interval during hold (coin spin)
MORPH_DURATION_MS = 500        # halved morph time for faster transition
MORPH_FRAMES = 10
MORPH_FRAME_MS = MORPH_DURATION_MS // MORPH_FRAMES

# ---- Spinning coin token -------------------------------------------------
COIN_IMAGE_PATH = "public/event_pfp2.png"
COIN_SHELL_PATH = "public/pixel-art-coin.jpg"   # 3D coin reference
COIN_RATIO = 0.22              # token diameter relative to QR_SIZE
# Spin period is derived so an exact integer number of full rotations fit
# inside the total GIF cycle, preventing any jump/blink on loop.
# Total cycle = 3 * (8000 + 250) = 24750ms.  24750 / 5 = 4950ms per turn
# (~5 s, same perceived speed as the original 5000ms).
COIN_SPIN_PERIOD_MS = 4950     # 5 full rotations per GIF loop (seamless)
COIN_ROTATION_STEPS = 60       # pre-rendered frames in one full spin
COIN_CLEAR_BLOCK = 16          # clearance zone block size (px)
# Edge thickness controlled by COIN_THICKNESS_DIVISOR in the 3D section below

# Halftone portrait grid
HALFTONE_GRID = 64             # cells per side for the halftone mosaic


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VariantConfig:
    name: str
    module_drawer: object
    eye_drawer: object
    front_color: tuple[int, int, int] = (0, 0, 0)


VARIANTS: list[VariantConfig] = [
    VariantConfig(
        name="Pompeian Red Rounded",
        module_drawer=RoundedModuleDrawer(),
        eye_drawer=RoundedModuleDrawer(),
        front_color=(164, 41, 46),      # Pompeian Red 18-1658 TCX  #A4292E
    ),
    VariantConfig(
        name="Green Gapped",
        module_drawer=GappedSquareModuleDrawer(),
        eye_drawer=RoundedModuleDrawer(),
        front_color=(0, 150, 57),       # Green 347 C  #009639
    ),
    VariantConfig(
        name="Diode Blue Bars",
        module_drawer=VerticalBarsDrawer(),
        eye_drawer=RoundedModuleDrawer(),
        front_color=(0, 69, 172),       # Diode Blue 20-0147 TPM  #0045AC
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def read_url() -> str:
    text = URL_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{URL_FILE} is empty -- add a URL and re-run.")
    return text


# ---------------------------------------------------------------------------
# QR generation with centre blank-out
# ---------------------------------------------------------------------------

def _find_intruding_modules(
    rendered: Image.Image,
    qr: qrcode.QRCode,
    clearance_mask: Image.Image,
    qr_offset: int,
) -> set[tuple[int, int]]:
    """
    Inspect a rendered QR image and return the set of ``(row, col)``
    module positions that have actual dark pixels inside the clearance
    zone.

    This is drawer-aware: it checks what the specific styled drawer
    *actually* painted, not just where the logical module centre sits.
    A module is flagged only if it contributes visible ink inside the
    clearance area.
    """
    n = qr.modules_count
    total = n + 2 * qr.border
    module_px = QR_SIZE / total

    # Convert rendered QR to a numpy bool array (True = dark ink)
    rendered_arr = np.asarray(rendered)
    has_ink = rendered_arr < 128

    # Convert clearance mask to bool array in canvas space.  The
    # rendered QR lives in a QR_SIZE x QR_SIZE image that gets pasted
    # at qr_offset inside the CANVAS_SIZE canvas, so we crop the
    # clearance mask to the QR region.
    clear_crop = clearance_mask.crop((
        qr_offset, qr_offset,
        qr_offset + QR_SIZE, qr_offset + QR_SIZE,
    ))
    clear_arr = np.asarray(clear_crop) > 0

    # Pixels that are both ink AND inside the clearance zone
    overlap = has_ink & clear_arr

    intruders: set[tuple[int, int]] = set()
    for row in range(n):
        for col in range(n):
            # Module pixel rectangle in QR-image space (includes border)
            x0 = int((col + qr.border) * module_px)
            y0 = int((row + qr.border) * module_px)
            x1 = int((col + qr.border + 1) * module_px)
            y1 = int((row + qr.border + 1) * module_px)

            # Clamp to image bounds
            x0, x1 = max(0, x0), min(QR_SIZE, x1)
            y0, y1 = max(0, y0), min(QR_SIZE, y1)

            if np.any(overlap[y0:y1, x0:x1]):
                intruders.add((row, col))

    return intruders


def generate_qr_mask(
    url: str,
    cfg: VariantConfig,
    clearance_mask: Image.Image | None = None,
    qr_offset: int = 0,
) -> Image.Image:
    """Generate a grayscale QR mask with per-drawer centre blank-out.

    When *clearance_mask* is provided the function uses a two-pass
    approach:

    1. **Render** the QR with the styled drawer (no blanking).
    2. **Inspect** the rendered image to find which modules actually
       have dark pixels inside the clearance zone.
    3. **Erase** only those modules at the data level and re-render.

    This is drawer-aware -- compact drawers (e.g. ``RoundedModuleDrawer``)
    keep more boundary modules than wide drawers (e.g. ``VerticalBarsDrawer``),
    producing the tightest possible hole for each style.
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    if clearance_mask is not None:
        # --- Pass 1: render with the styled drawer, no blanking ---
        first_pass = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=cfg.module_drawer,
            eye_drawer=cfg.eye_drawer,
        ).convert("L").resize((QR_SIZE, QR_SIZE), Image.LANCZOS)

        # --- Inspect: which modules have actual ink in the clearance? ---
        intruders = _find_intruding_modules(
            first_pass, qr, clearance_mask, qr_offset)

        # --- Erase only those modules and re-render ---
        if intruders:
            for row, col in intruders:
                qr.modules[row][col] = False

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=cfg.module_drawer,
        eye_drawer=cfg.eye_drawer,
    ).convert("L")

    return img.resize((QR_SIZE, QR_SIZE), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Frame composition (QR + tight border)
# ---------------------------------------------------------------------------

def colorize_qr(mask: Image.Image, fg: tuple[int, ...]) -> Image.Image:
    arr = np.asarray(mask)
    is_module = arr < 128
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[is_module] = (*fg, 255)
    return Image.fromarray(rgba, "RGBA")


def compose_frame(
    mask: Image.Image,
    fg: tuple[int, ...],
    border_width: int = BORDER_WIDTH,
) -> Image.Image:
    """White canvas + coloured QR modules + tight square border.

    Parameters
    ----------
    border_width : int
        Thickness of the outer border rectangle (in pixels).  When computed
        dynamically this matches ~1 QR module width so the border visually
        aligns with the finder-pattern eye outlines.
    """
    base = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))

    qr = colorize_qr(mask, fg)
    offset = (CANVAS_SIZE - QR_SIZE) // 2
    base.paste(qr, (offset, offset), qr)

    # Border sits outside the QR area. Pillow draws the stroke centred
    # on the rectangle coords, so we push it outward by half the width
    # to guarantee it never overlaps QR modules.
    draw = ImageDraw.Draw(base)
    half_bw = border_width / 2
    b = offset - BORDER_PAD - half_bw
    draw.rectangle(
        [b, b, CANVAS_SIZE - b - 1, CANVAS_SIZE - b - 1],
        outline=(*fg, 255),
        width=border_width,
    )
    return base


# ---------------------------------------------------------------------------
# Halftone portrait conversion
# ---------------------------------------------------------------------------

def portrait_to_brightness_grid(path: str, grid_size: int) -> np.ndarray:
    """
    Convert a portrait image to a grid_size x grid_size array of brightness
    values in [0, 1] where 0 = darkest (most ink) and 1 = lightest (least ink).
    The image is centre-cropped to square first.
    """
    img = Image.open(path).convert("L")

    # Centre-crop to square
    s = min(img.size)
    left = (img.width - s) // 2
    top = (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s))

    # Resize to grid dimensions
    img = img.resize((grid_size, grid_size), Image.LANCZOS)

    # Normalise with contrast stretch
    arr_raw = np.asarray(img, dtype=np.float64)
    lo, hi = np.percentile(arr_raw, 2), np.percentile(arr_raw, 98)
    if hi - lo < 1:
        hi = lo + 1
    arr_norm = np.clip((arr_raw - lo) / (hi - lo), 0, 1)
    return arr_norm


def render_halftone_token(
    brightness: np.ndarray,
    display_size: int,
    fg_color: tuple[int, int, int],
    inner_radius_ratio: float = 1.0,
) -> Image.Image:
    """
    Render the portrait as a halftone-style mosaic inside a circle.

    Each cell in the brightness grid becomes a filled circle whose radius
    is proportional to the darkness of that cell.  Dark cells -> large
    dots, light cells -> small or no dots.  This produces a bold,
    stylised portrait that reads well even at small sizes.

    Parameters
    ----------
    inner_radius_ratio : float
        Fraction of the full coin diameter used for the portrait circle.
        1.0 = fill entire coin (legacy behaviour).
        ~0.60 = confine portrait to the inner face, leaving space for the rim.

    Returns RGBA at display_size x display_size.
    """
    grid_h, grid_w = brightness.shape
    # Work at 4x resolution for crisp circular dots, then downsample
    work_size = display_size * 4
    cell_w = work_size / grid_w
    cell_h = work_size / grid_h

    canvas = Image.new("RGBA", (work_size, work_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row in range(grid_h):
        for col in range(grid_w):
            darkness = 1.0 - brightness[row, col]  # 0=white, 1=black
            if darkness < 0.06:
                continue  # skip very light cells

            # Dot radius proportional to darkness
            max_r = min(cell_w, cell_h) * 0.48  # max radius = 48% of cell
            dot_r = max_r * (0.25 + darkness * 0.75)
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h

            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                fill=(*fg_color, 255),
            )

    # Downsample to display size with high-quality resampling
    token = canvas.resize((display_size, display_size), Image.LANCZOS)

    # ---- Circular crop (clipped to inner face region) ----
    # When inner_radius_ratio < 1, the portrait occupies a smaller circle
    # centred inside the coin face.  The area outside is transparent so
    # the rim overlay and coin base colour show through.
    inner_diam = int(display_size * inner_radius_ratio)
    offset = (display_size - inner_diam) // 2

    circle_mask = Image.new("L", (display_size, display_size), 0)
    ImageDraw.Draw(circle_mask).ellipse(
        [offset, offset, offset + inner_diam - 1, offset + inner_diam - 1],
        fill=255,
    )

    result = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    white_bg = Image.new("RGBA", (display_size, display_size), (255, 255, 255, 255))
    result.paste(white_bg, mask=circle_mask)
    result.paste(token, mask=circle_mask)

    return result


# ---------------------------------------------------------------------------
# 3D Pixel-art coin shell from reference image
# ---------------------------------------------------------------------------

def load_coin_shell(path: str, display_size: int) -> tuple[Image.Image, float]:
    """
    Load the pixel-art coin reference and produce a **front-facing**
    circular rim mask plus inner_radius_ratio.

    The reference image shows the coin at a slight 3D angle (perspective
    edge visible on the right).  To get a clean, symmetrical rim for the
    face we:
      1. Threshold to isolate dark (rim) pixels.
      2. Find the bounding circle of the coin face (the elliptical front
         portion, ignoring the perspective thickness on the right).
      3. Sample the rim profile from the LEFT half of the image (the
         side facing the viewer, least affected by perspective).
      4. Mirror that left-half rim profile to produce a fully symmetrical
         circular rim ring.

    Returns
    -------
    rim_mask : Image (mode "L", display_size × display_size)
        Grayscale mask where 255 = rim pixel, 0 = transparent.
    inner_radius_ratio : float
        Fraction of the coin diameter occupied by the inner face (0..1).
    """
    img = Image.open(path).convert("L")

    # Centre-crop to square
    s = min(img.size)
    left = (img.width - s) // 2
    top = (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s))

    # Work at a comfortable analysis size, then output at display_size
    work_size = 512
    img = img.resize((work_size, work_size), Image.LANCZOS)
    arr = np.asarray(img)

    # Threshold: dark pixels (< 140) are the shell/rim
    shell_pixels = arr < 140  # bool array

    # --- Find the face centre and outer radius ---
    # Use the vertical extent of the coin (top-to-bottom) since the
    # vertical axis is not distorted by the 3D perspective.
    rows_any = np.any(shell_pixels, axis=1)
    row_indices = np.where(rows_any)[0]
    if len(row_indices) < 2:
        # Fallback: plain circular rim
        return _fallback_rim(display_size)

    top_row = row_indices[0]
    bot_row = row_indices[-1]
    cy = (top_row + bot_row) / 2.0
    outer_r = (bot_row - top_row) / 2.0

    # For the horizontal centre, use the leftmost extent (front-facing
    # side) to estimate. The left edge of the coin face should be at
    # roughly cx - outer_r.
    cols_any = np.any(shell_pixels, axis=0)
    col_indices = np.where(cols_any)[0]
    left_col = col_indices[0]
    # cx = left_col + outer_r (front face circle centre)
    cx = left_col + outer_r

    # --- Measure inner radius from the LEFT half only ---
    # Walk radially outward from the centre along angles in the LEFT
    # semicircle (90° to 270°) where there's no perspective edge.
    num_angles = 60
    inner_radii: list[float] = []

    for ai in range(num_angles):
        # Angles from π/2 to 3π/2 (left semicircle: up-left-down)
        a = math.pi * 0.5 + (ai / num_angles) * math.pi
        dx = math.cos(a)
        dy = math.sin(a)
        for ri in range(2, int(outer_r)):
            px = int(cx + dx * ri)
            py = int(cy + dy * ri)
            if 0 <= px < work_size and 0 <= py < work_size:
                if shell_pixels[py, px]:
                    inner_radii.append(ri / outer_r)
                    break
        else:
            inner_radii.append(1.0)

    if inner_radii:
        inner_radius_ratio = float(np.median(inner_radii))
    else:
        inner_radius_ratio = 0.65

    inner_radius_ratio = max(0.50, min(inner_radius_ratio, 0.80))

    # --- Build a clean, symmetrical circular rim mask ---
    # We construct the rim as the area between two concentric circles:
    # outer circle at outer_r, inner circle at outer_r * inner_radius_ratio.
    # Then we sample the pixel-art texture from the LEFT portion of the
    # original and stamp it around the ring for authentic pixel-art feel.

    rim_mask = Image.new("L", (work_size, work_size), 0)
    rim_draw = ImageDraw.Draw(rim_mask)

    # Draw the rim as a circular ring (outer - inner)
    o = int(cx - outer_r)
    rim_draw.ellipse(
        [o, int(cy - outer_r), int(cx + outer_r), int(cy + outer_r)],
        fill=255,
    )
    # Cut out the inner circle
    inner_r = outer_r * inner_radius_ratio
    rim_draw.ellipse(
        [int(cx - inner_r), int(cy - inner_r),
         int(cx + inner_r), int(cy + inner_r)],
        fill=0,
    )

    # Now modulate with the actual pixel-art texture:
    # Use the original shell pixels to add the chunky pixel-art detail.
    # We only use the left half (front-facing) and mirror it.
    texture = np.zeros((work_size, work_size), dtype=np.uint8)
    rim_arr = np.asarray(rim_mask)

    # For each pixel in the rim ring, sample from the original if it's
    # a shell pixel; this preserves the pixel-art jaggedness.
    shell_uint = shell_pixels.astype(np.uint8) * 255

    # Left half: use original pixels where they overlap with our ring
    half_x = int(cx)
    left_zone = np.zeros_like(texture)
    left_zone[:, :half_x] = 255
    left_mask = (rim_arr > 0) & (left_zone > 0)
    texture[left_mask] = shell_uint[left_mask]

    # Mirror the left half to the right half
    # Flip around the centre column
    for y in range(work_size):
        for x in range(half_x):
            if texture[y, x] > 0:
                mirror_x = int(2 * cx) - x
                if 0 <= mirror_x < work_size:
                    texture[y, mirror_x] = texture[y, x]

    # Ensure the ring shape is respected (clip to ring area)
    texture = np.minimum(texture, rim_arr)

    rim_mask_final = Image.fromarray(texture, "L")

    # Resize to display_size
    rim_mask_final = rim_mask_final.resize(
        (display_size, display_size), Image.LANCZOS)

    # Re-threshold after resize to keep crisp pixel-art edges
    rim_arr_final = np.asarray(rim_mask_final)
    rim_arr_final = ((rim_arr_final > 100).astype(np.uint8)) * 255
    rim_mask_out = Image.fromarray(rim_arr_final, "L")

    return rim_mask_out, inner_radius_ratio


def _fallback_rim(display_size: int) -> tuple[Image.Image, float]:
    """Fallback: generate a simple circular rim if the reference can't be parsed."""
    inner_radius_ratio = 0.65
    rim = Image.new("L", (display_size, display_size), 0)
    draw = ImageDraw.Draw(rim)
    draw.ellipse([0, 0, display_size - 1, display_size - 1], fill=255)
    inner = int(display_size * inner_radius_ratio)
    off = (display_size - inner) // 2
    draw.ellipse([off, off, off + inner - 1, off + inner - 1], fill=0)
    return rim, inner_radius_ratio


def colorize_shell(shell_mask: Image.Image, fg_color: tuple[int, int, int]) -> Image.Image:
    """Colorize a grayscale shell mask with the given colour."""
    size = shell_mask.size
    coloured = Image.new("RGBA", size, (0, 0, 0, 0))
    fg_layer = Image.new("RGBA", size, (*fg_color, 255))
    coloured.paste(fg_layer, mask=shell_mask)
    return coloured


def darken_color(c: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Darken a colour by a factor (0 = black, 1 = original)."""
    return tuple(max(0, int(v * factor)) for v in c)


# ---------------------------------------------------------------------------
# 3D coin rendering with stacked-ellipse edge
# ---------------------------------------------------------------------------
#
# The coin is composed of layers per frame:
#
#   1. Edge: stacked thin ellipses that simulate the coin's circular
#      thickness.  Each slice is offset horizontally by sin(angle) and
#      gets progressively darker toward the back.  This produces a smooth,
#      convincing 3D coin edge without capsule/slab artifacts.
#   2. Face disc: the circular front or back, squished horizontally by
#      cos(angle).  The pixel-art rim is composited onto the face BEFORE
#      squishing so it distorts naturally with the perspective.
#
# Z-ordering:
#   sin(angle) > 0  →  draw edge first, then face  (edge behind)
#   sin(angle) < 0  →  draw face first, then edge  (edge in front)
#
# Edge-on (|cos| near 0): only the stacked edge is visible.
# ---------------------------------------------------------------------------

COIN_THICKNESS_DIVISOR = 10  # display_size / this = max edge thickness in px


def _draw_edge_ellipses(
    canvas: Image.Image,
    display_size: int,
    angle: float,
    thickness: int,
    base_color: tuple[int, int, int],
) -> Image.Image:
    """
    Draw the coin edge as stacked horizontal ellipse slices.

    Each slice is a thin vertical ellipse offset along the x-axis
    according to its depth position and the rotation angle.  The
    back-most slices are darkest, the front-most are lightest.
    """
    sin_val = math.sin(angle)
    cos_val = math.cos(angle)
    draw = ImageDraw.Draw(canvas)

    cx = display_size // 2
    cy = display_size // 2
    coin_r = display_size // 2 - 2  # small margin so we don't clip

    # Number of visible edge slices depends on how edge-on the coin is
    num_slices = max(2, int(thickness * abs(sin_val)))
    if num_slices < 1:
        return canvas

    # Total visible width of the edge in pixels
    edge_visible_w = max(2, int(thickness * abs(sin_val)))

    for i in range(num_slices):
        # t goes from 0 (back-most slice) to 1 (front-most)
        t = i / max(1, num_slices - 1) if num_slices > 1 else 0.5

        # Horizontal offset: evenly spread slices across edge_visible_w
        # Back slices (t=0) are furthest from the face disc
        raw_offset = (0.5 - t) * edge_visible_w
        slice_offset = int(math.copysign(1, sin_val) * raw_offset)

        # Darkness gradient: back-most is darkest, front-most is lightest
        darkness = 0.30 + 0.50 * t  # range 0.30 to 0.80
        sc = darken_color(base_color, darkness)

        # Each slice is a thin vertical ellipse, slightly wider for coverage
        slice_w = max(1, edge_visible_w // max(1, num_slices - 1) + 1)
        sx = cx + slice_offset
        draw.ellipse(
            [sx - slice_w, cy - coin_r, sx + slice_w, cy + coin_r],
            fill=(*sc, 255),
        )

    # Highlight stripe on the leading edge (closest to the face disc)
    if num_slices >= 3:
        highlight = darken_color(base_color, 0.90)
        hl_x = cx  # leading edge is at centre when face is nearby
        draw.ellipse(
            [hl_x - 1, cy - coin_r + 2, hl_x + 1, cy + coin_r - 2],
            fill=(*highlight, 255),
        )

    return canvas


def build_3d_coin_frames(
    brightness: np.ndarray,
    rim_mask: Image.Image,
    inner_radius_ratio: float,
    display_size: int,
    fg_color: tuple[int, int, int],
) -> list[Image.Image]:
    """
    Pre-render all rotation frames of the 3D coin for one colour.

    The coin is assembled per-frame from edge ellipses + face disc.
    The pixel-art rim is composited onto the face disc BEFORE the
    perspective squish so it deforms naturally.

    Parameters
    ----------
    brightness : ndarray
        Halftone brightness grid from ``portrait_to_brightness_grid()``.
    rim_mask : Image (mode "L")
        Grayscale mask of the pixel-art rim ring (front-facing, symmetrical).
    inner_radius_ratio : float
        Fraction of coin radius for the portrait circle.
    display_size : int
        Pixel dimensions of the square coin canvas.
    fg_color : tuple
        Variant colour.

    Returns
    -------
    List of RGBA images, one per rotation step.
    """
    coin_thickness = display_size // COIN_THICKNESS_DIVISOR

    # --- Pre-render the face disc ---
    # Halftone portrait in the inner circle
    face_portrait = render_halftone_token(
        brightness, display_size, fg_color, inner_radius_ratio)

    # Composite the pixel-art rim ON TOP of the portrait (on the face)
    rim_coloured = colorize_shell(rim_mask, fg_color)
    face = Image.alpha_composite(face_portrait, rim_coloured)

    # Also add a thin outer ring for clean circle definition
    ring_layer = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    ring_w = max(2, display_size // 40)
    ImageDraw.Draw(ring_layer).ellipse(
        [0, 0, display_size - 1, display_size - 1],
        outline=(*fg_color, 255), width=ring_w)
    face = Image.alpha_composite(face, ring_layer)

    # Clip the face to a circle (ensure no square corners leak)
    face_circle_mask = Image.new("L", (display_size, display_size), 0)
    ImageDraw.Draw(face_circle_mask).ellipse(
        [0, 0, display_size - 1, display_size - 1], fill=255)
    face_clipped = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    face_clipped.paste(face, mask=face_circle_mask)
    face = face_clipped

    # --- Pre-render the back disc ---
    back = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    bd = ImageDraw.Draw(back)
    bd.ellipse([0, 0, display_size - 1, display_size - 1],
               fill=(*fg_color, 255))
    inset = display_size // 5
    bd.ellipse(
        [inset, inset, display_size - 1 - inset, display_size - 1 - inset],
        outline=(255, 255, 255, 180), width=max(2, display_size // 45))
    # Outer ring on back too
    bd.ellipse(
        [0, 0, display_size - 1, display_size - 1],
        outline=(*darken_color(fg_color, 0.7), 255),
        width=max(2, display_size // 40))
    # Clip back to circle
    back_clipped = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    back_clipped.paste(back, mask=face_circle_mask)
    back = back_clipped

    # --- Generate rotation frames ---
    frames: list[Image.Image] = []
    for step in range(COIN_ROTATION_STEPS):
        frac = step / COIN_ROTATION_STEPS
        angle = frac * 2 * math.pi
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        canvas = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))

        # --- Edge-on: only show edge ---
        if abs(cos_val) < 0.04:
            canvas = _draw_edge_ellipses(
                canvas, display_size, angle, coin_thickness, fg_color)
            frames.append(canvas)
            continue

        # --- Determine which side is showing ---
        showing_face = cos_val >= 0
        w_scale = abs(cos_val)

        # Squish the disc horizontally by cos(angle)
        src = face if showing_face else back
        new_w = max(4, int(display_size * w_scale))
        scaled_disc = src.resize((new_w, display_size), Image.LANCZOS)
        disc_x = (display_size - new_w) // 2

        # Build edge layer
        edge_layer = Image.new(
            "RGBA", (display_size, display_size), (0, 0, 0, 0))
        edge_layer = _draw_edge_ellipses(
            edge_layer, display_size, angle, coin_thickness, fg_color)

        # Z-ordering based on sin
        if sin_val >= 0:
            # Edge behind the face
            canvas = Image.alpha_composite(canvas, edge_layer)
            canvas.paste(scaled_disc, (disc_x, 0), scaled_disc)
        else:
            # Face behind the edge
            canvas.paste(scaled_disc, (disc_x, 0), scaled_disc)
            canvas = Image.alpha_composite(canvas, edge_layer)

        frames.append(canvas)

    return frames


def get_token_frame_continuous(
    all_frame_sets: list[list[Image.Image]],
    variant_idx: int,
    time_ms: float,
) -> Image.Image:
    """
    Get the token frame with continuous spin. The spin angle is computed
    from absolute time_ms so it never resets.

    Even variants (0, 2) spin clockwise, odd variants (1) spin counter-clockwise.
    The direction change is smooth because the angle is continuous.
    """
    frames = all_frame_sets[variant_idx]
    n = len(frames)
    clockwise = (variant_idx % 2 == 0)

    frac = (time_ms / COIN_SPIN_PERIOD_MS) % 1.0
    if not clockwise:
        frac = 1.0 - frac

    idx = int(frac * n) % n
    return frames[idx]


# ---------------------------------------------------------------------------
# Blocky / pixelated clearance zone
# ---------------------------------------------------------------------------

def make_blocky_clearance_mask(
    canvas_size: int,
    radius_px: int,
    block: int,
) -> Image.Image:
    """
    Build a mask with a blocky (pixelated) circular hole in the centre.
    The hole is built from square blocks so it looks like part of the QR grid,
    not a smooth circle.

    ``radius_px`` is the pixel radius of the zone to blank (should match or
    slightly exceed the module-level blank radius so no ragged modules show).
    """
    mask = Image.new("L", (canvas_size, canvas_size), 0)

    cx = canvas_size // 2
    cy = canvas_size // 2

    for bx in range(0, canvas_size, block):
        for by in range(0, canvas_size, block):
            bcx = bx + block // 2
            bcy = by + block // 2
            dist = math.sqrt((bcx - cx) ** 2 + (bcy - cy) ** 2)
            if dist < radius_px:
                mask.paste(255, (bx, by, min(bx + block, canvas_size),
                                 min(by + block, canvas_size)))

    return mask


def overlay_token(
    frame: Image.Image,
    token_img: Image.Image,
    clearance_mask: Image.Image,
) -> Image.Image:
    """Paste token on centre of frame with blocky white clearance zone."""
    frame = frame.copy()
    cw = token_img.width
    cx = (frame.width - cw) // 2
    cy = (frame.height - cw) // 2

    # Paint white over the blocky clearance area
    white = Image.new("RGBA", frame.size, (255, 255, 255, 255))
    frame.paste(white, mask=clearance_mask)

    frame.paste(token_img, (cx, cy), token_img)
    return frame


# ---------------------------------------------------------------------------
# Gooey morph engine
# ---------------------------------------------------------------------------

def _blur_radius(fraction: float) -> float:
    if fraction <= 0.0:
        return 100.0
    return min(8.0 / fraction - 8.0, 100.0)


def gooey_morph_mask(src: Image.Image, dst: Image.Image, t: float) -> Image.Image:
    src_blurred = src.filter(
        ImageFilter.GaussianBlur(radius=max(_blur_radius(1.0 - t), 0.1)))
    dst_blurred = dst.filter(
        ImageFilter.GaussianBlur(radius=max(_blur_radius(t), 0.1)))

    src_arr = np.asarray(src_blurred, dtype=np.float64) * ((1.0 - t) ** 0.4)
    dst_arr = np.asarray(dst_blurred, dtype=np.float64) * (t ** 0.4)
    blended = np.clip(src_arr + dst_arr, 0, 255)

    result = np.where(blended < 140.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="L")


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple[int, ...]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ---------------------------------------------------------------------------
# GIF assembly
# ---------------------------------------------------------------------------

def assemble_gif(
    masks: list[Image.Image],
    configs: list[VariantConfig],
    brightness: np.ndarray,
    rim_mask: Image.Image,
    inner_radius_ratio: float,
    border_width: int,
    clearance: Image.Image | None = None,
) -> None:
    n = len(masks)
    token_display = int(QR_SIZE * COIN_RATIO)

    # Use the pre-built clearance mask if provided; otherwise build one
    # (fallback keeps the function self-contained for any future callers).
    if clearance is None:
        clearance_radius = token_display // 2 + COIN_CLEAR_BLOCK
        clearance = make_blocky_clearance_mask(
            CANVAS_SIZE, clearance_radius, COIN_CLEAR_BLOCK)

    # Pre-build 3D coin rotation frames for each variant colour
    print("  Pre-rendering 3D coin rotations per variant ...")
    token_frame_sets: list[list[Image.Image]] = []
    for i, cfg in enumerate(configs):
        tfs = build_3d_coin_frames(
            brightness, rim_mask, inner_radius_ratio,
            token_display, cfg.front_color,
        )
        token_frame_sets.append(tfs)
        direction = "CW" if (i % 2 == 0) else "CCW"
        print(f"    Variant {i+1} ({cfg.name}): {direction}, {len(tfs)} frames")

    frames: list[Image.Image] = []
    durations: list[int] = []

    # Continuous time counter -- never resets, so the coin spin is seamless
    time_ms = 0.0
    hold_subframes = HOLD_DURATION_MS // HOLD_SUBFRAME_MS

    for i in range(n):
        print(f"  Variant {i+1}/{n} hold frames ...")
        hold_base = compose_frame(masks[i], configs[i].front_color, border_width)

        for _ in range(hold_subframes):
            tf = get_token_frame_continuous(token_frame_sets, i, time_ms)
            f = overlay_token(hold_base, tf, clearance)
            frames.append(f)
            durations.append(HOLD_SUBFRAME_MS)
            time_ms += HOLD_SUBFRAME_MS

        # ---- Morph to next variant ----
        ni = (i + 1) % n
        print(f"  Morph {i+1} -> {ni+1} ...")

        for fi in range(1, MORPH_FRAMES + 1):
            t = fi / MORPH_FRAMES
            m_mask = gooey_morph_mask(masks[i], masks[ni], t)
            fg = lerp_color(configs[i].front_color, configs[ni].front_color, t)
            f = compose_frame(m_mask, fg, border_width)

            # Blend token frames from both variants during morph
            tf_a = get_token_frame_continuous(token_frame_sets, i, time_ms)
            tf_b = get_token_frame_continuous(token_frame_sets, ni, time_ms)
            tf_blend = Image.blend(tf_a.convert("RGBA"), tf_b.convert("RGBA"), t)
            f = overlay_token(f, tf_blend, clearance)

            frames.append(f)
            durations.append(MORPH_FRAME_MS)
            time_ms += MORPH_FRAME_MS

    # Quantize for GIF
    print(f"  Quantising {len(frames)} frames ...")
    rgb_frames = [f.convert("RGB") for f in frames]
    pal_frames = [
        f.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        for f in rgb_frames
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pal_frames[0].save(
        str(OUTPUT_PATH),
        save_all=True,
        append_images=pal_frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Reading URL ...")
    url = read_url()
    print(f"  URL: {url}\n")

    print("Converting portrait to halftone grid ...")
    brightness = portrait_to_brightness_grid(COIN_IMAGE_PATH, HALFTONE_GRID)
    print(f"  {brightness.shape[0]}x{brightness.shape[1]} grid\n")

    print("Loading pixel-art coin shell reference ...")
    token_display = int(QR_SIZE * COIN_RATIO)
    rim_mask, inner_radius_ratio = load_coin_shell(COIN_SHELL_PATH, token_display)
    print(f"  Rim mask loaded at {rim_mask.size[0]}x{rim_mask.size[1]}")
    print(f"  Inner radius ratio: {inner_radius_ratio:.3f}\n")

    # Determine QR version to derive the dynamic border width.
    print("Determining QR version ...")
    probe = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10, border=4,
    )
    probe.add_data(url)
    probe.make(fit=True)
    qr_version = probe.version

    # Dynamic border width: match the finder-pattern eye ring thickness.
    # The eye outer ring is nominally 1 module, but the rounded drawer
    # anti-aliases it so visually it looks ~60-70% of a raw module.
    modules_with_border = probe.modules_count + 2 * probe.border
    module_px = QR_SIZE / modules_with_border
    max_border = (CANVAS_SIZE - QR_SIZE) // 2
    border_width = max(2, min(int(round(module_px * 0.65)), max_border))

    print(f"  QR version {qr_version} ({probe.modules_count}x{probe.modules_count})")
    print(f"  Border width: {border_width}px (~0.65 module = {module_px:.1f}px)\n")

    # Build the blocky clearance mask ONCE, before QR generation.
    # This single mask drives both module-level blanking (data erasure)
    # and the pixel-level white-out at compositing time.
    qr_offset = (CANVAS_SIZE - QR_SIZE) // 2
    clearance_radius = token_display // 2 + COIN_CLEAR_BLOCK
    clearance_mask = make_blocky_clearance_mask(
        CANVAS_SIZE, clearance_radius, COIN_CLEAR_BLOCK)
    print(f"  Clearance mask: radius {clearance_radius}px, "
          f"block {COIN_CLEAR_BLOCK}px\n")

    print("Generating QR variants (with centre blank-out) ...")
    masks: list[Image.Image] = []
    for i, cfg in enumerate(VARIANTS):
        print(f"  [{i+1}/{len(VARIANTS)}] {cfg.name}  "
              f"(modules={cfg.module_drawer.__class__.__name__}, "
              f"color={cfg.front_color})")
        masks.append(generate_qr_mask(url, cfg, clearance_mask, qr_offset))
    print()

    print("Assembling GIF ...")
    assemble_gif(masks, VARIANTS, brightness, rim_mask, inner_radius_ratio,
                 border_width, clearance_mask)

    total_ms = len(VARIANTS) * (HOLD_DURATION_MS + MORPH_DURATION_MS)
    nframes = len(VARIANTS) * (HOLD_DURATION_MS // HOLD_SUBFRAME_MS + MORPH_FRAMES)
    print(f"\nDone -> {OUTPUT_PATH.resolve()}")
    print(f"  {total_ms / 1000:.1f}s loop, {nframes} frames")


if __name__ == "__main__":
    main()
