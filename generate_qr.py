"""
Animated Gooey QR Code GIF Generator (v8)

Reads a URL from url.txt, generates 3 visually distinct QR code variants
(all with rounded eyes), morphs between them with a gooey transition,
overlays a spinning 3D pixel-art-style coin with halftone portrait in
the centre, and writes a looping animated GIF.

Key changes in v8:
  - Proper slab-based 3D coin rendering with correct z-ordering:
      * Thickness slab behind/in-front based on sin(angle)
      * Face disc squished by cos(angle)
      * Pixel-art rim overlay composited on top
  - Halftone portrait clipped to inner face area (~65% of coin radius)
  - QR border thickness dynamically matches finder-pattern eye width (~1 module)
  - COIN_EDGE_LAYERS removed; replaced by single COIN_THICKNESS slab
  - Pixel-art reference parsed into rim ring and inner face boundary
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

HOLD_DURATION_MS = 4000
HOLD_SUBFRAME_MS = 80          # sub-frame interval during hold (coin spin)
MORPH_DURATION_MS = 1000
MORPH_FRAMES = 15
MORPH_FRAME_MS = MORPH_DURATION_MS // MORPH_FRAMES

# ---- Spinning coin token -------------------------------------------------
COIN_IMAGE_PATH = "public/event_pfp2.png"
COIN_SHELL_PATH = "public/pixel-art-coin.jpg"   # 3D coin reference
COIN_RATIO = 0.34              # token diameter relative to QR_SIZE
COIN_SPIN_PERIOD_MS = 5000     # 5 seconds per full rotation (slower)
COIN_ROTATION_STEPS = 60       # pre-rendered frames in one full spin
COIN_CLEAR_BLOCK = 20          # clearance zone block size (px)
# (COIN_EDGE_LAYERS removed — replaced by slab-based 3D rendering)

# How many QR modules (radius) to blank in the centre for the token.
CENTER_BLANK_EXTRA = 0         # keep tight; clearance mask handles visual padding

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
        name="Coral Rounded",
        module_drawer=RoundedModuleDrawer(),
        eye_drawer=RoundedModuleDrawer(),
        front_color=(210, 60, 50),
    ),
    VariantConfig(
        name="Pink Gapped",
        module_drawer=GappedSquareModuleDrawer(),
        eye_drawer=RoundedModuleDrawer(),
        front_color=(220, 50, 140),
    ),
    VariantConfig(
        name="Purple Bars",
        module_drawer=VerticalBarsDrawer(),
        eye_drawer=RoundedModuleDrawer(),
        front_color=(100, 40, 180),
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

def _blank_center_modules(qr: qrcode.QRCode, blank_radius_modules: int) -> None:
    """
    Set modules in the centre of the QR to False (white) so they are not
    rendered. This prevents half-drawn modules around the token area.
    The QR uses ERROR_CORRECT_H (30% redundancy) so blanking the centre
    is safe for scanning.
    """
    n = qr.modules_count
    cx = n / 2.0
    cy = n / 2.0
    r2 = blank_radius_modules ** 2

    for row in range(n):
        for col in range(n):
            dx = col + 0.5 - cx
            dy = row + 0.5 - cy
            if dx * dx + dy * dy <= r2:
                qr.modules[row][col] = False


def generate_qr_mask(url: str, cfg: VariantConfig, blank_radius_modules: int = 0) -> Image.Image:
    """Generate a grayscale QR mask with optional centre blank-out."""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Blank centre modules BEFORE rendering
    if blank_radius_modules > 0:
        _blank_center_modules(qr, blank_radius_modules)

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=cfg.module_drawer,
        eye_drawer=cfg.eye_drawer,
    ).convert("L")

    return img.resize((QR_SIZE, QR_SIZE), Image.LANCZOS)


def compute_blank_radius(qr_version: int) -> int:
    """
    Compute how many modules (radius) to blank in the centre based on
    the token size relative to the QR grid.
    """
    modules_count = qr_version * 4 + 17  # standard QR formula
    border = 4  # QR spec quiet zone
    total = modules_count + 2 * border

    # Token diameter in pixels = QR_SIZE * COIN_RATIO
    # Module size in pixels = QR_SIZE / total
    module_px = QR_SIZE / total
    token_diameter_px = QR_SIZE * COIN_RATIO
    # Clearance must be tight enough to stay well under 30% erasure limit
    # (ERROR_CORRECT_H). One block of margin is sufficient since the
    # blocky clearance mask already adds visual padding.
    clearance_px = COIN_CLEAR_BLOCK * 1.5
    total_blank_diameter_px = token_diameter_px + clearance_px

    blank_radius_modules = int(math.ceil(total_blank_diameter_px / (2 * module_px))) + CENTER_BLANK_EXTRA
    return blank_radius_modules


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
    Load the pixel-art coin reference and parse it into a rim mask and
    an inner_radius_ratio.

    The reference is black-on-white:
      * Black pixels → rim (the thick circular border ring)
      * White pixels in the centre → inner face area (where the portrait goes)

    We threshold, then measure how far inward the white centre extends to
    compute inner_radius_ratio (typically ~0.60-0.70).

    Returns
    -------
    rim_mask : Image (mode "L", display_size × display_size)
        Grayscale mask where 255 = rim pixel, 0 = transparent.
    inner_radius_ratio : float
        Fraction of the coin radius occupied by the inner face (0..1).

    Requirements for the reference image:
      * Black-on-white, square aspect ratio preferred.
    """
    img = Image.open(path).convert("L")

    # Centre-crop to square
    s = min(img.size)
    left = (img.width - s) // 2
    top = (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s))

    img = img.resize((display_size, display_size), Image.LANCZOS)

    arr = np.asarray(img)

    # Threshold: dark pixels (< 160) are the shell/rim
    shell_pixels = arr < 160  # bool array

    # --- Compute inner_radius_ratio by measuring the white centre ---
    # The reference image may show the coin at a slight 3D angle, so
    # some radial directions will hit the edge detail (dark pixels)
    # much earlier than others.  To get a stable measurement we sample
    # many radial lines and take the 75th percentile — this ignores
    # the short "early hit" rays caused by the perspective edge.
    cx = display_size / 2.0
    cy = display_size / 2.0
    max_r = display_size / 2.0

    num_angles = 72  # sample every 5°
    boundary_radii: list[float] = []
    for ai in range(num_angles):
        a = (ai / num_angles) * 2 * math.pi
        dx = math.cos(a)
        dy = math.sin(a)
        # Walk outward from centre until we hit shell pixels
        for ri in range(1, int(max_r)):
            px = int(cx + dx * ri)
            py = int(cy + dy * ri)
            if 0 <= px < display_size and 0 <= py < display_size:
                if shell_pixels[py, px]:
                    boundary_radii.append(ri / max_r)
                    break
        else:
            # Never hit shell on this ray — use full radius
            boundary_radii.append(1.0)

    if boundary_radii:
        # 75th percentile ignores short rays from 3D perspective details
        inner_radius_ratio = float(np.percentile(boundary_radii, 75))
    else:
        inner_radius_ratio = 0.65  # fallback

    # Clamp to a sane range
    inner_radius_ratio = max(0.50, min(inner_radius_ratio, 0.85))

    rim_mask = (shell_pixels.astype(np.uint8)) * 255
    rim_mask_img = Image.fromarray(rim_mask, "L")

    return rim_mask_img, inner_radius_ratio


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
# Slab-based 3D coin rendering
# ---------------------------------------------------------------------------
#
# The coin is composed of three layers per frame:
#
#   1. Edge/thickness slab  — a coloured rounded rectangle whose visible
#      width = coin_thickness * abs(sin(angle)).  Gives the depth cue.
#   2. Face disc            — the circular front or back, squished
#      horizontally by cos(angle).
#   3. Rim overlay          — the pixel-art rim ring, also squished by
#      cos(angle), composited on top.
#
# Z-ordering depends on which side of the rotation we are on:
#
#   sin(angle) > 0  →  draw slab first, then face  (edge trails behind)
#   sin(angle) < 0  →  draw face first, then slab  (edge in front)
#
# Edge-on frames (|cos| near 0): only the thickness slab is shown as a
# narrow coloured strip with a highlight line for definition.
# ---------------------------------------------------------------------------

def render_coin_slab(
    display_size: int,
    angle: float,
    thickness: int,
    edge_color: tuple[int, int, int],
    highlight_color: tuple[int, int, int],
) -> Image.Image:
    """
    Render the coin thickness slab visible during rotation.

    The slab is shaped like a vertical pill/capsule that matches the coin's
    circular silhouette.  Its width is ``thickness * abs(sin(angle))`` —
    widest when the coin is edge-on, zero when face-on.

    The slab is horizontally offset based on ``sin(angle)`` so it naturally
    trails behind or leads in front of the face disc.

    Parameters
    ----------
    display_size : int
        Canvas width and height.
    angle : float
        Current rotation angle in radians.
    thickness : int
        Maximum pixel thickness of the coin edge.
    edge_color : tuple
        Base colour for the slab body.
    highlight_color : tuple
        Lighter colour for the centre highlight line.

    Returns
    -------
    RGBA image at display_size × display_size.
    """
    sin_val = math.sin(angle)
    slab_w = max(2, int(thickness * abs(sin_val)))

    canvas = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    cx = display_size // 2
    cy = display_size // 2
    # Offset the slab in the direction of the trailing edge
    # When sin > 0 the edge is to the right of centre, when < 0 to the left
    offset_x = int(sin_val * thickness * 0.45)

    # Draw the slab as a pill shape (stadium / capsule) that matches
    # the coin's circular height. The pill top/bottom are semicircles.
    pad = max(2, display_size // 40)  # small top/bottom margin
    pill_cx = cx + offset_x
    pill_top = pad
    pill_bot = display_size - pad
    pill_left = pill_cx - slab_w // 2
    pill_right = pill_cx + slab_w // 2

    # Use rounded_rectangle with large radius to get capsule shape
    corner_r = slab_w // 2
    if pill_right > pill_left and pill_bot > pill_top:
        draw.rounded_rectangle(
            [pill_left, pill_top, pill_right, pill_bot],
            radius=corner_r,
            fill=(*edge_color, 255),
        )

        # Centre highlight line for visual definition
        if slab_w >= 4:
            hl_w = max(1, slab_w // 5)
            hl_x = pill_cx
            draw.rounded_rectangle(
                [hl_x - hl_w // 2, pill_top + 3, hl_x + hl_w // 2, pill_bot - 3],
                radius=max(1, hl_w // 2),
                fill=(*highlight_color, 255),
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

    The coin is assembled per-frame from three layers (slab, face/back,
    rim) with z-ordering that depends on the rotation angle.  See the
    module-level comment block above for the full description.

    Parameters
    ----------
    brightness : ndarray
        Halftone brightness grid from ``portrait_to_brightness_grid()``.
    rim_mask : Image (mode "L")
        Grayscale mask of the pixel-art rim ring.
    inner_radius_ratio : float
        Fraction of coin radius for the portrait circle (from ``load_coin_shell``).
    display_size : int
        Pixel dimensions of the square coin canvas.
    fg_color : tuple
        Variant colour.

    Returns
    -------
    List of RGBA images, one per rotation step.
    """
    coin_thickness = display_size // 8  # adjustable: change divisor for thicker/thinner

    # --- Pre-render static layers ---

    # Face (front): halftone portrait clipped to inner face + coloured border ring
    face = render_halftone_token(brightness, display_size, fg_color, inner_radius_ratio)
    ring = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    ring_width = max(4, display_size // 25)
    ImageDraw.Draw(ring).ellipse(
        [0, 0, display_size - 1, display_size - 1],
        outline=(*fg_color, 255), width=ring_width)
    face = Image.alpha_composite(face, ring)

    # Back: solid colour circle with white decorative inner ring
    back = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    bd = ImageDraw.Draw(back)
    bd.ellipse([0, 0, display_size - 1, display_size - 1], fill=(*fg_color, 255))
    inset = display_size // 5
    bd.ellipse(
        [inset, inset, display_size - 1 - inset, display_size - 1 - inset],
        outline=(255, 255, 255, 255), width=max(3, display_size // 40))
    ring_w = max(3, display_size // 35)
    bd.ellipse(
        [0, 0, display_size - 1, display_size - 1],
        outline=(*fg_color, 255), width=ring_w)

    # Edge colours
    edge_color = darken_color(fg_color, 0.55)
    edge_highlight = darken_color(fg_color, 0.75)

    # Coloured rim overlay from pixel-art reference
    rim_coloured = colorize_shell(rim_mask, fg_color)

    # --- Generate rotation frames ---
    frames: list[Image.Image] = []
    for step in range(COIN_ROTATION_STEPS):
        frac = step / COIN_ROTATION_STEPS
        angle = frac * 2 * math.pi
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        canvas = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))

        # --- Edge-on case: only show thickness slab ---
        if abs(cos_val) < 0.05:
            slab = render_coin_slab(
                display_size, angle, coin_thickness, edge_color, edge_highlight)
            canvas = Image.alpha_composite(canvas, slab)
            frames.append(canvas)
            continue

        # --- Normal rotation: assemble slab + face/back + rim ---
        showing_face = cos_val >= 0
        w_scale = abs(cos_val)

        # Squish the face or back disc horizontally
        src = face if showing_face else back
        new_w = max(4, int(display_size * w_scale))
        scaled_face = src.resize((new_w, display_size), Image.LANCZOS)
        face_x = (display_size - new_w) // 2

        # Build the slab for this angle (slab self-offsets via sin)
        slab = render_coin_slab(
            display_size, angle, coin_thickness, edge_color, edge_highlight)

        # Apply z-ordering:
        #   sin > 0  →  slab behind face  (edge trails to the right)
        #   sin < 0  →  face behind slab  (edge appears in front, to the left)
        if sin_val > 0:
            canvas = Image.alpha_composite(canvas, slab)
            canvas.paste(scaled_face, (face_x, 0), scaled_face)
        else:
            canvas.paste(scaled_face, (face_x, 0), scaled_face)
            canvas = Image.alpha_composite(canvas, slab)

        # Overlay the pixel-art rim (always on top for style consistency).
        # The rim is squished the same as the face disc — it sits on the face.
        if w_scale >= 0.12:
            rim_w = max(4, int(display_size * w_scale))
            scaled_rim = rim_coloured.resize((rim_w, display_size), Image.LANCZOS)
            rx = (display_size - rim_w) // 2
            canvas.paste(scaled_rim, (rx, 0), scaled_rim)

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
    token_display: int,
    block: int,
) -> Image.Image:
    """
    Build a mask with a blocky (pixelated) circular hole in the centre.
    The hole is built from square blocks so it looks like part of the QR grid,
    not a smooth circle.
    """
    mask = Image.new("L", (canvas_size, canvas_size), 0)

    cx = canvas_size // 2
    cy = canvas_size // 2
    # Radius: token + 1 block of margin (tight to avoid covering
    # alignment patterns that sit near the centre)
    radius = token_display // 2 + block

    for bx in range(0, canvas_size, block):
        for by in range(0, canvas_size, block):
            bcx = bx + block // 2
            bcy = by + block // 2
            dist = math.sqrt((bcx - cx) ** 2 + (bcy - cy) ** 2)
            if dist < radius:
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
) -> None:
    n = len(masks)
    token_display = int(QR_SIZE * COIN_RATIO)

    # Build blocky clearance mask (same geometry for all frames)
    clearance = make_blocky_clearance_mask(CANVAS_SIZE, token_display, COIN_CLEAR_BLOCK)

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
            t = fi / (MORPH_FRAMES + 1)
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

    # Determine QR version first so we can compute centre blank radius
    # and derive the dynamic border width.
    print("Determining QR version ...")
    probe = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10, border=4,
    )
    probe.add_data(url)
    probe.make(fit=True)
    qr_version = probe.version
    blank_radius = compute_blank_radius(qr_version)

    # Dynamic border width: match the finder-pattern eye ring thickness.
    # The eye outer ring is nominally 1 module, but the rounded drawer
    # anti-aliases it so visually it looks ~60-70% of a raw module.
    modules_with_border = probe.modules_count + 2 * probe.border
    module_px = QR_SIZE / modules_with_border
    max_border = (CANVAS_SIZE - QR_SIZE) // 2
    border_width = max(2, min(int(round(module_px * 0.65)), max_border))

    print(f"  QR version {qr_version} ({probe.modules_count}x{probe.modules_count})")
    print(f"  Centre blank radius: {blank_radius} modules")
    print(f"  Border width: {border_width}px (~0.65 module = {module_px:.1f}px)\n")

    print("Generating QR variants (with centre blank-out) ...")
    masks: list[Image.Image] = []
    for i, cfg in enumerate(VARIANTS):
        print(f"  [{i+1}/{len(VARIANTS)}] {cfg.name}  "
              f"(modules={cfg.module_drawer.__class__.__name__}, "
              f"color={cfg.front_color})")
        masks.append(generate_qr_mask(url, cfg, blank_radius_modules=blank_radius))
    print()

    print("Assembling GIF ...")
    assemble_gif(masks, VARIANTS, brightness, rim_mask, inner_radius_ratio, border_width)

    total_ms = len(VARIANTS) * (HOLD_DURATION_MS + MORPH_DURATION_MS)
    nframes = len(VARIANTS) * (HOLD_DURATION_MS // HOLD_SUBFRAME_MS + MORPH_FRAMES)
    print(f"\nDone -> {OUTPUT_PATH.resolve()}")
    print(f"  {total_ms / 1000:.1f}s loop, {nframes} frames")


if __name__ == "__main__":
    main()
