"""
Animated Gooey QR Code GIF Generator (v6)

Reads a URL from url.txt, generates 3 visually distinct QR code variants
(all with rounded eyes), morphs between them with a gooey transition,
overlays a spinning halftone-portrait token (variant colour + white) in
the centre, and writes a looping animated GIF.

Key changes in v6:
  - Replaced faint ASCII-art with bold halftone mosaic (variable-size squares)
    for much better contrast and readability at small sizes
  - QR modules in the centre are blanked BEFORE rendering
  - Spin is continuous across variants, reversing direction smoothly
  - Token is variant colour + white only
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
BORDER_WIDTH = 2               # border line thickness

HOLD_DURATION_MS = 4000
HOLD_SUBFRAME_MS = 120         # sub-frame interval during hold (coin spin)
MORPH_DURATION_MS = 1000
MORPH_FRAMES = 15
MORPH_FRAME_MS = MORPH_DURATION_MS // MORPH_FRAMES

# ---- Spinning ASCII token ------------------------------------------------
COIN_IMAGE_PATH = "public/event_pfp2.png"
COIN_RATIO = 0.34              # token diameter relative to QR_SIZE
COIN_SPIN_PERIOD_MS = 4000     # 4 seconds per rotation
COIN_ROTATION_STEPS = 48       # pre-rendered frames in one full spin
COIN_CLEAR_BLOCK = 20          # clearance zone block size (px)

# How many QR modules (radius) to blank in the centre for the token.
# The token occupies COIN_RATIO of QR_SIZE. We need to figure out how many
# modules that covers and add a small margin.
CENTER_BLANK_EXTRA = 1         # extra modules beyond the token edge

# Halftone portrait grid
HALFTONE_GRID = 40             # cells per side for the halftone mosaic


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
            # Distance from centre of module grid
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
        border=2,
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
    border = 2
    total = modules_count + 2 * border

    # Token diameter in pixels = QR_SIZE * COIN_RATIO
    # Module size in pixels = QR_SIZE / total
    module_px = QR_SIZE / total
    token_diameter_px = QR_SIZE * COIN_RATIO
    # Add clearance for the blocky zone
    clearance_px = COIN_CLEAR_BLOCK * 2
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


def compose_frame(mask: Image.Image, fg: tuple[int, ...]) -> Image.Image:
    """White canvas + coloured QR modules + tight square border."""
    base = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))

    qr = colorize_qr(mask, fg)
    offset = (CANVAS_SIZE - QR_SIZE) // 2
    base.paste(qr, (offset, offset), qr)

    # Border sits BORDER_PAD pixels outside the QR area
    draw = ImageDraw.Draw(base)
    b = offset - BORDER_PAD
    draw.rectangle(
        [b, b, CANVAS_SIZE - b - 1, CANVAS_SIZE - b - 1],
        outline=(*fg, 255),
        width=BORDER_WIDTH,
    )
    return base


# ---------------------------------------------------------------------------
# ASCII-art portrait conversion
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
) -> Image.Image:
    """
    Render the portrait as a halftone-style mosaic inside a circle.

    Each cell in the brightness grid becomes a filled square whose size is
    proportional to the darkness of that cell. Dark cells -> large filled
    squares, light cells -> small or no squares. This produces a bold,
    stylised portrait that reads well even at small sizes.

    Returns RGBA at display_size x display_size.
    """
    grid_h, grid_w = brightness.shape
    # Work at 2x resolution for crisp edges, then downsample
    work_size = display_size * 2
    cell_w = work_size / grid_w
    cell_h = work_size / grid_h

    canvas = Image.new("RGBA", (work_size, work_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row in range(grid_h):
        for col in range(grid_w):
            darkness = 1.0 - brightness[row, col]  # 0=white, 1=black
            if darkness < 0.08:
                continue  # skip very light cells

            # Square size proportional to darkness (min 30%, max 95% of cell)
            fill_frac = 0.3 + darkness * 0.65
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            half_w = cell_w * fill_frac / 2
            half_h = cell_h * fill_frac / 2

            draw.rectangle(
                [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
                fill=(*fg_color, 255),
            )

    # Downsample to display size
    token = canvas.resize((display_size, display_size), Image.LANCZOS)

    # ---- Circular crop ----
    circle_mask = Image.new("L", (display_size, display_size), 0)
    ImageDraw.Draw(circle_mask).ellipse(
        [0, 0, display_size - 1, display_size - 1], fill=255)

    result = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    white_bg = Image.new("RGBA", (display_size, display_size), (255, 255, 255, 255))
    result.paste(white_bg, mask=circle_mask)
    result.paste(token, mask=circle_mask)

    # ---- Colour border ring ----
    ring = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    ring_width = max(4, display_size // 30)
    ImageDraw.Draw(ring).ellipse(
        [0, 0, display_size - 1, display_size - 1],
        outline=(*fg_color, 255), width=ring_width)
    result = Image.alpha_composite(result, ring)

    return result


# ---------------------------------------------------------------------------
# Spinning token (3D Y-axis rotation) with continuous direction reversal
# ---------------------------------------------------------------------------

def build_token_rotation_frames(
    brightness: np.ndarray,
    display_size: int,
    fg_color: tuple[int, int, int],
) -> list[Image.Image]:
    """
    Pre-render all rotation frames of the halftone token for one colour.
    Direction is handled at lookup time, not at build time.

    Face = halftone portrait.  Back = solid colour disc with white inner ring.
    Edge = thin coloured rim.
    """
    face = render_halftone_token(brightness, display_size, fg_color)

    # Back: solid colour circle with white inner ring
    back = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))
    bd = ImageDraw.Draw(back)
    bd.ellipse([0, 0, display_size - 1, display_size - 1], fill=(*fg_color, 255))
    inset = display_size // 5
    bd.ellipse(
        [inset, inset, display_size - 1 - inset, display_size - 1 - inset],
        outline=(255, 255, 255, 255), width=max(3, display_size // 50))
    ring_w = max(3, display_size // 40)
    bd.ellipse(
        [0, 0, display_size - 1, display_size - 1],
        outline=(*fg_color, 255), width=ring_w)

    frames: list[Image.Image] = []
    for step in range(COIN_ROTATION_STEPS):
        frac = step / COIN_ROTATION_STEPS
        angle = frac * 2 * math.pi
        cos_val = math.cos(angle)
        showing_face = cos_val >= 0
        w_scale = abs(cos_val)

        canvas = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))

        if w_scale < 0.06:
            # Edge-on: thin coloured rim
            d = ImageDraw.Draw(canvas)
            rim_w = max(3, int(display_size * 0.04))
            cx = display_size // 2
            d.rectangle(
                [cx - rim_w // 2, 2, cx + rim_w // 2, display_size - 3],
                fill=(*fg_color, 255))
        else:
            src = face if showing_face else back
            new_w = max(4, int(display_size * w_scale))
            scaled = src.resize((new_w, display_size), Image.LANCZOS)
            x_off = (display_size - new_w) // 2
            canvas.paste(scaled, (x_off, 0), scaled)

        frames.append(canvas)

    return frames


def get_token_frame_continuous(
    all_frame_sets: list[list[Image.Image]],
    variant_idx: int,
    time_ms: float,
) -> Image.Image:
    """
    Get the token frame with continuous spin that reverses direction
    each variant. The spin angle is computed from absolute time_ms so
    it never resets — only the direction flips.

    Even variants (0, 2) spin clockwise, odd variants (1) spin counter-clockwise.
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
    # Radius: token + 2 blocks of margin
    radius = token_display // 2 + block * 2

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
) -> None:
    n = len(masks)
    token_display = int(QR_SIZE * COIN_RATIO)

    # Build blocky clearance mask (same geometry for all frames)
    clearance = make_blocky_clearance_mask(CANVAS_SIZE, token_display, COIN_CLEAR_BLOCK)

    # Pre-build token rotation frames for each variant colour
    # (direction is handled at lookup time via get_token_frame_continuous)
    print("  Pre-rendering token rotations per variant ...")
    token_frame_sets: list[list[Image.Image]] = []
    for i, cfg in enumerate(configs):
        tfs = build_token_rotation_frames(brightness, token_display, cfg.front_color)
        token_frame_sets.append(tfs)
        direction = "CW" if (i % 2 == 0) else "CCW"
        print(f"    Variant {i+1} ({cfg.name}): {direction}, {len(tfs)} frames")

    frames: list[Image.Image] = []
    durations: list[int] = []

    # Continuous time counter — never resets, so the coin spin is seamless
    time_ms = 0.0
    hold_subframes = HOLD_DURATION_MS // HOLD_SUBFRAME_MS

    for i in range(n):
        print(f"  Variant {i+1}/{n} hold frames ...")
        hold_base = compose_frame(masks[i], configs[i].front_color)

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
            f = compose_frame(m_mask, fg)

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

    # Determine QR version first so we can compute centre blank radius
    print("Determining QR version ...")
    probe = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10, border=2,
    )
    probe.add_data(url)
    probe.make(fit=True)
    qr_version = probe.version
    blank_radius = compute_blank_radius(qr_version)
    print(f"  QR version {qr_version} ({probe.modules_count}x{probe.modules_count})")
    print(f"  Centre blank radius: {blank_radius} modules\n")

    print("Generating QR variants (with centre blank-out) ...")
    masks: list[Image.Image] = []
    for i, cfg in enumerate(VARIANTS):
        print(f"  [{i+1}/{len(VARIANTS)}] {cfg.name}  "
              f"(modules={cfg.module_drawer.__class__.__name__}, "
              f"color={cfg.front_color})")
        masks.append(generate_qr_mask(url, cfg, blank_radius_modules=blank_radius))
    print()

    print("Assembling GIF ...")
    assemble_gif(masks, VARIANTS, brightness)

    total_ms = len(VARIANTS) * (HOLD_DURATION_MS + MORPH_DURATION_MS)
    nframes = len(VARIANTS) * (HOLD_DURATION_MS // HOLD_SUBFRAME_MS + MORPH_FRAMES)
    print(f"\nDone -> {OUTPUT_PATH.resolve()}")
    print(f"  {total_ms / 1000:.1f}s loop, {nframes} frames")


if __name__ == "__main__":
    main()
