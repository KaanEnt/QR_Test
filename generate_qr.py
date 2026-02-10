"""
Animated Gooey QR Code GIF Generator (v11)

Reads a URL from url.txt, generates 3 visually distinct QR code variants
(all with rounded eyes), morphs between them with a gooey transition,
overlays a spinning 3D coin with halftone portrait in the centre, and
writes a looping animated GIF.

Key changes in v11:
  - Coin rendering uses pure Pillow perspective transforms (no OpenGL).
  - Coin face displays halftone mosaic portrait texture.
  - Metallic sheen simulated with a composited lighting overlay.
  - No dependency on trimesh, pyrender, or PyOpenGL.

Previous version (v10):
  - Increased hold duration per color to 8000ms (doubled)
  - Reduced morph duration to 250ms (halved)
  - Coin now completes exactly one full rotation per complete color cycle
  - Coin returns to starting position when loop restarts for seamless animation
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
COIN_RATIO = 0.22              # token diameter relative to QR_SIZE
# Spin period is derived so an exact integer number of full rotations fit
# inside the total GIF cycle, preventing any jump/blink on loop.
# Total cycle = 3 * (8000 + 250) = 24750ms.  24750 / 5 = 4950ms per turn
# (~5 s, same perceived speed as the original 5000ms).
COIN_SPIN_PERIOD_MS = 4950     # 5 full rotations per GIF loop (seamless)
COIN_ROTATION_STEPS = 60       # pre-rendered frames in one full spin
COIN_CLEAR_BLOCK = 16          # clearance zone block size (px)
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

    # Border: outer edge flush with the canvas (square corners), inner
    # edge rounded.  Built as a separate RGBA layer where only the border
    # band is opaque, then composited on top of the base.
    border_layer = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    bw = border_width
    # 1) Fill entire canvas with border colour
    ImageDraw.Draw(border_layer).rectangle(
        [0, 0, CANVAS_SIZE - 1, CANVAS_SIZE - 1],
        fill=(*fg, 255),
    )
    # 2) Punch out the interior with a rounded rect (transparent hole)
    inner_mask = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
    ImageDraw.Draw(inner_mask).rounded_rectangle(
        [bw, bw, CANVAS_SIZE - bw - 1, CANVAS_SIZE - bw - 1],
        radius=int(round(bw * 1.5)),
        fill=0,
    )
    border_layer.putalpha(inner_mask)
    # 3) Paste the border on top of the QR content
    base = Image.alpha_composite(base, border_layer)
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


def darken_color(c: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Darken a colour by a factor (0 = black, 1 = original)."""
    return tuple(max(0, int(v * factor)) for v in c)


# ---------------------------------------------------------------------------
# 3D coin rendering — pure Pillow (no OpenGL / pyrender needed)
# ---------------------------------------------------------------------------
#
# Simulates a spinning coin using perspective transforms in Pillow.
# The face shows the halftone mosaic portrait; the edge is a solid-colour
# strip whose visible width varies with the rotation angle.  A subtle
# lighting gradient is composited on top for a metallic look.
# 60 rotation frames (0 to 2π) are pre-rendered per variant colour.
# ---------------------------------------------------------------------------


def _apply_ellipse_mask(img: Image.Image) -> Image.Image:
    """Clip *img* to a centred circle (ellipse inscribed in the bbox)."""
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse(
        [0, 0, img.size[0] - 1, img.size[1] - 1], fill=255,
    )
    out = img.copy()
    out.putalpha(mask)
    return out


def _make_lighting_overlay(size: int) -> Image.Image:
    """Create a radial highlight overlay that simulates metallic sheen."""
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Bright highlight in upper-right quadrant
    cx, cy = int(size * 0.6), int(size * 0.35)
    r = size // 4
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=(255, 255, 255, 90),
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=size // 5))
    return overlay


def render_coin_frame_pillow(
    angle: float,
    face_texture: Image.Image,
    display_size: int,
    edge_color: tuple[int, int, int],
    border_thickness: int = 0,
) -> Image.Image:
    """
    Render a single frame of a spinning coin using Pillow transforms.

    Parameters
    ----------
    angle : float
        Rotation angle in radians (0 = face-on, π = back-on).
    face_texture : PIL Image (RGBA)
        Portrait texture for the coin face (already circular-clipped).
    display_size : int
        Output image width and height in pixels (square).
    edge_color : tuple
        RGB colour for the coin edge strip.
    border_thickness : int
        When > 0, use this as the coin ring thickness (to match the
        outer QR border).  When 0, fall back to 6% of coin diameter.

    Returns
    -------
    PIL Image in RGBA mode at display_size × display_size.
    """
    canvas = Image.new("RGBA", (display_size, display_size), (0, 0, 0, 0))

    # Lighter back face: blend edge_color 40% toward white for contrast
    back_color = tuple(c + (255 - c) * 2 // 5 for c in edge_color)

    # How much of the face is visible: cos(angle) ranges from -1 to 1.
    cos_a = math.cos(angle)
    face_visible = cos_a > 0

    # Horizontal scale factor — simulates foreshortening
    h_scale = abs(cos_a)

    # Coin occupies ~90 % of the display area
    coin_diam = int(display_size * 0.90)
    coin_radius = coin_diam // 2

    # Edge thickness — matches the outer QR border when provided,
    # otherwise falls back to 6% of the coin diameter.
    edge_thickness = border_thickness if border_thickness > 0 else max(2, int(coin_diam * 0.06))

    # Scaled width of the face ellipse
    face_width = max(1, int(coin_diam * h_scale))
    face_height = coin_diam

    cx, cy = display_size // 2, display_size // 2

    # --- Draw the full coin disc (rim) at the original size ---
    # The rim fills the entire face_width x face_height ellipse.
    rim_img = Image.new("RGBA", (face_width, face_height), (0, 0, 0, 0))
    ImageDraw.Draw(rim_img).ellipse(
        [0, 0, face_width - 1, face_height - 1],
        fill=edge_color + (255,),
    )
    canvas.paste(
        rim_img,
        (cx - face_width // 2, cy - face_height // 2),
        rim_img,
    )

    # --- Draw the face (or back) inset by edge_thickness inside the rim ---
    # The border ring is visible as the gap between the outer rim and the
    # smaller inner disc / texture.  Coin stays the same overall size.
    inner_w = max(1, face_width - edge_thickness * 2)
    inner_h = max(1, face_height - edge_thickness * 2)

    if inner_w >= 2:
        if face_visible:
            face_resized = face_texture.resize(
                (inner_w, inner_h), Image.LANCZOS,
            )
            face_resized = _apply_ellipse_mask(face_resized)
            canvas.paste(
                face_resized,
                (cx - inner_w // 2, cy - inner_h // 2),
                face_resized,
            )
        else:
            # Back of the coin — lighter disc inset inside the rim
            back_disc = Image.new(
                "RGBA", (inner_w, inner_h), (0, 0, 0, 0),
            )
            ImageDraw.Draw(back_disc).ellipse(
                [0, 0, inner_w - 1, inner_h - 1],
                fill=back_color + (255,),
            )
            canvas.paste(
                back_disc,
                (cx - inner_w // 2, cy - inner_h // 2),
                back_disc,
            )

    # --- Lighting overlay for metallic sheen ---
    lighting = _make_lighting_overlay(display_size)
    coin_mask = Image.new("L", (display_size, display_size), 0)
    ImageDraw.Draw(coin_mask).ellipse(
        [cx - coin_radius, cy - coin_radius,
         cx + coin_radius, cy + coin_radius],
        fill=255,
    )
    lighting.putalpha(
        Image.fromarray(
            np.minimum(
                np.array(lighting.split()[3]),
                np.array(coin_mask),
            )
        )
    )
    canvas = Image.alpha_composite(canvas, lighting)

    return canvas


def build_3d_coin_frames_trimesh(
    brightness: np.ndarray,
    display_size: int,
    fg_color: tuple[int, int, int],
    border_thickness: int = 0,
) -> list[Image.Image]:
    """
    Pre-render all rotation frames of the spinning coin with halftone mosaic.

    Uses pure Pillow perspective transforms — no OpenGL required.

    Parameters
    ----------
    brightness : ndarray
        Halftone brightness grid from ``portrait_to_brightness_grid()``.
    display_size : int
        Output frame size in pixels (square).
    fg_color : tuple
        Foreground colour for the halftone dots and coin edge.
    border_thickness : int
        Coin ring thickness (passed through to render_coin_frame_pillow).

    Returns
    -------
    List of COIN_ROTATION_STEPS RGBA PIL Images.
    """
    texture_size = display_size * 2
    halftone_texture = render_halftone_token(
        brightness, texture_size, fg_color,
    )

    if halftone_texture.mode != "RGBA":
        halftone_texture = halftone_texture.convert("RGBA")

    # Clip the texture to a circle (coin face shape)
    halftone_texture = _apply_ellipse_mask(halftone_texture)

    frames: list[Image.Image] = []
    for step in range(COIN_ROTATION_STEPS):
        angle = (2.0 * math.pi * step) / COIN_ROTATION_STEPS
        frame = render_coin_frame_pillow(
            angle, halftone_texture, display_size, fg_color, border_thickness,
        )
        frames.append(frame)
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
        tfs = build_3d_coin_frames_trimesh(
            brightness, token_display, cfg.front_color, border_width,
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
    # The eye outer ring is exactly 1 module wide, so we use the full
    # module pixel size for the border.
    modules_with_border = probe.modules_count + 2 * probe.border
    module_px = QR_SIZE / modules_with_border
    max_border = (CANVAS_SIZE - QR_SIZE) // 2
    border_width = max(2, min(int(round(module_px)), max_border))

    print(f"  QR version {qr_version} ({probe.modules_count}x{probe.modules_count})")
    print(f"  Border width: {border_width}px (~1 module = {module_px:.1f}px)\n")

    # Build the blocky clearance mask ONCE, before QR generation.
    # This single mask drives both module-level blanking (data erasure)
    # and the pixel-level white-out at compositing time.
    token_display = int(QR_SIZE * COIN_RATIO)
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
    assemble_gif(masks, VARIANTS, brightness, border_width, clearance_mask)

    total_ms = len(VARIANTS) * (HOLD_DURATION_MS + MORPH_DURATION_MS)
    nframes = len(VARIANTS) * (HOLD_DURATION_MS // HOLD_SUBFRAME_MS + MORPH_FRAMES)
    print(f"\nDone -> {OUTPUT_PATH.resolve()}")
    print(f"  {total_ms / 1000:.1f}s loop, {nframes} frames")


if __name__ == "__main__":
    main()
