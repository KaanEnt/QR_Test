"""
Animated Gooey QR Code GIF Generator (v3)

Reads a URL from url.txt, generates 3 visually distinct QR code variants
(all with rounded eyes), morphs between them with a gooey transition,
overlays an 8-bit spinning coin portrait in the centre, and writes a
looping animated GIF.
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
CANVAS_SIZE = 640              # full output frame
BORDER_INSET = 20              # border line distance from canvas edge
BORDER_WIDTH = 2               # border line thickness

HOLD_DURATION_MS = 4000        # display time per variant
HOLD_SUBFRAME_MS = 100         # sub-frame interval during hold (smooth coin spin)
MORPH_DURATION_MS = 1000       # transition time between variants
MORPH_FRAMES = 15              # intermediate frames per transition
MORPH_FRAME_MS = MORPH_DURATION_MS // MORPH_FRAMES

# ---- 8-bit spinning coin -------------------------------------------------
COIN_IMAGE_PATH = "public/event_pfp2.png"
COIN_RATIO = 0.24              # coin diameter relative to QR_SIZE
COIN_PIXEL_RES = 28            # work resolution (low = chunkier 8-bit pixels)
COIN_SPIN_PERIOD_MS = 1500     # one full rotation per this many ms
COIN_ROTATION_STEPS = 60       # pre-rendered frames in one full spin
COIN_RIM_COLOR = (190, 160, 50, 255)
COIN_BACK_COLOR = (210, 180, 70, 255)
COIN_OUTLINE_COLOR = (130, 100, 20, 255)
COIN_CLEAR_PAD = 8             # white clearance pixels around the coin


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
# QR generation (grayscale mask)
# ---------------------------------------------------------------------------

def generate_qr_mask(url: str, cfg: VariantConfig) -> Image.Image:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=3,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=cfg.module_drawer,
        eye_drawer=cfg.eye_drawer,
    ).convert("L")

    return img.resize((QR_SIZE, QR_SIZE), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Frame composition (QR + border, no background)
# ---------------------------------------------------------------------------

def colorize_qr(mask: Image.Image, fg: tuple[int, ...]) -> Image.Image:
    arr = np.asarray(mask)
    is_module = arr < 128
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[is_module] = (*fg, 255)
    return Image.fromarray(rgba, "RGBA")


def compose_frame(mask: Image.Image, fg: tuple[int, ...]) -> Image.Image:
    """White canvas + coloured QR modules + thin square border."""
    base = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 255))

    qr = colorize_qr(mask, fg)
    offset = (CANVAS_SIZE - QR_SIZE) // 2
    base.paste(qr, (offset, offset), qr)

    # Thin square border matching the QR colour
    draw = ImageDraw.Draw(base)
    b = BORDER_INSET
    draw.rectangle(
        [b, b, CANVAS_SIZE - b - 1, CANVAS_SIZE - b - 1],
        outline=(*fg, 255),
        width=BORDER_WIDTH,
    )
    return base


# ---------------------------------------------------------------------------
# 8-bit spinning coin
# ---------------------------------------------------------------------------

def build_coin_frames() -> list[Image.Image]:
    """
    Pre-render all rotation frames of the 8-bit spinning coin.

    The coin does a full 3-D Y-axis rotation:
      0 deg   = face (portrait) at full width
      90 deg  = edge-on (gold rim)
      180 deg = back (gold disc)
      270 deg = edge-on
    Everything is rendered at COIN_PIXEL_RES (e.g. 28x28), then upscaled
    with NEAREST-neighbour to get the chunky pixel-art look.
    """
    path = pathlib.Path(COIN_IMAGE_PATH)
    if not path.exists():
        print(f"  Warning: coin image not found at {path}, skipping coin.")
        return []

    coin_display = int(QR_SIZE * COIN_RATIO)
    PX = COIN_PIXEL_RES

    # ---- prepare face (portrait) at pixel resolution ----
    raw = Image.open(path).convert("RGBA")
    s = min(raw.size)
    raw = raw.crop(((raw.width - s) // 2, (raw.height - s) // 2,
                    (raw.width + s) // 2, (raw.height + s) // 2))
    raw = raw.resize((PX, PX), Image.LANCZOS)

    circle = Image.new("L", (PX, PX), 0)
    ImageDraw.Draw(circle).ellipse([0, 0, PX - 1, PX - 1], fill=255)

    face = Image.new("RGBA", (PX, PX), (0, 0, 0, 0))
    face.paste(raw, mask=circle)
    # gold outline ring
    ring_layer = Image.new("RGBA", (PX, PX), (0, 0, 0, 0))
    ImageDraw.Draw(ring_layer).ellipse(
        [0, 0, PX - 1, PX - 1], outline=COIN_OUTLINE_COLOR, width=2)
    face = Image.alpha_composite(face, ring_layer)

    # ---- prepare back (gold disc with inner ring) ----
    back = Image.new("RGBA", (PX, PX), (0, 0, 0, 0))
    bd = ImageDraw.Draw(back)
    bd.ellipse([0, 0, PX - 1, PX - 1], fill=COIN_BACK_COLOR)
    bd.ellipse([0, 0, PX - 1, PX - 1], outline=COIN_OUTLINE_COLOR, width=2)
    inset = max(3, PX // 5)
    bd.ellipse([inset, inset, PX - 1 - inset, PX - 1 - inset],
               outline=COIN_OUTLINE_COLOR, width=1)

    # ---- render one frame per rotation step ----
    frames: list[Image.Image] = []
    for step in range(COIN_ROTATION_STEPS):
        angle = (step / COIN_ROTATION_STEPS) * 2 * math.pi
        cos_val = math.cos(angle)
        showing_face = cos_val >= 0
        w_scale = abs(cos_val)

        canvas = Image.new("RGBA", (PX, PX), (0, 0, 0, 0))

        if w_scale < 0.08:
            # Edge-on: thin gold rim
            d = ImageDraw.Draw(canvas)
            rim_w = max(2, int(PX * 0.10))
            cx = PX // 2
            d.rounded_rectangle(
                [cx - rim_w // 2, 1, cx + rim_w // 2, PX - 2],
                radius=1, fill=COIN_RIM_COLOR, outline=COIN_OUTLINE_COLOR, width=1)
        else:
            src = face if showing_face else back
            new_w = max(3, int(PX * w_scale))
            scaled = src.resize((new_w, PX), Image.LANCZOS)
            x_off = (PX - new_w) // 2
            canvas.paste(scaled, (x_off, 0), scaled)

        # Upscale with NEAREST for the 8-bit look
        frames.append(canvas.resize((coin_display, coin_display), Image.NEAREST))

    return frames


def get_coin_frame(coin_frames: list[Image.Image], time_ms: float) -> Image.Image:
    """Look up the coin rotation frame for a given time position."""
    angle_deg = (time_ms / COIN_SPIN_PERIOD_MS) * 360.0
    n = len(coin_frames)
    idx = int((angle_deg % 360) / 360 * n) % n
    return coin_frames[idx]


def overlay_coin(frame: Image.Image, coin_img: Image.Image) -> Image.Image:
    """Paste a coin frame on centre of the canvas, with a white clearance ring."""
    frame = frame.copy()
    cw = coin_img.width
    cx = (frame.width - cw) // 2
    cy = (frame.height - cw) // 2

    # White circle behind the coin so QR modules don't bleed in
    pad = COIN_CLEAR_PAD
    draw = ImageDraw.Draw(frame)
    draw.ellipse(
        [cx - pad, cy - pad, cx + cw + pad, cy + cw + pad],
        fill=(255, 255, 255, 255),
    )

    frame.paste(coin_img, (cx, cy), coin_img)
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
    coin_frames: list[Image.Image],
) -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []
    time_ms = 0.0
    n = len(masks)

    hold_subframes = HOLD_DURATION_MS // HOLD_SUBFRAME_MS

    for i in range(n):
        print(f"  Variant {i+1}/{n} hold frames ...")
        # ---- Hold: sub-frames for smooth coin spin ----
        hold_base = compose_frame(masks[i], configs[i].front_color)
        for _ in range(hold_subframes):
            if coin_frames:
                cf = get_coin_frame(coin_frames, time_ms)
                f = overlay_coin(hold_base, cf)
            else:
                f = hold_base
            frames.append(f)
            durations.append(HOLD_SUBFRAME_MS)
            time_ms += HOLD_SUBFRAME_MS

        # ---- Morph ----
        print(f"  Variant {i+1}->{(i+1)%n+1} morph frames ...")
        ni = (i + 1) % n
        for fi in range(1, MORPH_FRAMES + 1):
            t = fi / (MORPH_FRAMES + 1)
            m_mask = gooey_morph_mask(masks[i], masks[ni], t)
            fg = lerp_color(configs[i].front_color, configs[ni].front_color, t)
            f = compose_frame(m_mask, fg)
            if coin_frames:
                cf = get_coin_frame(coin_frames, time_ms)
                f = overlay_coin(f, cf)
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

    print("Building 8-bit coin frames ...")
    coin_frames = build_coin_frames()
    if coin_frames:
        print(f"  {len(coin_frames)} rotation steps, "
              f"{coin_frames[0].size[0]}px display size\n")

    print("Generating QR variants ...")
    masks: list[Image.Image] = []
    for i, cfg in enumerate(VARIANTS):
        print(f"  [{i+1}/{len(VARIANTS)}] {cfg.name}  "
              f"(modules={cfg.module_drawer.__class__.__name__}, "
              f"color={cfg.front_color})")
        masks.append(generate_qr_mask(url, cfg))
    print()

    print("Assembling GIF ...")
    assemble_gif(masks, VARIANTS, coin_frames)

    total_ms = len(VARIANTS) * (HOLD_DURATION_MS + MORPH_DURATION_MS)
    print(f"\nDone -> {OUTPUT_PATH.resolve()}")
    print(f"  {total_ms / 1000:.1f}s loop, "
          f"{len(VARIANTS) * (HOLD_DURATION_MS // HOLD_SUBFRAME_MS + MORPH_FRAMES)} frames")


if __name__ == "__main__":
    main()
