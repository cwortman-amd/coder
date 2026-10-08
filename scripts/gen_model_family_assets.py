#!/usr/bin/env python3
"""Pixel-matched split of the 2026 local-LLM poster.

Source is 848x1264. The poster splits at the section titles:
model-family-comparison.svg keeps the banner and the comparison table;
hardware-tier-guide.svg keeps the quick-pick bars and the pro tip.
"""

from functools import lru_cache
from pathlib import Path
import re

OUT = Path(__file__).resolve().parents[1] / "reports" / "assets"
LOGO_DIR = OUT / "brand-logos"
FONT = "Arial, Helvetica, sans-serif"
BG = "#0c1117"


def esc(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts, self.defs = w, h, [], []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=16, fill="#ffffff", weight=400, anchor="start", spacing=0):
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{esc(s)}</text>'
        )

    def save(self, name):
        body = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}">',
                "<defs>",
                *self.defs,
                "</defs>",
                f'<rect width="{self.w}" height="{self.h}" fill="{BG}"/>',
                *self.parts,
                "</svg>",
            ]
        )
        path = OUT / name
        path.write_text(body, encoding="utf-8")
        print(name, path.stat().st_size)


def grad(s, gid, c0, c1):
    s.defs.append(
        f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{c0}"/>'
        f'<stop offset="1" stop-color="{c1}"/>'
        f"</linearGradient>"
    )


def stroke_icon(d, x, y, color, sw=1.6, fill="none"):
    return (
        f'<path d="{d}" transform="translate({x:.1f} {y:.1f})" fill="{fill}" stroke="{color}" '
        f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def circuit(s, x, y, flip_x=1, flip_y=1):
    """Corner trace inside the banner, matching the source HUD corners."""
    s.add(
        f'<g transform="translate({x:.1f} {y:.1f}) scale({flip_x} {flip_y})" '
        f'fill="none" stroke="#3ad4e4" stroke-width="1.3" opacity="0.85">'
        f'<path d="M0 0 H46"/>'
        f'<path d="M0 0 V28"/>'
        f'<path d="M16 0 V12 H34"/>'
        f'<path d="M0 14 H12"/>'
        f'<circle cx="46" cy="0" r="2.2" fill="#3ad4e4" stroke="none"/>'
        f'<circle cx="34" cy="12" r="2" fill="#7ef6ff" stroke="none"/>'
        f'<circle cx="0" cy="28" r="2" fill="#3ad4e4" stroke="none"/>'
        f"</g>"
    )


def banner(s):
    s.defs.append(
        '<linearGradient id="banner" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#08303c"/>'
        '<stop offset="1" stop-color="#061820"/>'
        "</linearGradient>"
    )
    s.add(
        '<rect x="20" y="18" width="808" height="146" rx="18" '
        'fill="url(#banner)" stroke="#3ad4e4" stroke-width="2.2"/>'
    )
    circuit(s, 40, 36, 1, 1)
    circuit(s, 808, 36, -1, 1)
    circuit(s, 40, 146, 1, -1)
    circuit(s, 808, 146, -1, -1)
    s.text(424, 84, "BEST LOCAL LLM MODELS TO RUN IN 2026", 25, "#4ae3f5", 800, "middle", 0.4)
    s.text(424, 118, "Pick by memory first, then by task", 18, "#ffffff", 600, "middle")


def pill(s, x, y, w, h, fill, stroke):
    s.add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>'
    )


def ram_chip(s, x, y, color):
    s.add(
        f'<g transform="translate({x:.1f} {y:.1f})" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round">'
        f'<rect x="3" y="6" width="22" height="16" rx="2"/>'
        f'<path d="M7 6 V2 M12 6 V2 M17 6 V2 M22 6 V2 M7 22 V26 M12 22 V26 M17 22 V26 M22 22 V26"/>'
        f'<text x="14" y="17.5" font-family="{FONT}" font-size="7" font-weight="700" '
        f'fill="{color}" stroke="none" text-anchor="middle">RAM</text>'
        f"</g>"
    )


def cpu_chip(s, x, y, color):
    s.add(
        f'<g transform="translate({x:.1f} {y:.1f})" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round">'
        f'<rect x="7" y="7" width="16" height="16" rx="2"/>'
        f'<rect x="11" y="11" width="8" height="8"/>'
        f'<path d="M11 7 V3 M16 7 V3 M21 7 V3 M11 27 V23 M16 27 V23 M21 27 V23 '
        f'M7 11 H3 M7 16 H3 M7 21 H3 M27 11 H23 M27 16 H23 M27 21 H23"/>'
        f"</g>"
    )


BRAND_LOGOS = {
    "llama": "llama-meta-reference.svg",
    "mistral": "mistral-reference.svg",
    "qwen": "qwen-reference.svg",
    "whale": "deepseek-reference.svg",
    "gem": "gemma-reference.svg",
    "phi": "phi-microsoft-reference.svg",
}


@lru_cache(maxsize=None)
def _brand_svg(kind):
    """Return a reference mark's viewBox and inline children."""
    raw = (LOGO_DIR / BRAND_LOGOS[kind]).read_text(encoding="utf-8")
    match = re.search(r'viewBox="([^"]+)"', raw)
    view_box = match.group(1) if match else "0 0 24 24"
    children = raw[raw.find(">") + 1:raw.rfind("</svg>")]
    return view_box, children


def family_icon(kind, x, y, size=26):
    """Inline the saved brand mark so the generated SVG remains self-contained."""
    view_box, children = _brand_svg(kind)
    return (
        f'<svg x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" '
        f'viewBox="{view_box}" preserveAspectRatio="xMidYMid meet">'
        f"{children}</svg>"
    )


def mini(kind, x, y, color):
    """18px inline icon for the best-for and tradeoff columns."""
    ox, oy = x, y
    if kind == "chat":
        return stroke_icon("M2 3 h10 a2 2 0 0 1 2 2 v6 a2 2 0 0 1-2 2 H7 L4 16 V13 H2 a2 2 0 0 1-2-2 V5 a2 2 0 0 1 2-2 z", ox, oy, color, 1.4)
    if kind == "pen":
        return stroke_icon("M3 14 L12 3 L15 6 L6 17 H3 Z M11 4 L14 7", ox, oy, color, 1.4)
    if kind == "code":
        return stroke_icon("M6 4 L2 9 L6 14 M10 4 L14 9 L10 14", ox, oy, color, 1.5)
    if kind == "arrow":
        return stroke_icon("M2 9 H14 M10 5 L14 9 L10 13", ox, oy, color, 1.5)
    if kind == "globe":
        return stroke_icon("M9 2 a7 7 0 1 0 0.1 0 M2 9 H16 M9 2 C6 5 6 13 9 16 M9 2 C12 5 12 13 9 16", ox, oy, color, 1.3)
    if kind == "brain":
        return stroke_icon("M4 9 C4 5 7 3 9 5 C10 3 14 4 14 8 C16 8 16 12 14 13 C14 16 10 16 9 14 C6 16 3 14 4 11 Z", ox, oy, color, 1.3)
    if kind == "sigma":
        return (
            f'<text x="{ox + 8:.1f}" y="{oy + 13:.1f}" font-family="{FONT}" font-size="13" '
            f'font-weight="700" fill="{color}" text-anchor="middle">&#931;&#960;</text>'
        )
    if kind == "image":
        return stroke_icon("M2 4 h14 v11 H2 Z M2 12 l4-4 3 3 2-2 3 3", ox, oy, color, 1.3)
    if kind == "wrench":
        return stroke_icon("M4 14 L11 6 M12 4 a3 3 0 0 1 3 3 l-2 1", ox, oy, color, 1.5)
    if kind == "cpu":
        return stroke_icon("M6 6 h8 v8 H6 Z M8 2 V6 M12 2 V6 M8 14 V18 M12 14 V18 M2 8 H6 M2 12 H6 M14 8 H18 M14 12 H18", ox, oy, color, 1.3)
    if kind == "bolt":
        return (
            f'<path d="M11 1 L5 10 H9 L7 17 L14 7 H10 Z" transform="translate({ox:.1f} {oy:.1f})" '
            f'fill="{color}"/>'
        )
    if kind == "lock":
        return stroke_icon("M5 8 h8 v8 H5 Z M7 8 V6 a2.5 2.5 0 0 1 5 0 V8", ox, oy, color, 1.4)
    if kind == "clock":
        return stroke_icon("M9 2 a7 7 0 1 0 0.1 0 M9 5 V9 L12 11", ox, oy, color, 1.4)
    if kind == "chip":
        return stroke_icon("M5 5 h8 v8 H5 Z M7 5 V2 M11 5 V2 M7 16 V13 M11 16 V13 M2 7 H5 M2 11 H5 M13 7 H16 M13 11 H16", ox, oy, color, 1.3)
    if kind == "doc":
        return stroke_icon("M4 2 h7 l4 4 v10 H4 Z M11 2 V6 H15 M6 9 H12 M6 12 H10", ox, oy, color, 1.3)
    if kind == "warn":
        return (
            f'<path d="M9 2 L16 14 H2 Z" transform="translate({ox + 2:.1f} {oy:.1f})" '
            f'fill="none" stroke="{color}" stroke-width="1.3"/>'
            f'<path d="M11 7 V10" transform="translate({ox:.1f} {oy:.1f})" stroke="{color}" '
            f'stroke-width="1.3" stroke-linecap="round"/>'
        )
    if kind == "bubble":
        return stroke_icon("M2 3 h12 a2 2 0 0 1 2 2 v5 a2 2 0 0 1-2 2 H8 L5 16 V12 H2 a2 2 0 0 1-2-2 V5 a2 2 0 0 1 2-2 z", ox, oy, color, 1.4)
    return ""


def comparison():
    s = Svg(848, 752)
    banner(s)
    s.text(424, 204, "Model Family Comparison Table", 20, "#ffffff", 700, "middle")
    headers = (
        (128, "FAMILY NAME"),
        (360, "BEST FOR"),
        (545, "HARDWARE FIT"),
        (720, "MAIN TRADEOFF"),
    )
    for x, label in headers:
        s.text(x, 248, label, 11, "#9bb7c3", 700, "middle", 1.15)

    rows = [
        ("Llama", "llama", "#e07a28", "#3a220f", "#1a1008",
         [("chat", 248), ("pen", 272)], ["Chat & Writing"],
         "ram", ["8 GB to", "48 GB+"],
         [("lock", 648)], ["Less open", "license"]),
        ("Mistral", "mistral", "#5aa2f0", "#163056", "#0c1c34",
         [("code", 248), ("arrow", 274)], ["Coding &", "Autocomplete"],
         "ram", ["8 GB to", "24 GB+"],
         [("brain", 648)], ["Not top for", "hard reasoning"]),
        ("Qwen", "qwen", "#3dce78", "#12381f", "#0c2416",
         [("code", 248), ("globe", 276)], ["Coding +", "Multilingual", "(100+ languages)"],
         "ram", ["8 GB to", "48 GB+"],
         [("chip", 648)], ["Higher VRAM", "for large models"]),
        ("DeepSeek", "whale", "#8d78f2", "#2a2158", "#161230",
         [("brain", 248), ("sigma", 274)], ["Math & Logic", "Reasoning"],
         "ram", ["12 GB to", "48 GB+"],
         [("clock", 648)], ["Slower on", "simple prompts"]),
        ("Gemma", "gem", "#e15aa0", "#4a1a36", "#240e1c",
         [("image", 248), ("wrench", 274)], ["Multimodal &", "Tool Use"],
         "ram", ["8 GB to", "32 GB+"],
         [("doc", 648), ("warn", 670)], ["License needs", "close read"]),
        ("Phi", "phi", "#e0b030", "#3a3010", "#221c08",
         [("cpu", 248), ("bolt", 274)], ["Low-RAM /", "CPU-only"],
         "cpu", ["4 GB to", "16 GB"],
         [("bubble", 648)], ["Less polished", "chat"]),
    ]
    y = 268
    for name, icon, accent, fill, fill2, best_icons, best_lines, hw, hw_lines, trade_icons, trade_lines in rows:
        gid = "g" + name
        grad(s, gid, fill, fill2)
        pill(s, 36, y, 190, 64, f"url(#{gid})", accent)
        s.add(family_icon(icon, 50, y + 18))
        s.text(96, y + 38, name, 16, "#ffffff", 700)
        for kind, ix in best_icons:
            s.add(mini(kind, ix, y + 16, accent))
        n = len(best_lines)
        line_y = y + (36 if n == 1 else 28 if n == 2 else 22)
        step = 16
        for i, line in enumerate(best_lines):
            size = 13 if i < 2 else 11
            color = "#ffffff" if i < 2 else "#c5d0d6"
            weight = 600 if i == 0 else 400
            s.text(312, line_y + i * step, line, size, color, weight)
        if hw == "ram":
            ram_chip(s, 492, y + 16, accent)
        else:
            cpu_chip(s, 492, y + 16, accent)
        s.text(530, y + 30, hw_lines[0], 13, "#ffffff", 600)
        s.text(530, y + 48, hw_lines[1], 13, "#ffffff", 600)
        for kind, ix in trade_icons:
            s.add(mini(kind, ix, y + 16, accent))
        s.text(692, y + 30, trade_lines[0], 13, "#ffffff", 600)
        s.text(692, y + 48, trade_lines[1], 13, "#ffffff", 600)
        y += 80
    s.save("model-family-comparison.svg")


def device(kind, x, y, color):
    if kind == "laptop":
        return (
            f'<g transform="translate({x:.1f} {y:.1f})" fill="none" stroke="{color}" '
            f'stroke-width="1.6" stroke-linejoin="round">'
            f'<rect x="4" y="4" width="24" height="16" rx="2"/>'
            f'<path d="M1 22 H31 L28 20 H4 Z" fill="{color}" stroke="none" opacity="0.9"/>'
            f"</g>"
        )
    if kind == "gpu":
        return (
            f'<g transform="translate({x:.1f} {y:.1f})" fill="none" stroke="{color}" '
            f'stroke-width="1.6" stroke-linejoin="round">'
            f'<path d="M2 10 H22 L26 14 V22 H6 L2 18 Z"/>'
            f'<circle cx="12" cy="16" r="3.2"/>'
            f'<path d="M22 14 H26 M6 22 V26 M12 22 V26 M18 22 V26"/>'
            f"</g>"
        )
    return (
        f'<g transform="translate({x:.1f} {y:.1f})" fill="none" stroke="{color}" '
        f'stroke-width="1.6" stroke-linejoin="round">'
        f'<rect x="8" y="3" width="16" height="22" rx="2"/>'
        f'<rect x="11" y="7" width="10" height="7"/>'
        f'<path d="M12 18 H20 M14 21 H18"/>'
        f"</g>"
    )


def hardware():
    s = Svg(848, 504)
    s.text(424, 36, "Hardware Tier Quick-Pick Guide", 20, "#ffffff", 700, "middle")
    tiers = [
        ("laptop", "8 GB RAM", "Qwen3 8B (~5.5 GB VRAM) or Phi-4-mini (~2.5\u20132.8 GB VRAM)",
         "#5aa6f0", "#10233c", "#16304e"),
        ("gpu", "12\u201316 GB VRAM", "Qwen3 14B (~8\u20139 GB) or DeepSeek-R1-Distill-14B (~9\u201310 GB)",
         "#63e0ea", "#0e2c32", "#123c42"),
        ("gpu", "24 GB VRAM", "Qwen3 32B (~18\u201320 GB) or DeepSeek-R1-Distill-32B (72.6% on AIME)",
         "#b79af0", "#241c3e", "#322652"),
        ("tower", "48 GB+ VRAM", "Llama 3.3 70B (~40 GB at Q4) for general use | DeepSeek-R1 Distill 70B for reasoning",
         "#f09448", "#2c160c", "#3d2214"),
    ]
    y = 58
    for i, (kind, title, body, accent, fill, fill2) in enumerate(tiers):
        gid = f"tier{i}"
        grad(s, gid, fill, fill2)
        s.add(
            f'<rect x="36" y="{y}" width="776" height="68" rx="16" '
            f'fill="url(#{gid})" stroke="{accent}" stroke-width="1.7"/>'
        )
        s.add(device(kind, 52, y + 16, accent))
        s.text(108, y + 28, title, 16, accent, 700)
        s.text(108, y + 50, body, 14, "#e8eef2", 400)
        y += 76

    s.defs.append(
        '<linearGradient id="tip" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#083038"/>'
        '<stop offset="1" stop-color="#061820"/>'
        "</linearGradient>"
    )
    s.add(
        '<rect x="22" y="372" width="804" height="112" rx="18" '
        'fill="url(#tip)" stroke="#3ad4e4" stroke-width="2"/>'
    )
    circuit(s, 40, 388, 1, 1)
    circuit(s, 808, 388, -1, 1)
    circuit(s, 40, 468, 1, -1)
    circuit(s, 808, 468, -1, -1)
    # bulb
    s.add(
        '<g transform="translate(48 404)">'
        '<circle cx="16" cy="14" r="10" fill="#f6e27a" stroke="#e0b030" stroke-width="1.2"/>'
        '<path d="M16 2 V6 M6 6 L8.5 8.5 M26 6 L23.5 8.5 M4 16 H8 M24 16 H28" '
        'stroke="#f6e27a" stroke-width="1.4" stroke-linecap="round"/>'
        '<rect x="11" y="22" width="10" height="4" rx="1" fill="#d7c36a"/>'
        "</g>"
    )
    s.text(96, 424, "Pro Tip:", 16, "#4ae3f5", 700)
    s.text(168, 424, "Within the same memory budget, a larger", 16, "#ffffff", 600)
    s.text(96, 452, "Q4 model usually beats a smaller Q8 model.", 16, "#ffffff", 600)
    s.save("hardware-tier-guide.svg")


if __name__ == "__main__":
    comparison()
    hardware()
