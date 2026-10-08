#!/usr/bin/env python3
"""Presentation routing diagrams. 1280x720, AMD palette, source-matched layouts."""

import math
from dataclasses import dataclass
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "reports" / "assets"
TEAL, GOLD, ORANGE, RED = "#00c2de", "#C1A968", "#F26522", "#ed1c24"
WHITE, MUTED = "#ffffff", "#9aa3ad"
TF, GF, OF, RF, PANEL = "#00181c", "#14110C", "#1F0D05", "#1A080A", "#05080c"
INNER, DIM, SELECT = "#0b1016", "#0c5563", "#06343c"
FONT = "Arial, Helvetica, sans-serif"
PX, PY = 16, 12


def tw(s, size=16, weight=400):
    factor = 0.62 if int(weight) >= 600 else 0.54
    return int(len(s) * size * factor) + 8


def measure(title, desc="", ts=20, ds=15, px=PX, py=PY):
    w = max(tw(title, ts, 700), tw(desc, ds) if desc else 0) + px * 2
    h = py + ts + ((8 + ds) if desc else 0) + py
    return w, h


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def r(self):
        return self.x + self.w

    @property
    def b(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


class Svg:
    def __init__(self, w=1280, h=720):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=16, fill=WHITE, weight=400, anchor="start", spacing=0):
        extra = f' letter-spacing="{spacing}"' if spacing else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{s}</text>'
        )

    def box(self, x, y, w, h, fill, stroke, sw=1.6, r=10):
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, d, color=TEAL, sw=2, dashed=False):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.add(f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="none"{dash}/>')

    def arrow(self, d, color=TEAL, sw=2, dashed=False):
        mid = {TEAL: "arrT", GOLD: "arrG", ORANGE: "arrO", RED: "arrR"}[color]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.add(
            f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="none" '
            f'stroke-linecap="butt" stroke-linejoin="round"{dash} '
            f'marker-end="url(#{mid})"/>'
        )

    def loop_arc(self, a, b, origin, color, bulge=54):
        ox, oy = origin
        ang_a = math.atan2(a.cy - oy, a.cx - ox)
        ang_b = math.atan2(b.cy - oy, b.cx - ox)
        da = (ang_b - ang_a) % math.tau
        if da > math.pi:
            da -= math.tau
        r = (math.hypot(a.cx - ox, a.cy - oy) + math.hypot(b.cx - ox, b.cy - oy)) / 2 + bulge

        def polar(ang):
            return ox + r * math.cos(ang), oy + r * math.sin(ang)

        h1x, h1y = polar(ang_a + da * 0.36)
        h2x, h2y = polar(ang_a + da * 0.64)
        x0, y0 = _edge(a, (b.cx - a.cx) + (a.cx - ox) * 0.4, (b.cy - a.cy) + (a.cy - oy) * 0.4, 8)
        x3, y3 = _edge(b, (a.cx - b.cx) + (b.cx - ox) * 0.4, (a.cy - b.cy) + (b.cy - oy) * 0.4, 12)
        self.arrow(
            f"M{x0:.1f} {y0:.1f} C {h1x:.1f} {h1y:.1f}, {h2x:.1f} {h2y:.1f}, {x3:.1f} {y3:.1f}",
            color,
        )

    def card(self, x, y, title, desc, fill, stroke, sw=1.6, ts=20, ds=15, r=10,
             title_fill=WHITE, desc_fill=MUTED, min_w=0, px=PX, py=PY):
        w, h = measure(title, desc, ts, ds, px, py)
        w = max(w, min_w)
        self.box(x, y, w, h, fill, stroke, sw, r)
        self.text(x + px, y + py + ts - 2, title, ts, title_fill, 700)
        if desc:
            self.text(x + px, y + py + ts + 6 + ds, desc, ds, desc_fill)
        return Box(x, y, w, h)

    def pill(self, x, y, label, fill, stroke, tc=WHITE):
        w, h = tw(label, 14, 700) + 24, 32
        self.box(x, y, w, h, fill, stroke, 1.3, 8)
        self.text(x + w / 2, y + 21, label, 14, tc, 700, "middle")
        return Box(x, y, w, h)

    def nested(self, x, y, accent, name, desc="", selected=False, w=None):
        h = 36
        nw = tw(name, 15, 700)
        dw = tw(desc, 14) if desc else 0
        if w is None:
            w = 14 + 4 + 10 + nw + (12 + dw if desc else 0) + 14
        fill = SELECT if selected else INNER
        self.box(x, y, w, h, fill, accent, 2.0 if selected else 1.2, 8)
        self.add(
            f'<rect x="{x + 10:.1f}" y="{y + 8:.1f}" width="4" height="{h - 16}" rx="1" fill="{accent}"/>'
        )
        self.text(x + 24, y + 24, name, 15, WHITE, 700)
        if desc:
            self.text(x + w - 12, y + 24, desc, 14, MUTED, 400, "end")
        return Box(x, y, w, h)

    def wrap(self, inner, pad=16, fill=PANEL, stroke=TEAL, sw=1.6, r=12, title_h=0):
        x = min(b.x for b in inner) - pad
        y = min(b.y for b in inner) - pad - title_h
        w = max(b.r for b in inner) - x + pad
        h = max(b.b for b in inner) - y + pad
        self.parts.insert(
            0,
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>',
        )
        return Box(x, y, w, h)

    def hexagon(self, cx, cy, r, fill, stroke, sw=2):
        pts = []
        for i in range(6):
            a = math.pi / 6 + i * math.pi / 3
            pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
        self.add(
            f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def icon(self, kind, x, y, color, size=20):
        sw = 1.45
        s = size
        def path(d, fill="none"):
            self.add(
                f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        if kind == "doc":
            path(f"M{x+4} {y+3} h{s-10} l6 6 v{s-12} h-{s-4} z")
            path(f"M{x+s-6} {y+3} v6 h6")
        elif kind == "brain":
            path(f"M{x+4} {y+11} c0-6 4-8 7-8 c3 0 4 2 5 2 c1 0 4-2 6 2 c2 4-1 6-1 8 c0 4-4 6-7 6 c-3 0-4-2-5-2 c-2 0-5 2-5-2 c0-2 0-4 0-6z")
        elif kind == "star":
            cx, cy, r = x + s / 2, y + s / 2, s / 2 - 2
            pts = []
            for i in range(5):
                a = -math.pi / 2 + i * 2 * math.pi / 5
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
                a2 = a + math.pi / 5
                pts.append((cx + r * 0.42 * math.cos(a2), cy + r * 0.42 * math.sin(a2)))
            d = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts) + " Z"
            path(d, color)
        elif kind == "bolt":
            path(
                f"M{x+s*0.58} {y+2} L{x+5} {y+s*0.52} h{s*0.32} L{x+s*0.38} {y+s-2} "
                f"L{x+s-4} {y+s*0.46} h-{s*0.32} Z",
                color,
            )
        elif kind == "db":
            path(f"M{x+4} {y+6} c0-3 {s-8} -3 {s-8} 0 v{s-12} c0 3-{s-8} 3-{s-8} 0 z")
            path(f"M{x+4} {y+6} c0 3 {s-8} 3 {s-8} 0")
        elif kind == "image":
            path(f"M{x+3} {y+5} h{s-6} v{s-10} h-{s-6} z")
            path(f"M{x+6} {y+s-8} l4-5 3 3 3-4 3 6")
        elif kind == "gear":
            path(f"M{x+s/2:.1f} {y+4} v3 M{x+s/2:.1f} {y+s-4} v-3 M{x+4} {y+s/2:.1f} h3 M{x+s-4} {y+s/2:.1f} h-3")
            self.add(
                f'<circle cx="{x+s/2:.1f}" cy="{y+s/2:.1f}" r="{s*0.22:.1f}" fill="none" '
                f'stroke="{color}" stroke-width="{sw}"/>'
            )
        elif kind == "wrench":
            path(f"M{x+5} {y+s-5} L{x+s-8} {y+8} M{x+s-6} {y+6} l3 3")
        elif kind == "shield":
            path(f"M{x+s/2:.1f} {y+3} L{x+s-4} {y+7} v{s*0.32:.1f} c0 {s*0.28:.1f}-{s*0.22:.1f} {s*0.42:.1f}-{s/2-2:.1f} {s*0.5:.1f} "
                 f"C{x+6} {y+s*0.74:.1f} {x+4} {y+s*0.6:.1f} {x+4} {y+7} Z")
        elif kind == "check":
            path(f"M{x+4} {y+s*0.52:.1f} l{s*0.22:.1f} {s*0.22:.1f} l{s*0.42:.1f}-{s*0.48:.1f}", color)
        elif kind == "users":
            path(f"M{x+7} {y+8} a3 3 0 1 1 0.1 0 M{x+13} {y+8} a3 3 0 1 1 0.1 0")
            path(f"M{x+4} {y+s-4} c0-4 3-6 6-6 c3 0 6 2 6 6")
        elif kind == "funnel":
            path(f"M{x+4} {y+5} h{s-8} l-{s*0.28:.1f} {s*0.38:.1f} v{s*0.28:.1f} h-{s*0.18:.1f} v-{s*0.28:.1f} z")
        elif kind == "net":
            nodes = (
                (x + s * 0.50, y + s * 0.18),
                (x + s * 0.18, y + s * 0.74),
                (x + s * 0.82, y + s * 0.74),
            )
            cx, cy = x + s / 2, y + s / 2
            for nx, ny in nodes:
                path(f"M{cx:.1f} {cy:.1f} L{nx:.1f} {ny:.1f}")
            self.add(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{s * 0.13:.1f}" fill="{color}"/>'
            )
            for nx, ny in nodes:
                self.add(
                    f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{s * 0.11:.1f}" fill="none" '
                    f'stroke="{color}" stroke-width="{sw}"/>'
                )
        elif kind == "warn":
            path(f"M{x+s/2:.1f} {y+3} L{x+s-3} {y+s-4} H{x+3} Z")
            path(f"M{x+s/2:.1f} {y+8} v{s*0.28:.1f}")
        elif kind == "clock":
            self.add(
                f'<circle cx="{x+s/2:.1f}" cy="{y+s/2:.1f}" r="{s*0.36:.1f}" fill="none" '
                f'stroke="{color}" stroke-width="{sw}"/>'
            )
            path(f"M{x+s/2:.1f} {y+s/2:.1f} v-{s*0.2:.1f} M{x+s/2:.1f} {y+s/2:.1f} h{s*0.16:.1f}")
        elif kind == "coin":
            self.add(
                f'<ellipse cx="{x+s/2:.1f}" cy="{y+7}" rx="{s*0.32:.1f}" ry="3" fill="none" '
                f'stroke="{color}" stroke-width="{sw}"/>'
            )
            path(f"M{x+s*0.18:.1f} {y+7} v{s-12} c0 3 {s*0.64:.1f} 3 {s*0.64:.1f} 0 v-{s-12}")
        else:
            path(f"M{x+4} {y+4} h{s-8} v{s-8} h-{s-8} z")

    def tile(self, x, y, w, h, kind, title, desc, color, fill=INNER):
        self.box(x, y, w, h, fill, color, 1.3, 10)
        self.icon(kind, x + 10, y + (h - 20) / 2, color, 20)
        self.text(x + 36, y + h / 2 - 4, title, 13, WHITE, 700)
        self.text(x + 36, y + h / 2 + 14, desc, 11, MUTED)
        return Box(x, y, w, h)

    def save(self, name):
        markers = []
        for mid, color in (("arrT", TEAL), ("arrG", GOLD), ("arrO", ORANGE), ("arrR", RED)):
            markers.append(
                f'<marker id="{mid}" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" '
                f'refX="9" refY="5" markerWidth="11" markerHeight="11" orient="auto" '
                f'overflow="visible">'
                f'<path d="M0 0.6 L9 5 L0 9.4 Z" fill="{color}"/></marker>'
            )
        chrome = []
        for x in range(0, self.w + 1, 40):
            chrome.append(
                f'<line x1="{x}" y1="0" x2="{x}" y2="{self.h}" stroke="#00c2de" '
                f'stroke-width="0.5" opacity="0.07"/>'
            )
        for y in range(0, self.h + 1, 40):
            chrome.append(
                f'<line x1="0" y1="{y}" x2="{self.w}" y2="{y}" stroke="#00c2de" '
                f'stroke-width="0.5" opacity="0.07"/>'
            )

        def arm(x, y, sx, sy):
            chrome.append(
                f'<path d="M{x + sx * 22} {y} H{x} V{y + sy * 22}" stroke="#1a7a88" '
                f'stroke-width="1.4" fill="none"/>'
            )

        arm(18, 18, 1, 1)
        arm(self.w - 18, 18, -1, 1)
        arm(18, self.h - 18, 1, -1)
        arm(self.w - 18, self.h - 18, -1, -1)
        body = "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}">',
                "<defs>",
                *markers,
                "</defs>",
                f'<rect width="{self.w}" height="{self.h}" fill="#000000"/>',
                *chrome,
                *self.parts,
                "</svg>",
            ]
        )
        path = OUT / name
        path.write_text(body, encoding="utf-8")
        print(name, path.stat().st_size)


def _edge(box, dx, dy, outset=0):
    length = math.hypot(dx, dy) or 1
    ux, uy = dx / length, dy / length
    ts = []
    if abs(dx) > 1e-9:
        for edge_x in (box.x, box.r):
            t = (edge_x - box.cx) / dx
            if t > 0:
                yy = box.cy + t * dy
                if box.y - 0.05 <= yy <= box.b + 0.05:
                    ts.append(t)
    if abs(dy) > 1e-9:
        for edge_y in (box.y, box.b):
            t = (edge_y - box.cy) / dy
            if t > 0:
                xx = box.cx + t * dx
                if box.x - 0.05 <= xx <= box.r + 0.05:
                    ts.append(t)
    t = min(ts) if ts else 1.0
    return box.cx + t * dx + ux * outset, box.cy + t * dy + uy * outset


def kicker(s, label):
    s.text(48, 42, label, 12, TEAL, 700, spacing=2.2)


def footer(s, items):
    s.text(48, 702, "  |  ".join(items), 11, "#5b6570", 600, spacing=1.1)


def classify():
    s = Svg()
    kicker(s, "LLM ROUTING")
    s.text(48, 150, "Route each prompt", 36, WHITE, 700)
    s.text(48, 196, "to the model that fits.", 36, WHITE, 700)
    s.text(48, 242, "Classify capability, context,", 16, MUTED)
    s.text(48, 264, "quality, cost, and latency first.", 16, MUTED)

    chips = [
        ("brain", "Capability", TEAL),
        ("doc", "Context", TEAL),
        ("star", "Quality", GOLD),
        ("db", "Cost", ORANGE),
        ("bolt", "Latency", GOLD),
    ]
    x = 48
    for kind, label, color in chips:
        s.box(x, 430, 72, 72, INNER, color, 1.2, 10)
        s.icon(kind, x + 26, 442, color, 20)
        s.text(x + 36, 488, label, 11, MUTED, 600, "middle")
        x += 84

    factors = [
        ("brain", "Capability", "What the task needs", TEAL),
        ("doc", "Context", "Domain, length, format", TEAL),
        ("star", "Quality", "Output requirements", GOLD),
        ("db", "Cost", "Budget constraints", ORANGE),
        ("bolt", "Latency", "Speed requirements", GOLD),
    ]
    fx, fy, fw = 548, 64, 288
    s.box(fx, fy, fw, 200, PANEL, TEAL, 1.4, 12)
    s.text(fx + 16, fy + 22, "Routing Factors", 15, WHITE, 700)
    for i, (kind, name, desc, color) in enumerate(factors):
        yy = fy + 36 + i * 31
        s.icon(kind, fx + 16, yy, color, 16)
        s.text(fx + 40, yy + 13, name, 13, WHITE, 700)
        s.text(fx + 126, yy + 13, desc, 12, MUTED)

    prompt = s.card(548, 292, "User prompt", "Task, context, goals", TF, TEAL, 1.6, 16, 13, px=14, py=10)
    hx, hy, hr = 858, prompt.cy, 68
    s.hexagon(hx, hy, hr, TF, TEAL, 2)
    s.icon("net", hx - 11, hy - 32, TEAL, 22)
    s.text(hx, hy + 12, "Routing Layer", 13, WHITE, 700, "middle")
    s.text(hx, hy + 28, "Analyze and select", 11, MUTED, 400, "middle")
    hub = Box(hx - hr, hy - hr, hr * 2, hr * 2)

    models = [
        ("Model A", "General purpose", TF, DIM, 1.4, False),
        ("Model B", "Long context", SELECT, TEAL, 2.4, True),
        ("Model C", "Reasoning", TF, DIM, 1.4, False),
        ("Fallback", "If unavailable or unsuitable", RF, RED, 1.6, False),
    ]
    mw = max(measure(t, d, 16, 13)[0] for t, d, *_ in models)
    mh = measure(models[0][0], models[0][1], 16, 13)[1]
    mx = 1024
    y = hy - mh / 2 - 12 - mh
    drawn = []
    for i, (title, desc, fill, stroke, sw, _sel) in enumerate(models):
        if i == 3:
            y += 16
        card = s.card(mx, y, title, desc, fill, stroke, sw, 16, 13, min_w=mw)
        drawn.append(card)
        y = card.b + 12
    a, b, c, fb = drawn
    s.arrow(f"M{prompt.r + 4:.0f} {prompt.cy:.0f} H{hub.x + 18:.0f}")
    spine = b.x - 18
    s.line(f"M{hub.r - 8:.0f} {b.cy:.0f} H{spine:.0f}")
    s.line(f"M{spine:.0f} {a.cy:.0f} V{c.cy:.0f}")
    s.arrow(f"M{spine:.0f} {a.cy:.0f} H{a.x - 4:.0f}")
    s.arrow(f"M{spine:.0f} {b.cy:.0f} H{b.x - 4:.0f}")
    s.arrow(f"M{spine:.0f} {c.cy:.0f} H{c.x - 4:.0f}")
    s.arrow(f"M{hx:.0f} {hub.b - 8:.0f} V{fb.cy:.0f} H{fb.x - 4:.0f}", TEAL, dashed=True)
    s.text((hx + fb.x) / 2, fb.cy - 14, "If unavailable", 12, "#f5b4b8", 600, "middle")
    footer(s, ["CAPABILITY", "QUALITY", "COST", "LATENCY", "FALLBACK"])
    s.save("routing-classify.svg")


def architecture():
    s = Svg()
    kicker(s, "LLM ROUTING")
    s.text(48, 132, "Routing", 36, WHITE, 700)
    s.text(48, 176, "architecture", 36, WHITE, 700)
    s.text(48, 214, "Route by requirements, not brand.", 16, GOLD, 600)
    s.text(48, 430, "Convert each request into", 14, MUTED)
    s.text(48, 450, "explicit requirements, filter", 14, MUTED)
    s.text(48, 470, "ineligible models, then choose", 14, MUTED)
    s.text(48, 490, "the best eligible model.", 14, MUTED)

    reqs = [
        ("doc", "Task type", "What to do", TEAL),
        ("gear", "Capability", "Skills needed", TEAL),
        ("image", "Modality", "Text / image", TEAL),
        ("doc", "Context size", "Input length", TEAL),
        ("brain", "Reasoning", "Shallow to deep", GOLD),
        ("wrench", "Tool needs", "Tools, APIs", ORANGE),
        ("bolt", "Latency", "Max time", GOLD),
        ("coin", "Cost budget", "Max cost", ORANGE),
        ("shield", "Risk / policy", "Safety, policy", RED),
    ]
    cw, ch, gap = 148, 70, 10
    gx, gy = 392, 56
    req_frame = Box(gx - 14, gy - 28, cw * 3 + gap * 2 + 28, ch * 3 + gap * 2 + 44)
    s.box(req_frame.x, req_frame.y, req_frame.w, req_frame.h, PANEL, TEAL, 1.4, 12)
    s.text(gx, gy - 10, "Requirements", 14, WHITE, 700)
    for i, (kind, title, desc, color) in enumerate(reqs):
        col, row = i % 3, i // 3
        s.tile(gx + col * (cw + gap), gy + 8 + row * (ch + gap), cw, ch, kind, title, desc, color)

    y = 328
    steps = [
        ("1. Incoming", "Prompt and context", TEAL, TF),
        ("2. Classify", "Extract requirements", TEAL, TF),
        ("3. Filter", "Drop ineligible", ORANGE, OF),
    ]
    x = 380
    drawn = []
    for title, desc, color, fill in steps:
        drawn.append(s.card(x, y + 36, title, desc, fill, color, 1.5, 13, 11, px=10, py=8))
        x = drawn[-1].r + 10
    nest_w = 140
    elig = Box(x, y, nest_w + 16, 176)
    s.box(elig.x, elig.y, elig.w, elig.h, TF, TEAL, 1.5)
    s.text(elig.x + 10, elig.y + 16, "4. Candidates", 12, WHITE, 700)
    s.nested(elig.x + 8, elig.y + 24, DIM, "Fast model", w=nest_w)
    s.nested(elig.x + 8, elig.y + 60, GOLD, "Reasoning", w=nest_w)
    s.nested(elig.x + 8, elig.y + 96, TEAL, "Tool-use", selected=True, w=nest_w)
    s.nested(elig.x + 8, elig.y + 132, DIM, "Multimodal", w=nest_w)
    x = elig.r + 10
    selected = s.card(x, y + 36, "5. Selected", "Best eligible fit", TF, TEAL, 1.5, 13, 11, px=10, py=8)
    check = s.card(selected.r + 10, y + 36, "6. Check", "Safety, quality", GF, GOLD, 1.5, 13, 11, px=10, py=8)

    span = check.r - selected.x
    ow, oh, og = 72, 48, 8
    inner = ow * 3 + og * 2
    ox = selected.x + max(0, (span - inner) / 2)
    y_out = elig.b + 72
    outs = []
    for i, (title, desc, color, fill) in enumerate(
        (("Accept", "Use it", GOLD, GF), ("Escalate", "Review", ORANGE, OF), ("Fallback", "Next best", RED, RF))
    ):
        bx = ox + i * (ow + og)
        s.box(bx, y_out, ow, oh, fill, color, 1.4, 8)
        s.text(bx + ow / 2, y_out + 20, title, 12, WHITE, 700, "middle")
        s.text(bx + ow / 2, y_out + 36, desc, 11, MUTED, 400, "middle")
        outs.append(Box(bx, y_out, ow, oh))
    frame = s.wrap(outs, pad=12, fill=PANEL, stroke=GOLD, sw=1.4, r=10, title_h=18)
    s.text(frame.x + 12, frame.y + 16, "7. Outcomes", 13, WHITE, 700)

    s.line(f"M{req_frame.cx:.0f} {req_frame.b:.0f} V{drawn[1].y - 8:.0f}")
    s.arrow(f"M{req_frame.cx:.0f} {drawn[1].y - 8:.0f} H{drawn[1].cx:.0f} V{drawn[1].y - 4:.0f}")
    s.arrow(f"M{drawn[0].r + 3:.0f} {drawn[0].cy:.0f} H{drawn[1].x - 4:.0f}")
    s.arrow(f"M{drawn[1].r + 3:.0f} {drawn[1].cy:.0f} H{drawn[2].x - 4:.0f}")
    s.arrow(f"M{drawn[2].r + 3:.0f} {drawn[2].cy:.0f} H{elig.x - 4:.0f}")
    s.arrow(f"M{elig.r + 3:.0f} {selected.cy:.0f} H{selected.x - 4:.0f}")
    s.arrow(f"M{selected.r + 3:.0f} {selected.cy:.0f} H{check.x - 4:.0f}")
    s.line(f"M{check.cx:.0f} {check.b:.0f} V{frame.y:.0f}", GOLD)
    s.line(f"M{outs[0].cx:.0f} {frame.y + 22:.0f} H{outs[2].cx:.0f}", GOLD)
    s.arrow(f"M{outs[0].cx:.0f} {frame.y + 22:.0f} V{outs[0].y - 4:.0f}", GOLD)
    s.arrow(f"M{outs[1].cx:.0f} {frame.y + 22:.0f} V{outs[1].y - 4:.0f}", ORANGE)
    s.arrow(f"M{outs[2].cx:.0f} {frame.y + 22:.0f} V{outs[2].y - 4:.0f}", RED)
    s.arrow(
        f"M{drawn[2].cx:.0f} {drawn[2].b:.0f} V{frame.cy:.0f} H{frame.x - 4:.0f}",
        ORANGE,
        dashed=True,
    )
    s.text((drawn[2].cx + frame.x) / 2, (elig.b + frame.y) / 2, "If needed, try the next eligible model", 12, "#F6C2AE", 500, "middle")
    footer(s, ["TASK", "CAPABILITY", "COST", "LATENCY", "POLICY"])
    s.save("routing-architecture.svg")


def factors():
    s = Svg()
    kicker(s, "LLM ROUTING")
    s.text(48, 148, "Key factors in", 34, WHITE, 700)
    s.text(48, 194, "LLM routing", 34, WHITE, 700)
    s.text(48, 236, "Match the right model to the", 16, GOLD, 600)
    s.text(48, 258, "task with the right criteria.", 16, GOLD, 600)
    s.text(48, 430, "Different tasks.", 15, MUTED)
    s.text(48, 452, "Different requirements.", 15, MUTED)
    s.text(48, 474, "A better model choice.", 15, MUTED)

    hard = [
        ("doc", "Task type", "Writing, coding, analysis"),
        ("image", "Modality", "Text, image, video, audio"),
        ("wrench", "Tool requirements", "Function calling, browsing"),
        ("db", "Context length", "Short, medium, long"),
        ("shield", "Policy and risk", "Privacy, compliance, safety"),
    ]
    soft = [
        ("star", "Quality", "Higher accuracy and reasoning"),
        ("coin", "Cost", "Lower cost per request"),
        ("bolt", "Latency", "Faster response time"),
        ("check", "Consistency", "Stable and reliable outputs"),
        ("users", "Availability", "Region and uptime"),
    ]
    row_w = 320
    col_w = 16 + row_w + 16
    x0, x1, y0 = 430, 820, 72

    def column(x, title, subtitle, stroke, items, accent):
        s.box(x, y0, col_w, 348, PANEL, stroke, 1.7, 12)
        s.text(x + col_w / 2, y0 + 28, title, 14, stroke, 700, "middle", 1.3)
        s.text(x + col_w / 2, y0 + 48, subtitle, 13, MUTED, 400, "middle")
        rows = []
        for i, (kind, name, desc) in enumerate(items):
            yy = y0 + 66 + i * 54
            s.box(x + 16, yy, row_w, 46, INNER, accent, 1.2, 8)
            s.icon(kind, x + 26, yy + 13, stroke, 18)
            s.text(x + 52, yy + 20, name, 14, WHITE, 700)
            s.text(x + 52, yy + 36, desc, 12, MUTED)
            rows.append(Box(x + 16, yy, row_w, 46))
        return Box(x, y0, col_w, 348)

    left = column(x0, "HARD CONSTRAINTS", "Must be met", TEAL, hard, DIM)
    right = column(x1, "SOFT PREFERENCES", "Optimize after the hard rules", GOLD, soft, "#6E5C38")

    y_chain = 500
    prompt = s.card(470, y_chain, "Your prompt", "Task, context, goals", TF, TEAL, 1.6, 16, 13)
    route = s.card(prompt.r + 36, y_chain, "Model routing", "Analyze, rank, select the best fit", TF, TEAL, 1.8, 16, 13)
    model = s.card(route.r + 36, y_chain, "Right model", "Hard rules, quality floor", GF, GOLD, 1.6, 16, 13)
    bus = y_chain - 18
    s.arrow(f"M{left.cx:.0f} {left.b:.0f} V{bus:.0f} H{route.cx:.0f} V{route.y - 4:.0f}", TEAL)
    s.arrow(f"M{right.cx:.0f} {right.b:.0f} V{bus:.0f} H{route.cx:.0f} V{route.y - 4:.0f}", GOLD)
    s.arrow(f"M{prompt.r + 4:.0f} {prompt.cy:.0f} H{route.x - 4:.0f}")
    s.arrow(f"M{route.r + 4:.0f} {route.cy:.0f} H{model.x - 4:.0f}")
    footer(s, ["PROMPTS", "MODELS", "CONTEXT", "REASONING", "BETTER RESULTS"])
    s.save("routing-factors.svg")


def cascade():
    s = Svg()
    kicker(s, "LLM ROUTING")
    s.text(48, 140, "Cascade vs.", 34, WHITE, 700)
    s.text(48, 186, "fallback routing", 34, WHITE, 700)
    s.text(48, 228, "Escalation solves quality.", 16, GOLD, 600)
    s.text(48, 250, "Fallback solves availability.", 16, TEAL, 600)
    s.text(48, 430, "Use cascade when the first", 14, MUTED)
    s.text(48, 450, "answer is not good enough.", 14, MUTED)
    s.text(48, 478, "Use fallback when the", 14, MUTED)
    s.text(48, 498, "preferred model cannot respond.", 14, MUTED)

    lx = 430
    y = 168
    inner_w = 300
    fast_b = s.card(lx, y, "Fast model", "Quick and cost-effective", GF, GOLD, 1.5, 16, 13, min_w=inner_w, px=14, py=10)
    qc_b = s.card(lx, fast_b.b + 16, "Quality check", "Evaluate answer quality", TF, TEAL, 1.5, 16, 13, min_w=inner_w, px=14, py=10)
    pass_p = s.pill(lx + 20, qc_b.b + 16, "Pass", GF, GOLD, GOLD)
    fail_p = s.pill(lx + inner_w - 20 - tw("Fail", 14, 700) - 24, qc_b.b + 16, "Fail", RF, RED, RED)
    acc_b = s.card(lx, pass_p.b + 14, "Accept", "Use the answer", GF, GOLD, 1.5, 16, 13, px=14, py=10)
    str_b = s.card(lx + acc_b.w + 16, pass_p.b + 14, "Strong model", "Higher capability", OF, ORANGE, 1.5, 16, 13, px=14, py=10)
    acc2_b = s.card(str_b.x, str_b.b + 14, "Accept", "Use the better answer", GF, GOLD, 1.5, 16, 13, min_w=str_b.w, px=14, py=10)
    left = s.wrap([fast_b, qc_b, pass_p, fail_p, acc_b, str_b, acc2_b], pad=18, fill=PANEL, stroke=TEAL, sw=1.7, r=12, title_h=58)
    s.text(left.cx, left.y + 22, "CASCADE ROUTING", 13, TEAL, 700, "middle", 1.2)
    s.text(left.cx, left.y + 42, "Quality-driven escalation", 14, WHITE, 600, "middle")
    s.text(left.cx, left.y + 58, "Try a fast model first. Escalate if needed.", 12, MUTED, 400, "middle")

    fb_items = [
        ("Primary model", "Best fit for the task", TEAL, TF),
        ("Timeout or error", "No response, or a failure", RED, RF),
        ("Backup model", "Next model that is up", GOLD, GF),
        ("Respond", "Continue with the task", GOLD, GF),
    ]
    fw = max(measure(t, d, 16, 13)[0] for t, d, _, _ in fb_items)
    rx = left.r + 36
    y = 168
    fbs = []
    for title, desc, color, fill in fb_items:
        fbs.append(s.card(rx, y, title, desc, fill, color, 1.5, 16, 13, min_w=fw, px=14, py=10))
        y = fbs[-1].b + 14
    right = s.wrap(fbs, pad=18, fill=PANEL, stroke=TEAL, sw=1.7, r=12, title_h=58)
    s.text(right.cx, right.y + 22, "FALLBACK ROUTING", 13, TEAL, 700, "middle", 1.2)
    s.text(right.cx, right.y + 42, "Availability-driven recovery", 14, WHITE, 600, "middle")
    s.text(right.cx, right.y + 58, "If the preferred model cannot respond.", 12, MUTED, 400, "middle")

    s.arrow(f"M{fast_b.cx:.0f} {fast_b.b + 3:.0f} V{qc_b.y - 4:.0f}", GOLD)
    s.line(f"M{pass_p.cx:.0f} {qc_b.b + 8:.0f} H{fail_p.cx:.0f}", GOLD)
    s.line(f"M{qc_b.cx:.0f} {qc_b.b:.0f} V{qc_b.b + 8:.0f}", GOLD)
    s.arrow(f"M{pass_p.cx:.0f} {qc_b.b + 8:.0f} V{pass_p.y - 4:.0f}", GOLD)
    s.arrow(f"M{fail_p.cx:.0f} {qc_b.b + 8:.0f} V{fail_p.y - 4:.0f}", RED)
    s.arrow(f"M{pass_p.cx:.0f} {pass_p.b + 3:.0f} V{acc_b.y - 4:.0f}", GOLD)
    s.arrow(f"M{fail_p.cx:.0f} {fail_p.b + 3:.0f} V{str_b.y - 4:.0f}", RED)
    s.arrow(f"M{str_b.cx:.0f} {str_b.b + 3:.0f} V{acc2_b.y - 4:.0f}", GOLD)
    for a, b, color in ((fbs[0], fbs[1], TEAL), (fbs[1], fbs[2], RED), (fbs[2], fbs[3], GOLD)):
        s.arrow(f"M{a.cx:.0f} {a.b + 3:.0f} V{b.y - 4:.0f}", color)

    s.text((left.cx + right.cx) / 2, 598, "Different trigger. Same continuity.", 14, MUTED, 600, "middle")
    s.box(left.x, 616, 250, 48, GF, GOLD, 1.3, 8)
    s.text(left.x + 14, 634, "Cascade = quality escalation", 12, WHITE, 700)
    s.text(left.x + 14, 650, "Find the best answer.", 11, MUTED)
    s.box(right.r - 250, 616, 250, 48, TF, TEAL, 1.3, 8)
    s.text(right.r - 236, 634, "Fallback = availability recovery", 12, WHITE, 700)
    s.text(right.r - 236, 650, "Keep the request moving.", 11, MUTED)
    footer(s, ["QUALITY", "AVAILABILITY", "RESILIENCE", "BETTER ROUTING"])
    s.save("routing-cascade.svg")


def evaluation():
    s = Svg()
    kicker(s, "LLM ROUTING")
    s.text(48, 140, "Evaluation", 34, WHITE, 700)
    s.text(48, 186, "loop", 34, WHITE, 700)
    s.text(48, 228, "Measure quality, cost, latency,", 16, GOLD, 600)
    s.text(48, 250, "then update the policy.", 16, GOLD, 600)
    s.text(48, 430, "Evaluate routing on", 14, MUTED)
    s.text(48, 450, "representative traffic, then", 14, MUTED)
    s.text(48, 470, "update the router from", 14, MUTED)
    s.text(48, 490, "real outcomes.", 14, MUTED)

    tiles = [
        ("star", "Quality", "Correct and useful", GOLD),
        ("bolt", "Escalation", "Upgrade needed", ORANGE),
        ("coin", "Cost", "Cost of the request", GOLD),
        ("db", "Fallback", "Did a backup run?", TEAL),
        ("clock", "Latency", "First and last token", TEAL),
        ("shield", "Policy", "Hard rules held", RED),
    ]
    twid, th, g = 156, 52, 10
    mx, my = 700, 300
    hub = Box(mx - 16, my - 36, twid * 2 + g + 32, th * 3 + g * 2 + 52)
    s.box(hub.x, hub.y, hub.w, hub.h, PANEL, TEAL, 1.7, 12)
    s.text(hub.cx, my - 14, "5  Evaluation metrics", 15, WHITE, 700, "middle")
    for i, (kind, title, desc, color) in enumerate(tiles):
        col, row = i % 2, i // 2
        x = mx + col * (twid + g)
        y = my + row * (th + g)
        s.box(x, y, twid, th, INNER, color, 1.2, 8)
        s.icon(kind, x + 8, y + 16, color, 18)
        s.text(x + 32, y + 22, title, 13, WHITE, 700)
        s.text(x + 32, y + 38, desc, 11, MUTED)

    one_w = measure("1  Representative traffic", "Realistic prompts and tasks", 15, 12, 12, 8)[0]
    four_w = measure("4  Output", "The model response", 15, 12, 12, 8)[0]
    one = s.card(hub.cx - one_w / 2, 52, "1  Representative traffic", "Realistic prompts and tasks", TF, TEAL, 1.6, 15, 12, px=12, py=8)
    two = s.card(hub.r + 28, 132, "2  Router decision", "Policy, then quality", TF, TEAL, 1.6, 15, 12, px=12, py=8)
    three = s.card(hub.r + 28, 348, "3  Selected model", "Fast, strong, or backup", GF, GOLD, 1.6, 15, 12, px=12, py=8)
    four = s.card(hub.cx - four_w / 2, hub.b + 24, "4  Output", "The model response", OF, ORANGE, 1.6, 15, 12, px=12, py=8)
    six = s.card(400, hub.y - 8, "6  Update the policy", "Rules, thresholds, order", GF, GOLD, 1.6, 15, 12, px=12, py=8)
    origin = (hub.cx, hub.cy)
    s.loop_arc(one, two, origin, TEAL, 36)
    s.loop_arc(two, three, origin, GOLD, 32)
    s.loop_arc(three, four, origin, ORANGE, 36)
    s.arrow(f"M{four.cx:.0f} {four.y - 4:.0f} V{hub.b + 4:.0f}", ORANGE)
    s.arrow(
        f"M{hub.x:.0f} {min(max(six.cy, hub.y + 16), hub.b - 16):.0f} "
        f"C {hub.x - 36:.0f} {six.cy:.0f}, {six.r + 40:.0f} {six.cy:.0f}, {six.r + 12:.0f} {six.cy:.0f}",
        GOLD,
    )
    s.loop_arc(six, one, origin, TEAL, 40)
    footer(s, ["QUALITY", "COST", "LATENCY", "FALLBACK", "POLICY"])
    s.save("routing-eval.svg")


def route_flow():
    s = Svg()
    s.text(40, 42, "CLASSIFY THE REQUEST, THEN ROUTE", 14, MUTED, 700, spacing=1.4)
    req = s.card(40, 72, "Request", "One coding-agent turn", TF, TEAL, 1.8, 20, 14)
    clas = s.card(req.r + 36, 72, "Classify", "Capability, context, tools", GF, GOLD, 1.8, 20, 14)
    filt = s.card(clas.r + 36, 72, "Filter", "Hard rules before cost", OF, ORANGE, 1.8, 20, 14)
    s.arrow(f"M{req.r + 4:.0f} {req.cy:.0f} H{clas.x - 4:.0f}", GOLD)
    s.arrow(f"M{clas.r + 4:.0f} {clas.cy:.0f} H{filt.x - 4:.0f}", ORANGE)

    dw = max(
        measure("Local GPU", "OpenCode + Qwen 27B", 20, 14)[0],
        measure("Frontier", "Only after a quality miss", 20, 14)[0],
        tw("Routine code stays here", 14, 600) + PX * 2,
        tw("Cascade, not the default", 14) + PX * 2,
    )
    dh = PY + 20 + 8 + 14 + 6 + 14 + PY
    pair_gap = 48
    pair_w = dw * 2 + pair_gap
    lx = filt.cx - pair_w / 2
    lx = min(max(lx, 40), 1240 - pair_w)
    y = filt.b + 40
    local = Box(lx, y, dw, dh)
    s.box(local.x, local.y, local.w, local.h, GF, GOLD, 1.8)
    s.text(local.x + PX, local.y + PY + 18, "Local GPU", 20, WHITE, 700)
    s.text(local.x + PX, local.y + PY + 42, "OpenCode + Qwen 27B", 14, MUTED)
    s.text(local.x + PX, local.y + PY + 60, "Routine code stays here", 14, GOLD, 600)
    frontier = Box(local.r + pair_gap, y, dw, dh)
    s.box(frontier.x, frontier.y, frontier.w, frontier.h, RF, RED, 1.8)
    s.text(frontier.x + PX, frontier.y + PY + 18, "Frontier", 20, WHITE, 700)
    s.text(frontier.x + PX, frontier.y + PY + 42, "Only after a quality miss", 14, MUTED)
    s.text(frontier.x + PX, frontier.y + PY + 60, "Cascade, not the default", 14, "#f5b4b8", 600)
    s.line(f"M{filt.cx:.0f} {filt.b:.0f} V{y - 18:.0f}")
    s.arrow(f"M{filt.cx:.0f} {y - 18:.0f} H{local.cx:.0f} V{local.y - 4:.0f}", GOLD)
    s.arrow(f"M{filt.cx:.0f} {y - 18:.0f} H{frontier.cx:.0f} V{frontier.y - 4:.0f}", RED)

    checks = [("Accept", GOLD), ("Escalate", ORANGE), ("Fall back", RED)]
    cw = max(tw(t, 18, 700) + 32 for t, _ in checks)
    ch, cg = 44, 28
    yq = local.b + 36
    inner_w = cw * 3 + cg * 2
    qx = local.x + (frontier.r - local.x - inner_w) / 2
    pills = []
    for i, (label, color) in enumerate(checks):
        bx = qx + i * (cw + cg)
        s.box(bx, yq + 28, cw, ch, INNER, color, 1.5)
        s.text(bx + cw / 2, yq + 28 + 28, label, 18, WHITE, 700, "middle")
        pills.append(Box(bx, yq + 28, cw, ch))
    frame = s.wrap(pills, pad=16, fill="#001418", stroke=TEAL, sw=1.8, r=12, title_h=18)
    s.text(frame.cx, frame.y + 16, "QUALITY CHECK", 13, "#8fc8d2", 700, "middle", 1.4)
    s.arrow(f"M{local.cx:.0f} {local.b + 4:.0f} V{frame.y - 4:.0f}")
    s.arrow(f"M{frontier.cx:.0f} {frontier.b + 4:.0f} V{frame.y - 4:.0f}")
    s.line(f"M{pills[0].cx:.0f} {frame.y + 22:.0f} H{pills[2].cx:.0f}")
    for p, (_, color) in zip(pills, checks):
        s.arrow(f"M{p.cx:.0f} {frame.y + 22:.0f} V{p.y - 4:.0f}", color)
    s.save("route-flow.svg")


def router_diagram():
    s = Svg()
    s.text(40, 48, "ROUTE TO THE RIGHT MODEL.", 32, WHITE, 700)
    s.text(40, 90, "NOT THE BIGGEST.", 28, ORANGE, 700)
    lanes = [
        ("Explain this function", "Low complexity", TEAL, TF, "Small model", "Fast, if quality holds", GOLD, GF),
        ("Fix a type error", "Standard coding task", GOLD, GF, "Mid model", "Local 27B, most code", ORANGE, OF),
        ("Private data", "Hard policy: stay local", RED, RF, "Local only", "Policy blocks the frontier", RED, RF),
    ]
    rw = max(measure(a, b, 20, 15)[0] for a, b, *_ in lanes)
    dw = max(measure(e, f, 20, 15)[0] for *_, e, f, __, ___ in lanes)
    s.text(40, 140, "INCOMING REQUESTS", 13, MUTED, 700, spacing=1.2)
    s.text(40 + rw + 48, 140, "ROUTE TO BEST FIT", 13, MUTED, 700, spacing=1.2)
    y = 156
    drawn = []
    for title, desc, color, fill, dest, tdesc, tcolor, tfill in lanes:
        left = s.card(40, y, title, desc, fill, color, 1.8, 20, 15, min_w=rw)
        right = s.card(left.r + 48, y, dest, tdesc, tfill, tcolor, 1.8, 20, 15, min_w=dw)
        s.arrow(f"M{left.r + 4:.0f} {left.cy:.0f} H{right.x - 4:.0f}", color)
        drawn.append((left, right))
        y = left.b + 28
    pay_x = drawn[0][1].r + 40
    pay_y = drawn[0][0].y
    pay_w, pay_h = 236, max(drawn[-1][0].b - pay_y, 340)
    s.box(pay_x, pay_y, pay_w, pay_h, "#07070b", "#3a3a44", 1.6, 14)
    cx = pay_x + pay_w / 2
    s.text(cx, pay_y + 32, "THE PAYOFF", 13, WHITE, 700, "middle", 1.2)
    s.text(cx, pay_y + 92, "38&#215;", 42, GOLD, 700, "middle")
    s.text(cx, pay_y + 122, "lower $ / million", 15, WHITE, 400, "middle")
    s.text(cx, pay_y + 144, "vs Sonnet 5 at C4", 13, MUTED, 400, "middle")
    s.text(cx, pay_y + 200, "11%", 36, TEAL, 700, "middle")
    s.text(cx, pay_y + 230, "of a C4 workday", 15, WHITE, 400, "middle")
    s.text(cx, pay_y + 252, "covers the GPU share", 13, MUTED, 400, "middle")
    s.text(cx, pay_y + 300, "Quality floor", 18, RED, 700, "middle")
    s.text(cx, pay_y + 322, "before the cheap route", 14, WHITE, 400, "middle")
    s.save("router-diagram.svg")


def llm_router():
    s = Svg()
    kicker(s, "LLM ROUTING")
    s.text(48, 78, "Route the coding request.", 34, WHITE, 700)
    s.text(700, 78, "Local first. Frontier only after a miss.", 18, GOLD, 600)
    requests = [
        ("Explain function", ""),
        ("Fix type error", ""),
        ("Repository refactor", ""),
        ("Private code", "Stay on the local path"),
    ]
    rw = max(measure(t, d, 17, 13, 14, 10)[0] for t, d in requests)
    inner, y = [], 168
    for title, desc in requests:
        inner.append(s.card(64, y, title, desc, "#001418", DIM, 1.2, 17, 13, 8, px=14, py=10, min_w=rw))
        y = inner[-1].b + 10
    col1 = s.wrap(inner, pad=16, fill=TF, stroke=TEAL, sw=1.6, r=10, title_h=28)
    s.text(col1.x + 16, col1.y + 22, "1  Coding requests", 18, WHITE, 700)
    steps = [
        ("Analyze", "Intent and constraints"),
        ("Apply rules", "Policy before cost"),
        ("Build context", "Repository stays local"),
        ("Select", "Least expensive fit"),
    ]
    sw = max(measure(t, d, 18, 14, 14, 10)[0] for t, d in steps)
    y, s_boxes = 168, []
    for title, desc in steps:
        s_boxes.append(s.card(col1.r + 56, y, title, desc, "#1F0D05", "#723010", 1.2, 18, 14, 8, px=14, py=10, min_w=sw))
        y = s_boxes[-1].b + 16
    col2 = s.wrap(s_boxes, pad=16, fill=OF, stroke=ORANGE, sw=1.6, r=10, title_h=28)
    s.text(col2.x + 16, col2.y + 22, "2  LLM router", 18, WHITE, 700)
    for a, b in zip(s_boxes, s_boxes[1:]):
        s.arrow(f"M{a.cx:.0f} {a.b + 4:.0f} V{b.y - 4:.0f}", ORANGE)
    s.arrow(f"M{col1.r + 4:.0f} {inner[1].cy:.0f} H{col2.x - 4:.0f}")
    s.text((col1.r + col2.x) / 2, inner[1].cy - 14, "Query", 13, "#8fc8d2", 600, "middle")
    dests = [
        ("Local GPU", "Qwen 27B, routine code", TEAL, TF),
        ("Frontier", "After a quality miss", GOLD, GF),
        ("Specialized", "Tools and domain tasks", ORANGE, OF),
    ]
    dw = max(measure(t, d, 20, 14)[0] for t, d, _, _ in dests)
    dx, select, db = col2.r + 48, s_boxes[-1], []
    y0 = col2.y + 8
    span = col2.b - 16 - y0
    step_h = (span - 2 * 24) / 3
    for i, (title, desc, color, fill) in enumerate(dests):
        db.append(s.card(dx, y0 + i * (step_h + 24), title, desc, fill, color, 1.8, 20, 14, min_w=dw))
    s.line(f"M{col2.r:.0f} {select.cy:.0f} H{dx - 22:.0f}")
    s.line(f"M{dx - 22:.0f} {db[0].cy:.0f} V{db[2].cy:.0f}")
    s.arrow(f"M{dx - 22:.0f} {db[0].cy:.0f} H{db[0].x - 4:.0f}", TEAL)
    s.arrow(f"M{dx - 22:.0f} {db[1].cy:.0f} H{db[1].x - 4:.0f}", GOLD)
    s.arrow(f"M{dx - 22:.0f} {db[2].cy:.0f} H{db[2].x - 4:.0f}", ORANGE)
    s.save("llm-router.svg")


def llm_router_tools():
    s = Svg()
    kicker(s, "LLM ROUTING")
    s.text(48, 78, "Finish the response on the path.", 34, WHITE, 700)
    s.text(680, 78, "Tools stay beside the model.", 18, TEAL, 600)
    tools = [
        ("Repository", "Files the agent can edit"),
        ("Tests", "Checks that accept the change"),
        ("Search", "Docs and the web"),
        ("Internal APIs", "Services inside the network"),
    ]
    twid = max(measure(t, d, 17, 13, 14, 10)[0] for t, d in tools)
    y, tboxes = 176, []
    for title, desc in tools:
        tboxes.append(s.card(64, y, title, desc, "#001418", DIM, 1.2, 17, 13, 8, px=14, py=10, min_w=twid))
        y = tboxes[-1].b + 10
    col1 = s.wrap(tboxes, pad=16, fill=TF, stroke=TEAL, sw=1.6, r=10, title_h=28)
    s.text(col1.x + 16, col1.y + 22, "Tools and data", 18, WHITE, 700)
    steps = ["Validate", "Filter", "Format", "Guardrails"]
    sw = max(measure(t, "", 18, 14, 14, 10)[0] for t in steps)
    y, pboxes = 176, []
    for title in steps:
        pboxes.append(s.card(col1.r + 56, y, title, "", "#001418", DIM, 1.2, 18, 14, 8, px=14, py=10, min_w=sw))
        y = pboxes[-1].b + 16
    col2 = s.wrap(pboxes, pad=16, fill=TF, stroke=TEAL, sw=1.6, r=10, title_h=28)
    s.text(col2.x + 16, col2.y + 22, "Response processor", 18, WHITE, 700)
    for a, b in zip(pboxes, pboxes[1:]):
        s.arrow(f"M{a.cx:.0f} {a.b + 4:.0f} V{b.y - 4:.0f}")
    s.arrow(f"M{col1.r + 4:.0f} {col1.cy:.0f} H{col2.x - 4:.0f}")
    s.text((col1.r + col2.x) / 2, col1.cy - 14, "Fetch", 13, "#8fc8d2", 600, "middle")
    resp_lines = [
        "Refactor plan generated locally.",
        "Private code remains inside the network.",
    ]
    rw = max(tw("Final response", 20, 700), max(tw(ln, 14) for ln in resp_lines)) + PX * 2
    rh = PY + 20 + 8 + 14 + 6 + 14 + PY
    rx, ry = col2.r + 40, col2.cy - rh / 2
    s.box(rx, ry, rw, rh, GF, GOLD, 1.8)
    s.text(rx + PX, ry + PY + 18, "Final response", 20, "#F3E8C8", 700)
    s.text(rx + PX, ry + PY + 42, resp_lines[0], 14, WHITE)
    s.text(rx + PX, ry + PY + 62, resp_lines[1], 14, WHITE)
    resp = Box(rx, ry, rw, rh)
    s.arrow(f"M{col2.r + 4:.0f} {col2.cy:.0f} H{resp.x - 4:.0f}")
    s.arrow(
        f"M{resp.cx:.0f} {resp.b:.0f} C {resp.cx:.0f} {col1.b + 48:.0f}, {col1.cx:.0f} {col1.b + 48:.0f}, {col1.cx:.0f} {col1.b + 4:.0f}"
    )
    s.text((col1.cx + resp.cx) / 2, col1.b + 36, "Return to the coding agent", 15, "#E6D5A8", 600, "middle")
    s.save("llm-router-tools.svg")


if __name__ == "__main__":
    evaluation()
    factors()
    architecture()
    classify()
    cascade()
    route_flow()
    router_diagram()
    llm_router()
    llm_router_tools()
