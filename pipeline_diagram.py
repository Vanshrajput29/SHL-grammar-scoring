"""Draws the pipeline architecture figure used in the notebook and README (pipeline.png).

    python pipeline_diagram.py          # writes pipeline.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

TEXT, AUDIO, BOTH = "#cde2fb", "#fbe0d3", "#e7e6e1"     # light fills: text path, audio path, shared steps
TEXT_EDGE, AUDIO_EDGE, INK = "#2a78d6", "#eb6834", "#2b2b2a"

# name: (x centre, y centre, width, height, title, detail, fill, edge)
BOXES = {
    "audio":   (1.0, 3.0, 1.5, 1.0, "Audio file", ".wav, 16 kHz", BOTH, INK),
    "whisper": (3.6, 4.6, 2.6, 1.1, "Whisper large-v3-turbo", "transcript; a messy prompt\nkeeps fillers and errors", TEXT, TEXT_EDGE),
    "wavlm":   (3.6, 1.4, 2.6, 1.1, "WavLM-base-plus (frozen)", "speech embedding:\nfluency, pauses, pronunciation", AUDIO, AUDIO_EDGE),
    "deberta": (6.6, 5.3, 2.4, 0.8, "DeBERTa-v3-large", "fine-tuned on transcripts", TEXT, TEXT_EDGE),
    "feats":   (6.6, 3.5, 2.4, 1.1, "Hand-made features", "from the transcript: speaking rate,\nrepeats, confidence + CoEdIT\ngrammar-edit rate", BOTH, INK),
    "svr":     (6.6, 1.4, 2.4, 0.8, "SVR audio model", "WavLM embedding + features", AUDIO, AUDIO_EDGE),
    "combine": (9.7, 3.3, 2.5, 1.3, "Combine", "audio score < 1  →  audio score\n(unintelligible speech)\notherwise  →  ½ audio + ½ text", BOTH, INK),
    "final":   (12.2, 3.3, 1.5, 1.0, "Final score", "0 – 5", BOTH, INK),
}
# (from, to, label on the arrow, colour)
ARROWS = [
    ("audio", "whisper", "", TEXT_EDGE), ("audio", "wavlm", "", AUDIO_EDGE),
    ("whisper", "deberta", "", TEXT_EDGE), ("whisper", "feats", "", TEXT_EDGE),
    ("feats", "svr", "", AUDIO_EDGE), ("wavlm", "svr", "", AUDIO_EDGE),
    ("deberta", "combine", "text score", TEXT_EDGE), ("svr", "combine", "audio score", AUDIO_EDGE),
    ("combine", "final", "", INK),
]


def _edge(name, side):
    x, y, w, h = BOXES[name][:4]
    return {"left": (x - w / 2, y), "right": (x + w / 2, y), "top": (x, y + h / 2), "bottom": (x, y - h / 2)}[side]


def _sides(a, b):
    (xa, ya), (xb, yb) = BOXES[a][:2], BOXES[b][:2]
    if abs(xa - xb) < 0.1:                       # same column: vertical arrow
        return ("bottom", "top") if ya > yb else ("top", "bottom")
    return "right", "left"


def draw(path=None):
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13.2); ax.set_ylim(0.4, 6.1); ax.axis("off")
    for x, y, w, h, title, detail, fill, edge in BOXES.values():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fill, edgecolor=edge, linewidth=1.6))
        ax.text(x, y + h / 2 - 0.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, y + h / 2 - 0.45, detail, ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.3)
    for a, b, label, colour in ARROWS:
        sa, sb = _sides(a, b)
        p, q = _edge(a, sa), _edge(b, sb)
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=colour,
                                     connectionstyle="arc3,rad=0", shrinkA=2, shrinkB=2))
        if label:
            ax.text(p[0] + 0.4 * (q[0] - p[0]) - 0.15, p[1] + 0.4 * (q[1] - p[1]), label, ha="center",
                    fontsize=8.6, color=colour, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text(0.2, 0.55, "■ text model (reads the transcript)", color=TEXT_EDGE, fontsize=9.5, fontweight="bold")
    ax.text(4.3, 0.55, "■ audio model (listens to the speech)", color=AUDIO_EDGE, fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


if __name__ == "__main__":
    draw("pipeline.png")
    print("wrote pipeline.png")
