"""
Architecture diagram for Masked Diffusion Summarizer.
Clean vertical layout optimized for diploma/paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

C = {
    "enc":     "#3B7DD8",
    "enc_l":   "#CADCF7",
    "dec":     "#E67E22",
    "dec_l":   "#FFF3E6",
    "noise":   "#8E44AD",
    "noise_l": "#E8D5F0",
    "emb":     "#27AE60",
    "emb_l":   "#D5F5E3",
    "loss":    "#E74C3C",
    "out":     "#2C3E50",
    "frozen":  "#7F8C8D",
    "bg":      "#FDFDFD",
    "txt":     "#1C1C1C",
    "arr":     "#444444",
    "self_a":  "#D4AC0D",
    "cross_a": "#C0392B",
    "ffn":     "#16A085",
    "mamba":   "#D35400",
}

fig, ax = plt.subplots(figsize=(16, 28))
ax.set_xlim(0, 16)
ax.set_ylim(0, 28)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C["bg"])


def box(x, y, w, h, text, fc, tc="white", fs=10, bold=False, ec=None, lw=1.5, alpha=1):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                        facecolor=fc, edgecolor=ec or fc, lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def arr(x1, y1, x2, y2, c=C["arr"], lw=1.8, ls="-", head="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=head, color=c, lw=lw, linestyle=ls), zorder=4)


def lbl(x, y, text, fs=9, c=C["txt"], ha="center", bold=False, italic=False):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=c,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", zorder=5)


# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(8, 27.4, "Masked Diffusion Summarizer", ha="center", fontsize=18,
        fontweight="bold", color=C["txt"])
ax.text(8, 26.9, "Frozen FRED-T5-1.7B Encoder  +  Trainable CrossMamba Decoder",
        ha="center", fontsize=11, color=C["frozen"], fontstyle="italic")

# ═══════════════════════════════════════════════════════════════
# INPUTS  (y=25.3)
# ═══════════════════════════════════════════════════════════════
box(0.5, 25.3, 4, 0.8, "Source Text", C["enc_l"], C["txt"], 11, bold=True)
box(6, 25.3, 4, 0.8, "Target Text", C["noise_l"], C["txt"], 11, bold=True)
box(12, 25.3, 3, 0.8, "t ~ U(0,T)", C["emb_l"], C["txt"], 11, bold=True)

# ═══════════════════════════════════════════════════════════════
# FROZEN ENCODER ZONE  (y=21.5–25)
# ═══════════════════════════════════════════════════════════════
frozen_bg = FancyBboxPatch((0.2, 21.2), 10.1, 3.9, boxstyle="round,pad=0.2",
    facecolor="#F0F4F8", edgecolor=C["frozen"], lw=2, ls="--", zorder=0)
ax.add_patch(frozen_bg)
lbl(5.25, 24.85, "FROZEN  (834M params)", fs=9, c=C["frozen"], bold=True)

box(0.5, 21.7, 4, 2.7, "FRED-T5-1.7B\nEncoder\n\nd_model=1536\n24 layers\n\noutput:\nhidden_states\n+ [CLS]",
    C["enc"], fs=9, bold=True)
box(6, 21.7, 4, 2.7, "FRED-T5-1.7B\nEncoder\n(same weights)\n\noutput:\nattention_scores aᵢ\n+ [CLS]",
    C["enc"], fs=9, bold=True)

arr(2.5, 25.3, 2.5, 24.45)
arr(8, 25.3, 8, 24.45)

# ═══════════════════════════════════════════════════════════════
# ENCODER OUTPUTS  (y=19.8–20.8)
# ═══════════════════════════════════════════════════════════════
box(0.5, 19.8, 2.3, 0.9, "Hidden\nStates", C["enc_l"], C["txt"], 9)
box(2.9, 19.8, 1.8, 0.9, "source\n[CLS]", C["enc_l"], C["txt"], 9)

box(6, 19.8, 2.3, 0.9, "Attention\nScores aᵢ", C["noise_l"], C["txt"], 9)
box(8.5, 19.8, 1.8, 0.9, "target\n[CLS]", C["noise_l"], C["txt"], 9)

arr(1.65, 21.7, 1.65, 20.75)
arr(3.5, 21.7, 3.8, 20.75)
arr(7.15, 21.7, 7.15, 20.75)
arr(8.5, 21.7, 9.4, 20.75)

# ═══════════════════════════════════════════════════════════════
# CLS PROJECTION  (y=18.7)
# ═══════════════════════════════════════════════════════════════
box(2.5, 18.6, 2.5, 0.8, "CLS Projection\n(trainable)", C["emb"], "white", 9, bold=True)
arr(3.8, 19.8, 3.75, 19.45)

# ═══════════════════════════════════════════════════════════════
# SEMANTIC NOISING  (y=17)
# ═══════════════════════════════════════════════════════════════
box(5.5, 17.0, 5, 1.1, "Semantic-Aware Noising\nPₜ = t/T − (1−t/T)·aᵢ",
    C["noise"], "white", 11, bold=True)

# labels → noising (from Target Text, going around)
arr(8, 25.3, 11.5, 25.7)
arr(11.5, 25.7, 11.5, 17.6)
arr(11.5, 17.6, 10.55, 17.55)
lbl(11.9, 22, "labels", fs=8, c=C["noise"], bold=True)

# attention scores → noising
arr(7.15, 19.8, 7.5, 18.15)

# ═══════════════════════════════════════════════════════════════
# NOISY TARGET  (y=15.6)
# ═══════════════════════════════════════════════════════════════
box(5, 15.5, 6, 1.0, "Noisy Target:  [M] важно [M] [M] текст [M]",
    C["noise_l"], C["txt"], 10)
arr(8, 17.0, 8, 16.55)

# ═══════════════════════════════════════════════════════════════
# TIMESTEP EMBEDDING  (y=21.5)
# ═══════════════════════════════════════════════════════════════
box(12, 22.0, 3, 2, "Timestep\nEmbedding\n\nsin/cos→MLP\nd=1536",
    C["emb"], "white", 10, bold=True)
arr(13.5, 25.3, 13.5, 24.05)

# ═══════════════════════════════════════════════════════════════
# DECODER ZONE  (y=2.5–14.8)
# ═══════════════════════════════════════════════════════════════
dec_bg = FancyBboxPatch((0.2, 2.5), 15.3, 12.5, boxstyle="round,pad=0.2",
    facecolor="#FFF8F0", edgecolor=C["dec"], lw=2, ls="--", zorder=0)
ax.add_patch(dec_bg)
lbl(8, 14.7, "TRAINABLE  CrossMamba Decoder  (462M params)", fs=10,
    c=C["dec"], bold=True)

# Token + Pos Embedding
box(3.5, 13.3, 6, 0.9, "Token Embedding + Position Embedding (d=1536)",
    C["emb_l"], C["txt"], 10, bold=True)
arr(8, 15.5, 6.5, 14.25)

# ─── CrossMamba Layer ×6 ──────────────────────────
ly_bg = FancyBboxPatch((0.8, 4.8), 14, 8.0, boxstyle="round,pad=0.2",
    facecolor="#FFF5EB", edgecolor=C["dec"], lw=2, zorder=0.5)
ax.add_patch(ly_bg)
lbl(8, 12.5, "CrossMamba Layer  ×6", fs=12, c=C["dec"], bold=True)

# Sublayer positions
sw, sh = 5.5, 0.9
sx = 1.5

y1 = 11.2
box(sx, y1, sw, sh, "① Mamba SSM Block\nO(n) bidirectional", C["mamba"], "white", 10, bold=True)
lbl(sx+sw+0.3, y1+0.45, "selective state space — local context", fs=8, c=C["txt"],
    ha="left", italic=True)

y2 = 10.0
box(sx, y2, sw, sh, "② Self-Attention\nbidirectional, all positions", C["self_a"], "white", 10, bold=True)
lbl(sx+sw+0.3, y2+0.45, "global target context (non-causal)", fs=8, c=C["txt"],
    ha="left", italic=True)

y3 = 8.8
box(sx, y3, sw, sh, "③ Cross-Attention\nQ=target, KV=source encoder", C["cross_a"], "white", 10, bold=True)
lbl(sx+sw+0.3, y3+0.45, "conditioning on source hidden states", fs=8, c=C["txt"],
    ha="left", italic=True)

y4 = 7.6
box(sx, y4, sw, sh, "④ Feed-Forward Network\nd_ff=6144, GELU", C["ffn"], "white", 10, bold=True)
lbl(sx+sw+0.3, y4+0.45, "nonlinear transformation", fs=8, c=C["txt"],
    ha="left", italic=True)

# Arrows between sublayers
mid_x = sx + sw/2
arr(mid_x, y1, mid_x, y1 - 0.1)
arr(mid_x, y1, mid_x, y2 + sh)
arr(mid_x, y2, mid_x, y3 + sh)
arr(mid_x, y3, mid_x, y4 + sh)

# AdaLN box
box(10.5, 5.5, 3.5, 3.0, "AdaLN\nTimestep\nConditioning\n\nscale & shift\nper sublayer",
    C["emb"], "white", 9, bold=True, alpha=0.9)

# Timestep → AdaLN
arr(13.5, 22.0, 13.5, 15.5)
arr(13.5, 15.5, 12.25, 8.55)
lbl(14.0, 17.5, "t_emb", fs=9, c=C["emb"], bold=True)

# AdaLN → sublayers (dashed)
arr(10.5, 7.8, 7.05, y1+0.45, c=C["emb"], lw=1.2, ls="--")
arr(10.5, 7.0, 7.05, y3+0.45, c=C["emb"], lw=1.2, ls="--")
arr(10.5, 6.2, 7.05, y4+0.45, c=C["emb"], lw=1.2, ls="--")

# Encoder hidden states → cross-attention (from left side)
arr(1.65, 19.8, 0.6, 19.0)
arr(0.6, 19.0, 0.6, 9.25)
arr(0.6, 9.25, 1.5, 9.25)
lbl(0.6, 16, "encoder\nhidden\nstates", fs=8, c=C["enc"], bold=True)

# Token emb → first layer
arr(6.5, 13.3, mid_x, y1+sh+0.15)

# ═══════════════════════════════════════════════════════════════
# OUTPUT HEAD  (y=3.5)
# ═══════════════════════════════════════════════════════════════
box(3.5, 3.3, 6, 0.9, "LayerNorm → Linear → Vocab Logits",
    C["out"], "white", 10, bold=True)
arr(mid_x, y4, mid_x, 5.0)
arr(mid_x, 5.0, 6.5, 4.25)

# ═══════════════════════════════════════════════════════════════
# LOSSES  (y=1.0)
# ═══════════════════════════════════════════════════════════════
box(0.3, 0.8, 3.2, 1.1, "L_vb\nCE on [MASK]\npositions", C["loss"], "white", 9, bold=True)
box(3.8, 0.8, 3.2, 1.1, "L_recon\nCE on ALL\nnon-pad", C["loss"], "white", 9, bold=True)
box(7.3, 0.8, 3.2, 1.1, "L_cls\n1−cos(Cₛ, Cₜ)\nsimilarity", C["loss"], "white", 9, bold=True)
box(11.5, 0.8, 3.7, 1.1, "L = L_vb +\nL_recon + L_cls",
    "#1A1A2E", "white", 10, bold=True)

arr(6.5, 3.3, 1.9, 1.95)
arr(6.5, 3.3, 5.4, 1.95)

# CLS → similarity loss
arr(3.75, 18.6, 3.75, 17.8)
arr(3.75, 17.8, 3.75, 15.2)
arr(3.75, 15.2, 8.9, 1.95)
lbl(3.2, 16.5, "source\nCLS", fs=7, c=C["emb"], bold=True)

arr(9.4, 19.8, 9.4, 15.2)
arr(9.4, 15.2, 8.9, 1.95)
lbl(9.9, 17.5, "target\nCLS\n(detach)", fs=7, c=C["noise"], bold=True)

# Arrows: individual losses → total
arr(3.5, 1.35, 11.5, 1.35)
arr(7.0, 1.35, 11.5, 1.35)
arr(10.5, 1.35, 11.5, 1.35)

# ═══════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════
legend_items = [
    (C["enc"], "Encoder (frozen, 834M)"),
    (C["dec"], "Decoder (trainable, 462M)"),
    (C["noise"], "Semantic noising"),
    (C["emb"], "Embeddings & conditioning"),
    (C["loss"], "Loss functions"),
]
for i, (color, text) in enumerate(legend_items):
    lx, ly = 12.3, 26.3 - i * 0.38
    r = FancyBboxPatch((lx, ly-0.12), 0.28, 0.28, boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=color, lw=1, zorder=2)
    ax.add_patch(r)
    ax.text(lx+0.4, ly+0.02, text, fontsize=8, va="center", color=C["txt"])

# Parameter summary
ax.text(8, 0.2, "Total: 1,296M params  |  Trainable: 462M  |  Frozen encoder: 834M",
        ha="center", fontsize=9, color=C["frozen"], fontstyle="italic")

plt.tight_layout()
plt.savefig("architecture_diagram.png", dpi=200, bbox_inches="tight", facecolor=C["bg"])
plt.savefig("architecture_diagram.pdf", bbox_inches="tight", facecolor=C["bg"])
print("Saved: architecture_diagram.png / .pdf")
