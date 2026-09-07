"""Backbone metadata shared across the RQ1-RQ5 plotting scripts.

Source of truth for architecture/paradigm/domain is the user's own paper
table (see plots/results_plots_layout.md discussion); this is the one file
meant to be hand-edited before/while iterating on the figures.
"""

# ---------------------------------------------------------------------------
# Per-backbone metadata, keyed by archive folder name (as it appears under
# archive/Pooled-Embeddings/<name>, archive/XCM-Generalization/<name>, etc.)
# ---------------------------------------------------------------------------
# architecture: "CNN" | "Transformer"
# paradigm: "SL" | "SSL"  (AST is SSL/Semi-SL upstream but rendered as SSL here)
# domain: "Bird" | "Other -> Bird" | "Cross-taxa" | "General audio" |
#         "General audio + cross-taxa" | "None"
#         ("Other -> Bird" = originally pretrained on a different domain --
#          ImageNet/AudioSet/speech -- then fine-tuned on BirdSet-XCL;
#          "None" = BEANS baseline, trained from scratch, no pretraining at all)

BACKBONE_META = {
    "EfficientNet-B1-BirdSet-XCL": dict(
        display="EfficientNet-B1", architecture="CNN", paradigm="SL", domain="Other -> Bird",
        identity_color="#4c72b0",
    ),
    "AudioProtoPNet-20-BirdSet-XCL": dict(
        display="AudioProtoPNet", architecture="CNN", paradigm="SL", domain="Bird",
        identity_color="#dd8452",
    ),
    "Birdnet_V2.3": dict(
        display="BirdNET v2.3", architecture="CNN", paradigm="SL", domain="Cross-taxa",
        identity_color="#55a868",
    ),
    "Birdnet_V2.4": dict(
        display="BirdNET v2.4", architecture="CNN", paradigm="SL", domain="Cross-taxa",
        identity_color="#c44e52",
    ),
    "perch_8": dict(
        display="Perch 1.0", architecture="CNN", paradigm="SL", domain="Bird",
        identity_color="#8172b3",
    ),
    "perch_v2_cpu": dict(
        display="Perch 2.0", architecture="CNN", paradigm="SL", domain="Cross-taxa",
        identity_color="#937860",
    ),
    "vggish": dict(
        display="VGGish", architecture="CNN", paradigm="SL", domain="General audio",
        identity_color="#da8bc3",
    ),
    "yamnet": dict(
        display="YAMNet", architecture="CNN", paradigm="SL", domain="General audio",
        identity_color="#8c8c8c",
    ),
    "BeansBaseline": dict(
        display="BEANS baseline", architecture="CNN", paradigm=None, domain="None",
        identity_color="#ccb974",
    ),
    "Bird-MAE-Huge": dict(
        display="Bird-MAE", architecture="Transformer", paradigm="SSL", domain="Bird",
        identity_color="#64b5cd",
    ),
    "AST-Birdset-XCL": dict(
        display="AST", architecture="Transformer", paradigm="SSL", domain="Other -> Bird",
        identity_color="#e41a1c",
    ),
    "Wav2Vec2-Base-BirdSet-XCL": dict(
        display="Wav2Vec2", architecture="Transformer", paradigm="SSL", domain="Other -> Bird",
        identity_color="#377eb8",
    ),
    "NatureLMBEATs": dict(
        display="NatureLM-audio", architecture="Transformer", paradigm="SSL",
        domain="General audio + cross-taxa", identity_color="#4daf4a",
    ),
}

# Fine-Tuning / XCM-Generalization-fine-tune studies use different folder
# names for some of the same backbones; map them to the canonical
# BACKBONE_META key above. perch_v2_old_lr is a deprecated LR variant and is
# intentionally excluded (not mapped).
FINETUNE_NAME_TO_CANONICAL = {
    "AudioProtoPNet": "AudioProtoPNet-20-BirdSet-XCL",
    "BirdSetBirdMAE": "Bird-MAE-Huge",
    "BirdSetEfficientNet": "EfficientNet-B1-BirdSet-XCL",
    "perch_v2": "perch_v2_cpu",
    "NatureLMBEATs": "NatureLMBEATs",
}

# RQ1 point color = domain (5 real categories + "None" for BEANS, which has
# no pretraining at all and is rendered as a distinct neutral gray rather
# than folded into any of the others).
DOMAIN_COLOR = {
    "Bird": "#2ca02c",
    "Other -> Bird": "#bcbd22",
    "Cross-taxa": "#ff7f0e",
    "General audio": "#1f77b4",
    "General audio + cross-taxa": "#9467bd",
    "None": "#7f7f7f",
}

# RQ1 point shape = architecture.
MARKER_BY_ARCH = {
    "CNN": "o",
    "Transformer": "^",
}


def ordered_backbones(subset=None):
    """Fixed backbone x-order (by domain, then insertion order), reused
    across figures for visual consistency. `subset` restricts to a given
    iterable of archive names (e.g. the 11 shared with XCM-Generalization)."""
    domain_rank = {d: i for i, d in enumerate(DOMAIN_COLOR)}
    names = list(BACKBONE_META) if subset is None else [n for n in BACKBONE_META if n in subset]
    return sorted(names, key=lambda n: (domain_rank[BACKBONE_META[n]["domain"]], n))
