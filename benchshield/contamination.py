"""Contamination probes for image-classification benchmarks.

Two probes, both dependency-light (numpy + PIL only):

1. near_duplicate_overlap(train_images, test_images)
   Perceptual dHash (difference hash) on every image; counts test images
   with a near-duplicate (hamming distance <= threshold) in train.
   Returns overlap fraction + the offending pairs. Train/test leakage is
   the classic way a benchmark silently overstates accuracy.

2. canary_probe(...)
   The canary-string idea adapted to vision: insert K synthetic "canary"
   samples (fixed random-noise patterns with assigned labels) into the
   TRAINING set, train the model, then measure canary accuracy vs
   held-out accuracy. A model that memorized its training data scores far
   above chance on canaries it has "seen"; a generalizing model scores
   near chance. The gap is the memorization signal.
   We provide: make_canaries(), and canary_gap() which given per-sample
   predictions computes the probe statistic + a permutation p-value.
"""
from __future__ import annotations

import math
import random

import numpy as np

try:
    from PIL import Image
    _PIL = True
except ImportError:  # pragma: no cover
    _PIL = False


# ---------------------------------------------------------------- dHash
def _to_gray_9x8(img):
    """img: HxW(xC) array -> 9x8 grayscale float array."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    a = a.astype(np.float64)
    if _PIL:
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((9, 8))
        return np.asarray(im, dtype=np.float64)
    # fallback: block-average downsample
    h, w = a.shape
    out = np.zeros((8, 9))
    for i in range(8):
        for j in range(9):
            out[i, j] = a[i * h // 8:(i + 1) * h // 8,
                           j * w // 9:(j + 1) * w // 9].mean()
    return out


def dhash(img):
    """64-bit difference hash as Python int."""
    g = _to_gray_9x8(img)
    diff = g[:, :-1] > g[:, 1:]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count("1")


def near_duplicate_overlap(train_images, test_images, threshold=5,
                           max_pairs=50):
    """Fraction of test images with a near-duplicate in train.

    Returns dict(fraction, n_test, n_flagged, pairs=[(test_idx, train_idx, dist)]).

    See distance_histogram() for calibrating `threshold` to a new domain:
    a bimodal histogram (duplicate cluster well below the bulk) means the
    threshold is separating real duplicates; a unimodal histogram means the
    threshold is just slicing natural similarity - recalibrate.
    """
    train_hashes = [dhash(im) for im in train_images]
    n_test = len(test_images)
    flagged = []
    for ti, tim in enumerate(test_images):
        th = dhash(tim)
        best = min(hamming_distance(th, hh) for hh in train_hashes)
        if best <= threshold:
            # find one witness
            wi = next(i for i, hh in enumerate(train_hashes)
                      if hamming_distance(th, hh) == best)
            flagged.append((ti, wi, best))
            if len(flagged) >= max_pairs:
                # keep counting cheaply
                pass
    return {
        "n_test": n_test,
        "n_train": len(train_images),
        "n_flagged": len(flagged),
        "fraction": len(flagged) / n_test if n_test else 0.0,
        "threshold": threshold,
        "pairs": flagged[:max_pairs],
    }


# ---------------------------------------------------------------- calibration
def distance_histogram(train_images, test_images, max_dist=32):
    """Best-match Hamming distance histogram: test -> nearest train image.

    Use to calibrate the near-duplicate threshold for a new domain.
    Bimodal (a low-distance cluster separated from the bulk) => real
    duplicate subpopulation; unimodal => threshold would slice natural
    similarity, pick lower or verify pairs visually.
    Returns dict(dist -> count).
    """
    from collections import Counter
    train_hashes = [dhash(im) for im in train_images]
    hist = Counter()
    for tim in test_images:
        h = dhash(tim)
        d = min(hamming_distance(h, x) for x in train_hashes)
        hist[min(d, max_dist)] += 1
    return dict(sorted(hist.items()))


# ---------------------------------------------------------------- canaries
def make_canaries(n, image_shape, n_classes, seed=0, label_mode="fixed"):
    """Generate n canary samples: fixed random patterns + labels.

    image_shape like (28, 28) or (28, 28, 1). label_mode "fixed" assigns
    labels round-robin; "random" assigns uniformly at random.
    Returns (images uint8 array, labels int array).
    """
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(n,) + tuple(image_shape)).astype(np.uint8)
    if label_mode == "fixed":
        labels = np.arange(n) % n_classes
    else:
        labels = rng.integers(0, n_classes, size=n)
    return imgs, labels.astype(int)


def canary_probe(canary_correct, canary_total, heldout_correct, heldout_total,
                n_perm=2000, seed=0, gap_warn=0.1, gap_fail=0.2, alpha=0.05):
    """High-level canary probe: gap statistic + p-value + verdict.

    Returns dict with canary_acc, heldout_acc, gap, p_value, verdict
    ("CLEAN" / "SUSPICIOUS" / "MEMORIZED").
    """
    stat = canary_gap(canary_correct, canary_total, heldout_correct,
                      heldout_total, n_perm=n_perm, seed=seed)
    g, p = stat["gap"], stat["p_value"]
    if g > gap_fail and p < alpha:
        verdict = "MEMORIZED"
    elif g > gap_warn and p < alpha:
        verdict = "SUSPICIOUS"
    else:
        verdict = "CLEAN"
    stat["verdict"] = verdict
    return stat


def canary_gap(canary_correct, canary_total, heldout_correct, heldout_total,
               n_perm=2000, seed=0):
    """Probe statistic: canary_acc - heldout_acc, with permutation p-value.

    A large positive gap with small p-value => the model memorized training
    data (canaries were IN train) rather than generalizing.
    Returns dict(gap, canary_acc, heldout_acc, p_value).
    """
    rng = random.Random(seed)
    ca = canary_correct / canary_total if canary_total else 0.0
    ha = heldout_correct / heldout_total if heldout_total else 0.0
    gap = ca - ha
    # permutation test on the pooled 0/1 outcomes
    y = [1] * canary_correct + [0] * (canary_total - canary_correct)
    z = [1] * heldout_correct + [0] * (heldout_total - heldout_correct)
    pooled = y + z
    n1 = len(y)
    obs = gap
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        m1 = sum(pooled[:n1]) / n1 if n1 else 0.0
        m2 = sum(pooled[n1:]) / (len(pooled) - n1) if len(pooled) > n1 else 0.0
        if m1 - m2 >= obs:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return {
        "canary_acc": round(ca, 4),
        "heldout_acc": round(ha, 4),
        "gap": round(gap, 4),
        "p_value": round(p, 4),
        "n_perm": n_perm,
    }
