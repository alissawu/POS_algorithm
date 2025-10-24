#!/usr/bin/env python3
"""
Advanced, category-level error analysis for POS tagging.

Usage:
  python3 advanced_error_analysis.py GOLD.pos PRED.pos [--limit 30]

Outputs:
  - Overall accuracy and top confusions
  - Particle-like window (short lowercase after verbs; with/without a particle list)
  - -ing nominalization after determiners
  - VBD/VBN confusion with auxiliary/subject contexts
  - Capitalized plural names (NNPS/NNP) with stem/next-token cues
  - Capitalization & sentence-initial slices

Notes:
  This is analysis only (no model changes). It helps identify robust, category-based
  rules to consider adding.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def read_tagged(path: Path) -> list[list[tuple[str, str]]]:
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with path.open("r", encoding="utf8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 2:
                continue
            w, t = parts
            cur.append((w, t))
    if cur:
        sents.append(cur)
    return sents


def accuracy_and_confusions(gold, pred, limit=20):
    N = 0
    correct = 0
    cm = defaultdict(Counter)  # gold->pred
    for gs, ps in zip(gold, pred):
        for (gw, gt), (pw, pt) in zip(gs, ps):
            # assume alignment
            if gw != pw:
                continue
            N += 1
            if gt == pt:
                correct += 1
            else:
                cm[gt][pt] += 1
    acc = 100.0 * correct / N if N else 0.0
    print(f"Overall Accuracy: {acc:.2f}% ({correct}/{N})\n")
    print("Top confusions (gold->pred):")
    conf = []
    for g, row in cm.items():
        for p, c in row.items():
            conf.append((c, g, p))
    for c, g, p in sorted(conf, reverse=True)[:limit]:
        print(f"  {g:5} -> {p:5}: {c}")
    print()
    return cm


def particle_window(gold, pred, limit=30):
    print("=== Particle/Preposition/Adverb Window (short-lower after verbs) ===")
    VERB_TAGS = {"VB","VBD","VBG","VBN","VBP","VBZ","MD"}
    rows = []
    for gs, ps in zip(gold, pred):
        for i in range(1, len(gs)):
            (gw, gt), (pw, pt) = gs[i], ps[i]
            (gw0, gt0), _ = gs[i-1], ps[i-1]
            if gt0 in VERB_TAGS and gw.isalpha() and gw.islower() and 2 <= len(gw) <= 4 and not gw.endswith("ly"):
                # collect next gold tag for context
                next_tag = gs[i+1][1] if i+1 < len(gs) else '</S>'
                rows.append((gw, gt, pt, gt0, next_tag))
    if not rows:
        print("(none)")
        return
    # Aggregate
    by_word = defaultdict(Counter)
    by_next = defaultdict(Counter)
    total = 0
    for w, gt, pt, prev_tag, next_tag in rows:
        by_word[w][(gt, pt)] += 1
        by_next[next_tag][(gt, pt)] += 1
        total += 1
    print(f"Total windows: {total}")
    print("Top short-lower words (gold->pred counts):")
    shown = 0
    for w, cnts in sorted(by_word.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
        items = ", ".join([f"{g}->{p}:{c}" for (g,p), c in cnts.most_common(3)])
        print(f"  {w:6} : {items}")
        shown += 1
        if shown >= limit:
            break
    print("\nNext gold-tag distributions:")
    shown = 0
    for t, cnts in by_next.items():
        items = ", ".join([f"{g}->{p}:{c}" for (g,p), c in cnts.most_common(3)])
        print(f"  next={t:5}: {items}")
        shown += 1
        if shown >= 12:
            break
    print()


def ing_nominalization(gold, pred, limit=30):
    print("=== -ing nominalization after determiners ===")
    DET = {"DT","PDT","PRP$"}
    rows = []
    for gs, ps in zip(gold, pred):
        for i in range(1, len(gs)):
            (gw, gt), (pw, pt) = gs[i], ps[i]
            (gw0, gt0), _ = gs[i-1], ps[i-1]
            if gt0 in DET and gw.lower().endswith("ing"):
                next_tag = gs[i+1][1] if i+1 < len(gs) else '</S>'
                rows.append((gw, gt, pt, gt0, next_tag))
    print(f"Candidates: {len(rows)}")
    if not rows:
        print()
        return
    # Head vs modifier proxy: next tag is nouny/verb/punct
    head_like = sum(1 for _,_,_,_,nt in rows if nt in {'.',',',':','VB','VBD','VBP','VBZ','MD','</S>'})
    print(f"  Head-like contexts (next is verb/punct/end): {head_like}")
    # Error counts
    cm = Counter()
    for w, gt, pt, prev, nt in rows:
        if gt != pt:
            cm[(gt, pt)] += 1
    print("  Top confusions:")
    for (g,p), c in cm.most_common(6):
        print(f"    {g}->{p}: {c}")
    print()


def vbn_vbd_contexts(gold, pred, limit=30):
    print("=== VBN/VBD confusion contexts ===")
    V = {"VB","VBD","VBG","VBN","VBP","VBZ","MD"}
    rows = []
    for gs, ps in zip(gold, pred):
        for i in range(len(gs)):
            (gw, gt), (pw, pt) = gs[i], ps[i]
            prev_tag = gs[i-1][1] if i>0 else '<S>'
            next_tag = gs[i+1][1] if i+1 < len(gs) else '</S>'
            if (gt, pt) in {("VBN","VBD"),("VBD","VBN")}:
                rows.append((gw.lower(), gt, pt, prev_tag, next_tag))
    print(f"Total VBN/VBD swaps: {len(rows)}")
    prev_ct = Counter(r[3] for r in rows)
    print("Prev-tag distribution:")
    for t, c in prev_ct.most_common(10):
        print(f"  {t:5}: {c}")
    print()


def plural_names(gold, pred, limit=30):
    print("=== Capitalized plurals (NNPS/NNP) ===")
    rows = []
    for gs, ps in zip(gold, pred):
        for i in range(len(gs)):
            (gw, gt), (pw, pt) = gs[i], ps[i]
            if gw[:1].isupper() and gw.endswith("s") and len(gw) > 2:
                stem = gw[:-1]
                next_cap = (gs[i+1][0][:1].isupper() if i+1 < len(gs) else False)
                rows.append((gw, gt, pt, stem, next_cap))
    gold_ct = Counter((gt, pt) for _, gt, pt, _, _ in rows)
    print("  Top (gold->pred):")
    for (g,p), c in gold_ct.most_common(6):
        print(f"    {g}->{p}: {c}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Advanced error analysis for POS taggers.")
    ap.add_argument("gold", type=Path)
    ap.add_argument("pred", type=Path)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    gold = read_tagged(args.gold)
    pred = read_tagged(args.pred)
    if sum(map(len, gold)) != sum(map(len, pred)):
        print("[WARN] Token counts differ; analysis may be off.")

    accuracy_and_confusions(gold, pred, limit=args.limit)
    particle_window(gold, pred, limit=args.limit)
    ing_nominalization(gold, pred, limit=args.limit)
    vbn_vbd_contexts(gold, pred, limit=args.limit)
    plural_names(gold, pred, limit=args.limit)


if __name__ == "__main__":
    main()

