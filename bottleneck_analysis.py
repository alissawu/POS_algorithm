#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Optional
import re


def read_tagged(path: Path) -> List[List[Tuple[str, str]]]:
    sents: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    with path.open('r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            cur.append((parts[0], parts[1]))
    if cur:
        sents.append(cur)
    return sents


def is_numeric_token(word: str) -> bool:
    lower = word.lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", lower):
        return True
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", lower):
        return True
    if lower.endswith('%') and lower[:-1].replace('.', '', 1).isdigit():
        return True
    # decades like 1980s, 1990s, 30s
    if re.fullmatch(r"\d{2,4}s", lower):
        return True
    return False


def analyze_pairs(gold_path: Path, pred_path: Path, train_path: Optional[Path], pairs: List[Tuple[str, str]], topn: int = 15) -> None:
    gold = read_tagged(gold_path)
    pred = read_tagged(pred_path)
    assert sum(map(len, gold)) == sum(map(len, pred)), 'Token counts must match'

    train_vocab = set()
    if train_path and train_path.exists():
        for sent in read_tagged(train_path):
            for w, _ in sent:
                train_vocab.add(w)

    # Build flat aligned list for convenience
    gold_flat = [(w, t) for s in gold for (w, t) in s]
    pred_flat = [(w, t) for s in pred for (w, t) in s]

    # Global confusion
    conf = defaultdict(Counter)
    for (_, gt), (_, pt) in zip(gold_flat, pred_flat):
        if gt != pt:
            conf[gt][pt] += 1

    print('Top Confusions (global):')
    allc = []
    for g, m in conf.items():
        for p, c in m.items():
            allc.append((c, g, p))
    for c, g, p in sorted(allc, reverse=True)[:20]:
        print(f'  {g:6} -> {p:6}: {c}')
    print('\n' + '=' * 80)

    # Build sentence-level index for prev/next contexts
    idx = 0
    for pair in pairs:
        gtag, ptag = pair
        print(f"\nDETAIL: {gtag} -> {ptag}")
        errors = []
        sent_id = 0
        for s_gold, s_pred in zip(gold, pred):
            for i, ((gw, gt), (pw, pt)) in enumerate(zip(s_gold, s_pred)):
                assert gw == pw
                if gt == gtag and pt == ptag:
                    prev_tag = s_gold[i - 1][1] if i > 0 else '<S>'
                    next_tag = s_gold[i + 1][1] if i + 1 < len(s_gold) else '</S>'
                    prev_word = s_gold[i - 1][0] if i > 0 else '<S>'
                    next_word = s_gold[i + 1][0] if i + 1 < len(s_gold) else '</S>'
                    errors.append({
                        'sent': sent_id,
                        'i': i,
                        'word': gw,
                        'prev_tag': prev_tag,
                        'next_tag': next_tag,
                        'prev_word': prev_word,
                        'next_word': next_word,
                        'oov': (gw not in train_vocab) if train_vocab else None,
                        'cap': gw[:1].isupper(),
                        'hyphen': '-' in gw,
                        'ing': gw.lower().endswith('ing'),
                        'ed': gw.lower().endswith('ed'),
                        'ly': gw.lower().endswith('ly'),
                        'len': len(gw),
                        'isnum': is_numeric_token(gw),
                    })
                idx += 1
            sent_id += 1

        print(f"Total errors: {len(errors)}")
        if not errors:
            continue

        # Aggregates
        prev_tags = Counter(e['prev_tag'] for e in errors)
        next_tags = Counter(e['next_tag'] for e in errors)
        words = Counter(e['word'] for e in errors)
        hyph = sum(1 for e in errors if e['hyphen'])
        cap = sum(1 for e in errors if e['cap'])
        ing = sum(1 for e in errors if e['ing'])
        ed = sum(1 for e in errors if e['ed'])
        ly = sum(1 for e in errors if e['ly'])
        num = sum(1 for e in errors if e['isnum'])
        oov = sum(1 for e in errors if e['oov']) if errors and errors[0]['oov'] is not None else None

        print('Prev tags:', prev_tags.most_common(8))
        print('Next tags:', next_tags.most_common(8))
        print(f"Hyphen:{hyph} Cap:{cap} -ing:{ing} -ed:{ed} -ly:{ly} Numeric:{num}")
        if oov is not None:
            print(f"OOV:{oov} Seen:{len(errors)-oov}")

        print('\nExamples:')
        for w, c in words.most_common(topn):
            # show a couple contexts for this word
            ctx = [e for e in errors if e['word'] == w][:2]
            for e in ctx:
                print(f"  {e['prev_word']}/{e['prev_tag']} → {w} → {e['next_word']}/{e['next_tag']}")
        print('\n' + '-' * 80)


def main():
    ap = argparse.ArgumentParser(description='Focused bottleneck error analysis')
    ap.add_argument('gold', type=Path)
    ap.add_argument('pred', type=Path)
    ap.add_argument('--train', type=Path, default=None)
    ap.add_argument('--pairs', type=str, nargs='*', default=['NN:JJ', 'IN:RB', 'VBD:VBN', 'VBN:VBD', 'CD:NNS'])
    args = ap.parse_args()

    pairs: List[Tuple[str, str]] = []
    for p in args.pairs:
        try:
            g, pr = p.split(':', 1)
            pairs.append((g, pr))
        except ValueError:
            raise SystemExit(f'Invalid pair: {p}; expected format GOLD:PRED')

    analyze_pairs(args.gold, args.pred, args.train, pairs)


if __name__ == '__main__':
    main()
