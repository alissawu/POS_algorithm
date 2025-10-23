#!/usr/bin/env python3
import sys
from collections import Counter, defaultdict

def analyze_specific_error(gold_file, pred_file, gold_tag, pred_tag):
    """Show all instances of a specific error type."""

    # Read files
    gold = []
    pred = []
    with open(gold_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    gold.append(parts)

    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    pred.append(parts)

    # Find specific errors
    errors = []
    for i, (g, p) in enumerate(zip(gold, pred)):
        if g[1] == gold_tag and p[1] == pred_tag:
            prev_word = gold[i-1][0] if i > 0 else '<START>'
            prev_tag = gold[i-1][1] if i > 0 else '<START>'
            next_word = gold[i+1][0] if i < len(gold)-1 else '<END>'
            next_tag = gold[i+1][1] if i < len(gold)-1 else '<END>'

            errors.append({
                'word': g[0],
                'prev_word': prev_word,
                'prev_tag': prev_tag,
                'next_word': next_word,
                'next_tag': next_tag,
                'index': i
            })

    print(f"{'='*70}")
    print(f"{gold_tag} -> {pred_tag} ERRORS: {len(errors)} total")
    print(f"{'='*70}\n")

    # Aggregate analysis
    prev_tags = Counter([e['prev_tag'] for e in errors])
    next_tags = Counter([e['next_tag'] for e in errors])
    words = Counter([e['word'] for e in errors])

    print("AGGREGATE ANALYSIS:")
    print("-" * 40)
    print("\nTop Previous Tags (what comes BEFORE the error):")
    for tag, count in prev_tags.most_common(10):
        examples = [e['word'] for e in errors if e['prev_tag'] == tag][:3]
        print(f"  After {tag:8}: {count:3} errors | Examples: {', '.join(examples)}")

    print("\nTop Next Tags (what comes AFTER the error):")
    for tag, count in next_tags.most_common(10):
        examples = [e['word'] for e in errors if e['next_tag'] == tag][:3]
        print(f"  Before {tag:8}: {count:3} errors | Examples: {', '.join(examples)}")

    print("\nMost Frequent Error Words:")
    for word, count in words.most_common(20):
        print(f"  \"{word}\": {count} times")

    print("\n" + "="*70)
    print("ALL ERRORS WITH CONTEXT:")
    print("-" * 40)

    # Group by previous tag for easier reading
    errors_by_prev = defaultdict(list)
    for e in errors:
        errors_by_prev[e['prev_tag']].append(e)

    for prev_tag in sorted(errors_by_prev.keys(), key=lambda x: -len(errors_by_prev[x])):
        print(f"\nAfter {prev_tag} ({len(errors_by_prev[prev_tag])} errors):")
        for e in errors_by_prev[prev_tag]:
            print(f"  {e['prev_word']}/{prev_tag} → \"{e['word']}\" (should be {gold_tag}, got {pred_tag}) → {e['next_word']}/{e['next_tag']}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python show_errors.py <gold_tag> <pred_tag>")
        print("Example: python show_errors.py NN JJ")
        sys.exit(1)

    gold_tag = sys.argv[1]
    pred_tag = sys.argv[2]

    analyze_specific_error("WSJ_24.pos", "viterbi_dev.pos", gold_tag, pred_tag)