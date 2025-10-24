#!/usr/bin/env python3
"""Detailed error analysis to find bottlenecks"""

from collections import Counter, defaultdict

gold_file = 'WSJ_24.pos'
pred_file = 'viterbi_dev.pos'

# Read files
gold_lines = []
pred_lines = []

with open(gold_file) as f:
    for line in f:
        gold_lines.append(line.strip())

with open(pred_file) as f:
    for line in f:
        pred_lines.append(line.strip())

# Collect errors with full context
errors = []
error_counts = Counter()

for i in range(len(gold_lines)):
    if not gold_lines[i] or not pred_lines[i]:
        continue

    try:
        gold_word, gold_tag = gold_lines[i].split('\t')
        pred_word, pred_tag = pred_lines[i].split('\t')

        if gold_tag != pred_tag:
            # Get context
            prev_word = gold_lines[i-1].split('\t')[0] if i > 0 and gold_lines[i-1] else "<START>"
            prev_tag = gold_lines[i-1].split('\t')[1] if i > 0 and gold_lines[i-1] else "<START>"
            next_word = gold_lines[i+1].split('\t')[0] if i+1 < len(gold_lines) and gold_lines[i+1] else "<END>"
            next_tag = gold_lines[i+1].split('\t')[1] if i+1 < len(gold_lines) and gold_lines[i+1] else "<END>"

            errors.append({
                'word': gold_word,
                'gold': gold_tag,
                'pred': pred_tag,
                'prev_word': prev_word,
                'prev_tag': prev_tag,
                'next_word': next_word,
                'next_tag': next_tag,
                'index': i
            })
            error_counts[(gold_tag, pred_tag)] += 1
    except:
        pass

print("=" * 80)
print("TOP ERROR CATEGORIES (sorted by frequency)")
print("=" * 80)

total_errors = sum(error_counts.values())
print(f"\nTotal errors: {total_errors}")
print(f"Total tokens: {len([l for l in gold_lines if l])}")
print(f"Accuracy: {(len([l for l in gold_lines if l]) - total_errors) / len([l for l in gold_lines if l]) * 100:.2f}%\n")

cumulative = 0
for i, ((gold, pred), count) in enumerate(error_counts.most_common(30), 1):
    cumulative += count
    pct = count / total_errors * 100
    cum_pct = cumulative / total_errors * 100
    print(f"{i:2d}. {gold:6s} -> {pred:6s} : {count:4d} ({pct:5.1f}%)  [cumulative: {cum_pct:5.1f}%]")

# Detailed analysis for top 5 errors
print("\n" + "=" * 80)
print("DETAILED CONTEXT ANALYSIS FOR TOP ERRORS")
print("=" * 80)

for (gold_tag, pred_tag), count in error_counts.most_common(5):
    print(f"\n{'=' * 80}")
    print(f"{gold_tag} -> {pred_tag} (should be {gold_tag}): {count} errors")
    print(f"{'=' * 80}")

    # Get examples of this error type
    examples = [e for e in errors if e['gold'] == gold_tag and e['pred'] == pred_tag]

    # Analyze patterns
    prev_tag_dist = Counter(e['prev_tag'] for e in examples)
    next_tag_dist = Counter(e['next_tag'] for e in examples)
    next_word_dist = Counter(e['next_word'].lower() for e in examples)

    print(f"\nMost common PREVIOUS tags:")
    for tag, cnt in prev_tag_dist.most_common(8):
        print(f"  {tag:8s}: {cnt:3d} ({cnt/len(examples)*100:.1f}%)")

    print(f"\nMost common NEXT tags:")
    for tag, cnt in next_tag_dist.most_common(8):
        print(f"  {tag:8s}: {cnt:3d} ({cnt/len(examples)*100:.1f}%)")

    print(f"\nMost common NEXT words:")
    for word, cnt in next_word_dist.most_common(10):
        print(f"  {word:15s}: {cnt:3d} ({cnt/len(examples)*100:.1f}%)")

    print(f"\nFirst 10 examples with context:")
    for e in examples[:10]:
        print(f"  {e['prev_word']}/{e['prev_tag']:6s}  *{e['word']}/{e['gold']:6s}*  {e['next_word']}/{e['next_tag']:6s}")

print("\n" + "=" * 80)
print("POTENTIAL HIGH-IMPACT FIXES")
print("=" * 80)

# Calculate potential impact of fixing errors
print("\nIf we fix top N error categories:")
for n in [1, 2, 3, 5, 10]:
    top_n_errors = sum(count for (_, _), count in error_counts.most_common(n))
    potential_accuracy = (len([l for l in gold_lines if l]) - total_errors + top_n_errors) / len([l for l in gold_lines if l]) * 100
    print(f"  Top {n:2d} categories ({top_n_errors:4d} errors): potential accuracy = {potential_accuracy:.2f}%")
