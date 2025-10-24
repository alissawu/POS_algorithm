#!/usr/bin/env python3
"""Cluster analysis to find high-precision targetable error patterns"""

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

# Collect errors with detailed context
errors = []

for i in range(len(gold_lines)):
    if not gold_lines[i] or not pred_lines[i]:
        continue

    try:
        gold_word, gold_tag = gold_lines[i].split('\t')
        pred_word, pred_tag = pred_lines[i].split('\t')

        if gold_tag != pred_tag:
            prev_word = gold_lines[i-1].split('\t')[0] if i > 0 and gold_lines[i-1] else "<START>"
            prev_tag = gold_lines[i-1].split('\t')[1] if i > 0 and gold_lines[i-1] else "<START>"
            next_word = gold_lines[i+1].split('\t')[0] if i+1 < len(gold_lines) and gold_lines[i+1] else "<END>"
            next_tag = gold_lines[i+1].split('\t')[1] if i+1 < len(gold_lines) and gold_lines[i+1] else "<END>"

            errors.append({
                'word': gold_word,
                'gold': gold_tag,
                'pred': pred_tag,
                'prev_word': prev_word.lower(),
                'prev_tag': prev_tag,
                'next_word': next_word.lower(),
                'next_tag': next_tag,
            })
    except:
        pass

print("=" * 80)
print("CLUSTER ANALYSIS: Finding High-Precision Targetable Patterns")
print("=" * 80)

# For each major error type, find strong patterns
for error_type in [('NN', 'JJ'), ('IN', 'RB'), ('VBN', 'JJ'), ('VBD', 'VBN'), ('VBN', 'VBD')]:
    gold_tag, pred_tag = error_type
    subset = [e for e in errors if e['gold'] == gold_tag and e['pred'] == pred_tag]

    if len(subset) < 20:
        continue

    print(f"\n{'='*80}")
    print(f"{gold_tag} -> {pred_tag}: {len(subset)} total errors")
    print(f"{'='*80}")

    # Analyze different clustering dimensions

    # 1. By next word (lexical)
    print(f"\n--- LEXICAL CLUSTERS (by next word) ---")
    next_word_groups = defaultdict(list)
    for e in subset:
        next_word_groups[e['next_word']].append(e)

    # Find high-frequency next words
    high_freq = [(word, errors) for word, errors in next_word_groups.items() if len(errors) >= 3]
    high_freq.sort(key=lambda x: len(x[1]), reverse=True)

    total_in_clusters = sum(len(errors) for _, errors in high_freq)
    print(f"High-frequency next words (>=3 occurrences): {len(high_freq)} clusters covering {total_in_clusters}/{len(subset)} errors ({total_in_clusters/len(subset)*100:.1f}%)")

    for word, error_list in high_freq[:10]:
        print(f"  next='{word}': {len(error_list)} errors")
        # Show a few examples
        for e in error_list[:3]:
            print(f"    {e['prev_word']}/{e['prev_tag']} *{e['word']}/{e['gold']}* {e['next_word']}/{e['next_tag']}")

    # 2. By structural pattern (prev_tag + next_tag)
    print(f"\n--- STRUCTURAL CLUSTERS (by prev_tag + next_tag) ---")
    structure_groups = defaultdict(list)
    for e in subset:
        key = (e['prev_tag'], e['next_tag'])
        structure_groups[key].append(e)

    high_freq_struct = [(pattern, errors) for pattern, errors in structure_groups.items() if len(errors) >= 5]
    high_freq_struct.sort(key=lambda x: len(x[1]), reverse=True)

    total_in_struct = sum(len(errors) for _, errors in high_freq_struct)
    print(f"High-frequency patterns (>=5 occurrences): {len(high_freq_struct)} patterns covering {total_in_struct}/{len(subset)} errors ({total_in_struct/len(subset)*100:.1f}%)")

    for (prev, nxt), error_list in high_freq_struct[:8]:
        print(f"  [{prev}] WORD [{nxt}]: {len(error_list)} errors")
        for e in error_list[:3]:
            print(f"    {e['prev_word']}/{e['prev_tag']} *{e['word']}/{e['gold']}* {e['next_word']}/{e['next_tag']}")

    # 3. Combined: specific next words within structural patterns
    print(f"\n--- COMBINED CLUSTERS (structure + next word) ---")
    combined_groups = defaultdict(list)
    for e in subset:
        key = (e['prev_tag'], e['next_tag'], e['next_word'])
        combined_groups[key].append(e)

    high_freq_combined = [(pattern, errors) for pattern, errors in combined_groups.items() if len(errors) >= 2]
    high_freq_combined.sort(key=lambda x: len(x[1]), reverse=True)

    total_combined = sum(len(errors) for _, errors in high_freq_combined)
    print(f"High-precision patterns (>=2 occurrences): {len(high_freq_combined)} patterns covering {total_combined}/{len(subset)} errors ({total_combined/len(subset)*100:.1f}%)")

    for (prev, nxt_tag, nxt_word), error_list in high_freq_combined[:10]:
        if len(error_list) >= 3:  # Only show very strong patterns
            print(f"  [{prev}] WORD [{nxt_tag}='{nxt_word}']: {len(error_list)} errors")
            for e in error_list[:2]:
                print(f"    {e['prev_word']}/{e['prev_tag']} *{e['word']}/{e['gold']}* {e['next_word']}/{e['next_tag']}")

    # 4. Word-level patterns (specific error words)
    print(f"\n--- WORD-LEVEL CLUSTERS (repeated error words) ---")
    word_groups = defaultdict(list)
    for e in subset:
        word_groups[e['word'].lower()].append(e)

    repeated_words = [(word, errors) for word, errors in word_groups.items() if len(errors) >= 2]
    repeated_words.sort(key=lambda x: len(x[1]), reverse=True)

    total_repeated = sum(len(errors) for _, errors in repeated_words)
    print(f"Repeated error words (>=2 times): {len(repeated_words)} words covering {total_repeated}/{len(subset)} errors ({total_repeated/len(subset)*100:.1f}%)")

    for word, error_list in repeated_words[:8]:
        if len(error_list) >= 3:
            print(f"  word='{word}': {len(error_list)} errors - POTENTIAL LEXICON FIX")

print("\n" + "=" * 80)
print("SUMMARY: Most Targetable Error Clusters")
print("=" * 80)

# Find the single best targetable pattern for each error type
for error_type in [('NN', 'JJ'), ('IN', 'RB'), ('VBN', 'JJ'), ('VBD', 'VBN'), ('VBN', 'VBD')]:
    gold_tag, pred_tag = error_type
    subset = [e for e in errors if e['gold'] == gold_tag and e['pred'] == pred_tag]

    if len(subset) < 20:
        continue

    # Find best structural pattern
    structure_groups = defaultdict(list)
    for e in subset:
        key = (e['prev_tag'], e['next_tag'])
        structure_groups[key].append(e)

    best_pattern, best_errors = max(structure_groups.items(), key=lambda x: len(x[1]))

    print(f"\n{gold_tag} -> {pred_tag} ({len(subset)} errors):")
    print(f"  Best pattern: [{best_pattern[0]}] WORD [{best_pattern[1]}] - {len(best_errors)} errors ({len(best_errors)/len(subset)*100:.1f}%)")
    print(f"  Rule potential: if prev_tag={best_pattern[0]} and next_tag={best_pattern[1]} and pred={pred_tag}, boost gold={gold_tag}")
