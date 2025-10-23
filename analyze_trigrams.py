#!/usr/bin/env python3
"""Analyze trigram patterns for error cases."""

from collections import defaultdict, Counter

def analyze_trigram_errors():
    """Check what trigram patterns are causing NN->JJ errors."""

    # Load gold and predicted files
    gold = []
    pred = []

    with open("WSJ_24.pos", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    gold.append(parts)

    with open("viterbi_dev.pos", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    pred.append(parts)

    # Find NN->JJ errors with trigram context
    nn_jj_trigrams = []

    for i in range(2, len(gold)):  # Start at 2 to have prev_prev
        g_word, g_tag = gold[i]
        p_word, p_tag = pred[i]

        if g_tag == "NN" and p_tag == "JJ":
            prev_prev_tag = gold[i-2][1] if i >= 2 else '<START>'
            prev_tag = gold[i-1][1]
            next_tag = gold[i+1][1] if i < len(gold)-1 else '<END>'

            nn_jj_trigrams.append({
                'word': g_word,
                'trigram': (prev_prev_tag, prev_tag, g_tag),
                'predicted_trigram': (prev_prev_tag, prev_tag, p_tag),
                'next': next_tag
            })

    print("="*70)
    print("NN->JJ ERRORS: TRIGRAM ANALYSIS")
    print("="*70)
    print(f"Total NN->JJ errors: {len(nn_jj_trigrams)}\n")

    # Count trigram patterns
    trigram_counts = Counter()
    for error in nn_jj_trigrams:
        trigram_counts[error['trigram'][:2]] += 1  # Count (prev_prev, prev) pairs

    print("Most common (prev_prev, prev) patterns leading to NN->JJ errors:")
    print("-"*50)
    for (pp, p), count in trigram_counts.most_common(15):
        examples = [e['word'] for e in nn_jj_trigrams
                   if e['trigram'][:2] == (pp, p)][:3]
        print(f"  {pp:8} -> {p:8} -> NN:  {count:3} errors")
        print(f"    Examples: {', '.join(examples)}")

    # Look at specific DT patterns since those are most common
    print("\n" + "="*70)
    print("DETAILED DT PATTERNS (28 errors after DT):")
    print("-"*50)

    dt_errors = [e for e in nn_jj_trigrams if e['trigram'][1] == 'DT']
    dt_trigram_patterns = Counter()

    for error in dt_errors:
        full_pattern = f"{error['trigram'][0]} -> DT -> NN -> {error['next']}"
        dt_trigram_patterns[full_pattern] += 1

    print("Full patterns (prev_prev -> DT -> NN -> next):")
    for pattern, count in dt_trigram_patterns.most_common(10):
        print(f"  {pattern}: {count} errors")

    # Check what words appear after DT that confuse the model
    print("\nWords after DT that are confused:")
    dt_words = Counter([e['word'] for e in dt_errors])
    for word, count in dt_words.most_common(15):
        print(f"  '{word}': {count} times")

    # Check for compound noun patterns
    print("\n" + "="*70)
    print("COMPOUND NOUN PATTERNS:")
    print("-"*50)

    # Check if next word is also NN (suggesting compound noun)
    compound_patterns = []
    for error in nn_jj_trigrams:
        if error['next'] in ['NN', 'NNS', 'NNP', 'NNPS']:
            compound_patterns.append(error)

    print(f"Errors where next word is also a noun: {len(compound_patterns)}/{len(nn_jj_trigrams)}")
    print("Examples of likely compound nouns being split:")
    for e in compound_patterns[:10]:
        i = gold.index([e['word'], 'NN'])
        next_word = gold[i+1][0] if i < len(gold)-1 else ''
        print(f"  '{e['word']} {next_word}' - {e['word']} tagged as JJ instead of NN")

if __name__ == "__main__":
    analyze_trigram_errors()