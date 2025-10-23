#!/usr/bin/env python3
import sys
from collections import defaultdict, Counter
import math

def load_pos_file(filename):
    """Load a POS file and return list of (word, tag) pairs."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    data.append(parts)
    return data

def analyze_transition_errors(gold_file, pred_file):
    """Analyze errors in the context of trigram transitions."""

    gold = load_pos_file(gold_file)
    pred = load_pos_file(pred_file)

    # Build confusion matrix with transition context
    # Key: (prev_tag, gold_tag, pred_tag) -> count
    transition_errors = defaultdict(int)

    # Track overall accuracy by tag
    tag_correct = defaultdict(int)
    tag_total = defaultdict(int)

    # Track specific error contexts
    # Key: (gold_tag, pred_tag) -> list of contexts
    error_contexts = defaultdict(list)

    for i in range(len(gold)):
        g_word, g_tag = gold[i]
        p_word, p_tag = pred[i]

        prev_tag = gold[i-1][1] if i > 0 else '<START>'
        next_tag = gold[i+1][1] if i < len(gold)-1 else '<END>'

        tag_total[g_tag] += 1

        if g_tag == p_tag:
            tag_correct[g_tag] += 1
        else:
            # Record the error with context
            transition_errors[(prev_tag, g_tag, p_tag)] += 1
            error_contexts[(g_tag, p_tag)].append({
                'word': g_word,
                'prev': prev_tag,
                'next': next_tag,
                'index': i
            })

    # Calculate tag accuracies
    tag_accuracy = {}
    for tag in tag_total:
        if tag_total[tag] > 0:
            tag_accuracy[tag] = tag_correct[tag] / tag_total[tag]

    # Find the biggest bottlenecks
    print("="*80)
    print("TRANSITION ERROR ANALYSIS - Finding Safe Bias Opportunities")
    print("="*80)

    # Group errors by (gold_tag, pred_tag) pair
    error_pairs = defaultdict(lambda: defaultdict(int))
    for (prev, gold, pred), count in transition_errors.items():
        error_pairs[(gold, pred)][prev] += count

    # Sort by total error count
    sorted_errors = sorted(error_pairs.items(),
                          key=lambda x: sum(x[1].values()),
                          reverse=True)

    print("\nTOP ERROR PATTERNS WITH TRANSITION CONTEXT:")
    print("-"*80)

    opportunities = []

    for (gold_tag, pred_tag), prev_contexts in sorted_errors[:15]:
        total_errors = sum(prev_contexts.values())

        # Find dominant previous context
        sorted_prevs = sorted(prev_contexts.items(), key=lambda x: x[1], reverse=True)
        dominant_prev = sorted_prevs[0][0]
        dominant_count = sorted_prevs[0][1]
        dominant_pct = (dominant_count / total_errors) * 100

        print(f"\n{gold_tag} -> {pred_tag}: {total_errors} errors total")
        print(f"  Tag accuracy: {gold_tag}={tag_accuracy.get(gold_tag, 0):.3f}, "
              f"{pred_tag}={tag_accuracy.get(pred_tag, 0):.3f}")

        # Show top previous contexts
        print(f"  Previous tag distribution:")
        for prev, count in sorted_prevs[:5]:
            pct = (count / total_errors) * 100
            print(f"    After {prev:8}: {count:3} errors ({pct:5.1f}%)")

        # Analyze if this is a safe bias opportunity
        if dominant_pct >= 30 and tag_accuracy.get(gold_tag, 0) < 0.95:
            # Check how often dominant_prev -> gold_tag is correct
            correct_transitions = 0
            total_transitions = 0
            for i in range(1, len(gold)):
                if gold[i-1][1] == dominant_prev and gold[i][1] == gold_tag:
                    total_transitions += 1
                    if pred[i][1] == gold_tag:
                        correct_transitions += 1

            if total_transitions > 0:
                transition_accuracy = correct_transitions / total_transitions
                print(f"  *** OPPORTUNITY: {dominant_pct:.1f}% of errors occur after {dominant_prev}")
                print(f"      Accuracy of {dominant_prev}->{gold_tag}: {transition_accuracy:.3f}")

                if transition_accuracy < 0.9:  # Room for improvement
                    impact = dominant_count * 0.5  # Conservative estimate
                    opportunities.append({
                        'pattern': f"{dominant_prev} -> {gold_tag} (not {pred_tag})",
                        'errors': dominant_count,
                        'impact': impact,
                        'safety': 1 - tag_accuracy.get(pred_tag, 0)  # How safe to penalize pred_tag
                    })

    # Analyze word patterns for top errors
    print("\n" + "="*80)
    print("WORD PATTERN ANALYSIS FOR TOP ERRORS:")
    print("-"*80)

    for (gold_tag, pred_tag), contexts in sorted_errors[:5]:
        total = sum(prev_contexts.values())
        if total < 20:  # Skip small error groups
            continue

        print(f"\n{gold_tag} -> {pred_tag} ({total} errors):")

        # Analyze word patterns
        words = Counter([c['word'].lower() for c in error_contexts[(gold_tag, pred_tag)]])
        suffixes = Counter()
        prefixes = Counter()

        for c in error_contexts[(gold_tag, pred_tag)]:
            word = c['word'].lower()
            if len(word) >= 3:
                suffixes[word[-2:]] += 1
                suffixes[word[-3:]] += 1
            if len(word) >= 4:
                prefixes[word[:3]] += 1

        # Show most common error words
        print("  Most frequent error words:")
        for word, count in words.most_common(10):
            if count >= 2:
                print(f"    '{word}': {count} times")

        # Show suffix patterns if significant
        print("  Common suffixes:")
        for suffix, count in suffixes.most_common(5):
            if count >= 5:
                pct = (count / total) * 100
                print(f"    -*{suffix}: {count} errors ({pct:.1f}%)")

    # Recommend targeted fixes
    print("\n" + "="*80)
    print("RECOMMENDED TARGETED FIXES (sorted by impact):")
    print("-"*80)

    opportunities.sort(key=lambda x: x['impact'], reverse=True)

    for i, opp in enumerate(opportunities[:10], 1):
        print(f"\n{i}. Bias {opp['pattern']}")
        print(f"   Potential impact: ~{opp['impact']:.0f} errors fixed")
        print(f"   Safety score: {opp['safety']:.3f} (higher = safer to implement)")

    return opportunities

def generate_fix_code(opportunities):
    """Generate actual code fixes for the top opportunities."""
    print("\n" + "="*80)
    print("SUGGESTED CODE IMPLEMENTATIONS:")
    print("-"*80)

    for opp in opportunities[:3]:
        pattern = opp['pattern']
        print(f"\n# Fix for {pattern}:")
        print("```python")

        # Parse the pattern
        parts = pattern.split(" -> ")
        prev_tag = parts[0]
        rest = parts[1].split(" (not ")
        gold_tag = rest[0]
        pred_tag = rest[1].rstrip(")")

        print(f"if prev == '{prev_tag}':")
        print(f"    if curr == '{gold_tag}':")
        print(f"        bonus += math.log(1.3)  # Boost {gold_tag} after {prev_tag}")
        print(f"    elif curr == '{pred_tag}':")
        print(f"        bonus += math.log(0.7)  # Penalty for {pred_tag} after {prev_tag}")
        print("```")

if __name__ == "__main__":
    print("Analyzing error patterns for targeted improvements...")
    opportunities = analyze_transition_errors("WSJ_24.pos", "viterbi_dev.pos")
    generate_fix_code(opportunities)