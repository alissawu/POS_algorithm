#!/usr/bin/env python3
"""Analyze IN/RB confusions in detail with surrounding context."""

def analyze_in_rb_errors():
    # Load files
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

    print("="*80)
    print("IN -> RB ERRORS (Model thinks it's RB but it's actually IN)")
    print("="*80)

    in_to_rb_errors = []
    for i in range(len(gold)):
        g_word, g_tag = gold[i]
        p_word, p_tag = pred[i]

        if g_tag == "IN" and p_tag == "RB":
            # Get context window (2 before, 2 after)
            context = []
            for j in range(max(0, i-2), min(len(gold), i+3)):
                if j == i:
                    context.append(f"[{gold[j][0]}/IN->RB]")
                else:
                    context.append(f"{gold[j][0]}/{gold[j][1]}")

            in_to_rb_errors.append({
                'word': g_word,
                'context': " ".join(context),
                'prev': gold[i-1][1] if i > 0 else '<START>',
                'next': gold[i+1][1] if i < len(gold)-1 else '<END>'
            })

    print(f"\nTotal IN->RB errors: {len(in_to_rb_errors)}\n")

    # Group by word
    from collections import Counter
    word_counts = Counter([e['word'] for e in in_to_rb_errors])

    print("Most confused words:")
    for word, count in word_counts.most_common(10):
        print(f"  '{word}': {count} times")

    print("\n" + "-"*80)
    print("ALL IN->RB ERRORS WITH CONTEXT:")
    print("-"*80)

    for i, error in enumerate(in_to_rb_errors, 1):
        print(f"\n{i}. Word: '{error['word']}' (prev: {error['prev']}, next: {error['next']})")
        print(f"   {error['context']}")

    print("\n" + "="*80)
    print("RB -> IN ERRORS (Model thinks it's IN but it's actually RB)")
    print("="*80)

    rb_to_in_errors = []
    for i in range(len(gold)):
        g_word, g_tag = gold[i]
        p_word, p_tag = pred[i]

        if g_tag == "RB" and p_tag == "IN":
            # Get context window
            context = []
            for j in range(max(0, i-2), min(len(gold), i+3)):
                if j == i:
                    context.append(f"[{gold[j][0]}/RB->IN]")
                else:
                    context.append(f"{gold[j][0]}/{gold[j][1]}")

            rb_to_in_errors.append({
                'word': g_word,
                'context': " ".join(context),
                'prev': gold[i-1][1] if i > 0 else '<START>',
                'next': gold[i+1][1] if i < len(gold)-1 else '<END>'
            })

    print(f"\nTotal RB->IN errors: {len(rb_to_in_errors)}\n")

    # Group by word
    word_counts_rb = Counter([e['word'] for e in rb_to_in_errors])

    print("Most confused words:")
    for word, count in word_counts_rb.most_common(10):
        print(f"  '{word}': {count} times")

    print("\n" + "-"*80)
    print("ALL RB->IN ERRORS WITH CONTEXT:")
    print("-"*80)

    for i, error in enumerate(rb_to_in_errors, 1):
        print(f"\n{i}. Word: '{error['word']}' (prev: {error['prev']}, next: {error['next']})")
        print(f"   {error['context']}")

    # Analyze patterns
    print("\n" + "="*80)
    print("PATTERN ANALYSIS:")
    print("="*80)

    # Check what follows IN->RB errors
    print("\nWhat comes AFTER words that should be IN but tagged as RB:")
    next_tags_in = Counter([e['next'] for e in in_to_rb_errors])
    for tag, count in next_tags_in.most_common(10):
        pct = (count / len(in_to_rb_errors)) * 100
        print(f"  {tag:8}: {count:3} ({pct:5.1f}%)")
        # Show examples
        examples = [e['word'] for e in in_to_rb_errors if e['next'] == tag][:3]
        print(f"    Examples: {', '.join(examples)}")

    print("\nWhat comes BEFORE words that should be IN but tagged as RB:")
    prev_tags_in = Counter([e['prev'] for e in in_to_rb_errors])
    for tag, count in prev_tags_in.most_common(10):
        pct = (count / len(in_to_rb_errors)) * 100
        print(f"  {tag:8}: {count:3} ({pct:5.1f}%)")
        examples = [e['word'] for e in in_to_rb_errors if e['prev'] == tag][:3]
        print(f"    Examples: {', '.join(examples)}")

if __name__ == "__main__":
    analyze_in_rb_errors()