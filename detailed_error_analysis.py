#!/usr/bin/env python3
from collections import Counter, defaultdict
from pathlib import Path
import sys

def read_tagged_corpus(path):
    """Read a tagged corpus file into list of (word, tag) pairs."""
    sentences = []
    current = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            current.append(tuple(parts))
    if current:
        sentences.append(current)
    return sentences

def analyze_errors_detailed(gold_file, pred_file):
    """Perform comprehensive error analysis."""
    gold_sentences = read_tagged_corpus(gold_file)
    pred_sentences = read_tagged_corpus(pred_file)

    # Flatten for easier processing
    gold_flat = [(w, t) for sent in gold_sentences for w, t in sent]
    pred_flat = [(w, t) for sent in pred_sentences for w, t in sent]

    # Track all errors with full context
    all_errors = []
    confusion_matrix = defaultdict(Counter)

    # Collect all errors with context
    for i, ((gold_word, gold_tag), (pred_word, pred_tag)) in enumerate(zip(gold_flat, pred_flat)):
        assert gold_word == pred_word, f"Word mismatch at position {i}: {gold_word} != {pred_word}"

        if gold_tag != pred_tag:
            prev_word = gold_flat[i-1][0] if i > 0 else "<START>"
            prev_tag = gold_flat[i-1][1] if i > 0 else "<START>"
            next_word = gold_flat[i+1][0] if i < len(gold_flat)-1 else "<END>"
            next_tag = gold_flat[i+1][1] if i < len(gold_flat)-1 else "<END>"

            error_entry = {
                'position': i,
                'word': gold_word,
                'gold_tag': gold_tag,
                'pred_tag': pred_tag,
                'prev_word': prev_word,
                'prev_tag': prev_tag,
                'next_word': next_word,
                'next_tag': next_tag,
                'context': f"{prev_word}/{prev_tag} {gold_word}/{gold_tag}->{pred_tag} {next_word}/{next_tag}"
            }
            all_errors.append(error_entry)
            confusion_matrix[gold_tag][pred_tag] += 1

    # Print summary statistics
    total_tokens = len(gold_flat)
    total_errors = len(all_errors)
    accuracy = 100.0 * (total_tokens - total_errors) / total_tokens

    print("="*80)
    print(f"SUMMARY STATISTICS")
    print("="*80)
    print(f"Total tokens: {total_tokens}")
    print(f"Total errors: {total_errors}")
    print(f"Accuracy: {accuracy:.3f}%")
    print()

    # Print confusion matrix sorted by frequency
    print("="*80)
    print("TOP 30 CONFUSION PAIRS (Gold -> Pred: Count)")
    print("="*80)
    all_confusions = []
    for gold_tag, pred_counts in confusion_matrix.items():
        for pred_tag, count in pred_counts.items():
            all_confusions.append((count, gold_tag, pred_tag))

    for count, gold_tag, pred_tag in sorted(all_confusions, reverse=True)[:30]:
        # Get examples for this confusion
        examples = [e['word'] for e in all_errors
                   if e['gold_tag'] == gold_tag and e['pred_tag'] == pred_tag][:5]
        print(f"{gold_tag:6} -> {pred_tag:6}: {count:4} | Examples: {', '.join(examples)}")
    print()

    # Analyze patterns for major confusions
    print("="*80)
    print("DETAILED ANALYSIS OF TOP CONFUSIONS")
    print("="*80)

    # For each major confusion, show context patterns
    for count, gold_tag, pred_tag in sorted(all_confusions, reverse=True)[:10]:
        print(f"\n{gold_tag} -> {pred_tag} ({count} errors)")
        print("-" * 40)

        # Get all errors of this type
        specific_errors = [e for e in all_errors
                          if e['gold_tag'] == gold_tag and e['pred_tag'] == pred_tag]

        # Analyze previous tags
        prev_tags = Counter([e['prev_tag'] for e in specific_errors])
        print(f"  Most common previous tags:")
        for tag, cnt in prev_tags.most_common(5):
            pct = 100.0 * cnt / len(specific_errors)
            print(f"    {tag:10}: {cnt:3} ({pct:5.1f}%)")

        # Show sample contexts
        print(f"  Sample contexts:")
        for e in specific_errors[:3]:
            print(f"    {e['context']}")

    # Analyze errors by word characteristics
    print("\n" + "="*80)
    print("ERROR ANALYSIS BY WORD CHARACTERISTICS")
    print("="*80)

    # Capitalization patterns
    cap_errors = defaultdict(list)
    for e in all_errors:
        word = e['word']
        if word[0].isupper() and not word.isupper():
            cap_errors['InitCap'].append(e)
        elif word.isupper() and len(word) > 1:
            cap_errors['AllCap'].append(e)
        else:
            cap_errors['Lowercase'].append(e)

    print("\nErrors by capitalization:")
    for cap_type, errors in cap_errors.items():
        print(f"  {cap_type:10}: {len(errors):4} errors")
        # Show most common confusions for this type
        type_confusions = Counter([(e['gold_tag'], e['pred_tag']) for e in errors])
        for (g, p), cnt in type_confusions.most_common(3):
            print(f"    {g}->{p}: {cnt}")

    # Suffix patterns
    suffix_errors = defaultdict(list)
    suffixes = ['ed', 'ing', 's', 'ly', 'er', 'est', 'tion', 'ment', 'ness', 'able', 'ful']
    for e in all_errors:
        word_lower = e['word'].lower()
        for suffix in suffixes:
            if word_lower.endswith(suffix):
                suffix_errors[suffix].append(e)
                break

    print("\nErrors by suffix:")
    for suffix, errors in sorted(suffix_errors.items(), key=lambda x: len(x[1]), reverse=True):
        if len(errors) > 10:  # Only show significant suffixes
            print(f"  -{suffix:6}: {len(errors):4} errors")
            type_confusions = Counter([(e['gold_tag'], e['pred_tag']) for e in errors])
            for (g, p), cnt in type_confusions.most_common(3):
                print(f"    {g}->{p}: {cnt}")

    # Position in sentence
    print("\n" + "="*80)
    print("ERROR ANALYSIS BY POSITION")
    print("="*80)

    sentence_initial_errors = [e for e in all_errors if e['prev_tag'] == '<START>']
    print(f"\nSentence-initial errors: {len(sentence_initial_errors)}")
    si_confusions = Counter([(e['gold_tag'], e['pred_tag']) for e in sentence_initial_errors])
    for (g, p), cnt in si_confusions.most_common(10):
        examples = [e['word'] for e in sentence_initial_errors
                   if e['gold_tag'] == g and e['pred_tag'] == p][:3]
        print(f"  {g:6} -> {p:6}: {cnt:3} | {', '.join(examples)}")

    # After specific tags
    print("\n" + "="*80)
    print("ERRORS AFTER SPECIFIC TAGS")
    print("="*80)

    important_prev_tags = ['DT', 'VB', 'VBZ', 'VBP', 'VBD', 'IN', 'NN', 'JJ']
    for prev_tag in important_prev_tags:
        errors_after = [e for e in all_errors if e['prev_tag'] == prev_tag]
        if len(errors_after) > 20:  # Only show significant patterns
            print(f"\nAfter {prev_tag}: {len(errors_after)} errors")
            after_confusions = Counter([(e['gold_tag'], e['pred_tag']) for e in errors_after])
            for (g, p), cnt in after_confusions.most_common(5):
                pct = 100.0 * cnt / len(errors_after)
                examples = [e['word'] for e in errors_after
                           if e['gold_tag'] == g and e['pred_tag'] == p][:3]
                print(f"  {g:6} -> {p:6}: {cnt:3} ({pct:5.1f}%) | {', '.join(examples)}")

    # Show all errors for inspection if requested
    if '--all' in sys.argv:
        print("\n" + "="*80)
        print("ALL ERRORS (first 100)")
        print("="*80)
        for i, e in enumerate(all_errors[:100]):
            print(f"{i+1:4}. {e['context']}")

    return all_errors, confusion_matrix

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python detailed_error_analysis.py gold_file predicted_file [--all]")
        print("  Add --all to show all individual errors")
        sys.exit(1)

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    analyze_errors_detailed(gold_file, pred_file)