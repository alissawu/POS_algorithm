#!/usr/bin/env python3
from collections import Counter, defaultdict
from pathlib import Path
import sys
import re

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

def analyze_errors(gold_file, pred_file):
    """Perform detailed error analysis."""
    gold_sentences = read_tagged_corpus(gold_file)
    pred_sentences = read_tagged_corpus(pred_file)

    # Flatten for easier processing
    gold_flat = [(w, t) for sent in gold_sentences for w, t in sent]
    pred_flat = [(w, t) for sent in pred_sentences for w, t in sent]

    # Track different types of errors
    confusion_matrix = defaultdict(Counter)  # gold_tag -> pred_tag -> count
    error_words = defaultdict(list)  # (gold_tag, pred_tag) -> list of words
    error_contexts = []  # list of (prev_word, word, next_word, gold_tag, pred_tag)

    # Word frequency analysis
    word_freq = Counter(w for w, _ in gold_flat)
    error_by_freq = defaultdict(int)  # freq_bucket -> error_count
    total_by_freq = defaultdict(int)   # freq_bucket -> total_count

    # OOV analysis
    training_vocab = set()
    train_file = Path("WSJ_02-21.pos")
    if train_file.exists():
        train_sentences = read_tagged_corpus(train_file)
        training_vocab = {w for sent in train_sentences for w, _ in sent}

    oov_errors = []
    oov_confusion = defaultdict(Counter)

    # Morphological feature errors
    suffix_errors = defaultdict(Counter)
    capitalized_errors = Counter()

    total_correct = 0
    total_tokens = 0

    for i, ((gold_word, gold_tag), (pred_word, pred_tag)) in enumerate(zip(gold_flat, pred_flat)):
        assert gold_word == pred_word, f"Word mismatch at position {i}: {gold_word} != {pred_word}"

        total_tokens += 1

        if gold_tag == pred_tag:
            total_correct += 1
        else:
            # Record confusion
            confusion_matrix[gold_tag][pred_tag] += 1
            error_words[(gold_tag, pred_tag)].append(gold_word)

            # Record context
            prev_word = gold_flat[i-1][0] if i > 0 else "<START>"
            next_word = gold_flat[i+1][0] if i < len(gold_flat)-1 else "<END>"
            error_contexts.append((prev_word, gold_word, next_word, gold_tag, pred_tag))

            # Frequency-based analysis
            freq = word_freq[gold_word]
            if freq == 1:
                bucket = "hapax"
            elif freq <= 5:
                bucket = "rare"
            elif freq <= 20:
                bucket = "medium"
            else:
                bucket = "common"
            error_by_freq[bucket] += 1

            # OOV analysis
            if training_vocab and gold_word not in training_vocab:
                oov_errors.append((gold_word, gold_tag, pred_tag))
                oov_confusion[gold_tag][pred_tag] += 1

            # Morphological analysis
            lower = gold_word.lower()
            # Suffix analysis
            for suffix in ['ing', 'ed', 's', 'ly', 'er', 'est', 'tion', 'ment', 'ness', 'able']:
                if lower.endswith(suffix):
                    suffix_errors[suffix][(gold_tag, pred_tag)] += 1

            # Capitalization
            if gold_word and gold_word[0].isupper():
                capitalized_errors[(gold_tag, pred_tag)] += 1

    # Count totals by frequency
    for word, _ in gold_flat:
        freq = word_freq[word]
        if freq == 1:
            bucket = "hapax"
        elif freq <= 5:
            bucket = "rare"
        elif freq <= 20:
            bucket = "medium"
        else:
            bucket = "common"
        total_by_freq[bucket] += 1

    # Print analysis results
    accuracy = 100.0 * total_correct / total_tokens
    print(f"Overall Accuracy: {accuracy:.2f}% ({total_correct}/{total_tokens})")
    print("\n" + "="*60)

    # Top confusion pairs
    print("\nTop 20 Confusion Pairs (Gold -> Pred: Count):")
    all_confusions = []
    for gold_tag, pred_counts in confusion_matrix.items():
        for pred_tag, count in pred_counts.items():
            all_confusions.append((count, gold_tag, pred_tag))

    for count, gold_tag, pred_tag in sorted(all_confusions, reverse=True)[:20]:
        examples = error_words[(gold_tag, pred_tag)][:5]
        print(f"  {gold_tag:6} -> {pred_tag:6}: {count:4} | Examples: {', '.join(examples)}")

    print("\n" + "="*60)
    print("\nError Rate by Word Frequency:")
    for bucket in ["hapax", "rare", "medium", "common"]:
        if total_by_freq[bucket] > 0:
            error_rate = 100.0 * error_by_freq[bucket] / total_by_freq[bucket]
            print(f"  {bucket:10}: {error_rate:5.2f}% ({error_by_freq[bucket]}/{total_by_freq[bucket]})")

    print("\n" + "="*60)
    print("\nOOV Error Analysis:")
    print(f"Total OOV errors: {len(oov_errors)}")
    if oov_errors:
        print("\nTop OOV Confusions:")
        oov_conf_list = []
        for gold_tag, pred_counts in oov_confusion.items():
            for pred_tag, count in pred_counts.items():
                oov_conf_list.append((count, gold_tag, pred_tag))

        for count, gold_tag, pred_tag in sorted(oov_conf_list, reverse=True)[:10]:
            oov_examples = [w for w, g, p in oov_errors if g == gold_tag and p == pred_tag][:3]
            print(f"  {gold_tag:6} -> {pred_tag:6}: {count:3} | Examples: {', '.join(oov_examples)}")

    print("\n" + "="*60)
    print("\nSuffix-based Errors (Top patterns per suffix):")
    for suffix in ['ing', 'ed', 's', 'ly', 'er']:
        if suffix_errors[suffix]:
            print(f"\nWords ending in '-{suffix}':")
            top_errors = suffix_errors[suffix].most_common(5)
            for (gold_tag, pred_tag), count in top_errors:
                examples = [w for w in error_words[(gold_tag, pred_tag)] if w.lower().endswith(suffix)][:3]
                if examples:
                    print(f"  {gold_tag:6} -> {pred_tag:6}: {count:3} | {', '.join(examples)}")

    print("\n" + "="*60)
    print("\nCapitalized Word Errors (Top 10):")
    cap_errors = capitalized_errors.most_common(10)
    for (gold_tag, pred_tag), count in cap_errors:
        examples = [w for w in error_words[(gold_tag, pred_tag)] if w and w[0].isupper()][:3]
        print(f"  {gold_tag:6} -> {pred_tag:6}: {count:3} | {', '.join(examples)}")

    # Analyze specific problematic patterns
    print("\n" + "="*60)
    print("\nSpecific Pattern Analysis:")

    # VBN vs VBD confusion
    vbn_vbd_errors = confusion_matrix.get("VBN", {}).get("VBD", 0)
    vbd_vbn_errors = confusion_matrix.get("VBD", {}).get("VBN", 0)
    print(f"\nVBN/VBD confusion: {vbn_vbd_errors + vbd_vbn_errors} errors")
    if vbn_vbd_errors > 0:
        examples = error_words[("VBN", "VBD")][:5]
        print(f"  VBN -> VBD: {vbn_vbd_errors} | {', '.join(examples)}")
    if vbd_vbn_errors > 0:
        examples = error_words[("VBD", "VBN")][:5]
        print(f"  VBD -> VBN: {vbd_vbn_errors} | {', '.join(examples)}")

    # NN vs NNP confusion
    nn_nnp_errors = confusion_matrix.get("NN", {}).get("NNP", 0)
    nnp_nn_errors = confusion_matrix.get("NNP", {}).get("NN", 0)
    print(f"\nNN/NNP confusion: {nn_nnp_errors + nnp_nn_errors} errors")
    if nn_nnp_errors > 0:
        examples = error_words[("NN", "NNP")][:5]
        print(f"  NN -> NNP: {nn_nnp_errors} | {', '.join(examples)}")
    if nnp_nn_errors > 0:
        examples = error_words[("NNP", "NN")][:5]
        print(f"  NNP -> NN: {nnp_nn_errors} | {', '.join(examples)}")

    # JJ vs NN confusion
    jj_nn_errors = confusion_matrix.get("JJ", {}).get("NN", 0)
    nn_jj_errors = confusion_matrix.get("NN", {}).get("JJ", 0)
    print(f"\nJJ/NN confusion: {jj_nn_errors + nn_jj_errors} errors")

    # IN vs RP confusion
    in_rp_errors = confusion_matrix.get("IN", {}).get("RP", 0)
    rp_in_errors = confusion_matrix.get("RP", {}).get("IN", 0)
    print(f"\nIN/RP (particle) confusion: {in_rp_errors + rp_in_errors} errors")
    if in_rp_errors > 0:
        examples = error_words[("IN", "RP")][:5]
        print(f"  IN -> RP: {in_rp_errors} | {', '.join(examples)}")
    if rp_in_errors > 0:
        examples = error_words[("RP", "IN")][:5]
        print(f"  RP -> IN: {rp_in_errors} | {', '.join(examples)}")

    return confusion_matrix, error_words, accuracy

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python error_analysis.py gold_file predicted_file")
        sys.exit(1)

    gold_file = sys.argv[1]
    pred_file = sys.argv[2]

    analyze_errors(gold_file, pred_file)