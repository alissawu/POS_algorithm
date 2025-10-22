# aw5571_viterbi_HW3.py
#
# Compact HMM tagger that trains a bigram model (with hapax-based
# OOV handling) and decodes a .words file via Viterbi.

from __future__ import annotations

import argparse
import collections
import math
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

from aw5571_train_HMM_HW3 import BEGIN, END, load_pos_counts, stable_tag_order
from aw5571_unk_utils_HW3 import UNK_CLASSES, word_to_unk_class


def build_model(
    train_pos_path: str,
    k_trans: float,
    k_emit: float,
    lambda_trans: float,
) -> Tuple[
    List[str],
    List[str],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, float],
    Dict[str, Dict[str, float]],
    set,
]:
    emit_raw, trans_counts, tag_count, _, _, _ = load_pos_counts(train_pos_path)

    total_counts = collections.Counter()
    for tag, counter in emit_raw.items():
        total_counts.update(counter)

    vocab = {w for w, c in total_counts.items() if c > 1}
    vocab_size = len(vocab)

    hapax_words = [w for w, c in total_counts.items() if c == 1]
    hapax_class_totals = collections.Counter(word_to_unk_class(w) for w in hapax_words)
    if not hapax_class_totals:
        hapax_class_totals = collections.Counter({cls: 1 for cls in UNK_CLASSES})
    total_hapax = sum(hapax_class_totals.values())
    class_prior = {
        cls: (hapax_class_totals.get(cls, 0) + 1) / (total_hapax + len(UNK_CLASSES))
        for cls in UNK_CLASSES
    }
    prior_mass = sum(class_prior.values())

    tags = stable_tag_order(tag_count)
    inner_tags = [t for t in tags if t not in (BEGIN, END)]
    num_tags = len(tags)
    total_tags = sum(tag_count.values())
    tag_priors = {
        tag: (tag_count[tag] / total_tags) if total_tags else 0.0 for tag in tags
    }

    log_trans = {}
    for prev in tags:
        denom = tag_count[prev] + k_trans * num_tags
        row = {}
        for cur in tags:
            num = trans_counts[prev][cur] + k_trans
            bigram = num / denom
            prior = tag_priors.get(cur, 0.0)
            prob = lambda_trans * bigram + (1.0 - lambda_trans) * prior
            if prob <= 0.0:
                prob = 1e-300
            row[cur] = math.log(prob)
        log_trans[prev] = row

    emit_known_log: Dict[str, Dict[str, float]] = {}
    emit_default_log: Dict[str, float] = {}
    emit_class_log: Dict[str, Dict[str, float]] = {}

    for tag in tags:
        if tag in (BEGIN, END):
            emit_known_log[tag] = {}
            emit_default_log[tag] = float("-inf")
            emit_class_log[tag] = {cls: float("-inf") for cls in UNK_CLASSES}
            continue

        known_counts = {w: c for w, c in emit_raw[tag].items() if total_counts[w] > 1}
        class_counts = collections.Counter()
        for w, c in emit_raw[tag].items():
            if total_counts[w] == 1:
                class_counts[word_to_unk_class(w)] += c

        denom = tag_count[tag] + k_emit * (vocab_size + prior_mass)

        emit_known_log[tag] = {
            w: math.log((c + k_emit) / denom) for w, c in known_counts.items()
        }
        emit_default_log[tag] = math.log(k_emit / denom)
        emit_class_log[tag] = {
            cls: math.log((class_counts.get(cls, 0) + k_emit * class_prior[cls]) / denom)
            for cls in UNK_CLASSES
        }

    return (
        tags,
        inner_tags,
        log_trans,
        emit_known_log,
        emit_default_log,
        emit_class_log,
        vocab,
    )


def emission_log_prob(
    tag: str,
    word: str,
    emit_known_log: Dict[str, Dict[str, float]],
    emit_default_log: Dict[str, float],
    emit_class_log: Dict[str, Dict[str, float]],
    vocab: set,
) -> float:
    if tag in (BEGIN, END):
        return float("-inf")
    if word in vocab:
        return emit_known_log[tag].get(word, emit_default_log[tag])
    unk_class = word_to_unk_class(word)
    return emit_class_log[tag][unk_class]


def viterbi_tag_sentence(
    words: Sequence[str],
    inner_tags: Sequence[str],
    log_trans: Dict[str, Dict[str, float]],
    emit_known_log: Dict[str, Dict[str, float]],
    emit_default_log: Dict[str, float],
    emit_class_log: Dict[str, Dict[str, float]],
    vocab: set,
) -> List[str]:
    if not words:
        return []

    dp: List[Dict[str, float]] = [{} for _ in words]
    backptr: List[Dict[str, str]] = [{} for _ in words]

    w0 = words[0]
    for tag in inner_tags:
        dp[0][tag] = log_trans[BEGIN][tag] + emission_log_prob(
            tag, w0, emit_known_log, emit_default_log, emit_class_log, vocab
        )
        backptr[0][tag] = BEGIN

    for i in range(1, len(words)):
        w = words[i]
        for tag in inner_tags:
            emit = emission_log_prob(
                tag, w, emit_known_log, emit_default_log, emit_class_log, vocab
            )
            best_prev = None
            best_score = float("-inf")
            for prev in inner_tags:
                prev_score = dp[i - 1].get(prev)
                if prev_score is None:
                    continue
                score = prev_score + log_trans[prev][tag]
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[i][tag] = best_score + emit
            backptr[i][tag] = best_prev if best_prev is not None else inner_tags[0]

    best_last = inner_tags[0]
    best_score = float("-inf")
    for tag in inner_tags:
        score = dp[-1].get(tag, float("-inf")) + log_trans[tag][END]
        if score > best_score:
            best_score = score
            best_last = tag

    seq = [best_last] * len(words)
    for i in range(len(words) - 1, 0, -1):
        seq[i - 1] = backptr[i][seq[i]]
    return seq


def tag_corpus(
    tags: Sequence[str],
    inner_tags: Sequence[str],
    log_trans: Dict[str, Dict[str, float]],
    emit_known_log: Dict[str, Dict[str, float]],
    emit_default_log: Dict[str, float],
    emit_class_log: Dict[str, Dict[str, float]],
    vocab: set,
    in_words_path: str,
    out_pos_path: str,
) -> None:
    with open(in_words_path, "r", encoding="utf-8") as fin, open(
        out_pos_path, "w", encoding="utf-8"
    ) as fout:
        sentence: List[str] = []
        for raw in fin:
            token = raw.rstrip("\n")
            if token == "":
                if sentence:
                    seq = viterbi_tag_sentence(
                        sentence,
                        inner_tags,
                        log_trans,
                        emit_known_log,
                        emit_default_log,
                        emit_class_log,
                        vocab,
                    )
                    for w, t in zip(sentence, seq):
                        fout.write(f"{w}\t{t}\n")
                    fout.write("\n")
                    sentence = []
                else:
                    fout.write("\n")
                continue
            sentence.append(token)

        if sentence:
            seq = viterbi_tag_sentence(
                sentence,
                inner_tags,
                log_trans,
                emit_known_log,
                emit_default_log,
                emit_class_log,
                vocab,
            )
            for w, t in zip(sentence, seq):
                fout.write(f"{w}\t{t}\n")
            fout.write("\n")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a bigram HMM POS tagger and decode a .words file."
    )
    parser.add_argument("train_pos", help="Training corpus with word<TAB>tag per line.")
    parser.add_argument("input_words", help="Input corpus containing one token per line.")
    parser.add_argument("output_pos", help="Where to write tagged output.")
    parser.add_argument(
        "--k-trans",
        type=float,
        default=0.06,
        help="Add-k smoothing constant for transitions (default: 0.06).",
    )
    parser.add_argument(
        "--k-emit",
        type=float,
        default=0.0001,
        help="Add-k smoothing constant for emissions (default: 1e-4).",
    )
    parser.add_argument(
        "--lambda-trans",
        type=float,
        default=0.93,
        help="Interpolation weight for transition bigram vs. unigram (default: 0.93).",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    (
        tags,
        inner_tags,
        log_trans,
        emit_known_log,
        emit_default_log,
        emit_class_log,
        vocab,
    ) = build_model(
        args.train_pos,
        k_trans=args.k_trans,
        k_emit=args.k_emit,
        lambda_trans=args.lambda_trans,
    )
    tag_corpus(
        tags,
        inner_tags,
        log_trans,
        emit_known_log,
        emit_default_log,
        emit_class_log,
        vocab,
        args.input_words,
        args.output_pos,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
