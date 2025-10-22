# aw5571_viterbi_HW3.py
#
# Straightforward HMM POS tagger:
#   1. collect counts from the tagged training corpus
#   2. convert counts to (log) transition/emission probabilities
#   3. decode sentences with Viterbi

from __future__ import annotations

import argparse
import collections
import math
import sys
from typing import Dict, Iterable, List, Sequence

from aw5571_train_HMM_HW3 import BEGIN, END, load_pos_counts, stable_tag_order
from aw5571_unk_utils_HW3 import UNK_CLASSES, word_to_unk_class

Model = Dict[str, object]


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(
    train_pos_path: str,
    k_trans: float,
    k_emit: float,
    lambda_trans: float,
) -> Model:
    emit_raw, trans_counts, tag_count, _, _, _ = load_pos_counts(train_pos_path)

    token_totals = collections.Counter()
    for counter in emit_raw.values():
        token_totals.update(counter)

    vocab = {w for w, c in token_totals.items() if c > 1}

    # Replace hapax tokens with their UNK class so emissions cover rare words.
    emissions: Dict[str, collections.Counter] = {}
    for tag, counter in emit_raw.items():
        adjusted = collections.Counter()
        for word, count in counter.items():
            if token_totals[word] == 1:
                adjusted[word_to_unk_class(word)] += count
            else:
                adjusted[word] += count
        emissions[tag] = adjusted

    tags = stable_tag_order(tag_count)
    inner_tags = [t for t in tags if t not in (BEGIN, END)]
    num_tags = len(tags)

    # Transition probabilities (Laplace + interpolation with tag priors).
    total_tags = sum(tag_count.values())
    tag_priors = {
        tag: (tag_count[tag] / total_tags) if total_tags else 0.0 for tag in tags
    }

    log_trans: Dict[str, Dict[str, float]] = {}
    for prev in tags:
        row = trans_counts.get(prev, {})
        denom = sum(row.values()) + k_trans * num_tags
        out_row = {}
        for cur in tags:
            count = row.get(cur, 0)
            bigram = (count + k_trans) / denom
            prior = tag_priors.get(cur, 0.0)
            prob = lambda_trans * bigram + (1.0 - lambda_trans) * prior
            if prob <= 0.0:
                prob = 1e-300
            out_row[cur] = math.log(prob)
        log_trans[prev] = out_row

    # Emission probabilities (Laplace smoothing).
    vocab_size = len(vocab) + len(UNK_CLASSES)
    log_emit: Dict[str, Dict[str, float]] = {}
    log_emit_default: Dict[str, float] = {}
    log_emit_class: Dict[str, Dict[str, float]] = {}

    for tag in tags:
        if tag in (BEGIN, END):
            log_emit[tag] = {}
            log_emit_default[tag] = float("-inf")
            log_emit_class[tag] = {cls: float("-inf") for cls in UNK_CLASSES}
            continue

        counter = emissions.get(tag, collections.Counter())
        denom = tag_count[tag] + k_emit * vocab_size

        emit_row = {}
        for word, count in counter.items():
            emit_row[word] = math.log((count + k_emit) / denom)
        log_emit[tag] = emit_row
        log_emit_default[tag] = math.log(k_emit / denom)

        class_row = {}
        for cls in UNK_CLASSES:
            class_row[cls] = math.log((counter.get(cls, 0) + k_emit) / denom)
        log_emit_class[tag] = class_row

    return {
        "tags": tags,
        "inner_tags": inner_tags,
        "log_trans": log_trans,
        "log_emit": log_emit,
        "log_emit_default": log_emit_default,
        "log_emit_class": log_emit_class,
        "vocab": vocab,
    }


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def emission_log_prob(model: Model, tag: str, word: str) -> float:
    if tag in (BEGIN, END):
        return 0.0

    log_emit = model["log_emit"][tag]  # type: ignore[index]
    if word in model["vocab"]:  # type: ignore[operator]
        return log_emit.get(word, model["log_emit_default"][tag])  # type: ignore[index]

    unk_class = word_to_unk_class(word)
    return model["log_emit_class"][tag].get(unk_class, model["log_emit_default"][tag])  # type: ignore[index]


def decode_sentence(model: Model, words: Sequence[str]) -> List[str]:
    if not words:
        return []

    inner_tags: List[str] = model["inner_tags"]  # type: ignore[assignment]
    log_trans = model["log_trans"]  # type: ignore[assignment]

    dp: List[Dict[str, float]] = [{} for _ in words]
    backptr: List[Dict[str, str]] = [{} for _ in words]

    first_word = words[0]
    for tag in inner_tags:
        dp[0][tag] = log_trans[BEGIN][tag] + emission_log_prob(model, tag, first_word)
        backptr[0][tag] = BEGIN

    for i in range(1, len(words)):
        word = words[i]
        for tag in inner_tags:
            best_prev = inner_tags[0]
            best_score = float("-inf")
            emit = emission_log_prob(model, tag, word)
            for prev in inner_tags:
                prev_score = dp[i - 1].get(prev)
                if prev_score is None:
                    continue
                score = prev_score + log_trans[prev][tag]
                if score > best_score:
                    best_score = score
                    best_prev = prev
            dp[i][tag] = best_score + emit
            backptr[i][tag] = best_prev

    best_last = max(inner_tags, key=lambda t: dp[-1].get(t, float("-inf")) + log_trans[t][END])
    seq = [best_last] * len(words)
    for i in range(len(words) - 1, 0, -1):
        seq[i - 1] = backptr[i][seq[i]]
    return seq


def tag_corpus(model: Model, words_path: str, out_path: str) -> None:
    with open(words_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        sentence: List[str] = []
        for raw in fin:
            token = raw.rstrip("\n")
            if token == "":
                if sentence:
                    tags = decode_sentence(model, sentence)
                    for word, tag in zip(sentence, tags):
                        fout.write(f"{word}\t{tag}\n")
                    fout.write("\n")
                    sentence = []
                else:
                    fout.write("\n")
                continue
            sentence.append(token)

        if sentence:
            tags = decode_sentence(model, sentence)
            for word, tag in zip(sentence, tags):
                fout.write(f"{word}\t{tag}\n")
            fout.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Viterbi HMM POS tagger")
    parser.add_argument("train_pos", help="Tagged training corpus (.pos)")
    parser.add_argument("input_words", help="Input corpus (.words)")
    parser.add_argument("output_pos", help="Where to write tagged output")
    parser.add_argument("--k-trans", type=float, default=0.06, help="Add-k constant for transitions")
    parser.add_argument("--k-emit", type=float, default=0.0001, help="Add-k constant for emissions")
    parser.add_argument(
        "--lambda-trans",
        type=float,
        default=0.93,
        help="Interpolation weight for transition bigram vs. unigram",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    model = train_model(
        args.train_pos,
        k_trans=args.k_trans,
        k_emit=args.k_emit,
        lambda_trans=args.lambda_trans,
    )
    tag_corpus(model, args.input_words, args.output_pos)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
