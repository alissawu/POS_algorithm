```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# -----------------------------
# Types & constants
# -----------------------------

Sentence = List[Tuple[str, str]]
TokenSequence = List[str]

START_TAG = "<s>"
END_TAG = "</s>"

# Transition smoothing
TRANSITION_ALPHA = 0.1  # add-k for tag bigrams (and transitions to </s>)

# Hapax class smoothing
CLASS_ALPHA = 0.5
CLASS_BACKOFF = 5.0

# Low-frequency emission blending
LOW_FREQ_BLEND_THRESHOLD = 2
LOW_FREQ_BLEND_WEIGHT = 0.7

# Interpolation weights (computed globally via deleted interpolation)
TRIGRAM_BACKOFF = 2.0  # kept for compatibility (unused)
BIGRAM_BACKOFF = 1.0   # kept for compatibility (unused)

# EMISSION smoothing for ALL seen words (critical for >96.5%)
EMISSION_ALPHA = 1e-4  # tune in [1e-5, 5e-4]

# Heuristic clamp for context/known-word bonuses
HEURISTIC_LO = math.log(0.6)  # ~ -0.51
HEURISTIC_HI = math.log(1.6)  # ~ +0.47

# Tiny floor for closed-class canonical tags (prevents drift)
CLOSED_FLOOR = 1e-6

SUFFIX_CLASSES = [
    "less", "ness", "able", "ible", "tion", "sion", "ment", "ship", "hood",
    "ity", "ous", "ive", "est", "ies", "ing", "ed", "es", "s", "ly", "er",
]

ADJ_SUFFIXES = (
    "able", "ible", "al", "ial", "ic", "ical", "ish", "ive", "less", "ous",
    "ful", "ant", "ent", "ary", "ory",
)

NOUN_SUFFIXES = (
    "ment", "ness", "tion", "sion", "ship", "hood", "dom", "ism", "ist",
    "ity", "age",
)

VERB_TAGS = {"VB", "VBD", "VBN", "VBP", "VBZ", "MD"}
DETERMINER_TAGS = {"DT", "PDT", "PRP$"}
CAPITAL_TAGS = {"NNP", "NNPS"}
PUNCT_TAGS = {".", ",", ":", "``", "''", "-LRB-", "-RRB-", "$", "#"}

# Closed-class canonical hints (weak priors)
CLOSED_HINTS: Dict[str, set[str]] = {
    "to": {"TO"},
    "and": {"CC"}, "or": {"CC"}, "but": {"CC"}, "nor": {"CC"}, "yet": {"CC"},
    "the": {"DT"}, "a": {"DT"}, "an": {"DT"},
    "there": {"EX"},
    "not": {"RB"}, "n't": {"RB"},
    "who": {"WP"}, "whom": {"WP"}, "what": {"WP"}, "whose": {"WP$"},
    "which": {"WDT"}, "when": {"WRB"}, "where": {"WRB"}, "why": {"WRB"}, "how": {"WRB"},
    # “that” is ambiguous; we do not force it to a single tag
}


@dataclass
class HMMParameters:
    tags: List[str]
    vocabulary: set[str]
    transition_probs: Dict[str, Dict[str, float]]               # smoothed bigrams P(t_i | t_{i-1})
    emission_probs: Dict[str, Dict[str, float]]                 # smoothed emissions P(w | t)
    tag_priors: Dict[str, float]                                # P(t)
    tag_counts: Dict[str, int]
    emission_counts: Dict[str, Counter[str]]
    transition_counts: Dict[str, Counter[str]]                  # bigram counts
    word_tag_counts: Dict[str, Counter[str]]
    default_tag: str
    log_transition_probs: Dict[str, Dict[str, float]]
    log_emission_probs: Dict[str, Dict[str, float]]
    log_tag_priors: Dict[str, float]
    unk_log_emissions: Dict[str, float]                         # hapax-based global OOV log probs per tag
    hapax_tag_probs: Dict[str, float]
    hapax_class_probs: Dict[str, Dict[str, float]]
    hapax_class_totals: Dict[str, int]
    trigram_counts: Dict[Tuple[str, str], Counter[str]]
    trigram_totals: Dict[Tuple[str, str], int]
    word_frequencies: Dict[str, int]
    bigram_totals: Dict[str, int]
    lambda_trigram: float
    lambda_bigram: float
    lambda_unigram: float
    sentence_count: int


# -----------------------------
# Utilities
# -----------------------------

def clamp_log(x: float, lo: float = HEURISTIC_LO, hi: float = HEURISTIC_HI) -> float:
    return min(max(x, lo), hi)


def read_tagged_corpus(path: Path) -> List[Sentence]:
    sentences: List[Sentence] = []
    current: Sentence = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            try:
                word, tag = line.split("\t")
            except ValueError:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Unexpected line format: {line!r}")
                word, tag = parts
            current.append((word, tag))
    if current:
        sentences.append(current)
    return sentences


def read_word_sequences(path: Path) -> List[TokenSequence]:
    sequences: List[TokenSequence] = []
    current: TokenSequence = []
    with path.open("r", encoding="utf8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                if current:
                    sequences.append(current)
                    current = []
                continue
            current.append(line)
    if current:
        sequences.append(current)
    return sequences


# -----------------------------
# Feature helpers
# -----------------------------

def classify_hapax(word: str) -> str:
    lower = word.lower()
    features: List[str] = []

    if re.fullmatch(r"\d+(?:\.\d+)?", lower):
        features.append("NUMERIC")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", lower):
        features.append("NUMERIC_COMMA")
    elif lower.endswith("%") and lower[:-1].replace(".", "", 1).isdigit():
        features.append("NUMERIC_PERCENT")
    elif any(ch.isdigit() for ch in word):
        features.append("ALNUM")

    if word and re.fullmatch(r"[^\w]+", word):
        features.append("PUNCT")

    if "-" in word:
        features.append("HYPHEN")
    if "'" in word or "’" in word:
        features.append("APOSTROPHE")

    if word.isupper() and any(ch.isalpha() for ch in word):
        features.append("ALLCAP")
    elif word and word[0].isupper():
        features.append("INITCAP")
    elif any(ch.isupper() for ch in word):
        features.append("MIXEDCAP")

    suffix = None
    for candidate in sorted(SUFFIX_CLASSES, key=len, reverse=True):
        if lower.endswith(candidate) and len(lower) > len(candidate):
            suffix = candidate
            break
    if suffix:
        features.append(f"SUF_{suffix.upper()}")

    if not features:
        return "OOV_OTHER"
    signature = "+".join(features)
    return f"OOV_{signature}"


def is_adjective_like(word: str) -> bool:
    lower = word.lower()
    return any(lower.endswith(suf) for suf in ADJ_SUFFIXES)


def is_noun_like(word: str) -> bool:
    lower = word.lower()
    return any(lower.endswith(suf) for suf in NOUN_SUFFIXES)


def is_numeric_token(word: str) -> bool:
    lower = word.lower()
    return bool(
        re.fullmatch(r"\d+(?:\.\d+)?", lower)
        or re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", lower)
        or (lower.endswith("%") and lower[:-1].replace(".", "", 1).isdigit())
    )


def is_symbol_token(word: str) -> bool:
    return bool(word) and bool(re.fullmatch(r"[^\w]+", word))


# -----------------------------
# Training
# -----------------------------

def train_hmm_parameters(sentences: Sequence[Sentence]) -> HMMParameters:
    emission_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    transition_counts: Dict[str, Counter[str]] = defaultdict(Counter)  # bigram counts
    word_tag_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    tag_counts: Counter[str] = Counter()
    vocabulary: set[str] = set()
    word_counts: Counter[str] = Counter()
    trigram_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    sentence_count = 0

    # Count
    for sentence in sentences:
        sentence_count += 1
        prev_tag = START_TAG
        prev_prev_tag = START_TAG
        transition_counts[START_TAG]  # ensure key exists
        for word, tag in sentence:
            vocabulary.add(word)
            emission_counts[tag][word] += 1
            word_tag_counts[word][tag] += 1
            tag_counts[tag] += 1
            transition_counts[prev_tag][tag] += 1
            trigram_counts[(prev_prev_tag, prev_tag)][tag] += 1
            prev_prev_tag, prev_tag = prev_tag, tag
            word_counts[word] += 1
        transition_counts[prev_tag][END_TAG] += 1
        trigram_counts[(prev_prev_tag, prev_tag)][END_TAG] += 1

    tag_list = sorted(tag_counts.keys())

    # Smoothed transition bigrams (add-α)
    transition_probs: Dict[str, Dict[str, float]] = {}
    log_transition_probs: Dict[str, Dict[str, float]] = {}
    all_prev_tags = [START_TAG] + tag_list
    for prev_tag in all_prev_tags:
        counts = transition_counts.get(prev_tag, Counter())
        # From START, we never go to END directly
        next_tags = tag_list if prev_tag == START_TAG else tag_list + [END_TAG]
        denom = sum(counts.get(t, 0) for t in next_tags) + TRANSITION_ALPHA * len(next_tags)
        probs: Dict[str, float] = {}
        logs: Dict[str, float] = {}
        for next_tag in next_tags:
            prob = (counts.get(next_tag, 0) + TRANSITION_ALPHA) / denom
            probs[next_tag] = prob
            logs[next_tag] = math.log(prob)
        transition_probs[prev_tag] = probs
        log_transition_probs[prev_tag] = logs

    # Unigram tag priors
    total_tags = sum(tag_counts.values())
    tag_priors = {t: (tag_counts[t] / total_tags) for t in tag_list} if total_tags else {}
    default_tag = max(tag_priors.items(), key=lambda kv: kv[1])[0] if tag_priors else "NN"
    log_tag_priors = {t: (math.log(p) if p > 0 else float("-inf")) for t, p in tag_priors.items()}

    # Hapax distributions
    hapax_tag_counts: Counter[str] = Counter()
    hapax_class_tag_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    hapax_class_totals: Counter[str] = Counter()
    for w, c in word_counts.items():
        if c == 1:
            tag = word_tag_counts[w].most_common(1)[0][0]
            hapax_tag_counts[tag] += 1
            cls = classify_hapax(w)
            hapax_class_totals[cls] += 1
            hapax_class_tag_counts[cls][tag] += 1

    hapax_total = sum(hapax_tag_counts.values())
    tiny = 1e-6
    hapax_tag_probs: Dict[str, float] = {}
    for t in tag_list:
        if hapax_total > 0:
            hapax_tag_probs[t] = (hapax_tag_counts.get(t, 0.0) + tiny) / (hapax_total + tiny * len(tag_list))
        else:
            hapax_tag_probs[t] = 1.0 / len(tag_list)

    unk_log_emissions = {t: math.log(p) for t, p in hapax_tag_probs.items()}

    hapax_class_probs: Dict[str, Dict[str, float]] = {}
    for cls, total in hapax_class_totals.items():
        tag_counter = hapax_class_tag_counts[cls]
        denom = total + CLASS_ALPHA
        dist: Dict[str, float] = {}
        for t in tag_list:
            numerator = tag_counter.get(t, 0.0) + CLASS_ALPHA * hapax_tag_probs[t]
            dist[t] = numerator / denom if denom > 0 else hapax_tag_probs[t]
        hapax_class_probs[cls] = dist

    # Emission probabilities with a small floor for ALL seen words
    emission_probs: Dict[str, Dict[str, float]] = {}
    log_emission_probs: Dict[str, Dict[str, float]] = {}
    vocab_size = max(len(vocabulary), 1)
    for t, counts in emission_counts.items():
        total = tag_counts[t]
        probs: Dict[str, float] = {}
        logs: Dict[str, float] = {}
        # Hapax-informed floor to keep alternatives alive
        floor_h = EMISSION_ALPHA * hapax_tag_probs[t]
        denom = total + EMISSION_ALPHA * vocab_size
        for w, c in counts.items():
            p = (c + EMISSION_ALPHA) / denom
            # small mixture with hapax-informed mass
            p = 0.98 * p + 0.02 * floor_h
            p = max(min(p, 1.0), 1e-15)
            probs[w] = p
            logs[w] = math.log(p)
        emission_probs[t] = probs
        log_emission_probs[t] = logs

    # Trigram counts/ totals for deleted interpolation
    trigram_totals: Dict[Tuple[str, str], int] = {
        ctx: sum(counter.values()) for ctx, counter in trigram_counts.items()
    }
    bigram_totals: Dict[str, int] = {prev: sum(c.values()) for prev, c in transition_counts.items()}

    # Global deleted interpolation lambdas (stable)
    lam3 = lam2 = lam1 = 0.0
    total_tags_minus_one = max(total_tags - 1, 1)
    for (t_2, t_1), counter in trigram_counts.items():
        tri_den = max(trigram_totals.get((t_2, t_1), 0) - 1, 1)
        bi_den = max(bigram_totals.get(t_1, 0) - 1, 1)
        for t, cnt in counter.items():
            c = max(cnt - 1, 0)
            p3 = c / tri_den if tri_den > 0 else 0.0
            bcnt = transition_counts.get(t_1, Counter()).get(t, 0)
            p2 = max(bcnt - 1, 0) / bi_den if bi_den > 0 else 0.0
            p1 = max(tag_counts.get(t, 0) - 1, 0) / total_tags_minus_one if total_tags_minus_one > 0 else 0.0
            # increment the bucket that wins
            if p3 >= p2 and p3 >= p1:
                lam3 += cnt
            elif p2 >= p3 and p2 >= p1:
                lam2 += cnt
            else:
                lam1 += cnt
    s = lam3 + lam2 + lam1
    if s == 0.0:
        lambda_trigram, lambda_bigram, lambda_unigram = 0.6, 0.3, 0.1
    else:
        lambda_trigram, lambda_bigram, lambda_unigram = lam3 / s, lam2 / s, lam1 / s

    return HMMParameters(
        tags=tag_list,
        vocabulary=vocabulary,
        transition_probs=transition_probs,
        emission_probs=emission_probs,
        tag_priors=tag_priors,
        tag_counts=dict(tag_counts),
        emission_counts=emission_counts,
        transition_counts=transition_counts,
        word_tag_counts=word_tag_counts,
        default_tag=default_tag,
        log_transition_probs=log_transition_probs,
        log_emission_probs=log_emission_probs,
        log_tag_priors=log_tag_priors,
        unk_log_emissions=unk_log_emissions,
        hapax_tag_probs=hapax_tag_probs,
        hapax_class_probs=hapax_class_probs,
        hapax_class_totals=dict(hapax_class_totals),
        trigram_counts=trigram_counts,
        trigram_totals=trigram_totals,
        word_frequencies=dict(word_counts),
        bigram_totals=bigram_totals,
        lambda_trigram=lambda_trigram,
        lambda_bigram=lambda_bigram,
        lambda_unigram=lambda_unigram,
        sentence_count=sentence_count,
    )


# -----------------------------
# Baseline
# -----------------------------

def tag_frequency_baseline(
    word_sequences: Sequence[TokenSequence],
    params: HMMParameters,
) -> List[Sentence]:
    tagged: List[Sentence] = []
    for sequence in word_sequences:
        sentence: Sentence = []
        for word in sequence:
            if word in params.vocabulary:
                word_counts = params.word_tag_counts[word]
                best_tag = word_counts.most_common(1)[0][0]
            else:
                best_tag = params.default_tag
            sentence.append((word, best_tag))
        tagged.append(sentence)
    return tagged


# -----------------------------
# Heuristic bonuses (small, clamped)
# -----------------------------

def known_word_bonus(word: str, tag: str, params: HMMParameters) -> float:
    bonus = 0.0
    lower = word.lower()

    if is_numeric_token(word):
        if tag == "CD":
            bonus += math.log(1.2)
        elif tag not in {"LS", "JJ"}:
            bonus += math.log(0.85)

    if is_symbol_token(word):
        if tag in PUNCT_TAGS:
            bonus += math.log(1.25)
        else:
            bonus += math.log(0.8)

    # Gentle capitalization hints
    if word[:1].isupper() and not word.isupper():
        if tag in CAPITAL_TAGS:
            bonus += math.log(1.1)
        elif tag in {"NN", "JJ"}:
            bonus += math.log(0.9)

    if word.isupper() and len(word) > 1:
        if tag in CAPITAL_TAGS | {"NN"}:
            bonus += math.log(1.05)
        elif tag == "JJ":
            bonus += math.log(0.9)

    if lower.endswith("ly"):
        if tag == "RB":
            bonus += math.log(1.15)
        elif tag in {"NN", "JJ", "VB"}:
            bonus += math.log(0.85)

    if tag == "JJ":
        bonus += math.log(1.1) if is_adjective_like(word) else math.log(0.9)

    if tag == "NN" and is_noun_like(word):
        bonus += math.log(1.05)

    if tag == "NNPS" and word[:1].isupper() and word.endswith("s"):
        bonus += math.log(1.1)

    # Ambiguity-aware: reinforce dominant tag lightly if strong
    counts = params.word_tag_counts.get(word)
    if counts:
        total = sum(counts.values())
        if total >= 3:
            best_tag, best_count = counts.most_common(1)[0]
            best_freq = best_count / total
            freq = counts.get(tag, 0) / total
            if tag == best_tag and best_freq >= 0.55:
                bonus += math.log(1.0 + 0.7 * max(0.0, best_freq - 0.55))
            elif tag != best_tag and freq <= 0.10:
                bonus += math.log(0.85)

    return clamp_log(bonus)


def context_bonus(prev_prev: str, prev: str, curr: str, word: str,
                  idx: int, words: TokenSequence, params: HMMParameters) -> float:
    bonus = 0.0
    lower = word.lower()

    # After determiners, NN is slightly favored; non-adj JJ penalized a bit
    if prev in DETERMINER_TAGS:
        if curr == "NN":
            bonus += math.log(1.1)
        elif curr == "JJ" and not is_adjective_like(word):
            bonus += math.log(0.9)

    # Past ed after nominal subject often VBD, but keep it weak (plenty of VBN-as-adj)
    if lower.endswith("ed") and prev in {"NN", "NNS", "PRP"}:
        if curr == "VBD":
            bonus += math.log(1.1)
        elif curr == "VBN":
            bonus += math.log(0.95)

    # RP vs IN: lookahead for fixed PP heads; keep light
    if idx + 1 < len(words):
        nxt = words[idx + 1].lower()
        if nxt in {"of", "from", "until", "to"}:
            if curr == "IN":
                bonus += math.log(1.25)
            elif curr == "RP":
                bonus += math.log(0.85)

    # Perfect auxiliaries prefer VBN (soft)
    if idx > 0 and words[idx - 1].lower() in {"has", "have", "had", "been", "being"}:
        if lower.endswith(("ed", "en")):
            if curr == "VBN":
                bonus += math.log(1.25)
            elif curr == "VBD":
                bonus += math.log(0.9)

    # 'to' context: if next looks like base verb (seen as VB often), gently help TO
    if lower == "to" and idx + 1 < len(words):
        nxt = words[idx + 1]
        tags = params.word_tag_counts.get(nxt, Counter())
        if tags:
            vb_freq = tags.get("VB", 0) / max(1, sum(tags.values()))
            if vb_freq > 0.5 and curr == "TO":
                bonus += math.log(1.2)

    # Existential there + BE
    if lower == "there" and idx + 1 < len(words):
        nxt = words[idx + 1].lower()
        if nxt in {"is", "are", "was", "were", "'s", "'re"} and curr == "EX":
            bonus += math.log(1.2)

    return clamp_log(bonus)


# -----------------------------
# Viterbi
# -----------------------------

def viterbi_tag(
    word_sequences: Sequence[TokenSequence],
    params: HMMParameters,
) -> List[Sentence]:
    tagged_sentences: List[Sentence] = []
    states = params.tags
    log_emit = params.log_emission_probs
    hapax_class_probs = params.hapax_class_probs
    hapax_class_totals = params.hapax_class_totals
    global_hapax = params.hapax_tag_probs
    trigram_counts = params.trigram_counts
    trigram_totals = params.trigram_totals
    transition_probs = params.transition_probs
    lambda_trigram = params.lambda_trigram
    lambda_bigram = params.lambda_bigram
    lambda_unigram = params.lambda_unigram
    tag_priors = params.tag_priors
    default_tag = params.default_tag
    neg_inf = float("-inf")

    def oov_log_prob(oov_class: str | None, tag: str) -> float:
        # global hapax
        prob = global_hapax.get(tag, 0.0)
        # class-specific interpolation
        if oov_class and oov_class in hapax_class_probs:
            class_probs = hapax_class_probs[oov_class]
            class_total = hapax_class_totals.get(oov_class, 0)
            mix = class_total / (class_total + CLASS_BACKOFF) if CLASS_BACKOFF > 0 else 1.0
            class_prob = class_probs.get(tag, 0.0)
            prob = mix * class_prob + (1.0 - mix) * prob

        # small special-cases for numeric/punct
        if oov_class:
            if "NUMERIC" in oov_class:
                if tag == "CD":
                    prob = max(prob, 0.9)
                else:
                    prob = max(prob * 0.05, 1e-8)
            elif "PUNCT" in oov_class:
                if tag in PUNCT_TAGS:
                    prob = max(prob, 0.8)
                else:
                    prob = max(prob * 0.02, 1e-8)

        return math.log(prob) if prob > 0.0 else neg_inf

    def get_known_emission_log(word: str, tag: str) -> float:
        # Standard emission
        le = log_emit.get(tag, {}).get(word)
        if le is not None:
            return le
        # Closed-class tiny floor for canonical tags (prevents drift on function words)
        low = word.lower()
        if low in CLOSED_HINTS and tag in CLOSED_HINTS[low]:
            return math.log(CLOSED_FLOOR)
        return neg_inf

    def transition_log(prev_prev: str, prev: str, curr: str) -> float:
        # Deleted-interpolated P(curr | prev_prev, prev)
        # trigram MLE (unsmoothed)
        tri_total = trigram_totals.get((prev_prev, prev), 0)
        p3 = (trigram_counts.get((prev_prev, prev), {}).get(curr, 0) / tri_total) if tri_total > 0 else 0.0
        # smoothed bigram
        p2 = transition_probs.get(prev, {}).get(curr, 0.0)
        # unigram
        if curr == END_TAG:
            # tiny uniform-ish mass for END in unigram part (rare)
            p1 = 1e-8
        else:
            p1 = tag_priors.get(curr, 0.0)
        prob = lambda_trigram * p3 + lambda_bigram * p2 + lambda_unigram * p1
        return math.log(prob) if prob > 0.0 else neg_inf

    for words in word_sequences:
        if not words:
            tagged_sentences.append([])
            continue

        viterbi: List[Dict[Tuple[str, str], float]] = []
        backpointer: List[Dict[Tuple[str, str], Tuple[str, str] | None]] = []

        # First word column
        w0 = words[0]
        w0_class = None if w0 in params.vocabulary else classify_hapax(w0)
        col0: Dict[Tuple[str, str], float] = {}
        back0: Dict[Tuple[str, str], Tuple[str, str] | None] = {}
        for curr in states:
            trans = transition_log(START_TAG, START_TAG, curr)
            if trans == neg_inf:
                continue
            if w0 in params.vocabulary:
                emit = get_known_emission_log(w0, curr)
                if emit != neg_inf:
                    # keep low-freq blend for very rare tokens
                    if params.word_frequencies.get(w0, 0) <= LOW_FREQ_BLEND_THRESHOLD:
                        base_p = math.exp(emit)
                        cls = classify_hapax(w0)
                        dist = params.hapax_class_probs.get(cls, {})
                        class_p = dist.get(curr, params.hapax_tag_probs.get(curr, 0.0))
                        mixed = LOW_FREQ_BLEND_WEIGHT * base_p + (1.0 - LOW_FREQ_BLEND_WEIGHT) * class_p
                        emit = math.log(max(mixed, 1e-15))
                    emit += known_word_bonus(w0, curr, params)
            else:
                emit = oov_log_prob(w0_class, curr)
            if emit == neg_inf:
                continue
            b = context_bonus(START_TAG, START_TAG, curr, w0, 0, words, params)
            score = trans + emit + b
            col0[(START_TAG, curr)] = score
            back0[(START_TAG, curr)] = None

        if not col0:
            # robust fallback: pick prior-best tag that has a valid transition
            best_tag = default_tag
            best_score = -1e9
            for t in states:
                trans = transition_log(START_TAG, START_TAG, t)
                if trans == neg_inf:
                    continue
                prior = params.log_tag_priors.get(t, -1e9)
                s = trans + prior
                if s > best_score:
                    best_score = s
                    best_tag = t
            col0[(START_TAG, best_tag)] = best_score
            back0[(START_TAG, best_tag)] = None

        viterbi.append(col0)
        backpointer.append(back0)

        # Subsequent columns
        for idx in range(1, len(words)):
            w = words[idx]
            w_class = None if w in params.vocabulary else classify_hapax(w)
            prev_col = viterbi[idx - 1]
            cur_col: Dict[Tuple[str, str], float] = {}
            cur_back: Dict[Tuple[str, str], Tuple[str, str] | None] = {}

            for (pp, p), prev_score in prev_col.items():
                if prev_score == neg_inf:
                    continue
                for curr in states:
                    if w in params.vocabulary:
                        emit = get_known_emission_log(w, curr)
                        if emit == neg_inf:
                            continue
                        if params.word_frequencies.get(w, 0) <= LOW_FREQ_BLEND_THRESHOLD:
                            base_p = math.exp(emit)
                            cls = classify_hapax(w)
                            dist = params.hapax_class_probs.get(cls, {})
                            class_p = dist.get(curr, params.hapax_tag_probs.get(curr, 0.0))
                            mixed = LOW_FREQ_BLEND_WEIGHT * base_p + (1.0 - LOW_FREQ_BLEND_WEIGHT) * class_p
                            emit = math.log(max(mixed, 1e-15))
                        emit += known_word_bonus(w, curr, params)
                    else:
                        emit = oov_log_prob(w_class, curr)
                        if emit == neg_inf:
                            continue

                    trans = transition_log(pp, p, curr)
                    if trans == neg_inf:
                        continue

                    b = context_bonus(pp, p, curr, w, idx, words, params)
                    cand = prev_score + trans + emit + b
                    key = (p, curr)
                    if cand > cur_col.get(key, neg_inf):
                        cur_col[key] = cand
                        cur_back[key] = (pp, p)

            if not cur_col:
                # robust fallback for empty column
                # choose frequent tag with valid transition from best previous state
                best_prev_pair, best_prev_score = max(prev_col.items(), key=lambda kv: kv[1])
                pp, p = best_prev_pair
                best_tag = default_tag
                best_score = -1e9
                for t in states:
                    trans = transition_log(pp, p, t)
                    if trans == neg_inf:
                        continue
                    prior = params.log_tag_priors.get(t, -1e9)
                    s = best_prev_score + trans + prior
                    if s > best_score:
                        best_score = s
                        best_tag = t
                cur_col[(p, best_tag)] = best_score
                cur_back[(p, best_tag)] = best_prev_pair

            viterbi.append(cur_col)
            backpointer.append(cur_back)

        # Termination
        last_idx = len(words) - 1
        best_score = neg_inf
        best_pair: Tuple[str, str] | None = None
        for pair, score in viterbi[last_idx].items():
            if score == neg_inf:
                continue
            pp, p = pair
            trans_end = transition_log(pp, p, END_TAG)
            if trans_end == neg_inf:
                continue
            cand = score + trans_end
            if cand > best_score:
                best_score = cand
                best_pair = pair

        if best_pair is None:
            best_pair = max(viterbi[last_idx].items(), key=lambda kv: kv[1])[0]

        # Backtrace
        idx = last_idx
        pair = best_pair
        tags_rev: List[str] = []
        while pair is not None and idx >= 0:
            prev_tag, curr_tag = pair
            tags_rev.append(curr_tag)
            pair = backpointer[idx].get(pair)
            idx -= 1
        predicted = list(reversed(tags_rev))
        if len(predicted) != len(words):
            predicted = [default_tag] * len(words)

        tagged_sentences.append(list(zip(words, predicted)))
    return tagged_sentences


# -----------------------------
# I/O
# -----------------------------

def write_tagged_corpus(sentences: Iterable[Sentence], path: Path) -> None:
    with path.open("w", encoding="utf8") as fh:
        for sentence in sentences:
            for word, tag in sentence:
                fh.write(f"{word}\t{tag}\n")
            fh.write("\n")


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Penn WSJ HMM POS Tagger (smoothed + stabilized).")
    parser.add_argument("--train", type=Path, required=True, help="Tagged training corpus")
    parser.add_argument("--test", type=Path, required=True, help="Untagged corpus to tag")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the tagged output corpus")
    parser.add_argument("--mode", choices=["baseline", "viterbi"], default="viterbi",
                        help="Tagging strategy to use.")
    parser.add_argument("--class-alpha", type=float, default=CLASS_ALPHA,
                        help="Smoothing strength for hapax class distributions.")
    parser.add_argument("--class-backoff", type=float, default=CLASS_BACKOFF,
                        help="Backoff weight controlling class/global interpolation for OOVs.")
    parser.add_argument("--transition-alpha", type=float, default=TRANSITION_ALPHA,
                        help="Additive smoothing applied to tag transition probabilities.")
    parser.add_argument("--emission-alpha", type=float, default=EMISSION_ALPHA,
                        help="Additive smoothing applied to emissions for ALL seen words.")
    return parser


def run_baseline(train_path: Path, test_path: Path, output_path: Path) -> None:
    tagged_sentences = read_tagged_corpus(train_path)
    word_sequences = read_word_sequences(test_path)
    params = train_hmm_parameters(tagged_sentences)
    tagged_output = tag_frequency_baseline(word_sequences, params)
    write_tagged_corpus(tagged_output, output_path)


def run_viterbi(train_path: Path, test_path: Path, output_path: Path,
                class_alpha: float, class_backoff: float, transition_alpha: float,
                emission_alpha: float) -> None:
    global CLASS_ALPHA, CLASS_BACKOFF, TRANSITION_ALPHA, EMISSION_ALPHA
    CLASS_ALPHA = class_alpha
    CLASS_BACKOFF = class_backoff
    TRANSITION_ALPHA = transition_alpha
    EMISSION_ALPHA = emission_alpha
    tagged_sentences = read_tagged_corpus(train_path)
    word_sequences = read_word_sequences(test_path)
    params = train_hmm_parameters(tagged_sentences)
    tagged_output = viterbi_tag(word_sequences, params)
    write_tagged_corpus(tagged_output, output_path)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "baseline":
        run_baseline(args.train, args.test, args.output)
    else:
        run_viterbi(
            args.train, args.test, args.output,
            class_alpha=args.class_alpha,
            class_backoff=args.class_backoff,
            transition_alpha=args.transition_alpha,
            emission_alpha=args.emission_alpha,
        )


if __name__ == "__main__":
    main()
```
