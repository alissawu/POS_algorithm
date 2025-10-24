
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Sentence = List[Tuple[str, str]]
TokenSequence = List[str]
START_TAG = "<s>"
END_TAG = "</s>"
TRANSITION_ALPHA = 0.1
CLASS_ALPHA = 0.5
CLASS_BACKOFF = 5.0
LOW_FREQ_BLEND_THRESHOLD = 2
LOW_FREQ_BLEND_WEIGHT = 0.7
TRIGRAM_BACKOFF = 2.0 # from 3 to 2, improve by 0.02%
BIGRAM_BACKOFF = 1.0

SUFFIX_CLASSES = [
    "ness",
    "able",
    "ible",
    "tion",
    "sion",
    "ment",
    "ship",
    "hood",
    "ity",
    "ous",
    "ive",
    "est",
    "ies",
    "ing",
    "ed",
    "es",
    "s",
    "ly",
    "er",

]

ADJ_SUFFIXES = (
    "ic",
    "ive",
)

NOUN_SUFFIXES = ()


VERB_TAGS = {"VB", "VBD", "VBN", "VBP", "VBZ", "MD"}
DETERMINER_TAGS = {"DT", "PDT", "PRP$"}
CAPITAL_TAGS = {"NNP", "NNPS"}
PUNCT_TAGS = {".", ",", ":", "``", "''", "-LRB-", "-RRB-", "$", "#"}


@dataclass
class HMMParameters:
    tags: List[str]
    vocabulary: set[str]
    transition_probs: Dict[str, Dict[str, float]]
    emission_probs: Dict[str, Dict[str, float]]
    tag_priors: Dict[str, float]
    tag_counts: Dict[str, int]
    emission_counts: Dict[str, Counter[str]]
    transition_counts: Dict[str, Counter[str]]
    word_tag_counts: Dict[str, Counter[str]]
    default_tag: str
    log_transition_probs: Dict[str, Dict[str, float]]
    log_emission_probs: Dict[str, Dict[str, float]]
    log_tag_priors: Dict[str, float]
    unk_log_emissions: Dict[str, float]
    hapax_tag_probs: Dict[str, float]
    hapax_class_probs: Dict[str, Dict[str, float]]
    hapax_class_totals: Dict[str, int]
    trigram_counts: Dict[Tuple[str, str], Counter[str]]
    trigram_totals: Dict[Tuple[str, str], int]
    word_frequencies: Dict[str, int]
    trigram_lambda_weights: Dict[Tuple[str, str], Tuple[float, float, float]]
    bigram_totals: Dict[str, int]
    lambda_trigram: float
    lambda_bigram: float
    lambda_unigram: float
    sentence_count: int
    end_unigram_prob: float


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
                # some files use spaces instead of tabs; fall back gracefully.
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


def classify_hapax(word: str) -> str:
    # Revert to earlier multi-feature signature (worked better on this corpus)
    lower = word.lower()
    features: List[str] = []

    if re.fullmatch(r"\d+(?:\.\d+)?", lower) or re.fullmatch(r"\d+(?:-\d+)?/\d+", lower):
        features.append("NUMERIC")
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", lower):
        features.append("NUMERIC")
    elif lower.endswith("%") and lower[:-1].replace(".", "", 1).isdigit():
        features.append("NUMERIC")
    elif re.fullmatch(r"\d{2,4}s", lower) or re.fullmatch(r"(?:early|mid|late)-\d{4}s", lower):
        features.append("NUMERIC")
    elif any(ch.isdigit() for ch in word):
        features.append("ALNUM")

    if word and re.fullmatch(r"[^\w]+", word):
        features.append("PUNCT")

    if "-" in word:
        features.append("HYPHEN")

    if word.isupper() and any(ch.isalpha() for ch in word):
        features.append("ALLCAP")
    elif word and word[0].isupper():
        features.append("INITCAP")

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


# (no independent OOV feature list; using single-class signatures)


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
        or re.fullmatch(r"\d+(?:-\d+)?/\d+", lower)
        or re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", lower)
        or (lower.endswith("%") and lower[:-1].replace(".", "", 1).isdigit())
        or re.fullmatch(r"\d{2,4}s", lower)
        or re.fullmatch(r"(?:early|mid|late)-\d{4}s", lower)
    )


def is_symbol_token(word: str) -> bool:
    return bool(word) and bool(re.fullmatch(r"[^\w]+", word))


def train_hmm_parameters(sentences: Sequence[Sentence]) -> HMMParameters:
    emission_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    transition_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    word_tag_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    tag_counts: Counter[str] = Counter()
    vocabulary: set[str] = set()
    word_counts: Counter[str] = Counter()
    trigram_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    trigram_lambda_weights: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    sentence_count = 0

    for sentence in sentences:
        sentence_count += 1
        prev_tag = START_TAG
        prev_prev_tag = START_TAG
        transition_counts[START_TAG]
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
    transition_probs: Dict[str, Dict[str, float]] = {}
    log_transition_probs: Dict[str, Dict[str, float]] = {}
    all_prev_tags = [START_TAG] + tag_list
    for prev_tag in all_prev_tags:
        counts = transition_counts.get(prev_tag, Counter())
        if prev_tag == START_TAG:
            next_tags = tag_list
        else:
            next_tags = tag_list + [END_TAG]
        denom = sum(counts.get(t, 0) for t in next_tags) + TRANSITION_ALPHA * len(next_tags)
        probs: Dict[str, float] = {}
        log_probs: Dict[str, float] = {}
        for next_tag in next_tags:
            prob = (counts.get(next_tag, 0) + TRANSITION_ALPHA) / denom
            probs[next_tag] = prob
            log_probs[next_tag] = math.log(prob)
        transition_probs[prev_tag] = probs
        log_transition_probs[prev_tag] = log_probs

    total_tags = sum(tag_counts.values())
    tag_priors = {tag: count / total_tags for tag, count in tag_counts.items()} if total_tags else {}
    default_tag = max(tag_priors.items(), key=lambda kv: kv[1])[0] if tag_priors else "NN"

    log_tag_priors = {tag: math.log(prob) if prob > 0.0 else float("-inf") for tag, prob in tag_priors.items()}
    hapax_tag_counts: Counter[str] = Counter()
    hapax_class_tag_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    hapax_class_totals: Counter[str] = Counter()
    for word, count in word_counts.items():
        if count == 1:
            tag = word_tag_counts[word].most_common(1)[0][0]
            hapax_tag_counts[tag] += 1
            cls = classify_hapax(word)
            hapax_class_totals[cls] += 1
            hapax_class_tag_counts[cls][tag] += 1

    hapax_total = sum(hapax_tag_counts.values())
    smoothing = 1e-6
    hapax_tag_probs: Dict[str, float] = {}
    for tag in tag_list:
        hapax_tag_probs[tag] = (
            (hapax_tag_counts.get(tag, 0.0) + smoothing)
            / (hapax_total + smoothing * len(tag_list))
            if hapax_total > 0
            else 1.0 / len(tag_list)
        )

    unk_log_emissions = {tag: math.log(prob) for tag, prob in hapax_tag_probs.items()}
    hapax_class_probs: Dict[str, Dict[str, float]] = {}
    for cls, total in hapax_class_totals.items():
        tag_counter = hapax_class_tag_counts[cls]
        denom = total + CLASS_ALPHA
        dist: Dict[str, float] = {}
        for tag in tag_list:
            numerator = tag_counter.get(tag, 0.0) + CLASS_ALPHA * hapax_tag_probs[tag]
            dist[tag] = numerator / denom if denom > 0 else hapax_tag_probs[tag]
        hapax_class_probs[cls] = dist

    # (no per-feature OOV distributions; using single-class signatures only)

    emission_probs: Dict[str, Dict[str, float]] = {}
    log_emission_probs: Dict[str, Dict[str, float]] = {}
    for tag, counts in emission_counts.items():
        total = tag_counts[tag]
        probs: Dict[str, float] = {}
        log_probs: Dict[str, float] = {}
        for word, count in counts.items():
            prob = count / total if total else 0.0
            if prob > 0.0:
                probs[word] = prob
                log_probs[word] = math.log(prob)
        emission_probs[tag] = probs
        log_emission_probs[tag] = log_probs

    trigram_totals: Dict[Tuple[str, str], int] = {
        context: sum(counter.values()) for context, counter in trigram_counts.items()
    }
    bigram_totals: Dict[str, int] = {
        prev: sum(counter.values()) for prev, counter in transition_counts.items()
    }

    lambda_trigram = 0.0
    lambda_bigram = 0.0
    lambda_unigram = 0.0
    total_tags_minus_one = max(total_tags - 1, 1)
    for (prev_prev_tag, prev_tag), counter in trigram_counts.items():
        trigram_den = trigram_totals.get((prev_prev_tag, prev_tag), 0) - 1
        bigram_den = bigram_totals.get(prev_tag, 0) - 1
        per_context_lambda = [0.0, 0.0, 0.0]
        for curr_tag, count in counter.items():
            trigram_num = max(count - 1, 0)
            trigram_prob = trigram_num / trigram_den if trigram_den > 0 else 0.0
            bigram_count = transition_counts.get(prev_tag, Counter()).get(curr_tag, 0)
            bigram_prob = (
                max(bigram_count - 1, 0) / bigram_den if bigram_den > 0 else 0.0
            )
            unigram_prob = (
                max(tag_counts[curr_tag] - 1, 0) / total_tags_minus_one if total_tags > 1 else 0.0
            )
            if trigram_prob >= bigram_prob and trigram_prob >= unigram_prob:
                lambda_trigram += count
                per_context_lambda[0] += count
            elif bigram_prob >= trigram_prob and bigram_prob >= unigram_prob:
                lambda_bigram += count
                per_context_lambda[1] += count
            else:
                lambda_unigram += count
                per_context_lambda[2] += count
        total = sum(per_context_lambda)
        if total > 0:
            trigram_lambda_weights[(prev_prev_tag, prev_tag)] = tuple(x / total for x in per_context_lambda)
        else:
            trigram_lambda_weights[(prev_prev_tag, prev_tag)] = (0.0, 0.0, 1.0)

    lambda_sum = lambda_trigram + lambda_bigram + lambda_unigram
    if lambda_sum == 0.0:
        lambda_trigram, lambda_bigram, lambda_unigram = 0.6, 0.3, 0.1
    else:
        lambda_trigram /= lambda_sum
        lambda_bigram /= lambda_sum
        lambda_unigram /= lambda_sum

    end_unigram_prob = (
        sentence_count / (sentence_count + total_tags)
        if (sentence_count + total_tags) > 0
        else 1.0
    )

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
        trigram_lambda_weights=trigram_lambda_weights,
        word_frequencies=dict(word_counts),
        bigram_totals=bigram_totals,
        lambda_trigram=lambda_trigram,
        lambda_bigram=lambda_bigram,
        lambda_unigram=lambda_unigram,
        sentence_count=sentence_count,
        end_unigram_prob=end_unigram_prob,
    )


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


def known_word_bonus(word: str, tag: str, params: HMMParameters) -> float:
    # TEST: Re-enable known word bonus
    bonus = 0.0
    lower = word.lower()
    if is_numeric_token(word):
        if tag == "CD":
            bonus += math.log(1.6)
        elif tag not in {"LS", "JJ"}:
            bonus += math.log(0.4)
    if is_symbol_token(word):
        if tag in PUNCT_TAGS:
            bonus += math.log(1.6)
        else:
            bonus += math.log(0.3)
    # Decade-like tokens are typically tagged as CD (e.g., 1960s, mid-1980s)
    if re.fullmatch(r"\d{2,4}s", word.lower()) or re.fullmatch(r"(?:early|mid|late)-\d{4}s", word.lower()):
        if tag == "CD":
            bonus += math.log(1.3)
        elif tag in {"NNS", "NNPS"}:
            bonus += math.log(0.9)
    if word[:1].isupper() and not word.isupper():
        if tag in CAPITAL_TAGS:
            bonus += math.log(1.2)
        elif tag in {"NN", "JJ"}:
            bonus += math.log(0.85)
    if word.isupper() and len(word) > 1:
        pass  # allcaps bonus removed (no measurable effect)
    if lower.endswith("ly"):
        if tag == "RB":
            bonus += math.log(1.25)
        elif tag in {"NN", "JJ", "VB"}:
            bonus += math.log(0.7)
    if tag == "JJ":
        if is_adjective_like(word):
            bonus += math.log(1.15)
        else:
            bonus += math.log(0.85)
    # removed: noun-like suffix bonus (no measurable effect)
    if tag == "NNPS" and word[:1].isupper() and word.endswith("s"):
        bonus += math.log(1.2)
    counts = params.word_tag_counts.get(word)
    if counts:
        total = sum(counts.values())
        if total >= 3:
            best_tag, best_count = counts.most_common(1)[0]
            best_freq = best_count / total
            freq = counts.get(tag, 0) / total
            if tag == best_tag and best_freq >= 0.50:  # Lowered threshold slightly
                bonus += math.log(1.0 + 1.2 * max(0.0, best_freq - 0.50))  # Stronger bonus
            elif tag != best_tag and freq <= 0.10:  # Slightly higher threshold
                bonus += math.log(0.75)  # Stronger penalty
    return bonus


def blend_low_freq_emission(word: str, tag: str, base_log: float, params: HMMParameters) -> float:
    if base_log == float("-inf"):
        return base_log
    freq = params.word_frequencies.get(word)
    if freq is None or freq > LOW_FREQ_BLEND_THRESHOLD:
        return base_log
    base_prob = math.exp(base_log)
    word_class = classify_hapax(word)
    class_probs = params.hapax_class_probs.get(word_class)
    if class_probs:
        class_prob = class_probs.get(tag, params.hapax_tag_probs.get(tag, 0.0))
    else:
        class_prob = params.hapax_tag_probs.get(tag, 0.0)
    prob = LOW_FREQ_BLEND_WEIGHT * base_prob + (1.0 - LOW_FREQ_BLEND_WEIGHT) * class_prob
    return math.log(prob) if prob > 0.0 else float("-inf")


def context_bonus(prev_prev: str, prev: str, curr: str, word: str, idx: int, words: TokenSequence, params: HMMParameters) -> float:
    # TEST: Only sentence-initial capitalization handling
    bonus = 0.0
    lower = word.lower()

    # Handle sentence-initial capitalization (idx == 0 means first word of sentence)
    if idx == 0 and word[0].isupper() and len(word) > 1 and not word.isupper():
        # Check if lowercase version exists in vocabulary
        if lower in params.vocabulary and lower != word:
            # The word exists in lowercase - check what it usually is
            lower_tags = params.word_tag_counts.get(lower, Counter())
            if lower_tags:
                most_common_tag = lower_tags.most_common(1)[0][0]
                total_count = sum(lower_tags.values())
                tag_freq = lower_tags[most_common_tag] / total_count

                # If lowercase version is commonly a non-proper noun tag
                if most_common_tag not in CAPITAL_TAGS and tag_freq > 0.6:
                    if curr in CAPITAL_TAGS:  # NNP, NNPS
                        bonus += math.log(0.3)  # Strong penalty for treating as proper noun
                    elif curr == most_common_tag:
                        bonus += math.log(1.4)  # Boost the expected tag

    # (no mid-sentence capitalization adjustment; initial-only rule retained)

    # Handle NNPS vs NNP - pluralized proper nouns
    if word.endswith("s") and word[0].isupper() and len(word) > 2:
        stem = word[:-1]
        if stem in params.vocabulary:
            stem_tags = params.word_tag_counts.get(stem, Counter())
            if stem_tags:
                stem_total = sum(stem_tags.values())
                nnp_count = stem_tags.get("NNP", 0)
                if nnp_count > 0 and nnp_count / stem_total > 0.5:
                    if curr == "NNPS":
                        bonus += math.log(1.5)  # Strong boost for NNPS
                    elif curr == "NNP":
                        bonus += math.log(0.6)  # Penalty for singular when plural expected
        # If followed by another capitalized token (e.g., "Farmers Group"), prefer NNPS
        if idx + 1 < len(words) and words[idx + 1][:1].isupper():
            if curr == "NNPS":
                bonus += math.log(1.2)
            elif curr == "NNP":
                bonus += math.log(0.9)

    # Fix for DT context: 
    if prev in DETERMINER_TAGS:
        if curr == "NN":
            bonus += math.log(1.1)  # Boost NN after determiners
        elif curr == "JJ" and not is_adjective_like(word):
            bonus += math.log(0.7)  # Penalty for non-adjective words as JJ
        # Strengthen when the following token is likely a noun head
        if idx + 1 < len(words):
            next_word = words[idx + 1]
            log_emit = params.log_emission_probs
            nn_log = max(
                log_emit.get("NN", {}).get(next_word, float("-inf")),
                log_emit.get("NNS", {}).get(next_word, float("-inf")),
            )
            jj_next = log_emit.get("JJ", {}).get(next_word, float("-inf"))
            if nn_log != float("-inf") and (jj_next == float("-inf") or (nn_log - jj_next) > math.log(1.2)):
                if curr == "NN":
                    bonus += math.log(1.05)
                elif curr == "JJ" and not is_adjective_like(word):
                    bonus += math.log(0.95)
            # (no extra punctuation-based bias beyond sentence-final case)
        else:
            pass  # no special sentence-final bias

    # Fix for VBD/VBN confusion: 
    if lower.endswith("ed") and prev in {"NN", "NNS", "PRP", "RB", "CC", ","}:
        if curr == "VBD":
            bonus += math.log(1.3)  # After subjects, -ed words are usually past tense VBD
        elif curr == "VBN":
            bonus += math.log(0.7)  # Penalty for past participle after subjects

    # Auxiliary + past participle: after auxiliaries/modals, prefer VBN over VBD
    if prev in {"VB", "VBP", "VBZ", "VBD", "MD"}:
        if lower.endswith("ed") or lower.endswith("en") or lower.endswith("wn") or lower.endswith("rn"):
            if curr == "VBN":
                bonus += math.log(1.2)
            elif curr == "VBD":
                bonus += math.log(0.9)

    # (no reduced participle special-case beyond auxiliary rule)

    # (no -ing nominalization heuristic; kept minimal)

    # (no generic VBZ vs NNS heuristic beyond transitions)

    # Compound noun heuristic: hyphenated modifier before a likely noun
    # If current token contains a hyphen and the next word is strongly noun-like
    # (based on emission likelihoods for NN/NNS), gently prefer NN over JJ.
    if "-" in word and idx + 1 < len(words):
        next_word = words[idx + 1]
        log_emit = params.log_emission_probs
        nn_log = max(
            log_emit.get("NN", {}).get(next_word, float("-inf")),
            log_emit.get("NNS", {}).get(next_word, float("-inf")),
        )
        jj_log = log_emit.get("JJ", {}).get(next_word, float("-inf"))
        # Require that NN/NNS emission exists and is stronger than JJ by a solid margin
        if nn_log != float("-inf") and (jj_log == float("-inf") or (nn_log - jj_log) > math.log(1.7)):
            if curr == "NN":
                bonus += math.log(1.15)
            elif curr == "JJ" and not is_adjective_like(word):
                bonus += math.log(0.9)

    # Preposition before numeric: prefer IN when the following token is numeric-like
    if idx + 1 < len(words):
        nxt = words[idx + 1]
        # Reuse numeric detector from known_word_bonus
        if is_numeric_token(nxt):
            if curr == "IN":
                bonus += math.log(1.25)
            elif curr == "RB":
                bonus += math.log(0.85)

    # (no particle heuristic; minimalism)

    # (no extra after-comma determiner bias)

    # Short lowercase token after a noun/number, likely preposition if next looks like NP start or punctuation
    if word.islower() and word.isalpha() and 2 <= len(word) <= 4 and prev in {"NN", "NNS", "CD"} and idx + 1 < len(words):
        nxt = words[idx + 1]
        log_emit = params.log_emission_probs
        dt_log = log_emit.get("DT", {}).get(nxt, float("-inf"))
        nn_log = max(log_emit.get("NN", {}).get(nxt, float("-inf")), log_emit.get("NNS", {}).get(nxt, float("-inf")))
        looks_np_start = (dt_log != float("-inf")) or (nn_log != float("-inf")) or is_numeric_token(nxt) or is_symbol_token(nxt)
        if looks_np_start:
            if curr == "IN":
                bonus += math.log(1.15)
            elif curr == "RB":
                bonus += math.log(0.95)
            elif curr == "RP":
                bonus += math.log(0.98)

    # (no additional preposition chain heuristic)

    # (no additional general short-lowercase preposition heuristic)

    # Compound noun heuristic: non-hyphen modifier before a likely noun
    # If the next word is strongly noun-like and current word isn't adjective-like,
    # nudge NN over JJ for the current token. Helps cases like "desktop computer".
    if idx + 1 < len(words):
        next_word = words[idx + 1]
        log_emit = params.log_emission_probs
        nn_log = max(
            log_emit.get("NN", {}).get(next_word, float("-inf")),
            log_emit.get("NNS", {}).get(next_word, float("-inf")),
        )
        jj_next = log_emit.get("JJ", {}).get(next_word, float("-inf"))
        if nn_log != float("-inf") and (jj_next == float("-inf") or (nn_log - jj_next) > math.log(1.3)):
            if word.islower() and word.isalpha() and not is_adjective_like(word):
                if curr == "NN":
                    bonus += math.log(1.08)
                elif curr == "JJ":
                    bonus += math.log(0.95)
        # (no triplet pattern adjustment; minimalism)
        # (no additional triplet pattern adjustment)
        # -ed before clear noun head: prefer VBN over JJ (attributive participles)
        if lower.endswith("ed") and nn_log != float("-inf") and (jj_next == float("-inf") or (nn_log - jj_next) > math.log(1.2)):
            # only in nominal contexts
            if prev in DETERMINER_TAGS or prev in {"NN", "NNS", "NNP", "JJ"}:
                if curr == "VBN":
                    bonus += math.log(1.15)
                elif curr == "JJ":
                    bonus += math.log(0.94)
        # -en/-wn/-rn before clear noun head: also prefer VBN over JJ (e.g., broken glass)
        if (lower.endswith("en") or lower.endswith("wn") or lower.endswith("rn")) and nn_log != float("-inf") and (jj_next == float("-inf") or (nn_log - jj_next) > math.log(1.2)):
            if prev in DETERMINER_TAGS or prev in {"NN", "NNS", "NNP", "JJ"}:
                if curr == "VBN":
                    bonus += math.log(1.10)
                elif curr == "JJ":
                    bonus += math.log(0.97)

    # (no additional -ed narrowing beyond existing rules)

    # (no mid-sentence capitalization demotion rule)

    return bonus


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
    trigram_lambda_weights = params.trigram_lambda_weights
    transition_probs = params.transition_probs
    lambda_trigram = params.lambda_trigram
    lambda_bigram = params.lambda_bigram
    lambda_unigram = params.lambda_unigram
    end_unigram_prob = params.end_unigram_prob
    tag_priors = params.tag_priors
    default_tag = params.default_tag
    neg_inf = float("-inf")

    def oov_log_prob(oov_class: str | None, tag: str) -> float:
        global_prob = global_hapax.get(tag, 0.0)
        if oov_class and oov_class in hapax_class_probs:
            class_probs = hapax_class_probs[oov_class]
            class_total = hapax_class_totals.get(oov_class, 0)
            mix = class_total / (class_total + CLASS_BACKOFF) if CLASS_BACKOFF > 0 else 1.0
            class_prob = class_probs.get(tag, 0.0)
            prob = mix * class_prob + (1.0 - mix) * global_prob
        else:
            prob = global_prob
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
        if prob <= 0.0:
            return neg_inf
        return math.log(prob)

    def transition_log(prev_prev: str, prev: str, curr: str) -> float:
        trigram_total = trigram_totals.get((prev_prev, prev), 0)
        trigram_prob = (
            trigram_counts.get((prev_prev, prev), {}).get(curr, 0) / trigram_total
            if trigram_total > 0
            else 0.0
        )
        bigram_prob = transition_probs.get(prev, {}).get(curr, 0.0)
        if curr == END_TAG:
            unigram_prob = end_unigram_prob
        else:
            unigram_prob = tag_priors.get(curr, 0.0)
        lambdas = trigram_lambda_weights.get((prev_prev, prev))
        if lambdas:
            lt, lb, lu = lambdas
        else:
            lt, lb, lu = lambda_trigram, lambda_bigram, lambda_unigram
        prob = lt * trigram_prob + lb * bigram_prob + lu * unigram_prob
        return math.log(prob) if prob > 0.0 else neg_inf

    for words in word_sequences:
        if not words:
            tagged_sentences.append([])
            continue
        viterbi: List[Dict[Tuple[str, str], float]] = []
        backpointer: List[Dict[Tuple[str, str], Tuple[str, str] | None]] = []

        first_word = words[0]
        first_class = None if first_word in params.vocabulary else classify_hapax(first_word)
        first_scores: Dict[Tuple[str, str], float] = {}
        first_back: Dict[Tuple[str, str], Tuple[str, str] | None] = {}
        for curr in states:
            trans_score = transition_log(START_TAG, START_TAG, curr)
            if first_word in params.vocabulary:
                emit_score = log_emit.get(curr, {}).get(first_word, neg_inf)
                if emit_score != neg_inf:
                    emit_score = blend_low_freq_emission(first_word, curr, emit_score, params)
                    emit_score += known_word_bonus(first_word, curr, params)
            else:
                emit_score = oov_log_prob(first_class, curr)
            if trans_score == neg_inf or emit_score == neg_inf:
                continue
            bonus = context_bonus(START_TAG, START_TAG, curr, first_word, 0, words, params)
            first_scores[(START_TAG, curr)] = trans_score + emit_score + bonus
            first_back[(START_TAG, curr)] = None
        if not first_scores:
            freq_tag = (
                params.word_tag_counts[first_word].most_common(1)[0][0]
                if first_word in params.vocabulary
                else default_tag
            )
            first_scores[(START_TAG, freq_tag)] = params.log_tag_priors.get(freq_tag, 0.0)
            first_back[(START_TAG, freq_tag)] = None
        viterbi.append(first_scores)
        backpointer.append(first_back)

        for idx in range(1, len(words)):
            word = words[idx]
            word_class = None if word in params.vocabulary else classify_hapax(word)
            prev_column = viterbi[idx - 1]
            curr_scores: Dict[Tuple[str, str], float] = {}
            curr_back: Dict[Tuple[str, str], Tuple[str, str] | None] = {}
            for (prev_prev, prev), prev_score in prev_column.items():
                if prev_score == neg_inf:
                    continue
                for curr in states:
                    if word in params.vocabulary:
                        emit_score = log_emit.get(curr, {}).get(word, neg_inf)
                        if emit_score != neg_inf:
                            emit_score = blend_low_freq_emission(word, curr, emit_score, params)
                            emit_score += known_word_bonus(word, curr, params)
                    else:
                        emit_score = oov_log_prob(word_class, curr)
                    if emit_score == neg_inf:
                        continue
                    trans_score = transition_log(prev_prev, prev, curr)
                    if trans_score == neg_inf:
                        continue
                    bonus = context_bonus(prev_prev, prev, curr, word, idx, words, params)
                    candidate = prev_score + trans_score + emit_score + bonus
                    key = (prev, curr)
                    if candidate > curr_scores.get(key, neg_inf):
                        curr_scores[key] = candidate
                        curr_back[key] = (prev_prev, prev)
            if not curr_scores:
                freq_tag = (
                    params.word_tag_counts[word].most_common(1)[0][0]
                    if word in params.vocabulary
                    else default_tag
                )
                best_prev_pair, best_prev_score = max(
                    prev_column.items(),
                    key=lambda kv: kv[1],
                    default=((START_TAG, START_TAG), neg_inf),
                )
                prev_prev, prev_state = best_prev_pair
                trans_score = transition_log(prev_prev, prev_state, freq_tag)
                if word in params.vocabulary:
                    emit_score = log_emit.get(freq_tag, {}).get(word, neg_inf)
                    if emit_score != neg_inf:
                        emit_score = blend_low_freq_emission(word, freq_tag, emit_score, params)
                        emit_score += known_word_bonus(word, freq_tag, params)
                else:
                    emit_score = oov_log_prob(word_class, freq_tag)
                if trans_score == neg_inf or emit_score == neg_inf:
                    total = best_prev_score
                else:
                    total = best_prev_score + trans_score + emit_score
                curr_scores[(prev_state, freq_tag)] = total
                curr_back[(prev_state, freq_tag)] = best_prev_pair
            viterbi.append(curr_scores)
            backpointer.append(curr_back)

        last_index = len(words) - 1
        best_score = neg_inf
        best_pair: Tuple[str, str] | None = None
        for pair, score in viterbi[last_index].items():
            if score == neg_inf:
                continue
            prev_prev, prev = pair
            trans_score = transition_log(prev_prev, prev, END_TAG)
            if trans_score == neg_inf:
                continue
            candidate = score + trans_score
            if candidate > best_score:
                best_score = candidate
                best_pair = pair
        if best_pair is None:
            best_pair = max(
                viterbi[last_index].items(), key=lambda kv: kv[1], default=((START_TAG, default_tag), 0.0)
            )[0]

        idx = len(words) - 1
        pair = best_pair
        tags_rev: List[str] = []
        while pair is not None and idx >= 0:
            prev_tag, curr_tag = pair
            tags_rev.append(curr_tag)
            pair = backpointer[idx].get(pair)
            idx -= 1
        predicted_tags = list(reversed(tags_rev))
        if len(predicted_tags) != len(words):
            predicted_tags = [default_tag] * len(words)

        tagged_sentences.append(list(zip(words, predicted_tags)))
    return tagged_sentences


def write_tagged_corpus(sentences: Iterable[Sentence], path: Path) -> None:
    with path.open("w", encoding="utf8") as fh:
        for sentence in sentences:
            for word, tag in sentence:
                fh.write(f"{word}\t{tag}\n")
            fh.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Components for HW3 POS tagging.")
    parser.add_argument("--train", type=Path, required=True, help="Tagged training corpus")
    parser.add_argument("--test", type=Path, required=True, help="Untagged corpus to tag")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the tagged output corpus",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "viterbi"],
        default="baseline",
        help="Tagging strategy to use.",
    )
    parser.add_argument(
        "--class-alpha",
        type=float,
        default=CLASS_ALPHA,
        help="Smoothing strength for hapax class distributions.",
    )
    parser.add_argument(
        "--class-backoff",
        type=float,
        default=CLASS_BACKOFF,
        help="Backoff weight controlling class/global interpolation for OOVs.",
    )
    parser.add_argument(
        "--transition-alpha",
        type=float,
        default=TRANSITION_ALPHA,
        help="Additive smoothing applied to tag transition probabilities.",
    )
    return parser


def run_baseline(train_path: Path, test_path: Path, output_path: Path) -> None:
    tagged_sentences = read_tagged_corpus(train_path)
    word_sequences = read_word_sequences(test_path)
    params = train_hmm_parameters(tagged_sentences)
    tagged_output = tag_frequency_baseline(word_sequences, params)
    write_tagged_corpus(tagged_output, output_path)


def run_viterbi(train_path: Path, test_path: Path, output_path: Path) -> None:
    tagged_sentences = read_tagged_corpus(train_path)
    word_sequences = read_word_sequences(test_path)
    params = train_hmm_parameters(tagged_sentences)
    tagged_output = viterbi_tag(word_sequences, params)
    write_tagged_corpus(tagged_output, output_path)


def main() -> None:
    args = build_arg_parser().parse_args()
    global CLASS_ALPHA, CLASS_BACKOFF, TRANSITION_ALPHA
    CLASS_ALPHA = args.class_alpha
    CLASS_BACKOFF = args.class_backoff
    TRANSITION_ALPHA = args.transition_alpha
    if args.mode == "baseline":
        run_baseline(args.train, args.test, args.output)
    elif args.mode == "viterbi":
        run_viterbi(args.train, args.test, args.output)
    else:
        raise NotImplementedError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
