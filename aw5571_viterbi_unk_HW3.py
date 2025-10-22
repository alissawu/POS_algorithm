# aw5571_viterbi_unk_HW3.py
# Viterbi + small decode-time bonuses + interpolated transitions
# Targeted at: NN↔JJ/NNP, NNPS→NNP, IN↔RB/RP, VBD/VBN/VBG, and "that".

import sys, re
from aw5571_probs_unk_HW3 import (
    make_log_prob_tables_hapax,
    emit_logprob_with_unk,
    interp_log_transition,
)
from aw5571_train_HMM_HW3 import BEGIN, END

# Punctuation mapping consistent with PTB
PUNCT_TAG = {",": ",", ".": ".", "``": "``", "''": "''", ":": ":", "(": "(", ")": ")"}

_is_num  = re.compile(r'.*\d.*').match
_acronym = re.compile(r'^(?:[A-Z]\.){2,}$').match     # e.g., U.S., U.K.
_initcap = re.compile(r'^[A-Z][a-z].*').match

# Small closed-class word sets for gentle nudges (allowed by HW)
PARTICLES = {"up","down","out","off","back","in","over","away","through","around"}
CLOSED_IN = {"as"}  # very often IN in WSJ

HAVE_WORDS = {"have","has","had","having"}
BE_WORDS   = {"be","am","is","are","was","were","been","being"}

def emission_with_bonuses(logB, tag, word, seen_tags, pos_in_sent):
    base = emit_logprob_with_unk(logB, tag, word)

    # punctuation (deterministic)
    if word in PUNCT_TAG and tag == PUNCT_TAG[word]:
        return base + 5.0
    if word in (";","--") and tag == ":":
        base += 2.5

    # numbers -> CD
    if _is_num(word) and tag == "CD":
        base += 3.0

    # tag dictionary bias (from training) + small negative for unseen tags
    tags_seen = seen_tags.get(word)
    if tags_seen:
        if len(tags_seen) == 1 and tag in tags_seen:
            base += 2.0
        elif len(tags_seen) == 2 and tag in tags_seen:
            base += 0.5
        if tag not in tags_seen:
            base -= 1.0

    # sentence-initial: be careful with NNP (reduce BEGIN→NN→NNP errors)
    if pos_in_sent == 0 and tag == "NNP" and not _acronym(word):
        # only penalize if the word isn't uniquely NNP in training
        tags_seen = tags_seen or set()
        if not (len(tags_seen) == 1 and "NNP" in tags_seen):
            base -= 0.6

    # NNPS vs NNP: mid-sentence InitCap + endswith 's' (not "'s") -> NNPS
    if pos_in_sent > 0 and word.endswith("s") and not word.endswith("'s"):
        if _initcap(word) and tag == "NNPS":
            base += 3.0

    # 'as' is almost always IN in WSJ
    if word.lower() == "as" and tag == "IN":
        base += 2.0

    # acronyms & mid-sentence proper names -> NNP
    if _acronym(word) and tag == "NNP":
        base += 3.0
    if pos_in_sent > 0 and _initcap(word) and tag == "NNP":
        base += 1.2

    # -ly adverbs (tiny)
    if word.lower().endswith("ly") and tag == "RB":
        base += 1.0

    return base


def ctx_bonus(prev_tag, cur_tag, prev_word=None, cur_word=None):
    b = 0.0
    w = (cur_word or "").lower()
    pw = (prev_word or "").lower()

    # After TO or a modal, prefer base-form verb
    if prev_tag in ("TO", "MD") and cur_tag == "VB":
        b += 0.8

    # Verb-particle: after a verb/aux, particle words -> RP (slightly reduced)
    if prev_tag.startswith("VB") or prev_tag in ("MD","VBD","VBP","VBZ","VBN","VBG"):
        if w in PARTICLES and cur_tag == "RP":
            b += 1.2   # was 1.5

    # If not after a verb, particle words are usually IN (comma cases, etc.)
    if w in PARTICLES and cur_tag == "IN":
        if not (prev_tag.startswith("VB") or prev_tag in ("MD","VBD","VBP","VBZ","VBN","VBG")):
            b += 0.8

    # Comma + particle specifically favors IN (handles ", down", ", up")
    if prev_word == "," and w in PARTICLES and cur_tag == "IN":
        b += 1.0

    # Perfect/progressive nudges
    if pw in {"have","has","had","having"} and cur_tag == "VBN":
        b += 1.0
    if pw in {"be","am","is","are","was","were","been","being"} and cur_tag == "VBG":
        b += 1.0
    # tag-only fallback
    if prev_tag in ("VBD","VBP","VBZ") and cur_tag == "VBN":
        b += 0.6
    if prev_tag in ("VBD","VBP","VBZ") and cur_tag == "VBG":
        b += 0.6

    # After adverb, past tense VBD > VBN a bit (fix RB→VBD→VBN)
    if prev_tag == "RB":
        if cur_tag == "VBD":
            b += 0.4
        elif cur_tag == "VBN":
            b -= 0.3

    # "that": IN after verbs (complementizer), WDT after nouns/',' (relative)
    if w == "that":
        if prev_tag in ("VBD","VBP","VBZ","VB","MD") and cur_tag == "IN":
            b += 1.0
        if prev_tag in ("NN","NNS","NNP",",") and cur_tag == "WDT":
            b += 1.0

    return b


def viterbi_tag_sentence(words, tags, logA, logB, seen_tags, tag_priors, lam=0.9):
    inner = [t for t in tags if t not in (BEGIN, END)]
    if not words:
        return []

    V  = [{} for _ in range(len(words))]
    BP = [{} for _ in range(len(words))]

    # init (BEGIN -> t) + emission
    w0 = words[0]
    for t in inner:
        trans = interp_log_transition(BEGIN, t, logA, tag_priors, lam)
        V[0][t] = trans + emission_with_bonuses(logB, t, w0, seen_tags, pos_in_sent=0)
        BP[0][t] = BEGIN

    # forward
    for i in range(1, len(words)):
        w = words[i]
        pw = words[i-1] if i > 0 else None
        for t in inner:
            best_s, best_p = -1e18, None
            e = emission_with_bonuses(logB, t, w, seen_tags, pos_in_sent=i)
            for p in inner:
                prev = V[i-1].get(p)
                if prev is None:
                    continue
                trans = interp_log_transition(p, t, logA, tag_priors, lam)
                s = prev + trans + e + ctx_bonus(p, t, prev_word=pw, cur_word=w)
                if s > best_s:
                    best_s, best_p = s, p
            V[i][t] = best_s
            BP[i][t] = best_p

    # end (t -> END)
    best_last, best_s = None, -1e18
    for t in inner:
        s = V[-1].get(t, -1e18) + interp_log_transition(t, END, logA, tag_priors, lam)
        if s > best_s:
            best_s, best_last = s, t

    # backtrack
    seq = [None] * len(words)
    seq[-1] = best_last if best_last is not None else inner[0]
    for i in range(len(words) - 1, 0, -1):
        prev = BP[i][seq[i]]
        seq[i-1] = prev if prev is not None else inner[0]
    return seq

def tag_file(train_pos_path, in_words_path, out_pos_path):
    tags, logA, logB, _, seen_tags, tag_priors = make_log_prob_tables_hapax(
        train_pos_path, k_trans=0.1, k_emit=0.1
    )
    with open(in_words_path, "r", encoding="utf-8") as fin, \
         open(out_pos_path, "w", encoding="utf-8") as fout:
        sent = []
        for raw in fin:
            line = raw.rstrip("\n")
            if line == "":
                if sent:
                    seq = viterbi_tag_sentence(sent, tags, logA, logB, seen_tags, tag_priors, lam=0.9)
                    for w, t in zip(sent, seq):
                        fout.write(f"{w}\t{t}\n")
                    fout.write("\n")
                    sent = []
                else:
                    fout.write("\n")
                continue
            sent.append(line)
        if sent:
            seq = viterbi_tag_sentence(sent, tags, logA, logB, seen_tags, tag_priors, lam=0.9)
            for w, t in zip(sent, seq):
                fout.write(f"{w}\t{t}\n")
            fout.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 aw5571_viterbi_unk_HW3.py <train.pos> <in.words> <out.pos>")
        sys.exit(1)
    tag_file(sys.argv[1], sys.argv[2], sys.argv[3])
