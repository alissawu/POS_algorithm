# aw5571_probs_unk_HW3.py
# Train-time hapax -> UNK(class) + light Laplace (k) + transition interpolation.
import math, collections
from aw5571_train_HMM_HW3 import load_pos_counts, BEGIN, END
from aw5571_unk_utils_HW3 import word_to_unk_class, UNK_CLASSES

def _word_totals(emit):
    tot = collections.Counter()
    for tag, cnt in emit.items():
        for w, c in cnt.items():
            tot[w] += c
    return tot

def make_log_prob_tables_hapax(pos_path, k_trans=0.05, k_emit=0.05):
    """
    Returns: tags, logA, logB, vocab2, seen_tags, tag_priors
      - tags: all tags incl. BEGIN/END
      - logA: log P(t2|t1) (Laplace-smoothed & interpolated)
      - logB: log P(w|t) over (seen>1 words ∪ UNK classes)
      - vocab2: set of emission keys
      - seen_tags: word -> set(tags) from raw counts (pre-UNK)
      - tag_priors: unigram P(t)
    """
    emit_raw, trans, tag_count, vocab, hapax, _ = load_pos_counts(pos_path)
    tags = list(tag_count.keys())
    T = len(tags)

    # seen_tags from raw counts (pre-UNK), used as a learned tag dictionary
    seen_tags = collections.defaultdict(set)
    for tag, cnt in emit_raw.items():
        for w in cnt.keys():
            seen_tags[w].add(tag)

    # unigram tag priors
    total_tags = sum(tag_count.values())
    tag_priors = {t: (tag_count[t] / total_tags) for t in tags}

    # base transitions with Laplace
    logA_base = {}
    for t1 in tags:
        row = {}
        denom = tag_count[t1] + k_trans * T
        for t2 in tags:
            num = trans[t1][t2] + k_trans
            row[t2] = math.log(num / denom)
        logA_base[t1] = row

    # interpolate bigram with unigram: λ*P_bigram + (1-λ)*P_unigram
    def interp_logA(logA, priors, lam=0.95):
        out = {}
        for t1, row in logA.items():
            out_row = {}
            for t2, lp in row.items():
                p_big = math.exp(lp)
                p_uni = priors.get(t2, 1e-12)
                p = lam * p_big + (1.0 - lam) * p_uni
                if p <= 0: p = 1e-300
                out_row[t2] = math.log(p)
            out[t1] = out_row
        return out
    logA = interp_logA(logA_base, tag_priors, lam=0.95)

    # replace hapax with UNK class at training time
    totals = _word_totals(emit_raw)
    emit = {tag: collections.Counter() for tag in tags}
    for tag, cnt in emit_raw.items():
        for w, c in cnt.items():
            if totals[w] == 1:
                emit[tag][word_to_unk_class(w)] += c
            else:
                emit[tag][w] += c

    # vocab2 = all seen>1 words + UNK classes
    vocab2 = set()
    for tag in tags:
        vocab2.update(emit[tag].keys())
    vocab2.update(UNK_CLASSES)
    V = len(vocab2)

    # emissions with light Laplace
    logB = {}
    for tag in tags:
        row = {}
        denom = tag_count[tag] + k_emit * V
        for w, c in emit[tag].items():
            row[w] = math.log((c + k_emit) / denom)
        for w in vocab2:
            if w not in row:
                row[w] = math.log(k_emit / denom)
        logB[tag] = row

    return tags, logA, logB, vocab2, seen_tags, tag_priors

def emit_logprob_with_unk(logB, tag, word):
    row = logB.get(tag, {})
    if word in row:  # seen>1 word
        return row[word]
    # back off to its UNK class
    return row.get(word_to_unk_class(word), -1e9)

def interp_log_transition(prev_tag, cur_tag, logA, tag_priors, lam=0.95):
    p_big = math.exp(logA[prev_tag][cur_tag])
    p_uni = tag_priors.get(cur_tag, 1e-12)
    p = lam * p_big + (1.0 - lam) * p_uni
    if p <= 0: p = 1e-300
    return math.log(p)
