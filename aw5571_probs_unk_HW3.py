# aw5571_probs_unk_HW3.py
# Train-time hapax -> UNK(class) + light Laplace (k=0.1)

import math, collections
from aw5571_train_HMM_HW3 import load_pos_counts, BEGIN, END
from aw5571_unk_utils_HW3 import word_to_unk_class, UNK_CLASSES

def _word_totals(emit):
    tot = collections.Counter()
    for tag, cnt in emit.items():
        for w, c in cnt.items():
            tot[w] += c
    return tot

def make_log_prob_tables_hapax(pos_path, k_trans=0.1, k_emit=0.1):
    """
    Returns: tags, logA, logB, vocab2
      - tags: all tags incl. BEGIN/END
      - logA: P(t2|t1) in log
      - logB: P(w|t) in log  (w includes UNK classes)
      - vocab2: set of observed>1 words ∪ UNK_CLASSES
    """
    emit_raw, trans, tag_count, vocab, hapax, _ = load_pos_counts(pos_path)
    tags = list(tag_count.keys())
    T = len(tags)

    # --- transitions ---
    logA = {}
    for t1 in tags:
        row = {}
        denom = tag_count[t1] + k_trans * T
        for t2 in tags:
            num = trans[t1][t2] + k_trans
            row[t2] = math.log(num / denom)
        logA[t1] = row

    # --- replace hapax with UNK class at training time ---
    totals = _word_totals(emit_raw)
    emit = {tag: collections.Counter() for tag in tags}
    for tag, cnt in emit_raw.items():
        for w, c in cnt.items():
            if totals[w] == 1:
                emit[tag][word_to_unk_class(w)] += c
            else:
                emit[tag][w] += c

    # vocab2 includes all seen>1 words plus all UNK classes
    vocab2 = set()
    for tag in tags:
        vocab2.update(emit[tag].keys())
    vocab2.update(UNK_CLASSES)
    V = len(vocab2)

    # --- emissions with light Laplace ---
    logB = {}
    for tag in tags:
        row = {}
        denom = tag_count[tag] + k_emit * V
        for w, c in emit[tag].items():
            row[w] = math.log((c + k_emit) / denom)
        # guarantee every w has at least k_emit mass
        for w in vocab2:
            if w not in row:
                row[w] = math.log(k_emit / denom)
        logB[tag] = row

    return tags, logA, logB, vocab2

def emit_logprob_with_unk(logB, tag, word):
    row = logB.get(tag, {})
    if word in row:
        return row[word]
    # unseen surface form: map to UNK class learned at train time
    return row.get(word_to_unk_class(word), -1e9)
