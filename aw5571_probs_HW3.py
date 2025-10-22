# aw5571_probs_HW3.py
# Step 3: convert raw counts -> Laplace-smoothed log-probabilities

import math
from aw5571_train_HMM_HW3 import load_pos_counts, BEGIN, END

def make_log_prob_tables(pos_path, k_trans=1.0, k_emit=1.0):
    """Turn counts into log probabilities with Laplace (add-k) smoothing."""
    emit, trans, tag_count, vocab, hapax, num_sent = load_pos_counts(pos_path)

    tags = list(tag_count.keys())
    V = len(vocab) + 1   # +1 for <UNK>
    T = len(tags)

    # Transition probabilities:  P(tag2 | tag1)
    logA = {}
    for t1 in tags:
        row = {}
        denom = tag_count[t1] + k_trans * T
        for t2 in tags:
            num = trans[t1][t2] + k_trans
            row[t2] = math.log(num / denom)
        logA[t1] = row

    # Emission probabilities:  P(word | tag)
    logB = {}
    for tag in tags:
        row = {}
        denom = tag_count[tag] + k_emit * V
        for w, c in emit[tag].items():
            row[w] = math.log((c + k_emit) / denom)
        row["<UNK>"] = math.log(k_emit / denom)  # baseline for OOVs
        logB[tag] = row

    return tags, logA, logB, vocab

def preview_tables(pos_path):
    tags, logA, logB, vocab = make_log_prob_tables(pos_path)
    print(f"Tags: {len(tags)}  |  Vocab: {len(vocab)}")
    print("\nSample transition log-probs from Begin_Sent:")
    for t, lp in list(logA["Begin_Sent"].items())[:5]:
        print(f"  Begin_Sent → {t:5s} : {lp:.4f}")
    print("\nSample emission log-probs for NN:")
    for w, lp in list(logB["NN"].items())[:5]:
        print(f"  {w:10s} | NN : {lp:.4f}")
    print("UNK example:", logB["NN"]["<UNK>"])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 aw5571_probs_HW3.py <train.pos>")
        sys.exit(1)
    preview_tables(sys.argv[1])
