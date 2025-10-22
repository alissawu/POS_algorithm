# aw5571_viterbi_HW3.py
# Step 4: Viterbi decode .words → .pos using the tables from Step 3

import sys, math
from aw5571_probs_HW3 import make_log_prob_tables
from aw5571_train_HMM_HW3 import BEGIN, END

def emit_logprob(logB, tag, word):
    row = logB.get(tag, {})
    # fallback to learned <UNK> for OOVs
    return row.get(word, row.get("<UNK>", -1e9))

def viterbi_tag_sentence(words, tags, logA, logB):
    """Return best tag sequence (list[str]) for a tokenized sentence."""
    inner = [t for t in tags if t not in (BEGIN, END)]
    if not words:
        return []

    # Viterbi DP tables
    V = [{} for _ in range(len(words))]
    BP = [{} for _ in range(len(words))]

    # init (from BEGIN)
    w0 = words[0]
    for t in inner:
        V[0][t] = logA[BEGIN][t] + emit_logprob(logB, t, w0)
        BP[0][t] = BEGIN

    # forward
    for i in range(1, len(words)):
        w = words[i]
        for t in inner:
            best_score, best_prev = -1e18, None
            e = emit_logprob(logB, t, w)
            for p in inner:  # previous tag
                prev_score = V[i-1].get(p)
                if prev_score is None:
                    continue
                s = prev_score + logA[p][t] + e
                if s > best_score:
                    best_score, best_prev = s, p
            V[i][t] = best_score
            BP[i][t] = best_prev

    # end (to END)
    best_last, best_score = None, -1e18
    for t in inner:
        s = V[-1].get(t, -1e18) + logA[t][END]
        if s > best_score:
            best_score, best_last = s, t

    # backtrack
    seq = [None] * len(words)
    seq[-1] = best_last if best_last is not None else inner[0]
    for i in range(len(words) - 1, 0, -1):
        seq[i-1] = BP[i][seq[i]] or inner[0]
    return seq

def tag_file(train_pos_path, in_words_path, out_pos_path):
    tags, logA, logB, vocab = make_log_prob_tables(train_pos_path, k_trans=1.0, k_emit=1.0)

    with open(in_words_path, "r", encoding="utf-8") as fin, \
         open(out_pos_path, "w", encoding="utf-8") as fout:
        sent_tokens = []
        for raw in fin:
            line = raw.rstrip("\n")
            if line == "":  # sentence boundary
                if sent_tokens:
                    tags_seq = viterbi_tag_sentence(sent_tokens, tags, logA, logB)
                    for w, t in zip(sent_tokens, tags_seq):
                        fout.write(f"{w}\t{t}\n")
                    fout.write("\n")
                    sent_tokens = []
                else:
                    # preserve empty line (robustness)
                    fout.write("\n")
                continue

            tok = line
            # we keep raw tok for output; emissions use <UNK> internally if unseen
            sent_tokens.append(tok)

        # flush last sentence if file didn't end with a blank line
        if sent_tokens:
            tags_seq = viterbi_tag_sentence(sent_tokens, tags, logA, logB)
            for w, t in zip(sent_tokens, tags_seq):
                fout.write(f"{w}\t{t}\n")
            fout.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 aw5571_viterbi_HW3.py <train.pos> <in.words> <out.pos>")
        sys.exit(1)
    tag_file(sys.argv[1], sys.argv[2], sys.argv[3])
