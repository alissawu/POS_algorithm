# aw5571_viterbi_unk_HW3.py
# Viterbi that uses the hapax-based UNK emissions

import sys
from aw5571_probs_unk_HW3 import make_log_prob_tables_hapax, emit_logprob_with_unk
from aw5571_train_HMM_HW3 import BEGIN, END

def viterbi_tag_sentence(words, tags, logA, logB):
    inner = [t for t in tags if t not in (BEGIN, END)]
    if not words: return []

    V = [{} for _ in range(len(words))]
    BP = [{} for _ in range(len(words))]

    # init
    w0 = words[0]
    for t in inner:
        V[0][t] = logA[BEGIN][t] + emit_logprob_with_unk(logB, t, w0)
        BP[0][t] = BEGIN

    # forward
    for i in range(1, len(words)):
        w = words[i]
        for t in inner:
            best_s, best_p = -1e18, None
            e = emit_logprob_with_unk(logB, t, w)
            for p in inner:
                prev = V[i-1].get(p)
                if prev is None: continue
                s = prev + logA[p][t] + e
                if s > best_s:
                    best_s, best_p = s, p
            V[i][t] = best_s
            BP[i][t] = best_p

    # end
    best_last, best_s = None, -1e18
    for t in inner:
        s = V[-1].get(t, -1e18) + logA[t][END]
        if s > best_s:
            best_s, best_last = s, t

    # backtrack
    seq = [None]*len(words)
    seq[-1] = best_last if best_last is not None else inner[0]
    for i in range(len(words)-1, 0, -1):
        prev = BP[i][seq[i]]
        seq[i-1] = prev if prev is not None else inner[0]
    return seq

def tag_file(train_pos_path, in_words_path, out_pos_path):
    tags, logA, logB, _ = make_log_prob_tables_hapax(train_pos_path, k_trans=0.1, k_emit=0.1)
    with open(in_words_path, "r", encoding="utf-8") as fin, \
         open(out_pos_path, "w", encoding="utf-8") as fout:
        sent = []
        for raw in fin:
            line = raw.rstrip("\n")
            if line == "":
                if sent:
                    seq = viterbi_tag_sentence(sent, tags, logA, logB)
                    for w, t in zip(sent, seq):
                        fout.write(f"{w}\t{t}\n")
                    fout.write("\n")
                    sent = []
                else:
                    fout.write("\n")
                continue
            sent.append(line)
        if sent:
            seq = viterbi_tag_sentence(sent, tags, logA, logB)
            for w, t in zip(sent, seq):
                fout.write(f"{w}\t{t}\n")
            fout.write("\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 aw5571_viterbi_unk_HW3.py <train.pos> <in.words> <out.pos>")
        sys.exit(1)
    tag_file(sys.argv[1], sys.argv[2], sys.argv[3])
