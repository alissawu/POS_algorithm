# aw5571_train_HMM_HW3.py
# emissions P(word|tag), transitions P(tag_i | tag_{i-1})
# BEGIN/END are synthetic tags used only for transitions.

import sys
import collections

BEGIN, END = "Begin_Sent", "End_Sent"

def stable_tag_order(tag_count):
    """
    Deterministic tag order: BEGIN first, END last, others by
    descending frequency then alphabetical for ties.
    """
    inner = [t for t in tag_count if t not in (BEGIN, END)]
    inner.sort(key=lambda t: (-tag_count[t], t))
    return [BEGIN] + inner + [END]

def load_pos_counts(pos_path):
    """
    Reads a Penn Treebank-style .pos file (token<TAB>tag, blank line between sentences)
    and returns:
      emit:        dict[tag][word] -> count
      trans:       dict[prev_tag][cur_tag] -> count
      tag_count:   Counter over tags (incl. BEGIN/END)
      vocab:       set of seen surface tokens
      hapax:       set of words seen exactly once (useful for OOV classes)
      num_sent:    number of sentences
    """
    emit = collections.defaultdict(collections.Counter)   # emissions
    trans = collections.defaultdict(collections.Counter)  # transitions
    tag_count = collections.Counter()
    word_count = collections.Counter()
    vocab = set()

    num_sent = 0
    prev = BEGIN
    start_of_sentence = True  # count BEGIN exactly once per sentence

    with open(pos_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # sentence boundary
            if line == "":
                if prev != BEGIN:  # only close if sentence had tokens
                    trans[prev][END] += 1
                    tag_count[END] += 1
                    num_sent += 1
                prev = BEGIN
                start_of_sentence = True
                continue

            # token<TAB>tag (robust fallback to whitespace split)
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Bad line (need token<TAB>tag): {line!r}")
            tok, tag = parts[0], parts[1]

            if start_of_sentence:
                tag_count[BEGIN] += 1
                start_of_sentence = False

            # emissions
            emit[tag][tok] += 1
            vocab.add(tok)
            word_count[tok] += 1
            tag_count[tag] += 1

            # transitions
            trans[prev][tag] += 1
            prev = tag

    # close final sentence if file missing trailing blank line
    if prev != BEGIN:
        trans[prev][END] += 1
        tag_count[END] += 1
        num_sent += 1

    hapax = {w for w, c in word_count.items() if c == 1}
    return emit, trans, tag_count, vocab, hapax, num_sent

# ---------- Optional dev audit helpers ----------

def dev_oov_report(dev_words_path, vocab):
    """Compute token/type OOV rates for a .words file against a given vocab."""
    total_tokens = 0
    total_types = set()
    oov_tokens = 0
    oov_types = set()

    with open(dev_words_path, "r", encoding="utf-8") as f:
        for raw in f:
            w = raw.rstrip("\n")
            if w == "":
                continue
            total_tokens += 1
            total_types.add(w)
            if w not in vocab:
                oov_tokens += 1
                oov_types.add(w)

    tok_rate = 100.0 * oov_tokens / max(1, total_tokens)
    type_rate = 100.0 * len(oov_types) / max(1, len(total_types))
    return {
        "tokens": total_tokens,
        "types": len(total_types),
        "oov_tokens": oov_tokens,
        "oov_types": len(oov_types),
        "oov_token_rate": tok_rate,
        "oov_type_rate": type_rate,
    }

def print_transition_headroom(trans, tag_count):
    """
    Prints average outgoing branchiness (nonzero successors per tag) and
    top-5 most 'branchy' tags. Useful as a sparsity sanity check.
    """
    nonzero = []
    for t1, row in trans.items():
        k = sum(1 for c in row.values() if c > 0)
        nonzero.append((t1, k))
    if not nonzero:
        return
    avg = sum(k for _, k in nonzero) / len(nonzero)
    nonzero.sort(key=lambda x: (-x[1], x[0]))
    print(f"\nAvg nonzero transitions per tag: {avg:.2f}")
    print("Most branchy tags (top 5):")
    for t, k in nonzero[:5]:
        print(f"  {t:>5s}: {k} successors (count={tag_count[t]})")

# --------------- CLI ---------------

def main():
    if not (2 <= len(sys.argv) <= 3):
        print("Usage: python3 aw5571_train_HMM_HW3.py <train.pos> [dev.words]")
        sys.exit(1)

    pos_path = sys.argv[1]
    dev_words = sys.argv[2] if len(sys.argv) == 3 else None

    emit, trans, tag_count, vocab, hapax, num_sent = load_pos_counts(pos_path)
    tags_order = stable_tag_order(tag_count)

    print("=== Training summary ===")
    print(f"Sentences: {num_sent}")
    print(f"Distinct tags (incl. BEGIN/END): {len(tag_count)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Hapax (count==1) words: {len(hapax)}")
    print(f"BEGIN count (should equal sentences): {tag_count[BEGIN]}")
    print(f"END   count (should equal sentences): {tag_count[END]}")
    print(f"Stable tag order (first 10): {tags_order[:10]}")

    # Top tags
    inner = [(t, c) for t, c in tag_count.items() if t not in (BEGIN, END)]
    inner.sort(key=lambda x: -x[1])
    print("\nTop 10 tags by frequency:")
    for t, c in inner[:10]:
        print(f"  {t:>5s} : {c}")

    # Most common successors of BEGIN
    if BEGIN in trans:
        print("\nMost common tags after BEGIN:")
        for t, c in trans[BEGIN].most_common(5):
            print(f"  {BEGIN} -> {t}: {c}")

    # Sample emissions for the most common tag
    if inner:
        top_tag = inner[0][0]
        print(f"\nSample emissions for top tag `{top_tag}`:")
        for w, c in emit[top_tag].most_common(5):
            print(f"  {w!r} | {top_tag}: {c}")

    print_transition_headroom(trans, tag_count)

    # Optional dev OOV audit
    if dev_words:
        rep = dev_oov_report(dev_words, vocab)
        print("\n=== Dev OOV audit ===")
        print(f"Tokens: {rep['tokens']}  |  Types: {rep['types']}")
        print(f"OOV tokens: {rep['oov_tokens']} ({rep['oov_token_rate']:.2f}%)")
        print(f"OOV types : {rep['oov_types']} ({rep['oov_type_rate']:.2f}%)")

if __name__ == "__main__":
    main()
