# aw5571_train_HMM_HW3.py
# Step 2: load .pos, build raw counts, print stats
# Emissions: P(word | tag), Transitions: P(tag_i | tag_{i-1})
# BEGIN/END are synthetic tags used only for transitions.

import sys
import collections

BEGIN, END = "Begin_Sent", "End_Sent"

def load_pos_counts(pos_path):
    """
    Reads a Penn Treebank-style .pos file (token<TAB>tag, blank line between sentences)
    and returns:
      emit:        dict[tag][word] -> count
      trans:       dict[prev_tag][cur_tag] -> count
      tag_count:   Counter over tags (incl. BEGIN/END)
      vocab:       set of seen surface tokens
      hapax:       set of words seen exactly once (useful later for OOV)
      num_sent:    number of sentences
    """
    emit = collections.defaultdict(collections.Counter)   # emissions
    trans = collections.defaultdict(collections.Counter)  # transitions
    tag_count = collections.Counter()
    word_count = collections.Counter()
    vocab = set()

    num_sent = 0
    prev = BEGIN
    start_of_sentence = True  # we count BEGIN exactly once per sentence

    with open(pos_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # --- sentence boundary ---
            if line == "":
                # only close a sentence if we actually saw at least one token
                if prev != BEGIN:
                    trans[prev][END] += 1
                    tag_count[END] += 1
                    num_sent += 1
                # reset for the next sentence
                prev = BEGIN
                start_of_sentence = True
                continue

            # --- parse token<TAB>tag (robust fallback to whitespace split) ---
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Bad line (need token<TAB>tag): {line!r}")
            tok, tag = parts[0], parts[1]

            # count BEGIN exactly once per sentence: at the first token
            if start_of_sentence:
                tag_count[BEGIN] += 1
                start_of_sentence = False

            # emissions
            emit[tag][tok] += 1
            vocab.add(tok)
            word_count[tok] += 1
            tag_count[tag] += 1

            # transitions (from previous tag → current tag)
            trans[prev][tag] += 1
            prev = tag

    # file may not end with a blank line; close any open sentence
    if prev != BEGIN:
        trans[prev][END] += 1
        tag_count[END] += 1
        num_sent += 1

    hapax = {w for w, c in word_count.items() if c == 1}

    # --- sanity checks (won't raise; only computed for your info) ---
    # BEGIN should be counted exactly once per sentence
    # END should be counted exactly once per sentence
    # You can uncomment to enforce strictly:
    # assert tag_count[BEGIN] == num_sent, (tag_count[BEGIN], num_sent)
    # assert tag_count[END]   == num_sent, (tag_count[END], num_sent)

    return emit, trans, tag_count, vocab, hapax, num_sent

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 aw5571_train_HMM_HW3.py <train.pos>")
        sys.exit(1)

    pos_path = sys.argv[1]
    emit, trans, tag_count, vocab, hapax, num_sent = load_pos_counts(pos_path)

    print("=== Training summary ===")
    print(f"Sentences: {num_sent}")
    print(f"Distinct tags (incl. BEGIN/END): {len(tag_count)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Hapax (count==1) words: {len(hapax)}")
    print(f"BEGIN count (should equal sentences): {tag_count[BEGIN]}")
    print(f"END   count (should equal sentences): {tag_count[END]}")

    # Top 10 most frequent tags (excluding BEGIN/END)
    inner = [(t, c) for t, c in tag_count.items() if t not in (BEGIN, END)]
    inner.sort(key=lambda x: -x[1])
    print("\nTop 10 tags by frequency:")
    for t, c in inner[:10]:
        print(f"  {t:>5s} : {c}")

    # Most common tags after BEGIN
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

if __name__ == "__main__":
    main()
