# aw5571_train_HMM_HW3.py
# Step 2: load .pos, build raw counts, print stats
# Emissions: P(word | tag), Transitions: P (tag_i | tag_i-1)
import sys, collections

BEGIN, END = "Begin_Sent", "End_Sent"

def load_pos_counts(pos_path):
    """
    Reads a Penn Treebank-style .pos file (token<TAB>tag, blank line between sentences)
    and returns raw count tables + vocab + sentence count.
    """
    # emission counts: count[tag][word]
    emit = collections.defaultdict(lambda: collections.Counter())
    # transition counts: count[prev_tag][cur_tag]
    trans = collections.defaultdict(lambda: collections.Counter())
    # total occurrences per tag (as emitters or structural markers)
    tag_count = collections.Counter()
    # token counts (to detect hapax/OOV)
    word_count = collections.Counter()

    vocab = set()
    num_sent = 0

    prev = BEGIN
    tag_count[BEGIN] += 1  # mark sentence start for the first sentence

    with open(pos_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:  # sentence boundary
                trans[prev][END] += 1
                tag_count[END] += 1
                num_sent += 1
                prev = BEGIN
                tag_count[BEGIN] += 1
                continue

            # expect token \t tag ; fall back to split on whitespace if needed
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Bad line (need token<TAB>tag): {line!r}")

            tok, tag = parts[0], parts[1]

            emit[tag][tok] += 1
            trans[prev][tag] += 1
            tag_count[tag] += 1
            word_count[tok] += 1
            vocab.add(tok)
            prev = tag

    # hapax (words that occur once in training) — useful later for OOV handling
    hapax = {w for w, c in word_count.items() if c == 1}

    return emit, trans, tag_count, vocab, hapax, num_sent

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 aw5571_train_HMM_HW3.py <train.pos>")
        sys.exit(1)

    pos_path = sys.argv[1]
    emit, trans, tag_count, vocab, hapax, num_sent = load_pos_counts(pos_path)

    print("-----Training summary-----")
    print(f"Sentences: {num_sent}")
    print(f"Distinct tags (incl. BEGIN/END): {len(tag_count)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Hapax (count==1) words: {len(hapax)}")

    # Top 10 most frequent tags (excluding BEGIN/END)
    inner = [(t, c) for t, c in tag_count.items() if t not in (BEGIN, END)]
    inner.sort(key=lambda x: -x[1])
    print("\nTop 10 tags by frequency:")
    for t, c in inner[:10]:
        print(f"  {t:>5s} : {c}")

    # spot-check: a few example transition counts out of BEGIN
    if BEGIN in trans:
        sample_next = list(trans[BEGIN].most_common(5))
        print("\nMost common tags after BEGIN:")
        for t, c in sample_next:
            print(f"  {BEGIN} -> {t}: {c}")

    # and a couple emission samples for the most common tag
    if inner:
        top_tag = inner[0][0]
        most_words = emit[top_tag].most_common(5)
        print(f"\nSample emissions for top tag `{top_tag}`:")
        for w, c in most_words:
            print(f"  {w!r} | {top_tag}: {c}")

if __name__ == "__main__":
    main()
