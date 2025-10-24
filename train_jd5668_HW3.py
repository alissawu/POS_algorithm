# train_jd5668_HW3.py
from collections import defaultdict
import argparse, math

BOS = "<s>"
EOS = "</s>"

KNOWN_VOCAB = set()  # filled at runtime from training vocab

NUM_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million","billion","trillion",
    "dozen","half","quarter"
}

VERB_TAGS = {"VB","VBD","VBG","VBN","VBP","VBZ","MD"}
AUX_BE = {"is","are","am","was","were","be","been","being"}
AUX_HAVE = {"have","has","had"}
AUX_GET = {"get","gets","got","getting","gotten"}
PARTICLES = {"up","out","off","in","over","back","down","through","around","away"}

def trans_context_bonus(prev_tag: str, cur_tag: str, prev_word: str, cur_word: str) -> float:
    """Small log-bonus for helpful local contexts."""
    wprev = prev_word.lower()
    wcur  = cur_word.lower()
    bonus = 0.0
    # Bias VBN after BE/HAVE/GET (passive/perfect)
    if cur_tag == "VBN" and (wprev in AUX_BE or wprev in AUX_HAVE or wprev in AUX_GET):
        bonus += 0.4
    # Bias RP after verb for particles like "up/out/off/..."
    if cur_tag == "RP" and wcur in PARTICLES and prev_tag in VERB_TAGS:
        bonus += 0.3
    # Bias "as" to IN (common)
    if wcur == "as" and cur_tag == "IN":
        bonus += 0.2
    return bonus

# ===== Rule-based prior (for both IV/OOV) =====
def rule_prior(word: str, is_sent_start: bool = False) -> dict:
    """
    Lightweight hard rules:
      - numeric/percent/float -> CD
      - non-alnum -> '.' fallback
      - capitalized (weaken at sentence start) -> NNP
      - endswith 's' -> NNS
      - endswith 'ing/ed/ly' -> small verb/adv bias
    """
    w = word
    wl = w.lower()
    B = {}

    # number-like
    t = wl.replace(",", "")
    if t.endswith("%"):
        t = t[:-1]
    is_num_like = False
    try:
        float(t)
        is_num_like = True
    except:
        pass
    if is_num_like:
        B["CD"] = 1.0
        return B

    # non-alphanumeric
    if not any(ch.isalnum() for ch in w):
        B["."] = 0.5
        return B

    # capitalization
    if w[:1].isupper() and len(w) > 1:
        B["NNP"] = 0.5 if not is_sent_start else 0.2

    # plural-ish
    if wl.endswith("s"):
        B["NNS"] = max(B.get("NNS", 0.0), 0.5)

    # common morphology
    if wl.endswith("ing"):
        B["VBG"] = max(B.get("VBG", 0.0), 0.5)
    if wl.endswith("ed"):
        B["VBD"] = max(B.get("VBD", 0.0), 0.4)
    if wl.endswith("ly"):
        B["RB"] = max(B.get("RB", 0.0), 0.5)

    # 0) spelled-out numbers -> CD
    if wl in NUM_WORDS:
        B["CD"] = max(B.get("CD", 0.0), 0.9)
        return B

    ...
    # 1) capitalization: weaken for known vocab to avoid Big/Heavy->NNP
    if w[:1].isupper() and len(w) > 1:
        if w not in KNOWN_VOCAB:  # prefer NNP only when OOV/proper-name-like
            B["NNP"] = 0.5 if not is_sent_start else 0.2

    # 2) hyphen alnum mix -> JJ (ratings, 12th-worst, triple-B-plus, Baa-1)
    if "-" in w:
        has_alpha = any(ch.isalpha() for ch in w)
        has_digit = any(ch.isdigit() for ch in w)
        if has_alpha and (has_digit or any(ch in "+%" for ch in w)):
            B["JJ"] = max(B.get("JJ", 0.0), 0.6)

    return B

def rule_logbias(word: str, tag: str, all_tags, *, is_sent_start=False,
                 eta=0.2, eps=1e-6) -> float:
    """Convert rule prior to a small log-bias added to emissions."""
    prior = rule_prior(word, is_sent_start=is_sent_start)
    if not prior:
        return 0.0
    base = {t: eps for t in all_tags}
    for k, v in prior.items():
        base[k] = base.get(k, eps) + v
    Z = sum(base.values())
    p_rule = base.get(tag, eps) / (Z if Z > 0 else 1.0)
    return eta * math.log(p_rule)

# ===== Transitions (tag bigrams) =====
def bigram_counts(corpus_path):
    counts = defaultdict(int)     # (prev, curr) -> freq
    row_totals = defaultdict(int) # prev -> total
    tags = [BOS]
    with open(corpus_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if len(tags) > 1:
                    tags.append(EOS)
                    for i in range(1, len(tags)):
                        prev, cur = tags[i-1], tags[i]
                        counts[(prev, cur)] += 1
                        row_totals[prev] += 1
                tags = [BOS]
            else:
                try:
                    _, pos = line.rsplit(None, 1)
                except ValueError:
                    continue
                tags.append(pos)
    if len(tags) > 1:
        tags.append(EOS)
        for i in range(1, len(tags)):
            prev, cur = tags[i-1], tags[i]
            counts[(prev, cur)] += 1
            row_totals[prev] += 1
    return counts, row_totals

def build_priors(corpus_path, k_trans=0.5):
    counts, row_totals = bigram_counts(corpus_path)
    next_tags = sorted({cur for (_, cur) in counts} | {EOS})
    if next_tags and next_tags[-1] != EOS:
        next_tags = [t for t in next_tags if t != EOS] + [EOS]
    prev_tags = [BOS] + sorted({prev for (prev, _) in counts} - {BOS, EOS}) + [EOS]
    V = len(next_tags)

    priors = {prev: {} for prev in prev_tags}
    for prev in prev_tags:
        denom = row_totals.get(prev, 0) + k_trans * V
        for cur in next_tags:
            num = counts.get((prev, cur), 0) + k_trans
            priors[prev][cur] = (num / denom) if V else 0.0
    states = [t for t in next_tags if t != EOS]
    return states, priors

# ===== Emissions (word|tag counts) =====
def likelihood_counts(corpus_path):
    counts = defaultdict(lambda: defaultdict(int))  # tag -> {word: cnt}
    totals = defaultdict(int)                       # tag -> total tokens
    vocab = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            try:
                token, pos = line.rsplit(None, 1)
            except ValueError:
                continue
            counts[pos][token] += 1
            totals[pos] += 1
            vocab.add(token)
    return counts, totals, sorted(vocab)

# ===== Dominant-tag dictionary (B) =====
def build_dominant_tag_map(like_counts, min_ratio=0.95, min_count=5):
    """If a word is ≥min_ratio in one tag and occurs ≥min_count -> keep only that tag."""
    word_tag = defaultdict(lambda: defaultdict(int))
    for tag, row in like_counts.items():
        for w, c in row.items():
            word_tag[w][tag] += c

    dom = {}
    for w, row in word_tag.items():
        total = sum(row.values())
        if total < min_count:
            continue
        tag, cnt = max(row.items(), key=lambda kv: kv[1])
        if cnt / total >= min_ratio:
            dom[w] = tag
    return dom

# ===== Read .words as sentences =====
def read_words_sentences(path):
    sent = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if sent:
                    yield sent
                    sent = []
            else:
                sent.append(line)
    if sent:
        yield sent

# ===== OOV buckets =====
def oov_bucket(word: str) -> str:
    w = word
    wl = w.lower()
    if all(ch.isdigit() for ch in w): return "NUM"
    if any(ch.isdigit() for ch in w): return "HASDIGIT"
    if not any(ch.isalpha() for ch in w): return "PUNCT"
    if "-" in w: return "HYPH"
    if w.isupper() and len(w) > 1: return "ALLCAPS"
    if w[0].isupper(): return "INITCAP"
    if wl.endswith(("ment","ness","ity")): return "NOUN_SUFFIX"
    if wl.endswith(("able","ible","ous","ive","al")): return "ADJ_SUFFIX"
    if wl.endswith(("er","or","est")): return "ER_OR_EST"
    if wl.endswith("ing"): return "ING"
    if wl.endswith("ed"):  return "ED"
    if wl.endswith("ly"):  return "LY"
    if wl.endswith(("tion","sion","ion")): return "ION"
    if wl.endswith("s"):   return "S"
    return "OTHER"

def build_hapax_bucket_weights(like_counts, tag_totals, alpha=1.0, lowfreq=2):
    """Bucket weights from low-frequency words (≤lowfreq)."""
    word_total = defaultdict(int)
    for pos, row in like_counts.items():
        for w, c in row.items():
            word_total[w] += c

    bucket_tag = defaultdict(lambda: defaultdict(int))  # bucket -> {tag: cnt}
    for pos, row in like_counts.items():
        for w, c in row.items():
            if word_total[w] <= lowfreq:
                b = oov_bucket(w)
                bucket_tag[b][pos] += 1

    tags = set(like_counts.keys())
    T = len(tags)
    weights = {}
    for b, row in bucket_tag.items():
        total = sum(row.values()) + alpha * T
        weights[b] = {t: (row.get(t, 0) + alpha) / total for t in tags}
    uniform = {t: 1.0 / T for t in tags} if T else {}
    return weights, uniform

# ===== Emission log-prob (with rule-bias + OOV) =====
def emission_logprob(word, tag, like_counts, tag_totals, vocab_set,
                     k_emit=0.5,
                     all_states=None, is_sent_start=False, rule_eta=0.2,
                     oov_mode="hapaxbucket", oov_const=1e-6,
                     bucket_w=None, uniform_w=None):
    """Return log P(word|tag) + rule log-bias (works for IV and OOV)."""
    # rule bias
    rule_lb = 0.0
    if all_states is not None and rule_eta > 0.0:
        rule_lb = rule_logbias(word, tag, all_states,
                               is_sent_start=is_sent_start,
                               eta=rule_eta, eps=1e-6)

    # IV with add-k
    if word in vocab_set:
        V = len(vocab_set)
        c = like_counts.get(tag, {}).get(word, 0)
        denom = tag_totals.get(tag, 0) + k_emit * V
        log_emit = math.log((c + k_emit) / denom) if denom > 0 else -1e9
        return log_emit + rule_lb

    # OOV
    if oov_mode == "hapaxbucket":
        b = oov_bucket(word)
        w_row = (bucket_w.get(b) if bucket_w else None) or (uniform_w or {})
        V = len(vocab_set) + 1
        denom = tag_totals.get(tag, 0) + k_emit * V
        num = k_emit * w_row.get(tag, 0.0)
        log_emit = math.log(num / denom) if (denom > 0 and num > 0) else -1e9
        return log_emit + rule_lb

    if oov_mode == "constant":
        return math.log(oov_const) + rule_lb

    return -1e9 + rule_lb

# ===== Viterbi (log-space) =====
def viterbi_tag_sentence(tokens, states, priors, like_counts, tag_totals, vocab_set,
                         k_emit=0.5, oov_mode="hapaxbucket", oov_const=1e-6,
                         bucket_w=None, uniform_w=None,
                         rule_eta=0.2, dom_map=None):
    n = len(tokens)
    if n == 0:
        return []

    # per-position candidate tags (B)
    state_list = []
    for w in tokens:
        if dom_map and w in dom_map:
            state_list.append([dom_map[w]])
        else:
            state_list.append(states)

    # precompute log transitions
    logA = {prev: {cur: math.log(p) for cur, p in row.items()}
            for prev, row in priors.items()}

    delta = [dict() for _ in range(n)]
    psi   = [dict() for _ in range(n)]

    # init
    w0 = tokens[0]
    for s in state_list[0]:
        tlog = logA.get(BOS, {}).get(s, -1e9)
        elog = emission_logprob(w0, s, like_counts, tag_totals, vocab_set,
                                k_emit=k_emit,
                                all_states=states, is_sent_start=True, rule_eta=rule_eta,
                                oov_mode=oov_mode, oov_const=oov_const,
                                bucket_w=bucket_w, uniform_w=uniform_w)
        delta[0][s] = tlog + elog
        psi[0][s] = None

    # recursion
    for i in range(1, n):
        wi = tokens[i]
        for s in state_list[i]:
            best_prev, best_score = None, -1e18
            elog = emission_logprob(wi, s, like_counts, tag_totals, vocab_set,
                                    k_emit=k_emit,
                                    all_states=states, is_sent_start=False, rule_eta=rule_eta,
                                    oov_mode=oov_mode, oov_const=oov_const,
                                    bucket_w=bucket_w, uniform_w=uniform_w)
            prev_word = tokens[i-1]
            for sp in state_list[i-1]:
                score = (delta[i-1][sp] +
                        logA.get(sp, {}).get(s, -1e9) +
                        trans_context_bonus(sp, s, prev_word, wi))
                if score > best_score:
                    best_score, best_prev = score, sp
            delta[i][s] = best_score + elog
            psi[i][s] = best_prev

    # termination
    best_last, best_total = None, -1e18
    for s in state_list[-1]:
        score = delta[n-1][s] + logA.get(s, {}).get(EOS, -1e9)
        if score > best_total:
            best_total, best_last = score, s

    # backtrack
    seq = [best_last]
    for i in range(n-1, 0, -1):
        seq.append(psi[i][seq[-1]])
    seq.reverse()
    return seq

# ===== Driver =====
def tag_corpus_with_viterbi(train_pos_path, words_path, out_path,
                            k_trans=0.5, k_emit=0.5,
                            oov_mode="hapaxbucket", oov_const=1e-6,
                            lowfreq=2, alpha=1.0,
                            rule_eta=0.2, dom_ratio=0.95, dom_min_count=5):
    # build models
    states, priors = build_priors(train_pos_path, k_trans=k_trans)
    like_counts, tag_totals, vocab = likelihood_counts(train_pos_path)
    vocab_set = set(vocab)
    KNOWN_VOCAB.clear()
    KNOWN_VOCAB.update(vocab_set)

    # dominant-tag dictionary (B)
    dom_map = build_dominant_tag_map(like_counts, min_ratio=dom_ratio, min_count=dom_min_count)

    # OOV weights
    bucket_w = uniform_w = None
    if oov_mode == "hapaxbucket":
        bucket_w, uniform_w = build_hapax_bucket_weights(
            like_counts, tag_totals, alpha=alpha, lowfreq=lowfreq
        )

    # decode
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        for tokens in read_words_sentences(words_path):
            tags = viterbi_tag_sentence(tokens, states, priors,
                                        like_counts, tag_totals, vocab_set,
                                        k_emit=k_emit, oov_mode=oov_mode, oov_const=oov_const,
                                        bucket_w=bucket_w, uniform_w=uniform_w,
                                        rule_eta=rule_eta, dom_map=dom_map)
            assert len(tags) == len(tokens)
            assert all(t is not None for t in tags)
            for tok, tg in zip(tokens, tags):
                out.write(f"{tok}\t{tg}\n")
            out.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Bigram HMM POS tagger with rule-bias, pruning, and OOV.")
    ap.add_argument("train", help="Path to WSJ_*.pos (training or merged POS file)")
    ap.add_argument("--tag", help="Path to WSJ_*.words to decode")
    ap.add_argument("--out_tags", default="out.pos", help="Tagged output (.pos)")
    ap.add_argument("--k", type=float, default=0.5, help="Add-k for transitions")
    ap.add_argument("--emit_k", type=float, default=0.5, help="Add-k for emissions")
    ap.add_argument("--oov", choices=["hapaxbucket", "constant"], default="hapaxbucket",
                    help="OOV handling")
    ap.add_argument("--oov_const", type=float, default=1e-6, help="Const prob for OOV if --oov=constant")
    ap.add_argument("--lowfreq", type=int, default=2, help="Hapax threshold (≤N)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Smoothing for hapax buckets")
    ap.add_argument("--rule_eta", type=float, default=0.15, help="Rule-bias strength (0..1)")
    ap.add_argument("--dom_ratio", type=float, default=0.97, help="Pruning: dominant tag ratio")
    ap.add_argument("--dom_min_count", type=int, default=8, help="Pruning: min occurrences")
    args = ap.parse_args()

    if args.tag:
        tag_corpus_with_viterbi(args.train, args.tag, args.out_tags,
                                k_trans=args.k, k_emit=args.emit_k,
                                oov_mode=args.oov, oov_const=args.oov_const,
                                lowfreq=args.lowfreq, alpha=args.alpha,
                                rule_eta=args.rule_eta,
                                dom_ratio=args.dom_ratio, dom_min_count=args.dom_min_count)
        print(f"[OK] Tagged file saved to: {args.out_tags}")
    else:
        states, priors = build_priors(args.train, k_trans=args.k)
        like_counts, tag_totals, vocab = likelihood_counts(args.train)
        print(f"[OK] Built priors (|states|={len(states)}) and likelihood counts (|vocab|={len(vocab)}).")
        row = priors.get(BOS, {}); s = sum(row.values()) if row else 0.0
        print(f"  Row sum check <s>: {s:.6f}")

if __name__ == "__main__":
    main()
