# usage: python3 aw5571_error_analysis_HW3.py WSJ_24.pos WSJ_24_sys.pos
import sys, collections

def read_pos(path):
    gold = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            if not line: gold.append(("", "")); continue
            w, t = line.split("\t")
            gold.append((w, t))
    return gold

def main(gold_p, sys_p):
    gold = read_pos(gold_p)
    sys  = read_pos(sys_p)
    assert len(gold)==len(sys), "length mismatch"
    conf = collections.Counter()       # (gold,sys)
    wrong_words = collections.Counter()# word
    per_tag = collections.Counter()    # gold tag counts
    errs_by_tag = collections.Counter()# gold tag errors
    for (w_g, t_g), (w_s, t_s) in zip(gold, sys):
        if w_g=="" and w_s=="": continue
        if w_g!=w_s: 
            print("TOKEN MISMATCH, check formatting")
            sys.exit(1)
        per_tag[t_g]+=1
        if t_g!=t_s:
            conf[(t_g,t_s)]+=1
            wrong_words[w_g]+=1
            errs_by_tag[t_g]+=1

    print("\nTop 15 confusions (gold→sys):")
    for (g,s),c in conf.most_common(15):
        print(f"{g:>5s} → {s:<5s} : {c}")

    print("\nMost error-prone words (top 20):")
    for w,c in wrong_words.most_common(20):
        print(f"{w!r}: {c}")

    print("\nPer-tag error rates (gold count / errors / %):")
    for t,cnt in per_tag.most_common():
        e = errs_by_tag[t]
        if cnt>0:
            print(f"{t:>5s}: {cnt:6d}  {e:6d}  {100*e/cnt:6.2f}%")

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
