# usage: python3 aw5571_error_analysis_plus_HW3.py WSJ_24.pos WSJ_24_sys.pos
import sys, collections

def read_pos(path):
    sents = []
    cur = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.rstrip("\n")
            if not line:
                if cur: sents.append(cur); cur=[]
                else:   sents.append([])
                continue
            w, t = line.split("\t")
            cur.append((w, t))
    if cur: sents.append(cur)
    return sents

def main(gold_p, sys_p):
    gold_s = read_pos(gold_p)
    sys_s  = read_pos(sys_p)
    assert len(gold_s)==len(sys_s), "length mismatch in #sentences"

    conf = collections.Counter()         # (gold,sys)
    word_conf = collections.Counter()    # word
    prevtag_conf = collections.Counter() # (prev_gold_tag, gold_tag, sys_tag)
    that_split = collections.Counter()   # ('prevTag->sysTag')

    particles = {"up","down","out","off","back","in","over","away","through","around"}
    particle_ctx = collections.Counter() # (word, prev_gold_tag, gold_tag, sys_tag)

    for gs, ss in zip(gold_s, sys_s):
        assert len(gs)==len(ss), "sentence length mismatch"
        for i, ((wg,tg),(ws,ts)) in enumerate(zip(gs,ss)):
            if wg!=ws: 
                print("TOKEN mismatch; check formatting")
                sys.exit(1)
            if tg!=ts:
                conf[(tg,ts)]+=1
                word_conf[wg]+=1
                prev = gs[i-1][1] if i>0 else "BEGIN"
                prevtag_conf[(prev,tg,ts)]+=1
                if wg.lower()=="that":
                    that_split[(prev, tg, ts)] += 1
                if wg.lower() in particles:
                    particle_ctx[(wg.lower(), prev, tg, ts)] += 1

    print("\nTop 20 confusions (gold→sys):")
    for (g,s),c in conf.most_common(20):
        print(f"{g:>5s} → {s:<5s} : {c}")

    print("\nMost error-prone words (top 20):")
    for w,c in word_conf.most_common(20):
        print(f"{w!r}: {c}")

    print("\nPrev-tag-conditioned confusions (top 20): prevGold→gold→sys  : count")
    for (p,g,s),c in prevtag_conf.most_common(20):
        print(f"{p:>6s} → {g:>5s} → {s:<5s} : {c}")

    print("\n' that ' breakdown (prevGold→gold→sys):")
    for k,c in that_split.most_common():
        p,g,s = k
        print(f"{p:>6s} → {g:>5s} → {s:<5s} : {c}")

    print("\nParticles breakdown (word, prevGold→gold→sys):")
    for (w,p,g,s),c in particle_ctx.most_common(40):
        print(f"{w:>8s} ; {p:>6s} → {g:>5s} → {s:<5s} : {c}")

if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])
