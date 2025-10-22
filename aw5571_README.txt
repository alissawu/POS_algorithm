tracking things done:
Step 2 aw5571_train_HMM_HW3.py
got sanity stats, aggregated data like # sentences, vocab, # unique words, etc
Step 3
- laplace smoothing, add 1 to every possible pair so nothing is p=0
log probabilities so we can add instead of multiply
logA: transitions btwn tags
logB: emissions of words
<UNK> = fallback emission for unseen words
