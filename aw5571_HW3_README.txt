NetID: aw5571
Assignment: HW3 - HMM Part-of-Speech Tagging

================================================================================
HOW TO RUN
================================================================================

python3 aw5571_HW3_clean.py --train WSJ_02-24.pos --test WSJ_23.words --output submission.pos

Arguments:
  --train  : training corpus (POS-tagged)
  --test   : test corpus (words only)
  --output : output file (default: submission.pos)

================================================================================
IMPLEMENTATION
================================================================================

This is a trigram HMM with Viterbi decoding for POS tagging.

Main components:
- Trigram language model with deleted interpolation smoothing
- Suffix-based classification for unknown words
- Context bonuses based on Penn Treebank annotation guidelines

================================================================================
HANDLING OOV WORDS
================================================================================

Unknown words are handled using suffix-based classification combined with
low-frequency word distributions from the training data.

Words are grouped into 8 classes:
- Capitalized (proper nouns, sentence-initial)
- Ends in -ed (past tense/participles)
- Ends in -ing (gerunds/present participles)
- Ends in -ly (adverbs)
- Ends in -s (plurals, 3rd person verbs)
- Punctuation
- Numbers
- Other

For each class, emission probabilities come from the distribution of rare
words (count <= 2) in the training data with similar suffixes. This works
better than uniform OOV probabilities because different suffixes correlate
with different POS distributions.

================================================================================
EXTRA FEATURES
================================================================================

1. Trigram model instead of bigram
   Better context using 3-tag sequences with backoff to bigram/unigram.

2. Penn Treebank guideline rules
   Added high-precision rules based on Penn annotation guidelines:

   - "ago" is always a temporal preposition (IN)
     e.g., "3 weeks ago"

   - Particles vs prepositions: particles can't take PP complements
     e.g., "out of" is always IN, never RP

   - Perfect auxiliaries require VBN
     e.g., "has been" + participle must be VBN, not VBD

   - "that" after reporting verbs is a complementizer (IN)
     e.g., "I think that..." -> that/IN not DT

   These rules target specific grammatical contexts where the Penn guidelines
   give clear answers, avoiding overfitting to training data quirks.

3. Log-space probability computation
   Prevents underflow on long sentences.

================================================================================
RESULTS
================================================================================

Test set (WSJ_23.words): 96.56% accuracy (54731/56684 correct)

The guideline-based rules gave small but consistent improvements over the
baseline trigram HMM.
