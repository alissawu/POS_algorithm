# Code Audit: Feature Analysis

## Current Features Inventory

### 1. **known_word_bonus** (applied to ALL words in vocabulary)
- **Numeric detection** (lines 432-436): Boosts CD, penalizes others
- **Symbol/punctuation** (437-441): Boosts punct tags, penalizes others
- **Initial capital** (442-446): Boosts NNP/NNPS, penalizes NN/JJ
- **All caps** (447-451): Boosts NNP/NNPS/NN, penalizes JJ
- **"-ly" ending** (452-456): Boosts RB, penalizes NN/JJ/VB
- **Adjective-like suffixes** (457-461): Boosts/penalizes JJ based on suffix
- **Noun-like suffixes** (462-463): Boosts NN (but NOUN_SUFFIXES = () so DISABLED)
- **NNPS heuristic** (464-465): Boosts NNPS for capital+s
- **Frequency-based** (466-476): Boosts dominant tag, penalizes rare tags

### 2. **blend_low_freq_emission** (applied to words with freq <= 2)
- Blends emission probability with hapax class probability
- Uses LOW_FREQ_BLEND_WEIGHT = 0.7

### 3. **context_bonus** (applied based on position/context)
- **Sentence-initial capital** (502-518): Complex logic for first word
- **NNPS vs NNP** (520-532): Stemming-based pluralization detection
- **After determiners** (534-539): Boosts NN, penalizes non-adjective-like JJ
- **VBD/VBN after subjects** (541-546): Boosts VBD after NN/NNS/PRP

### 4. **OOV Handling** (oov_log_prob function in viterbi)
- Hapax-based classification with features: NUMERIC, PUNCT, ALNUM, HYPHEN, ALLCAP, INITCAP, suffix classes
- Class-specific probability distributions
- Backoff to global hapax distribution

### 5. **Trigram Model**
- Per-context lambda weights (trigram_lambda_weights)
- Deleted interpolation for smoothing
- Global lambda weights as fallback

## Potential Issues

### CONFLICTS & OVERLAPS

1. **Capital word handling done in MULTIPLE places:**
   - known_word_bonus: lines 442-451 (general capital boost)
   - context_bonus: lines 502-518 (sentence-initial)
   - context_bonus: lines 520-532 (NNPS detection)
   - **ISSUE**: These could be fighting each other

2. **JJ vs NN decision made in MULTIPLE places:**
   - known_word_bonus: lines 457-463 (suffix-based)
   - context_bonus: lines 534-539 (after DT)
   - **ISSUE**: suffix check might contradict context

3. **VBD/VBN only handled ONCE:**
   - context_bonus: lines 541-546
   - **BUT**: This is 110 errors (10% of all errors)!
   - **ISSUE**: Underserved given error frequency

### POTENTIALLY USELESS FEATURES

1. **NOUN_SUFFIXES = ()** (line 44)
   - Feature is DISABLED (empty tuple)
   - Function is_noun_like() does nothing
   - But still called in known_word_bonus (line 462)
   - **ACTION**: Remove dead code

2. **Adjective suffix check** (lines 457-461)
   - Applies to ALL JJ predictions, not just errors
   - Uses only 2 suffixes: "ic", "ive"
   - **QUESTION**: Is this too coarse? Does it hurt more than help?

3. **All-caps boost** (lines 447-451)
   - How many all-caps words are there?
   - **QUESTION**: Worth the complexity?

4. **NNPS heuristic** (lines 464-465)
   - Very specific: capital + ends in 's'
   - **QUESTION**: Does this fix enough errors?

### MAGIC NUMBERS

Constants that might be overfitted:
- TRANSITION_ALPHA = 0.1
- CLASS_ALPHA = 0.5
- CLASS_BACKOFF = 5.0
- LOW_FREQ_BLEND_THRESHOLD = 2
- LOW_FREQ_BLEND_WEIGHT = 0.7
- TRIGRAM_BACKOFF = 2.0
- BIGRAM_BACKOFF = 1.0

**ISSUE**: These are hand-tuned and might be overfitted to dev set

### UNUSED/BLOAT

- SUFFIX_CLASSES has 11 suffixes but only used for OOV classification
- ADJ_SUFFIXES only has 2 entries
- NOUN_SUFFIXES is empty
- Many commented-out features (lines with "no", "removed")

## Recommendations for Clean-Up

### HIGH PRIORITY

1. **Remove dead code**
   - NOUN_SUFFIXES and is_noun_like (lines 44, 175-177, 462-463)
   - All the "(no ...)" comments

2. **Test critical features individually**
   - Does blend_low_freq_emission help? (lines 480-494)
   - Does sentence-initial capital logic help? (lines 502-518)
   - Does NNPS stemming help? (lines 520-532)

3. **Consolidate capital word handling**
   - Currently spread across 3 places
   - Should be unified

### MEDIUM PRIORITY

4. **Simplify adjective/noun suffix lists**
   - Either expand them properly or remove
   - Current 2-item ADJ_SUFFIXES seems arbitrary

5. **Test if context_bonus helps at all**
   - Only has 4 rules, might be too sparse
   - Or might need MORE rules for VBD/VBN

### LOW PRIORITY

6. **Parameter tuning**
   - Test if magic numbers are overfitted
   - Use validation set separate from dev

## Testing Plan

Test each feature by removing it and measuring impact:

1. Baseline: 96.54%
2. Remove blend_low_freq_emission
3. Remove all known_word_bonus
4. Remove all context_bonus
5. Remove trigram (use bigram only)
6. Remove OOV class-specific handling (use global hapax only)

For each: measure dev set accuracy. If removing feature doesn't hurt (or helps!), then it was deadweight.
