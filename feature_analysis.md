# Principled Feature Analysis

## Evaluation Criteria
1. **Linguistic validity**: Does it make actual English sense?
2. **Potential impact**: How many errors could it possibly fix?
3. **Significance**: Is the measured difference meaningful or noise?

## Feature-by-Feature Analysis

### blend_low_freq_emission
- **What it does**: Blends emission prob with hapax class prob for words with freq ≤ 2
- **Accuracy**: Removing it: 96.542% → 96.551% (+0.009%)
- **Linguistic validity**: 🤔 Questionable - why should freq=2 words behave like freq=1 words?
- **Potential impact**: Low - only affects words seen ≤2 times in training
- **Significance**: +0.009% = ~3 errors. **NEGLIGIBLE**
- **Verdict**: ❌ **REMOVE** - adds complexity for negligible benefit, linguistically questionable

### context_bonus
Let me break down each sub-rule:

#### 1. Sentence-initial capitalization (lines 502-518)
- **What**: If first word is capitalized and lowercase version exists in vocab, check if lowercase is usually non-proper
- **Linguistic validity**: ✅ Strong - "The" vs "the", "Many" vs "many"
- **Potential impact**: Medium - only fixes sentence-initial errors
- **Test**: How many sentence-initial capital errors are there?
- **Verdict**: 🤔 **TEST IMPACT**

#### 2. NNPS vs NNP stemming (lines 520-532)
- **What**: If word ends in 's' and stem exists as NNP, boost NNPS
- **Linguistic validity**: ✅ Good - "American" → "Americans"
- **Errors**: NNPS→NNP: 38 errors, NNP→NNPS: 16 errors = 54 total (4.8% of errors)
- **Potential impact**: Medium-High - addresses 54 errors
- **Test**: How many does this rule actually catch?
- **Verdict**: ✅ **LIKELY GOOD** - addresses real error category

#### 3. DT context boost (lines 534-539)
- **What**: After DT, boost NN and penalize non-adjective-like JJ
- **Linguistic validity**: ⚠️ **DUBIOUS** - "the big house" is totally valid!
- **Errors**: NN→JJ: 105 errors, but [DT] WORD [NN] pattern has 49.3% legitimate JJ!
- **Potential impact**: High risk of false positives
- **Verdict**: ❌ **LIKELY BAD** - conflicts with legitimate English patterns

#### 4. VBD/VBN after subjects (lines 541-546)
- **What**: After NN/NNS/PRP, if word ends in -ed, boost VBD
- **Linguistic validity**: ✅ Good - "John walked" (VBD) not "John walked" (VBN)
- **Errors**: VBD→VBN: 60, VBN→VBD: 54 = 114 total (10% of all errors!)
- **But**: Analysis shows VBN→VBD errors are often AFTER nouns ("sales reported")
- **Verdict**: ⚠️ **CONFLICTED** - rule makes sense but might be wrong direction

### known_word_bonus
Let me analyze each:

#### 1. Numeric detection (lines 432-436)
- **Potential**: How many numeric tokens are mistagged?
- **Linguistic validity**: ✅ Excellent - "123" is always CD
- **Verdict**: ✅ **KEEP** - high precision, makes sense

#### 2. Symbol/punctuation (lines 437-441)
- **Potential**: How many punctuation errors?
- **Linguistic validity**: ✅ Excellent
- **Verdict**: ✅ **KEEP**

#### 3. Initial capital boost (lines 442-446)
- **What**: Boost NNP/NNPS, penalize NN/JJ
- **Linguistic validity**: ⚠️ Conflicts with sentence-initial rule!
- **Verdict**: ⚠️ **REDUNDANT/CONFLICTING** with context_bonus

#### 4. All-caps boost (lines 447-451)
- **Potential**: How many all-caps words are there?
- **Significance**: Probably very few words
- **Verdict**: 🤔 **TEST FREQUENCY**

#### 5. "-ly" ending (lines 452-456)
- **Linguistic validity**: ✅ Excellent - "quickly" is RB
- **Errors**: How many -ly errors?
- **Verdict**: ✅ **LIKELY GOOD**

#### 6. Adjective suffix check (lines 457-461)
- **What**: ALL JJ predictions get bonus/penalty based on 2 suffixes ("ic", "ive")
- **Linguistic validity**: ⚠️ Too coarse - only 2 suffixes
- **Impact**: Affects EVERY JJ prediction!
- **Errors**: JJ→NN: 50, NN→JJ: 105 = 155 errors
- **Verdict**: ⚠️ **LIKELY OVERFIT** - 2 suffixes is arbitrary

#### 7. Noun suffix check (lines 462-463)
- **What**: NOUN_SUFFIXES = () - DEAD CODE
- **Verdict**: ❌ **DELETE**

#### 8. NNPS heuristic (lines 464-465)
- **What**: Capital + ends in 's'
- **Errors**: NNPS→NNP: 38, NNP→NNPS: 16
- **Verdict**: ✅ **KEEP** - addresses real errors, makes sense

#### 9. Frequency-based bonus (lines 466-476)
- **What**: If word has dominant tag (≥50% freq), boost it; penalize rare tags
- **Linguistic validity**: ✅ Excellent - trust the data
- **Impact**: Affects all known words with freq ≥ 3
- **Verdict**: ✅ **KEEP** - fundamental principle

## Recommended Action Plan

### Phase 1: Remove Obvious Bloat
1. ❌ Delete blend_low_freq_emission (negligible impact, questionable)
2. ❌ Delete NOUN_SUFFIXES and is_noun_like (dead code)
3. ❌ Delete adjective suffix check (only 2 suffixes, too coarse)
4. ❌ Delete DT context boost (conflicts with legitimate patterns)

### Phase 2: Test Frequencies
Before deciding, measure:
1. How many all-caps errors are there?
2. How many sentence-initial capitalization errors?
3. How many -ly mistagging errors?

### Phase 3: Fix Conflicts
1. Capital word handling is in BOTH known_word_bonus AND context_bonus
   - Keep only ONE place
   - Sentence-initial is more specific, so keep that in context_bonus
   - Remove general capital boost from known_word_bonus

2. VBD/VBN rule might be backwards
   - Current rule boosts VBD after subjects
   - But analysis shows "noun + VBN" errors (passive voice)
   - Need to rethink this

### Expected Outcome
- Cleaner, more maintainable code
- Remove ~0.01% of overfitted gains
- But gain a FOUNDATION to add targeted, high-impact rules
- Can then address the 110 VBD/VBN errors properly

## Key Insight
**Small accuracy differences (< 0.05%) on dev set are NOISE, not signal.**
Only trust improvements > 0.1% (~30 errors) unless linguistically very sound.
