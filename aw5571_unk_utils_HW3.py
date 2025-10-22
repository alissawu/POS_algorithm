# aw5571_unk_utils_HW3.py
import re

UNK_CLASSES = [
    "UNK_NUM", "UNK_HYPH", "UNK_PUNCT",
    "UNK_ALLCAPS", "UNK_INITCAP",
    "UNK_ING", "UNK_ED", "UNK_S",
    "UNK_LOWER",
]

_re_allcaps = re.compile(r'^[A-Z]+$')
_re_initcap = re.compile(r'^[A-Z][a-z].*')
_re_num = re.compile(r'.*\d.*')
_re_hyph = re.compile(r'.*-.*')
_re_punct = re.compile(r'^\W+$')

def word_to_unk_class(w: str) -> str:
    if _re_num.match(w):     return "UNK_NUM"
    if _re_hyph.match(w):    return "UNK_HYPH"
    if _re_punct.match(w):   return "UNK_PUNCT"
    if _re_allcaps.match(w): return "UNK_ALLCAPS"
    if _re_initcap.match(w): return "UNK_INITCAP"
    if len(w) >= 3 and w.endswith("ing"): return "UNK_ING"
    if len(w) >= 2 and w.endswith("ed"):  return "UNK_ED"
    if len(w) >= 2 and w.endswith("s"):   return "UNK_S"
    return "UNK_LOWER"
