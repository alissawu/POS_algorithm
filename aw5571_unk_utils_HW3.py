# aw5571_unk_utils_HW3.py
import re

# Order matters: first match wins (most specific → most general)
UNK_CLASSES = [
    "UNK_NUM",        # contains a digit
    "UNK_PUNCT",      # all non-word chars
    "UNK_HYPH",       # has a hyphen
    "UNK_ACRONYM",    # U.S., I.B.M.
    "UNK_ALLCAPS",    # NASA
    "UNK_INITCAP",    # Apple

    # morphology / endings (longer suffixes first)
    "UNK_ABLE",       # notable, possible
    "UNK_AL",         # provincial
    "UNK_AN",         # artisan
    "UNK_AR",         # circular
    "UNK_ATE",        # fortunate
    "UNK_EN",         # beaten
    "UNK_ED",         # walked
    "UNK_ER",         # bigger / worker
    "UNK_OR",         # actor
    "UNK_EST",        # biggest
    "UNK_FUL",        # helpful
    "UNK_IC",         # economic
    "UNK_ING",        # running
    "UNK_ION",        # inflation
    "UNK_ISH",        # childish
    "UNK_IST",        # socialist
    "UNK_ITY",        # scarcity
    "UNK_IVE",        # expensive
    "UNK_LESS",       # powerless
    "UNK_LY",         # quickly
    "UNK_MENT",       # judgment
    "UNK_NESS",       # weakness
    "UNK_OUS",        # previous
    "UNK_S",          # plurals / 3sg
    "UNK_Y",          # rainy
    "UNK_LOWER",      # fallback
]

_re_allcaps = re.compile(r'^[A-Z]+$')
_re_initcap = re.compile(r'^[A-Z][a-z].*')
_re_acronym = re.compile(r'^(?:[A-Z]\.){2,}$')
_re_num     = re.compile(r'.*\d.*')
_re_hyph    = re.compile(r'.*-.*')
_re_punct   = re.compile(r'^\W+$')

_SUFFIX_MAP = [
    ("UNK_ABLE", ("able", "ible")),
    ("UNK_AL", ("al",)),
    ("UNK_AN", ("an", "ian")),
    ("UNK_AR", ("ar",)),
    ("UNK_ATE", ("ate",)),
    ("UNK_EN", ("en",)),
    ("UNK_ED", ("ed",)),
    ("UNK_ER", ("er",)),
    ("UNK_OR", ("or",)),
    ("UNK_EST", ("est",)),
    ("UNK_FUL", ("ful",)),
    ("UNK_IC", ("ic",)),
    ("UNK_ING", ("ing",)),
    ("UNK_ION", ("tion", "sion", "ion")),
    ("UNK_ISH", ("ish",)),
    ("UNK_IST", ("ist",)),
    ("UNK_ITY", ("ity",)),
    ("UNK_IVE", ("ive",)),
    ("UNK_LESS", ("less",)),
    ("UNK_LY", ("ly",)),
    ("UNK_MENT", ("ment",)),
    ("UNK_NESS", ("ness",)),
    ("UNK_OUS", ("ous",)),
    ("UNK_S", ("s", "es")),
    ("UNK_Y", ("y",)),
]

def word_to_unk_class(w: str) -> str:
    wl = w.lower()

    if _re_num.match(w):
        return "UNK_NUM"
    if _re_punct.match(w):
        return "UNK_PUNCT"
    if _re_hyph.match(w):
        return "UNK_HYPH"
    if _re_acronym.match(w):
        return "UNK_ACRONYM"

    if _re_allcaps.match(w):
        return "UNK_ALLCAPS"
    if _re_initcap.match(w):
        return "UNK_INITCAP"

    for cls, suffixes in _SUFFIX_MAP:
        for suf in suffixes:
            if wl.endswith(suf):
                return cls

    return "UNK_LOWER"
