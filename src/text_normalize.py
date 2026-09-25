"""
Text normalization for business_name and business_address fields.

Design principles (driven by EDA on the actual dataset):
  - Source 1 is always clean, romanized text; Source 2/3 are noisy and,
    for India specifically, ~20-25% of names/addresses are in native
    script (Devanagari, Kannada, Tamil, Telugu, Bengali, Gurmukhi) rather
    than romanized. We do NOT hand-write a transliteration table (fragile,
    incomplete, and effectively a hardcoded per-language lookup). Instead
    we (a) produce clean features from whatever script is present, and
    (b) flag script mix so the model/blocking can lean on the embedding
    channel (see embedding_blocking.py) for cross-script recall.
  - The country column is an OPEN SET (test adds France, unseen in train).
    Nothing here special-cases "US" / "India" strings — the address
    abbreviation map below has entries for US, Indian, and French address
    vocabulary, but adding a fourth country later is just adding rows to
    that dict, not branching logic.
  - Legal-suffix handling is a *feature signal*, not a hard filter: we
    compute a suffix-stripped "core name" and keep both the full and core
    versions, so a wrong or missed suffix never destroys the match — it
    just loses one feature's worth of signal.
"""
import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------
_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}


def script_flags(text: str) -> dict:
    """Return which scripts appear in `text`, plus a catch-all 'non_latin'."""
    flags = {k: False for k in _SCRIPT_RANGES}
    non_latin = False
    for ch in text:
        cp = ord(ch)
        if cp < 128:
            continue
        matched = False
        for name, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                flags[name] = True
                matched = True
                break
        if not matched and ch.isalpha():
            non_latin = True  # accented Latin (French) won't trip this
    flags["non_latin_other"] = non_latin
    flags["any_indic"] = any(flags[k] for k in _SCRIPT_RANGES)
    return flags


def ascii_fold(text: str) -> str:
    """Strip diacritics: 'Thénard' -> 'Thenard'. No-op on non-Latin scripts."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# ---------------------------------------------------------------------------
# Legal-suffix vocabulary (US / India / France + generic). This is basic,
# public linguistic knowledge about entity naming conventions — not a
# business-identity lookup — and is used purely to derive a "core name"
# feature, never to filter or exclude candidates.
# ---------------------------------------------------------------------------
_LEGAL_SUFFIXES = [
    # US
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "l l c", "llp", "l l p", "pllc", "pc", "ltd",
    # India
    "pvt", "private", "limited", "opc",
    # France
    "sarl", "sas", "sasu", "eurl", "sci", "sa", "ei", "auto entrepreneur",
    # Generic / cross-border
    "group", "holdings", "enterprises", "associates", "partners",
    "dba", "d b a", "t a", "trading as",
]
# Longest-first so multi-word suffixes match before their substrings.
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(s) for s in _LEGAL_SUFFIXES), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_URL_PATTERN = re.compile(
    r"(https?://\S+)|(\bwww\.\S+)|(\b[\w-]+\.(?:com|net|org|in|co|co\.in|fr|io)\b)",
    re.IGNORECASE,
)
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_PIPE_SPLIT_PATTERN = re.compile(r"\s*\|\s*")
_AMPERSAND_PATTERN = re.compile(r"\s*&\s*")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_DIGIT_PATTERN = re.compile(r"\d+")
_STRAY_EDGE_PATTERN = re.compile(r"^[\s'-]+|[\s'-]+$")

# NOTE: we deliberately do NOT use `[^\w\s]` to strip punctuation. Python's
# `\w` excludes Unicode *combining marks* (category Mn/Mc), which is how
# Devanagari/Kannada/Tamil/etc. vowel signs are encoded — a `\w`-based
# filter silently shreds Indic-script text into isolated consonants
# ("सूर्या" -> "स र य"). Instead we explicitly strip a known punctuation/
# symbol set and keep everything else (all letters, marks, digits survive,
# in every script).
_PUNCT_TO_STRIP = re.compile(
    r"[!\"$%()*+,./:;<=>?@\[\]\\^_`{|}~“”‘’«»—–…•]"
)


def _strip_punctuation(text: str) -> str:
    return _PUNCT_TO_STRIP.sub(" ", text)


def clean_name(raw: str) -> str:
    """Generic cleanup shared by every downstream name representation."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    # Extra metadata (URL, pipe-delimited trailer) is dropped from the name
    # itself; if you want it as a feature, capture it before calling this.
    text = _PIPE_SPLIT_PATTERN.split(text)[0]
    de_urled = _URL_PATTERN.sub(" ", text)
    # A name that is ENTIRELY a domain (e.g. "msbrothers.com") would
    # otherwise be stripped to nothing — fall back to the un-stripped
    # text in that case so the domain itself still serves as a name token.
    text = de_urled if de_urled.strip() else text
    text = _HASHTAG_PATTERN.sub(r"\1", text)
    text = _AMPERSAND_PATTERN.sub(" and ", text)
    text = _strip_punctuation(text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text.lower()


def strip_legal_suffix(cleaned: str) -> tuple:
    """Return (core_name, found_suffixes_joined_or_''). Order-independent —
    handles 'LLC Foo Bar', 'Foo Bar LLC', and multiple suffix tokens like
    'Eye Associates LLC LLC'."""
    found = _SUFFIX_PATTERN.findall(cleaned)
    if not found:
        return cleaned, None
    core = _SUFFIX_PATTERN.sub(" ", cleaned)
    core = _WHITESPACE_PATTERN.sub(" ", core).strip()
    core = _STRAY_EDGE_PATTERN.sub("", core)
    # Dedup while preserving order (e.g. "llc llc" -> "llc").
    seen = []
    for s in found:
        s = s.lower()
        if s not in seen:
            seen.append(s)
    return (core or cleaned), ",".join(seen)


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------
# Bidirectional abbreviation -> canonical form. Extend this dict per-country;
# nothing else in the pipeline needs to change to support a new country.
_ADDR_ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "dr": "drive",
    "ln": "lane", "apt": "apartment", "bldg": "building", "fl": "floor",
    "hwy": "highway", "ct": "court", "pl": "place", "sq": "square",
    "near": "near", "opp": "opposite", "opp.": "opposite",
    "chs": "cooperative housing society", "soc": "society",
    "ch": "chemin", "imp": "impasse", "bd.": "boulevard",
}
_LANDMARK_PATTERN = re.compile(
    r"\b(near|opp\.?|opposite|behind|next to|beside|adjacent to)\b\s+(.+?)(?:,|$)",
    re.IGNORECASE,
)


def clean_address(raw: str) -> str:
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = text.lower()
    text = re.sub(r"[.,]", " , ", text)  # keep commas as soft separators
    text = _strip_punctuation(text)
    tokens = text.split()
    expanded = [_ADDR_ABBREVIATIONS.get(t, t) for t in tokens]
    text = " ".join(expanded)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def extract_landmark_clause(raw_lower_address: str):
    """Pull out 'near X' / 'opp X' style landmark references so they can be
    excluded from the core address comparison (landmarks rarely appear
    identically across independently-sourced records)."""
    m = _LANDMARK_PATTERN.search(raw_lower_address)
    return m.group(0) if m else None


def address_tokens(cleaned_address: str) -> set:
    """Alpha tokens only, length >= 2, comma markers stripped."""
    return {t for t in cleaned_address.replace(",", " ").split()
            if len(t) >= 2 and not t.isdigit()}


def extract_numbers(text: str) -> set:
    """All digit runs (house/plot numbers, PIN/ZIP when present)."""
    return set(_DIGIT_PATTERN.findall(text))


def name_tokens(cleaned_name: str, min_len: int = 2) -> set:
    return {t for t in cleaned_name.split() if len(t) >= min_len}


def char_ngrams(text: str, n: int = 4) -> set:
    """Character n-grams over the space-joined token string. Works across
    scripts (two Devanagari strings still share n-grams) but two records in
    *different* scripts for the same business will NOT share n-grams —
    that gap is exactly what the embedding channel is for."""
    padded = f"  {text}  "
    if len(padded) < n:
        return {padded}
    return {padded[i:i + n] for i in range(len(padded) - n + 1)}


@dataclass
class NormalizedRecord:
    entity_id: str
    country: str
    name_clean: str
    name_core: str
    name_suffix: str
    name_ascii: str
    name_tokens: frozenset
    name_ngrams: frozenset
    addr_clean: str
    addr_tokens: frozenset
    addr_numbers: frozenset
    landmark: str
    scripts: dict


def normalize_record(entity_id: str, name: str, address: str, country: str,
                      ngram_n: int = 4) -> NormalizedRecord:
    name_clean = clean_name(name)
    name_core, name_suffix = strip_legal_suffix(name_clean)
    name_ascii = ascii_fold(name_clean)
    addr_clean = clean_address(address)
    landmark = extract_landmark_clause(addr_clean)
    return NormalizedRecord(
        entity_id=entity_id,
        country=country,
        name_clean=name_clean,
        name_core=name_core,
        name_suffix=name_suffix or "",
        name_ascii=name_ascii,
        name_tokens=frozenset(name_tokens(name_clean)),
        name_ngrams=frozenset(char_ngrams(name_core, ngram_n)),
        addr_clean=addr_clean,
        addr_tokens=frozenset(address_tokens(addr_clean)),
        addr_numbers=frozenset(extract_numbers(address or "")),
        landmark=landmark or "",
        scripts=script_flags((name or "") + " " + (address or "")),
    )
