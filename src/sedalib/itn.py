"""Inverse text normalization: spelled-out Turkish numbers -> digits.

    itn("bin dokuz yüz yirmilerde iki bin yirmi dört yılında") == "1920'lerde 2024 yılında"
    itn("yüzde elli beşi") == "%55'i"            itn("on dokuzuncu yüzyıl") == "19'uncu yüzyıl"
    itn("üç virgül beş metre") == "3,5 metre"    itn("yirmi milyon avro") == "20 milyon avro"

Conservative, because number words collide with ordinary words (bir "a", yüz "face", altı "under",
yedi "ate", on- "onu/ona", bin "get on"). A span is converted only if
  - it is two or more number words that spell one number exactly as the model writes it, or
  - it is a single tens word yirmi..doksan (with or without a suffix), or follows "yüzde" / precedes
    "virgül".
Small ordinals stay written out (birinci, üçüncü); multi-word ones become 19'uncu. The last word may
carry a suffix from a closed list, kept after an apostrophe; if the word after a span starts like a
number word but its suffix is unknown, the span is left alone rather than converted halfway.
"""

from __future__ import annotations

_ONES_LIST = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS_LIST = [
    "",
    "on",
    "yirmi",
    "otuz",
    "kırk",
    "elli",
    "altmış",
    "yetmiş",
    "seksen",
    "doksan",
]
_SCALE_LIST = [
    (10**12, "trilyon"),
    (10**9, "milyar"),
    (10**6, "milyon"),
    (10**3, "bin"),
]

ONES = {w: i for i, w in enumerate(_ONES_LIST) if w}
TENS = {w: i * 10 for i, w in enumerate(_TENS_LIST) if w}
SCALES = {name: scale for scale, name in _SCALE_LIST}
BIG = ("milyon", "milyar", "trilyon")  # written as words after the digits: 20 milyon
WORDS = {**ONES, **TENS, "yüz": 100, **SCALES}
# Suffixes allowed on the last number word (all vowel-harmony variants spelled out).
_SUFFIX_WORDS = """
de da te ta den dan ten tan e a ye ya i ı u ü yi yı yu yü in ın un ün nin nın nun nün le la yle yla
ler lar lerde larda lerden lardan lere lara leri ları lerin ların lerle larla
si sı su sü sine sına suna süne sini sını sunu sünü sinde sında sunda sünde sinden sından sundan sünden
inde ında unda ünde ine ına una üne ini ını unu ünü inden ından undan ünden
deki daki teki taki ndeki ndaki lik lık luk lük li lı lu lü er ar şer şar
inci ıncı uncu üncü nci ncı ncu ncü
"""
SUFFIXES = set(_SUFFIX_WORDS.split())


def _under_1000(n: int) -> str:
    parts = []
    h, rest = divmod(n, 100)
    if h:
        parts.append("yüz" if h == 1 else f"{_ONES_LIST[h]} yüz")
    t, o = divmod(rest, 10)
    if t:
        parts.append(_TENS_LIST[t])
    if o:
        parts.append(_ONES_LIST[o])
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """Spoken Turkish for a non-negative integer, as the model writes it: 'iki bin yirmi dört'."""
    if n == 0:
        return "sıfır"
    words = []
    for scale, name in _SCALE_LIST:
        q, n = divmod(n, scale)
        if q:
            # Turkish drops the leading 'bir' only for 'bin'.
            words.append(
                name if (scale == 1000 and q == 1) else f"{_under_1000(q)} {name}"
            )
    if n:
        words.append(_under_1000(n))
    return " ".join(w for w in words if w)


def _split_word(w: str) -> tuple[str, str] | None:
    """number word (+ allowed suffix) -> (number word, suffix), else None. dörd- is softened dört."""
    if w in WORDS:
        return w, ""
    for base in sorted(WORDS, key=len, reverse=True):
        stem = "dörd" if base == "dört" and w.startswith("dörd") else base
        suf = w[len(stem) :]
        if w.startswith(stem) and suf in SUFFIXES:
            if stem == "dört" and suf[0] in "aeıioöuü":
                return None  # unsoftened "dörtü" is not how the model writes it, so keep it as is
            return base, suf
    return None


def _looks_numeric(w: str) -> bool:
    return any(w.startswith(b) for b in WORDS if len(b) > 2) and _split_word(w) is None


def _value(words: list[str]) -> int | None:
    """int if the words spell it exactly the way number_to_words writes it, else None."""
    total = group = 0
    for w in words:
        if w in SCALES:
            total += (group or 1) * SCALES[w]
            group = 0
        elif w == "yüz":
            group = (group or 1) * 100
        else:
            group += WORDS[w]
    n = total + group
    return n if n and number_to_words(n).split() == words else None


def _number_at(toks: list[str], i: int, allow_single: bool, small: bool = False):
    """Longest number span starting at i -> (end, words, suffix), or None. small: no yüz/bin/scales."""
    vocab = {**ONES, **TENS} if small else WORDS
    j = i
    while j < len(toks) and toks[j] in vocab:
        j += 1
    for end in range(min(j + 1, len(toks)), i, -1):
        sp = _split_word(toks[end - 1])
        if sp is None or (small and sp[0] not in vocab):
            continue
        base, suf = sp
        words = toks[i : end - 1] + [base]
        if _value(words) is None:
            continue
        if end == j and end < len(toks) and _looks_numeric(toks[end]):
            return (
                None  # "iki bin on dokuzdaki": unknown suffix on the next number word
            )
        if len(words) == 1 and not (allow_single or (base in TENS and base != "on")):
            continue
        return end, words, suf
    return None


def _digits(n: int) -> str:
    return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)


def _render(words: list[str], suf: str) -> str:
    """digits (+ big scale word) (+ 'suffix)."""
    if (
        words[-1] in BIG and len(words) > 1
    ):  # yirmi milyon -> 20 milyon, otuz iki milyonu -> 32 milyonu
        return f"{_digits(_value(words[:-1]))} {words[-1]}{suf}"
    return _digits(_value(words)) + (f"'{suf}" if suf else "")


def itn(text: str) -> str:
    """Write the numbers of a spoken-form transcript as digits."""
    toks, out, i = text.split(), [], 0
    while i < len(toks):
        pct = toks[i] == "yüzde"
        start = i + 1 if pct else i
        m = _number_at(toks, start, allow_single=pct)
        if not m and start + 1 < len(toks) and toks[start + 1] == "virgül":
            m = _number_at(toks, start, allow_single=True)
        if not m:
            out.append(toks[i])
            i += 1
            continue
        end, words, suf = m
        r = _render(words, suf)
        if (
            not suf
            and words[-1] not in BIG
            and end + 1 < len(toks)
            and toks[end] == "virgül"
            and (f := _number_at(toks, end + 1, allow_single=True, small=True))
        ):
            fend, fwords, fsuf = f
            r, end = f"{r},{_value(fwords)}", fend
            if fsuf:
                r += f"'{fsuf}"
            elif end < len(toks) and (b := _split_word(toks[end])) and b[0] in BIG:
                r, end = (
                    f"{r} {toks[end]}",
                    end + 1,
                )  # yüz yedi virgül altı milyon -> 107,6 milyon
        out.append(("%" if pct else "") + r)
        i = end
    return " ".join(out)
