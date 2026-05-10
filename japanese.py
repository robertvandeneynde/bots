def uniname(x):
    import unicodedata
    return unicodedata.name(x, '?')

from funcoperators import postfix, infix, postfix as to

@infix
def iin(a: str, b: str) -> bool:
    return a.upper() in b.upper()

@infix
def notiin(a: str, b: str) -> bool:
    return a.upper() not in b.upper()

@infix
def icontains(a: str, b: str) -> bool:
    return b.upper() in a.upper()

@infix
def ifullmatches(x:str, reg:str):
    import re
    return re.fullmatch(reg, x, re.I)

def uniqdict(it) -> dict:
    D = {}
    for a, b in it:
        if a in D:
            raise ValueError(f"Duplicate key {a}")
        D[a] = b
    return D

main_hiraganas = [
    (x, uniname(x)) for x in map(chr, range(0xffff))
    if 'hiragana letter' /iin/ uniname(x)
    and not 'small' /iin/ uniname(x)
]

small_hiraganas = [
    (x, uniname(x)) for x in map(chr, range(0xffff))
    if 'hiragana letter' /iin/ uniname(x)
    and 'small' /iin/ uniname(x)
]

hiragana_map = uniqdict(
    ((name /ifullmatches/ '.*LETTER ([A-Z]+).*').group(1), x)
    for x, name in main_hiraganas
)

main_katakana = [
    (x, uniname(x)) for x in map(chr, range(0xffff))
    if 'katakana letter' /iin/ uniname(x)
    and not 'halfwidth' /iin/ uniname(x)
    and not 'small' /iin/ uniname(x)
]

small_katakanas = [
    (x, uniname(x)) for x in map(chr, range(0xffff))
    if 'katakana letter' /iin/ uniname(x)
    and not 'halfwidth' /iin/ uniname(x)
    and 'small' /iin/ uniname(x)
]

katana_map = uniqdict(
    (letter, s) for (i, (s, name, letter)) in (
        (i, (s, name, (name /ifullmatches/ '.*LETTER ([A-Z]+).*').group(1)))
        for (i, (s, name))
        in enumerate(main_katakana)))

comp = [
    (x, katana_map.get(x, '?'), hiragana_map.get(x, '?'))
    for x in katana_map | hiragana_map
]

autocall = lambda x: x()

@autocall
def grouped():
    unvoiced = 'K', 'S', 'T', 'W'
    voiced   = 'G', 'Z', 'D', 'V'
    # H B P is a voiced triplet

    triplet_lower = 'H',
    triplet_unvoiced = 'P',  
    triplet_voiced = 'B', 

    def to_unvoiced(x):
        try:
            return unvoiced[voiced.index(x)]
        except ValueError:
            return x 
    
    def to_triplet_lower(x):
        try:
            return triplet_lower[triplet_unvoiced.index(x)]
        except ValueError:
            try:
                return triplet_lower[triplet_voiced.index(x)]
            except ValueError:
                return x

    M = {}
    for x, k, h in ((x, katana_map.get(x, '?'), hiragana_map.get(x, '?')) for x in katana_map | hiragana_map):
        if len(x) == 2:
            if x.startswith(unvoiced + voiced):
                c = to_unvoiced(x[0]) + x[1]
                if c not in M:
                    M[c] = [None] * 4
                index = 0 if x.startswith(unvoiced) else 2 if x.startswith(voiced) else None
                M[c][index] = k
                M[c][index+1] = h
            elif x.startswith(triplet_lower + triplet_unvoiced + triplet_voiced) :
                c = to_triplet_lower(x[0]) + x[1]
                if c not in M:
                    M[x] = [None] * 6
                index = 0 if x.startswith(triplet_lower) else 2 if x.startswith(triplet_unvoiced) else 4 if x.startswith(triplet_voiced) else None
                M[c][index] = k 
                M[c][index+1] = h
            else:
                assert x not in M
                M[x] = [k, h]
        else:
            assert x not in M
            M[x] = [k, h]
    return M

@autocall
def grouped_vowel():
    Vowels = ('A', 'E', 'I', 'O', 'U')
    M = {}
    for x, y in grouped.items():
        c, v = x if len(x) == 2 else (x, ' ') if x == 'N' else (' ', x) if x in Vowels else None
        if c not in M:
            M[c] = {}
        M[c][v] = y
    return M

def make_html():
    import pandas as pd
    from pathlib import Path as path, Path
    df = pd.DataFrame.from_dict(grouped_vowel, columns=tuple('AEIOU'), orient='index')
    path('katakana-hiragana.html').write_text(df.to_html())

def romaji(s: str) -> str:
    import re
    Re = re.compile('|'.join(map(re.escape, sorted(katana_map, key=len, reverse=True))), re.I)
    return Re.sub(lambda m: katana_map[m.group(0).upper()], s)
