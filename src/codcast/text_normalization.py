"""Deutsche Textnormalisierung fuer lokale TTS-Modelle.

Chatterbox liest Ziffern und Grossbuchstaben-Kuerzel unzuverlaessig: gemessen wurde
"48-Volt" als "88 Volt" und "MY2025" als "Mai 1050". Solche Fehler sind inhaltlich
falsch, nicht nur unschoen. OpenAI TTS braucht das nicht, deshalb laeuft die
Normalisierung nur im lokalen Pfad und ist abschaltbar.

Bewusst konservativ: nur Zahlen, Dezimalkommas, Uhrzeit-artige Bindestrich-Komposita
und kurze Grossbuchstaben-Kuerzel werden angefasst. Alles andere bleibt unveraendert.
"""

from __future__ import annotations

import re

# Echte Umlaute, weil das Ergebnis gesprochen wird: "fuenf" liest Chatterbox als "fu-enf".
ONES = [
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
    "achtzehn", "neunzehn",
]
TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]

# Kuerzel, die als Wort gesprochen werden und deshalb nicht buchstabiert werden duerfen.
SPOKEN_ACRONYMS = {"ADAC", "ABS", "ESP", "GmbH", "TUEV", "TÜV", "USB", "LED", "SUV", "PDF", "KI", "IT", "EU", "USA"}


def _below_hundred(value: int) -> str:
    if value < 20:
        return ONES[value]
    tens, ones = divmod(value, 10)
    if ones == 0:
        return TENS[tens]
    return f"{'ein' if ones == 1 else ONES[ones]}und{TENS[tens]}"


def german_number(value: int) -> str:
    """Zahlwort fuer 0 bis 999999. Grosse Zahlen bleiben Ziffern (selten in Skripten)."""
    if value < 0:
        return f"minus {german_number(-value)}"
    if value < 100:
        return _below_hundred(value)
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        prefix = "einhundert" if hundreds == 1 else f"{ONES[hundreds]}hundert"
        return prefix if rest == 0 else f"{prefix}{_below_hundred(rest)}"
    if value < 1_000_000:
        thousands, rest = divmod(value, 1000)
        prefix = "eintausend" if thousands == 1 else f"{german_number(thousands)}tausend"
        return prefix if rest == 0 else f"{prefix}{german_number(rest)}"
    return str(value)


def _year(value: int) -> str:
    """Jahre bis 1999 werden im Deutschen als 'neunzehnhundert...' gesprochen."""
    if 1100 <= value < 2000:
        hundreds, rest = divmod(value, 100)
        prefix = f"{_below_hundred(hundreds)}hundert"
        return prefix if rest == 0 else f"{prefix}{_below_hundred(rest)}"
    return german_number(value)


def _spell(text: str) -> str:
    return " ".join(text)


def _acronym(match: re.Match[str]) -> str:
    word = match.group(0)
    if word in SPOKEN_ACRONYMS or len(word) > 5:
        return word
    return _spell(word)


def _decimal(match: re.Match[str]) -> str:
    whole, fraction = match.group(1), match.group(2)
    spoken_fraction = " ".join(ONES[int(digit)] for digit in fraction)
    # "1,2 Liter" heisst gesprochen "ein Komma zwei Liter", nicht "eins Komma zwei".
    spoken_whole = "ein" if int(whole) == 1 else german_number(int(whole))
    return f"{spoken_whole} Komma {spoken_fraction}"


def _integer(match: re.Match[str]) -> str:
    digits = match.group(0).replace(".", "")
    value = int(digits)
    if len(digits) == 4 and digits[0] in "12":
        return _year(value)
    return german_number(value)


def normalize_for_tts(text: str) -> str:
    """Text so umschreiben, dass lokale Modelle ihn zuverlaessig aussprechen."""
    result = text
    # "48-Volt-Hybridtechnik" -> "48 Volt Hybridtechnik": Bindestriche an Zahlen loesen.
    result = re.sub(r"(?<=\d)-(?=[A-Za-zÄÖÜäöü])", " ", result)
    result = re.sub(r"(?<=[A-Za-zÄÖÜäöü])-(?=\d)", " ", result)
    # Kuerzel mit Ziffer trennen: "MY2025" -> "MY 2025", "e-DCT6" -> "e-DCT 6".
    result = re.sub(r"(?<=[A-Z])(?=\d)", " ", result)
    # Kleinbuchstabe direkt vor Grossbuchstaben-Kuerzel trennen: "e-DCT" bleibt lesbar.
    result = re.sub(r"\b([a-z])-([A-Z]{2,5})\b", r"\1 \2", result)
    result = re.sub(r"(\d+),(\d+)", _decimal, result)
    result = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b|\b\d+\b", _integer, result)
    result = re.sub(r"\b[A-Z]{2,6}\b", _acronym, result)
    return re.sub(r"\s{2,}", " ", result).strip()
