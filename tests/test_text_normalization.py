import pytest

from codcast.text_normalization import german_number, normalize_for_tts


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "null"),
        (1, "eins"),
        (7, "sieben"),
        (12, "zwölf"),
        (21, "einundzwanzig"),
        (30, "dreißig"),
        (48, "achtundvierzig"),
        (100, "einhundert"),
        (145, "einhundertfünfundvierzig"),
        (1000, "eintausend"),
        (23450, "dreiundzwanzigtausendvierhundertfünfzig"),
    ],
)
def test_german_number(value: int, expected: str):
    assert german_number(value) == expected


def test_years_use_spoken_form():
    assert normalize_for_tts("gebaut seit 1998") == "gebaut seit neunzehnhundertachtundneunzig"
    assert normalize_for_tts("Baujahr 2024") == "Baujahr zweitausendvierundzwanzig"


def test_number_unit_compounds_lose_the_hyphen():
    # Der gemessene Fehler ohne Normalisierung war "88 Volt" statt "48-Volt".
    assert normalize_for_tts("48-Volt-Hybridtechnik") == "achtundvierzig Volt-Hybridtechnik"
    assert normalize_for_tts("6-Gang-Automatik") == "sechs Gang-Automatik"


def test_decimal_uses_ein_not_eins():
    assert normalize_for_tts("ein 1,2-Liter-Benziner") == "ein ein Komma zwei Liter-Benziner"
    assert normalize_for_tts("2,5 Sekunden") == "zwei Komma fünf Sekunden"


def test_model_year_code_is_split_and_spelled():
    assert normalize_for_tts("MY2025-Daten") == "M Y zweitausendfünfundzwanzig Daten"


def test_lowercase_prefixed_acronym_is_spelled():
    assert normalize_for_tts("e-DCT6-Automatik") == "e D C T sechs Automatik"


def test_wordlike_acronyms_are_left_alone():
    assert normalize_for_tts("mit ABS, ESP und Bremsassistent") == "mit ABS, ESP und Bremsassistent"


def test_single_letters_and_plain_prose_are_untouched():
    assert normalize_for_tts("P, R, N, D, Parkbremse") == "P, R, N, D, Parkbremse"
    assert normalize_for_tts("Heute geht es um eine Probefahrt.") == "Heute geht es um eine Probefahrt."


def test_result_never_contains_ascii_umlaut_substitutes():
    # "fuenf" wuerde als "fu-enf" gesprochen.
    spoken = normalize_for_tts("55 Kilometer in 5,5 Stunden bei 30 Grad")
    assert "ue" not in spoken and "oe" not in spoken and "ss " not in spoken
    assert "fünfundfünfzig" in spoken and "dreißig" in spoken
