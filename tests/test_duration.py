from codcast.duration import count_words, duration_status, estimate_minutes, target_word_range
from codcast.models import PodcastScript, ScriptLine, SpeakerSpec


def test_word_count_handles_german_words():
    assert count_words("Hallo Welt, kuenstliche Intelligenz und Software-Architektur.") == 6


def test_target_word_range_and_status():
    assert target_word_range(1, 2, 120) == (120, 240)
    script = PodcastScript(
        title="Test",
        topic="Test",
        target_min_minutes=0.1,
        target_max_minutes=1.0,
        speakers=[SpeakerSpec(id="s1", display_name="A", role="Host", voice_profile_id="v1")],
        lines=[ScriptLine(speaker_id="s1", text="eins zwei drei vier fuenf")],
    )
    status, words, minutes = duration_status(script, 100)
    assert status == "too_short"
    assert words == 5
    assert estimate_minutes(words, 100) == minutes

