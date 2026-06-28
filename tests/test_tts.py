import threading

from codcast.config import load_config
from codcast.models import PodcastScript, ScriptLine, SpeakerSpec
from codcast.tts import _api_key_from_env_or_file, _convert_kokoro_state_dict, _is_expected_missing_norm_key, fish_payload_for_voice
from codcast.tts import OpenAIBackend, group_single_speaker_openai_lines
from codcast.tts import ScriptRenderer


def test_convert_kokoro_state_dict_strips_module_and_weight_norm_parametrizations():
    converted = _convert_kokoro_state_dict(
        {
            "module.decode.0.conv1.parametrizations.weight.original0": "g",
            "module.decode.0.conv1.parametrizations.weight.original1": "v",
            "module.decode.0.conv1.bias": "bias",
        }
    )
    assert converted == {
        "decode.0.conv1.weight_g": "g",
        "decode.0.conv1.weight_v": "v",
        "decode.0.conv1.bias": "bias",
    }


def test_expected_missing_norm_keys_are_limited_to_norm_params():
    assert _is_expected_missing_norm_key("decode.0.norm1.norm.weight")
    assert _is_expected_missing_norm_key("decode.0.norm1.norm.bias")
    assert not _is_expected_missing_norm_key("decode.0.conv1.weight_g")


def test_fish_payload_uses_reference_audio_and_text(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voice = config.tts.voices[0]
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"wav-bytes")
    voice.speaker_wav = reference
    voice.ref_text = "Referenztext"

    payload = fish_payload_for_voice("Hallo Twitch.", voice, config)

    assert payload["text"] == "Hallo Twitch."
    assert payload["reference_id"] is None
    assert payload["references"] == [{"audio": b"wav-bytes", "text": "Referenztext"}]
    assert payload["format"] == "wav"


def test_fish_payload_requires_reference_text(tmp_path):
    config = load_config(tmp_path / "missing.yml")
    voice = config.tts.voices[0]
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"wav-bytes")
    voice.speaker_wav = reference
    voice.ref_text = None

    try:
        fish_payload_for_voice("Hallo.", voice, config)
    except ValueError as exc:
        assert "ref_text" in str(exc)
    else:
        raise AssertionError("expected missing ref_text to fail")


def test_api_key_from_env_file_reads_only_named_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_TTS_API_KEY", raising=False)
    env_file = tmp_path / ".env.tts.local"
    env_file.write_text(
        "OTHER_KEY=ignored\n"
        "export OPENAI_TTS_API_KEY='test-secret'\n",
        encoding="utf-8",
    )

    assert _api_key_from_env_or_file("OPENAI_TTS_API_KEY", env_file) == "test-secret"


def test_api_key_from_env_file_accepts_raw_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_TTS_API_KEY", raising=False)
    env_file = tmp_path / ".env.tts.local"
    env_file.write_text("raw-test-secret\n", encoding="utf-8")

    assert _api_key_from_env_or_file("OPENAI_TTS_API_KEY", env_file) == "raw-test-secret"


def test_openai_backend_posts_speech_request(tmp_path, monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"wav-bytes"
        text = ""

    calls = []

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    config = load_config(tmp_path / "missing.yml")
    config.tts.openai.env_file = tmp_path / ".env.tts.local"
    config.tts.openai.env_file.write_text("OPENAI_TTS_API_KEY=fake-key\n", encoding="utf-8")
    voice = next(item for item in config.tts.voices if item.backend == "openai")
    monkeypatch.setattr("requests.post", fake_post)

    output = tmp_path / "out.wav"
    OpenAIBackend(config).render("Hallo Welt.", voice, output)

    assert output.read_bytes() == b"wav-bytes"
    assert calls[0]["url"] == "https://api.openai.com/v1/audio/speech"
    assert calls[0]["json"]["model"] == "gpt-4o-mini-tts"
    assert calls[0]["json"]["voice"] == voice.openai_voice
    assert calls[0]["json"]["input"] == "Hallo Welt."
    assert calls[0]["json"]["response_format"] == "wav"
    assert calls[0]["headers"]["authorization"] == "Bearer fake-key"


def test_openai_single_speaker_lines_are_grouped_without_labels(tmp_path):
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def render(self, text, voice, output_path):
            self.calls.append(text)
            output_path.write_bytes(b"wav")

    config = load_config(tmp_path / "missing.yml")
    config.tts.openai.max_input_chars = 30
    voice = next(item for item in config.tts.voices if item.id == "openai-cedar")
    renderer = ScriptRenderer(config, {voice.id: voice})
    fake = FakeBackend()
    renderer.backends[voice.backend] = fake
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=[SpeakerSpec(id="s1", display_name="Host", role="Host", voice_profile_id=voice.id)],
        lines=[
            ScriptLine(speaker_id="s1", text="Alpha beta."),
            ScriptLine(speaker_id="s1", text="Gamma delta."),
            ScriptLine(speaker_id="s1", text="Epsilon zeta."),
        ],
    )

    rendered = renderer.render_script(script, tmp_path)

    assert len(rendered) == 2
    assert fake.calls == ["Alpha beta.\n\nGamma delta.", "Epsilon zeta."]
    assert all("s1:" not in item for item in fake.calls)


def test_openai_multi_speaker_stays_line_by_line(tmp_path):
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def render(self, text, voice, output_path):
            self.calls.append((voice.id, text))
            output_path.write_bytes(b"wav")

    config = load_config(tmp_path / "missing.yml")
    voice_a = next(item for item in config.tts.voices if item.id == "openai-cedar")
    voice_b = next(item for item in config.tts.voices if item.id == "openai-marin")
    renderer = ScriptRenderer(config, {voice_a.id: voice_a, voice_b.id: voice_b})
    fake = FakeBackend()
    renderer.backends["openai"] = fake
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=[
            SpeakerSpec(id="s1", display_name="A", role="Host", voice_profile_id=voice_a.id),
            SpeakerSpec(id="s2", display_name="B", role="Analyst", voice_profile_id=voice_b.id),
        ],
        lines=[
            ScriptLine(speaker_id="s1", text="Erste Zeile."),
            ScriptLine(speaker_id="s2", text="Zweite Zeile."),
            ScriptLine(speaker_id="s1", text="Dritte Zeile."),
        ],
    )

    rendered = renderer.render_script(script, tmp_path)

    assert len(rendered) == 3
    assert fake.calls == [
        ("openai-cedar", "Erste Zeile."),
        ("openai-marin", "Zweite Zeile."),
        ("openai-cedar", "Dritte Zeile."),
    ]


def test_openai_segments_render_in_parallel_and_keep_order(tmp_path):
    class BlockingBackend:
        def __init__(self):
            self.started = 0
            self.max_started_before_release = 0
            self.lock = threading.Lock()
            self.release = threading.Event()

        def render(self, text, voice, output_path):
            with self.lock:
                self.started += 1
                self.max_started_before_release = max(self.max_started_before_release, self.started)
                if self.started == 2:
                    self.release.set()
            self.release.wait(timeout=2)
            output_path.write_bytes(text.encode("utf-8"))

    config = load_config(tmp_path / "missing.yml")
    config.tts.openai.concurrency = 2
    config.tts.openai.max_input_chars = 15
    voice = next(item for item in config.tts.voices if item.id == "openai-cedar")
    renderer = ScriptRenderer(config, {voice.id: voice})
    fake = BlockingBackend()
    renderer.backends[voice.backend] = fake
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=[SpeakerSpec(id="s1", display_name="Host", role="Host", voice_profile_id=voice.id)],
        lines=[
            ScriptLine(speaker_id="s1", text="Segment eins."),
            ScriptLine(speaker_id="s1", text="Segment zwei."),
            ScriptLine(speaker_id="s1", text="Segment drei."),
        ],
    )

    rendered = renderer.render_script(script, tmp_path)

    assert fake.max_started_before_release >= 2
    assert [segment.index for segment in rendered] == [1, 2, 3]
    assert [segment.text for segment in rendered] == ["Segment eins.", "Segment zwei.", "Segment drei."]


def test_openai_grouping_respects_max_chars():
    chunks = group_single_speaker_openai_lines(
        [
            ScriptLine(speaker_id="s1", text="Eins zwei drei."),
            ScriptLine(speaker_id="s1", text="Vier fuenf sechs."),
            ScriptLine(speaker_id="s1", text="Sieben acht neun."),
        ],
        25,
    )

    assert [chunk.source_line_indexes for chunk in chunks] == [(1,), (2,), (3,)]
    assert all(len(chunk.text) <= 25 for chunk in chunks)


def test_script_renderer_reports_segment_progress(tmp_path):
    class FakeBackend:
        def render(self, text, voice, output_path):
            output_path.write_bytes(b"wav")

    config = load_config(tmp_path / "missing.yml")
    voice = next(item for item in config.tts.voices if item.backend == "kokoro")
    renderer = ScriptRenderer(config, {voice.id: voice})
    renderer.backends[voice.backend] = FakeBackend()
    events = []
    script = PodcastScript(
        title="Titel",
        topic="Thema",
        target_min_minutes=1,
        target_max_minutes=2,
        speakers=[SpeakerSpec(id="s1", display_name="Host", role="Host", voice_profile_id=voice.id)],
        lines=[
            ScriptLine(speaker_id="s1", text="Erste Zeile."),
            ScriptLine(speaker_id="s1", text="Zweite Zeile."),
        ],
    )

    rendered = renderer.render_script(script, tmp_path, progress=events.append)

    assert len(rendered) == 2
    assert events[-1].phase == "tts"
    assert events[-1].current == 2
    assert events[-1].total == 2
