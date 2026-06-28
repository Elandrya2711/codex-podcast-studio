from pathlib import Path

from codcast.config import EvidenceConfig
from codcast.local_evidence import LocalEvidenceCollector, extract_youtube_urls, normalize_youtube_url, vtt_to_plain_text


def test_extract_youtube_urls_normalizes_common_forms():
    urls = extract_youtube_urls(
        "Links: https://youtu.be/abc123?si=x und https://www.youtube.com/watch?v=xyz789&t=30s."
    )
    assert urls == [
        "https://www.youtube.com/watch?v=abc123",
        "https://www.youtube.com/watch?v=xyz789",
    ]


def test_normalize_youtube_url_rejects_non_video_urls():
    assert normalize_youtube_url("https://www.youtube.com/@channel") is None
    assert normalize_youtube_url("https://example.com/watch?v=abc") is None


def test_vtt_to_plain_text_removes_timestamps_and_tags():
    text = vtt_to_plain_text(
        """WEBVTT

00:00:00.000 --> 00:00:01.000
<c>Hallo</c> &amp; willkommen

00:00:01.000 --> 00:00:02.000
Anny the Duck auf Twitch
"""
    )
    assert text == "Hallo & willkommen\nAnny the Duck auf Twitch"


def test_youtube_transcript_rejects_oversized_subtitle_before_read(tmp_path: Path):
    fake_ytdlp = tmp_path / "yt-dlp"
    fake_ytdlp.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "root = Path(__import__('sys').argv[-2]).parent\n"
        "(root / 'video.de.vtt').write_text('x' * 128, encoding='utf-8')\n"
        "(root / 'video.info.json').write_text('{\"title\":\"T\"}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_ytdlp.chmod(0o755)
    config = EvidenceConfig(yt_dlp_executable=str(fake_ytdlp), max_subtitle_bytes=32)
    collector = LocalEvidenceCollector(config, tmp_path)
    run_dir = tmp_path / "run"

    try:
        collector._fetch_youtube_transcript("https://www.youtube.com/watch?v=abc123", run_dir, 1, 0)
    except RuntimeError as exc:
        assert "subtitle file is too large" in str(exc)
    else:
        raise AssertionError("expected oversized subtitle to fail")


def test_read_info_json_returns_existing_metadata(tmp_path: Path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "video.info.json").write_text('{"title":"Titel","uploader":"Kanal"}', encoding="utf-8")

    info = LocalEvidenceCollector(EvidenceConfig(), tmp_path)._read_info_json(root)

    assert info["title"] == "Titel"
    assert info["uploader"] == "Kanal"
