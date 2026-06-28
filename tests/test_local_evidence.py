from codcast.local_evidence import extract_youtube_urls, normalize_youtube_url, vtt_to_plain_text


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
