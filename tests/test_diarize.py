"""Tests fuer die Auswahl-Logik der Sprechertrennung.

Deckt bewusst nur die reinen Funktionen ab: sie und nicht das Clustering
entscheiden, ob im Ergebnis wirklich nur eine Stimme landet. Laeuft ohne
torch und ohne GPU in der Projekt-venv.
"""

from codcast.config import load_config
from codcast.diarize import (
    Chunk,
    apply_gates,
    build_runs,
    erode_runs,
    select_chunks,
    smooth_labels,
    speaker_label,
    split_run,
)


def diarize_config(tmp_path, **overrides):
    config = load_config(tmp_path / "missing.yml").diarize
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# --- smooth_labels ---------------------------------------------------------


def test_smooth_labels_flips_single_outlier():
    assert smooth_labels([0, 0, 1, 0, 0]) == [0, 0, 0, 0, 0]


def test_smooth_labels_keeps_real_speaker_change():
    # Ein echter Wechsel besteht aus mehreren Fenstern und darf nicht wegge-
    # glaettet werden, sonst verschwimmt die Grenze zwischen den Sprechern.
    assert smooth_labels([0, 0, 0, 1, 1, 1]) == [0, 0, 0, 1, 1, 1]


def test_smooth_labels_leaves_short_input_alone():
    assert smooth_labels([0, 1]) == [0, 1]


# --- build_runs ------------------------------------------------------------


def test_build_runs_merges_overlapping_windows_of_same_speaker():
    windows = [(0.0, 1.5, 0), (0.75, 2.25, 0), (1.5, 3.0, 0)]
    runs = build_runs(windows)
    assert len(runs) == 1
    assert (runs[0].start, runs[0].end, runs[0].speaker) == (0.0, 3.0, 0)


def test_build_runs_breaks_on_speaker_change():
    runs = build_runs([(0.0, 1.5, 0), (0.75, 2.25, 1)])
    assert [run.speaker for run in runs] == [0, 1]


def test_build_runs_breaks_on_long_gap_but_not_on_breath_pause():
    same_speaker_breathing = build_runs([(0.0, 1.5, 0), (2.0, 3.5, 0)], max_gap_sec=1.0)
    assert len(same_speaker_breathing) == 1

    after_long_silence = build_runs([(0.0, 1.5, 0), (10.0, 11.5, 0)], max_gap_sec=1.0)
    assert len(after_long_silence) == 2


# --- erode_runs ------------------------------------------------------------


def test_erode_runs_trims_both_ends():
    eroded = erode_runs([Chunk(start=10.0, end=20.0, speaker=0, speech_sec=0.0)], 0.75, 3.0)
    assert (eroded[0].start, eroded[0].end) == (10.75, 19.25)


def test_erode_runs_drops_runs_that_become_too_short():
    # 4s Lauf, 0.75s je Seite weg -> 2.5s, unter min_run_sec=3.0.
    assert erode_runs([Chunk(start=0.0, end=4.0, speaker=0, speech_sec=0.0)], 0.75, 3.0) == []


# --- split_run -------------------------------------------------------------


def test_split_run_returns_single_chunk_for_continuous_speech():
    run = Chunk(start=0.0, end=8.0, speaker=0, speech_sec=0.0)
    chunks = split_run(run, [(0.0, 8.0)], min_chunk_sec=4.0, max_chunk_sec=12.0)
    assert len(chunks) == 1
    assert chunks[0].speech_sec == 8.0
    assert chunks[0].speech_ratio == 1.0


def test_split_run_cuts_at_silence_instead_of_mid_word():
    # Zwei Sprachbloecke mit einer Pause dazwischen; zusammen laenger als max.
    run = Chunk(start=0.0, end=20.0, speaker=0, speech_sec=0.0)
    speech = [(0.0, 8.0), (9.0, 17.0)]
    chunks = split_run(run, speech, min_chunk_sec=4.0, max_chunk_sec=12.0)
    assert len(chunks) == 2
    assert (chunks[0].start, chunks[0].end) == (0.0, 8.0)
    assert (chunks[1].start, chunks[1].end) == (9.0, 17.0)


def test_split_run_records_pauses_in_speech_ratio():
    run = Chunk(start=0.0, end=12.0, speaker=0, speech_sec=0.0)
    # 4s Sprache, 2s Pause, 4s Sprache -> 8s Sprache in 10s Chunk.
    chunks = split_run(run, [(0.0, 4.0), (6.0, 10.0)], min_chunk_sec=4.0, max_chunk_sec=12.0)
    assert len(chunks) == 1
    assert chunks[0].duration == 10.0
    assert chunks[0].speech_sec == 8.0
    assert chunks[0].speech_ratio == 0.8


def test_split_run_hard_cuts_a_single_overlong_region():
    run = Chunk(start=0.0, end=30.0, speaker=0, speech_sec=0.0)
    chunks = split_run(run, [(0.0, 30.0)], min_chunk_sec=4.0, max_chunk_sec=12.0)
    assert all(chunk.duration <= 12.0 for chunk in chunks)
    assert sum(chunk.duration for chunk in chunks) == 30.0


def test_split_run_drops_material_shorter_than_min_chunk():
    run = Chunk(start=0.0, end=3.0, speaker=0, speech_sec=0.0)
    assert split_run(run, [(0.0, 3.0)], min_chunk_sec=4.0, max_chunk_sec=12.0) == []


def test_split_run_without_speech_returns_nothing():
    run = Chunk(start=100.0, end=110.0, speaker=0, speech_sec=0.0)
    assert split_run(run, [(0.0, 8.0)], min_chunk_sec=4.0, max_chunk_sec=12.0) == []


# --- apply_gates -----------------------------------------------------------


def clean_chunk(**overrides):
    values = dict(
        start=0.0, end=10.0, speaker=0, speech_sec=10.0, margin=0.5, peak=0.7, rms_dbfs=-20.0
    )
    values.update(overrides)
    return Chunk(**values)


def test_apply_gates_accepts_a_clean_chunk(tmp_path):
    assert apply_gates(clean_chunk(), diarize_config(tmp_path)).rejected is None


def test_apply_gates_rejects_too_much_silence(tmp_path):
    chunk = apply_gates(clean_chunk(speech_sec=5.0), diarize_config(tmp_path))
    assert chunk.rejected is not None and "speech_ratio" in chunk.rejected


def test_apply_gates_rejects_low_margin(tmp_path):
    # Niedriger Margin heisst: klingt fast so sehr nach dem anderen Sprecher.
    # Genau so faellt ueberlappende Rede durch, ohne eigenen Detektor.
    chunk = apply_gates(clean_chunk(margin=0.01), diarize_config(tmp_path))
    assert chunk.rejected is not None and "margin" in chunk.rejected


def test_apply_gates_rejects_clipping(tmp_path):
    chunk = apply_gates(clean_chunk(peak=1.0), diarize_config(tmp_path))
    assert chunk.rejected is not None and "peak" in chunk.rejected


def test_apply_gates_rejects_near_silence_and_overdrive(tmp_path):
    config = diarize_config(tmp_path)
    assert "rms_dbfs" in apply_gates(clean_chunk(rms_dbfs=-60.0), config).rejected
    assert "rms_dbfs" in apply_gates(clean_chunk(rms_dbfs=-2.0), config).rejected


# --- select_chunks ---------------------------------------------------------


def scored(start, margin, duration=10.0, speaker=0):
    return Chunk(
        start=start, end=start + duration, speaker=speaker, speech_sec=duration, margin=margin
    )


def test_select_chunks_stops_once_the_target_is_reached():
    pool = [scored(index * 20.0, 0.5) for index in range(50)]
    selected = select_chunks(pool, target_sec=100.0, time_bins=10, total_duration=1000.0)
    assert sum(chunk.duration for chunk in selected) >= 100.0
    # Nicht wesentlich mehr nehmen als noetig.
    assert sum(chunk.duration for chunk in selected) < 120.0


def test_select_chunks_spreads_across_the_recording():
    # Die besten Margins liegen alle am Anfang; reine Sortierung wuerde nur
    # dort zugreifen und die Prosodie-Varianz verschenken.
    early = [scored(index * 10.0, 0.9) for index in range(10)]
    late = [scored(500.0 + index * 10.0, 0.4) for index in range(10)]
    selected = select_chunks(early + late, target_sec=60.0, time_bins=10, total_duration=1000.0)
    assert any(chunk.start >= 500.0 for chunk in selected)


def test_select_chunks_prefers_higher_margin_within_a_bin():
    pool = [scored(0.0, 0.2), scored(10.0, 0.9)]
    selected = select_chunks(pool, target_sec=10.0, time_bins=1, total_duration=100.0)
    assert selected[0].margin == 0.9


def test_select_chunks_returns_what_exists_when_material_is_short():
    pool = [scored(0.0, 0.5), scored(20.0, 0.5)]
    selected = select_chunks(pool, target_sec=300.0, time_bins=10, total_duration=100.0)
    assert sum(chunk.duration for chunk in selected) == 20.0


def test_select_chunks_returns_results_in_temporal_order():
    pool = [scored(index * 50.0, 1.0 - index * 0.1) for index in range(8)]
    selected = select_chunks(pool, target_sec=60.0, time_bins=8, total_duration=400.0)
    assert [chunk.start for chunk in selected] == sorted(chunk.start for chunk in selected)


def test_select_chunks_handles_empty_pool():
    assert select_chunks([], target_sec=300.0, time_bins=10, total_duration=100.0) == []


# --- speaker_label ---------------------------------------------------------


def test_speaker_label_naming():
    assert speaker_label(0) == "speaker_A"
    assert speaker_label(1) == "speaker_B"
