"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import collections
import torchaudio as _torchaudio

if not hasattr(_torchaudio, "AudioMetaData"):
    _torchaudio.AudioMetaData = collections.namedtuple(
        "AudioMetaData",
        ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"]
    )
if not hasattr(_torchaudio, "list_audio_backends"):
    _torchaudio.list_audio_backends = lambda: ["soundfile"]
if not hasattr(_torchaudio, "get_audio_backend"):
    _torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(_torchaudio, "set_audio_backend"):
    _torchaudio.set_audio_backend = lambda x: None
try:
    import torchaudio as _ta
    _ta.set_audio_backend("soundfile")
except Exception:
    pass
import functools as _functools
import torch as _torch
_original_torch_load = _torch.load
@_functools.wraps(_original_torch_load)
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
_torch.load = _patched_load

import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError, AttributeError) as e:
        logger.warning("pyannote.audio import failed: %s", e)
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        import traceback
        logger.warning("Diarization failed for %s: %s\n%s", audio_path, exc, traceback.format_exc())
        return []

def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    result = []
    for seg in segments:
        seg_copy = seg.copy()
        seg_start = seg["start"]
        seg_end = seg["end"]
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for diar in diarization:
            overlap = max(0, min(seg_end, diar["end_s"]) - max(seg_start, diar["start_s"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar["speaker"]
        seg_copy["speaker"] = best_speaker
        result.append(seg_copy)
    return result
