"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""
import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


def _build_speaker_voice_map(segments: list[dict], lang: str = "en") -> dict[str, str]:
    """Map each unique speaker label to a reference WAV file.

    Uses round-robin assignment over available WAVs in
    pipeline_data/speakers/{lang}/. Falls back to empty string
    (default voice) if no WAVs are found.

    Args:
        segments: Translated segments with optional 'speaker' field.
        lang: Language subdirectory to look in (default 'en').

    Returns:
        Dict mapping speaker label to relative WAV path e.g.
        {'SPEAKER_00': 'en/SPEAKER_00.wav', 'SPEAKER_01': 'en/SPEAKER_01.wav'}
    """
    speakers_base = pathlib.Path(__file__).parent.parent.parent.parent / "pipeline_data" / "speakers"
    lang_dir = speakers_base / lang

    # Get sorted list of available WAV files
    available_wavs = sorted(lang_dir.glob("*.wav")) if lang_dir.exists() else []

    # Get unique speakers from segments in order of appearance
    seen = []
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker not in seen:
            seen.append(speaker)

    if not seen or not available_wavs:
        return {}

    # Round-robin assignment
    voice_map = {}
    for i, speaker in enumerate(seen):
        wav = available_wavs[i % len(available_wavs)]
        voice_map[speaker] = f"{lang}/{wav.name}"

    return voice_map


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    When speaker labels exist in segments, uses per-speaker voice cloning.
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"
    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = trans_dir / f"{title}.json"

    # Build speaker voice map if segments have speaker labels
    speaker_voice_map = {}
    if source_path.exists():
        with open(source_path) as f:
            trans_data = json.load(f)
        segments = trans_data.get("segments", [])
        has_speakers = any(seg.get("speaker") for seg in segments)
        if has_speakers:
            speaker_voice_map = _build_speaker_voice_map(segments, lang="en")
            if speaker_voice_map:
                print(f"[tts] Per-speaker voice map: {speaker_voice_map}")

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        str(source_path),
        str(audio_dir),
        alignment=alignment,
        speaker_voice_map=speaker_voice_map,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")