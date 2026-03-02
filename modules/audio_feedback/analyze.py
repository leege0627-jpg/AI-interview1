from __future__ import annotations
from .features import compute_audio_features
from .scoring import score_voice_stability, score_clarity, merge_total, MetricResult

def estimate_words_per_min(transcript: str | None, duration_sec: float) -> float:
    if not transcript or duration_sec <= 0:
        return 0.0
    # 아주 단순: 공백 기준 토큰 수
    words = [w for w in transcript.strip().split() if w]
    wpm = (len(words) / duration_sec) * 60.0
    return float(wpm)

def analyze_audio_feedback(
    wav_path: str,
    transcript: str | None = None,
    stt_confidence_avg: float | None = None,
) -> dict:
    """
    wav_path: 분석할 wav 파일 경로
    transcript: STT 결과 텍스트(있으면 좋음)
    stt_confidence_avg: STT confidence 평균(있으면 좋음; Google STT는 alternative.confidence 등으로 얻을 수 있음)
    """
    feats = compute_audio_features(wav_path)
    wpm = estimate_words_per_min(transcript, feats["duration_sec"])

    voice_res: MetricResult = score_voice_stability(
        pitch_cv=feats["pitch_cv"],
        energy_cv=feats["energy_cv"],
        silence_ratio=feats["silence_ratio"],
    )

    clarity_res: MetricResult = score_clarity(
        stt_confidence_avg=stt_confidence_avg,
        words_per_min=wpm,
        silence_ratio=feats["silence_ratio"],
    )

    total = merge_total(voice_res.score, clarity_res.score)

    # 최종 출력 문장 만들기(너가 원하는 형태)
    lines: list[str] = []
    lines.append("[음성 피드백]")
    lines.append(f"- 목소리 안정성: {voice_res.score}점 ({voice_res.label})")
    for t in voice_res.tips:
        lines.append(f"  · {t}")
    lines.append("")
    lines.append(f"- 발음/명료도: {clarity_res.score}점 ({clarity_res.label})")
    for t in clarity_res.tips:
        lines.append(f"  · {t}")
    lines.append("")
    lines.append(f"총점: {total}점")

    return {
        "features": feats,
        "words_per_min": wpm,
        "voice": {"score": voice_res.score, "label": voice_res.label, "tips": voice_res.tips},
        "clarity": {"score": clarity_res.score, "label": clarity_res.label, "tips": clarity_res.tips},
        "total_score": total,
        "report_text": "\n".join(lines),
    }
