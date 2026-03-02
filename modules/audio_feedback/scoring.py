from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MetricResult:
    score: int
    label: str
    tips: list[str]

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def score_to_label(score: int, mode: str = "default") -> str:
    if mode == "pronunciation":
        # 발음 정확도 라벨
        return "좋음" if score >= 75 else ("보통" if score >= 55 else "부정확")
    else:
        # 목소리 떨림/안정성 라벨
        return "좋음" if score >= 75 else ("보통" if score >= 55 else "나쁨")

def score_voice_stability(pitch_cv: float, energy_cv: float, silence_ratio: float) -> MetricResult:
    """
    pitch_cv/energy_cv/silence_ratio가 낮을수록 안정적.
    아래는 '초기 기준치'라서 너 데이터 쌓이면 꼭 튜닝해야 함.
    """
    # pitch_cv: 0.05~0.20 정도에서 구간 나뉨(사람/마이크에 따라 다름)
    pitch_pen = clamp01((pitch_cv - 0.06) / (0.20 - 0.06))  # 0이면 좋음, 1이면 나쁨
    energy_pen = clamp01((energy_cv - 0.10) / (0.35 - 0.10))
    silence_pen = clamp01((silence_ratio - 0.10) / (0.35 - 0.10))  # 침묵이 너무 많으면 감점

    # 가중치
    bad = 0.55*pitch_pen + 0.25*energy_pen + 0.20*silence_pen
    score = int(round(100 * (1.0 - bad)))
    score = max(0, min(100, score))

    tips: list[str] = []
    if score < 55:
        tips += [
            "호흡을 길게 내쉬면서 일정한 톤으로 말해보세요(급하게 시작하면 떨림이 커져요).",
            "말하는 도중 볼륨이 들쭉날쭉하면 떨림처럼 느껴질 수 있어요. 한 톤으로 유지해보세요.",
        ]
    elif score < 75:
        tips += [
            "전체적으로 괜찮지만 문장 끝에서 톤/볼륨이 흔들릴 수 있어요. 끝맺음을 또렷하게 해보세요.",
        ]
    else:
        tips += ["목소리 톤과 볼륨이 안정적입니다."]

    return MetricResult(score=score, label=score_to_label(score, "default"), tips=tips)

def score_clarity(
    stt_confidence_avg: float | None,
    words_per_min: float,
    silence_ratio: float,
) -> MetricResult:
    """
    자유발화에서는 '발음 정확도' 대신 '명료도'로 점수화.
    - STT confidence 평균(있으면 가장 강력한 신호)
    - 말속도(WPM): 너무 빠르거나 너무 느리면 감점
    - 침묵 비율이 너무 높으면 감점
    """
    tips: list[str] = []

    # confidence 점수(없으면 중립값)
    if stt_confidence_avg is None:
        conf_score = 70.0
    else:
        conf_score = max(0.0, min(100.0, stt_confidence_avg * 100.0))

    # 속도 점수: 110~170 WPM을 적정으로 가정(한국어는 측정/토큰화에 따라 달라서 참고용)
    # 너무 빠르면 - , 너무 느리면 -
    if words_per_min <= 1.0:
        speed_score = 50.0
    else:
        # 중앙 140, 허용폭 +-30
        diff = abs(words_per_min - 140.0)
        speed_score = max(0.0, 100.0 - (diff / 30.0) * 35.0)  # diff=30이면 65점 정도

    # 침묵 페널티
    silence_score = max(0.0, 100.0 - clamp01((silence_ratio - 0.10) / (0.40 - 0.10)) * 50.0)

    # 최종
    total = 0.55*conf_score + 0.25*speed_score + 0.20*silence_score
    score = int(round(total))
    score = max(0, min(100, score))

    # 팁
    if stt_confidence_avg is not None and stt_confidence_avg < 0.70:
        tips.append("발음이 뭉개질 수 있어요. 단어 끝소리를 살리고, 입을 더 크게 열어 말해보세요.")
    if words_per_min > 180:
        tips.append("말속도가 빠른 편이에요. 중요한 문장에서 한 박자 쉬어가면 명료도가 좋아져요.")
    elif 0 < words_per_min < 100:
        tips.append("말속도가 느린 편이에요. 문장 단위로 리듬감을 주면 더 자연스럽게 들려요.")
    if silence_ratio > 0.30:
        tips.append("중간 침묵이 많은 편이에요. 생각 정리는 짧게 하고, 연결어로 흐름을 이어보세요(예: '그리고', '또한').")

    if not tips:
        tips.append("전반적으로 발화가 또렷합니다.")

    return MetricResult(score=score, label=score_to_label(score, "pronunciation"), tips=tips)

def merge_total(voice_score: int, clarity_score: int) -> int:
    return int(round(voice_score * 0.5 + clarity_score * 0.5))
