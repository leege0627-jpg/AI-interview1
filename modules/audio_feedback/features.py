from __future__ import annotations
import numpy as np
import soundfile as sf

def load_wav_mono(wav_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    wav를 읽어서 mono float32 [-1,1]로 반환.
    target_sr로 리샘플은 여기서는 안 함(복잡도 줄이기).
    -> 네 파이프라인에서 16k로 저장하는 걸 권장.
    """
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, sr

def frame_rms(y: np.ndarray, sr: int, frame_ms: int = 30, hop_ms: int = 10) -> np.ndarray:
    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = max(1, int(sr * hop_ms / 1000))
    if len(y) < frame_len:
        return np.array([float(np.sqrt(np.mean(y**2)))], dtype=np.float32)

    rms = []
    for i in range(0, len(y) - frame_len + 1, hop_len):
        frame = y[i:i+frame_len]
        rms.append(float(np.sqrt(np.mean(frame**2) + 1e-12)))
    return np.array(rms, dtype=np.float32)

def simple_voiced_mask(rms: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """
    아주 간단한 VAD 대용: RMS가 일정 이상이면 유성(말하는 구간)으로 간주
    threshold 미지정이면 RMS 중앙값 기반으로 자동 설정
    """
    if threshold is None:
        med = float(np.median(rms))
        threshold = max(1e-4, med * 1.3)
    return rms > threshold

def pitch_track_parselmouth(y: np.ndarray, sr: int, f0_min: float = 75.0, f0_max: float = 300.0) -> np.ndarray:
    """
    Praat(parselmouth)로 pitch 추적. 무성 구간은 0에 가까운 값으로 나오므로 후처리에서 제거.
    """
    import parselmouth
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0 = pitch.selected_array["frequency"].astype(np.float32)  # Hz
    return f0

def compute_audio_features(wav_path: str) -> dict:
    y, sr = load_wav_mono(wav_path)

    rms = frame_rms(y, sr)
    voiced = simple_voiced_mask(rms)

    # 침묵 비율(프레임 기준)
    silence_ratio = 1.0 - float(np.mean(voiced))

    # 볼륨 변동(유성 프레임에서)
    voiced_rms = rms[voiced] if np.any(voiced) else rms
    energy_cv = float(np.std(voiced_rms) / (np.mean(voiced_rms) + 1e-12))

    # pitch 변동
    f0 = pitch_track_parselmouth(y, sr)
    # pitch 배열은 10ms step. RMS voiced mask는 hop 10ms라 대략 길이 맞는 편이지만 완벽 일치 아님.
    # 그래서 f0에서 유효값만 써서 변동계수 계산.
    f0_valid = f0[(f0 > 50.0) & (f0 < 500.0)]
    if len(f0_valid) < 10:
        pitch_cv = 0.25  # 정보 부족이면 보수적으로
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) else 0.0
    else:
        f0_mean = float(np.mean(f0_valid))
        pitch_cv = float(np.std(f0_valid) / (f0_mean + 1e-12))

    duration_sec = float(len(y) / sr)

    return {
        "sr": sr,
        "duration_sec": duration_sec,
        "silence_ratio": silence_ratio,
        "energy_cv": energy_cv,
        "pitch_cv": pitch_cv,
        "f0_mean": f0_mean,
    }
