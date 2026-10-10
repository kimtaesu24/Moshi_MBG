import torch
from glob import glob
import os
import soundfile as sf


def balance_left_channel(waveform, threshold=0.8):
    """
    waveform: [2, time] 형태의 Tensor
    threshold: 채널 0이 채널 1의 이 비율보다 작을 때 보정 실행
    """
    # 1. 각 채널의 RMS 계산
    waveform = torch.from_numpy(waveform)
    rms = waveform.pow(2).mean(dim=1).sqrt()
    left_rms = rms[0]
    right_rms = rms[1]

    # 2. 왼쪽(0번)이 오른쪽(1번)보다 작을 경우에만 처리
    if left_rms < right_rms * threshold:
        # 보정 계수 계산 (오른쪽 크기에 맞춤)
        gain_factor = right_rms / (left_rms + 1e-8)
        
        # 왼쪽 채널에만 Gain 적용
        waveform[0] = waveform[0] * gain_factor
        
        # 3. 중요: 소리 찢어짐(Clipping) 방지 프로세스
        # 보정 후 왼쪽 채널이나 전체 waveform에 1.0을 넘는 피크가 생겼는지 확인
        max_peak = waveform.abs().max()
        if max_peak > 0.95:
            # 전체 볼륨을 안전 범위(0.95)로 다시 스케일 다운
            waveform = waveform * (0.95 / max_peak)
            
    return waveform

if __name__ == "__main__":
    paths = glob("/home2/s20235100/Conversational-AI/Moshi/moshi-finetune/stereo_ami/data_stereo/*.wav")

    save_path = "/home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune/data/stereo_ami_balanced/data_stereo/"

    os.makedirs(save_path, exist_ok=True)

    for path in paths:
        waveform, sr = sf.read(path)
        waveform_balanced = balance_left_channel(waveform)
        sf.write(save_path + path.split("data_stereo/")[-1], waveform_balanced, sr)

    os.system(f"cp stereo_ami/data.jsonl /home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune/data/stereo_ami_balanced/data.jsonl")
    os.system(f"cp stereo_ami/data_stereo/*.json /home2/s20235100/Conversational-AI/personaplex_MBG/moshi-finetune/data/stereo_ami_balanced/data_stereo/")