import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any

class CachedDataset(Dataset):
    def __init__(self, cache_paths: str | List[str]):
        if isinstance(cache_paths, str):
            cache_paths = [cache_paths]
            
        self.data = []
        self.vap_hop_s = 0.08
        
        for i, path in enumerate(cache_paths):
            print(f"Loading cached dataset from {path}...")
            payload = torch.load(path, map_location="cpu")
            shard_data = payload["data"]
            # 각 샘플에 어떤 데이터셋 출신인지 인덱스 부여
            for item in shard_data:
                item["dataset_idx"] = i
            
            self.data.extend(shard_data)
            if "vap_hop_s" in payload:
                self.vap_hop_s = payload["vap_hop_s"]
                
        print(f"Loaded total {len(self.data)} samples from {len(cache_paths)} files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        T = 125
        T_face = 250
        
        # ── 모든 구성 요소를 125/250 프레임으로 강제 절삭 ──────────────────
        # audio_latents: [C, T, D] -> [2, T, D]로 통일
        latents = item["audio_latents"]
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        # 1채널(모노)이면 2채널로 패딩
        if latents.shape[0] == 1:
            pad = torch.zeros(1, latents.shape[1], latents.shape[2], dtype=latents.dtype)
            latents = torch.cat([latents, pad], dim=0)
        latents = latents[:2, :T, :]
        
        # audio_codes: [C, K, T] -> [2, K, T]로 통일
        audio_codes = item["audio_codes"]
        if audio_codes.ndim == 2:
            audio_codes = audio_codes.unsqueeze(0)
        # 1채널(모노)이면 2채널로 패딩
        if audio_codes.shape[0] == 1:
            pad = torch.zeros(1, audio_codes.shape[1], audio_codes.shape[2], dtype=audio_codes.dtype)
            audio_codes = torch.cat([audio_codes, pad], dim=0)
        audio_codes = audio_codes[:2, :, :T]
        # vap_targets: [T]
        vap_targets = item["vap_targets"][:T]
        # face_motion: [T_face, 54]
        face_motion = item["face_motion"][:T_face, :]
        
        # 각 프레임의 모든 채널/D차원이 0인지 체크하여 유효 마스크 생성
        is_nonzero = (latents.abs().sum(dim=(0, 2)) > 1e-5)
        nz_indices = torch.where(is_nonzero)[0]
        valid_len = nz_indices[-1].item() + 1 if len(nz_indices) > 0 else T
        
        valid_mask = torch.zeros(T, dtype=torch.bool)
        valid_mask[:valid_len] = True
        
        # Face 프레임 유효 개수 (Mimi 12.5fps -> Face 25fps)
        valid_face_frames = min(valid_len * 2, T_face)
        
        # bc_timing_targets 생성: vap_targets를 기반으로 backchannel 이벤트 타겟 생성
        # vap_targets가 0이 아니면 backchannel 이벤트로 간주 (임시 로직)
        bc_timing_targets = (vap_targets != 0).float()  # [T] 형태의 바이너리 타겟, float 타입
        
        return {
            "audio_codes": audio_codes,
            "audio_latents": latents,
            "face_motion": face_motion,
            "vap_targets": vap_targets,
            "bc_timing_targets": bc_timing_targets,
            "voice_prompt_codes": item.get("voice_prompt_codes"),
            "valid_mask": valid_mask,
            "valid_face_frames": torch.tensor(valid_face_frames, dtype=torch.long),
            "text_tokens": item.get("text_tokens")
        }




def collate_cached_batch(batch: List[Dict[str, Any]], text_padding_id: int = 3):
    # 1. 오디오 코드 수집: [B, 2, 8, T]
    raw_audio_codes = torch.stack([item["audio_codes"] for item in batch])
    B, S, K, T = raw_audio_codes.shape
    
    # 2. 에이전트/유저 배치: [B, 2, 8, T] -> [B, 16, T]
    # 사용자 확인 결과 앞 8개(0-7)가 에이전트, 뒤 8개(8-15)가 유저입니다.
    audio_codes = raw_audio_codes.reshape(B, S * K, T)
    
    # 3. 텍스트 토큰 추가: [B, T] -> [B, 1, T]
    # 캐시된 실제 텍스트 토큰을 사용합니다. (없을 경우 패딩 처리)
    text_list = []
    for item in batch:
        t = item.get("text_tokens")
        if t is None:
            t = torch.full((T,), text_padding_id, dtype=torch.long)
        text_list.append(t[:T])
    text_tokens = torch.stack(text_list).unsqueeze(1).to(audio_codes.device) # [B, 1, T]
    codes = torch.cat([text_tokens, audio_codes], dim=1)
    
    audio_latents = torch.stack([item["audio_latents"] for item in batch])
    face_motion = torch.stack([item["face_motion"] for item in batch])
    vap_targets = torch.stack([item["vap_targets"] for item in batch])
    bc_timing_targets = torch.stack([item["bc_timing_targets"] for item in batch])
    
    # Handle voice_prompt_codes — pad to max T in batch
    vp_list = [item.get("voice_prompt_codes") for item in batch]
    if any(vp is not None for vp in vp_list):
        # 모든 VP를 [T, K] 형태로 통일 (필요시 transpose)
        processed_vps = []
        for vp in vp_list:
            if vp is None:
                processed_vps.append(None)
                continue
            # 만약 [8, T] 형태라면 [T, 8]로 변환
            if vp.shape[0] < vp.shape[1] and vp.shape[0] < 100: # K는 보통 작음 (8, 16 등)
                vp = vp.transpose(0, 1)
            processed_vps.append(vp)
            
        max_t = max((vp.shape[0] if vp is not None else 0) for vp in processed_vps)
        if max_t > 0:
            padded = []
            for vp in processed_vps:
                if vp is None:
                    # 보이스 프롬프트가 없는 경우 패딩 토큰(3)으로 가득 찬 더미 생성
                    K = next(v.shape[1] for v in processed_vps if v is not None)
                    vp = torch.full((max_t, K), 3, dtype=torch.long)
                elif vp.shape[0] < max_t:
                    # 시간 차원(0번) 패딩 (값은 3 사용)
                    vp = F.pad(vp, (0, 0, 0, max_t - vp.shape[0]), value=3)
                padded.append(vp)
            voice_prompt_codes = torch.stack(padded)
        else:
            voice_prompt_codes = None
    else:
        voice_prompt_codes = None

    valid_mask = torch.stack([item["valid_mask"] for item in batch])
    valid_face_frames = torch.stack([item["valid_face_frames"] for item in batch])
    dataset_idx = torch.tensor([item.get("dataset_idx", 0) for item in batch], dtype=torch.long)
        
    return {
        "audio_codes": codes, # 이제 17채널 (1 text + 16 interleaved audio)
        "audio_latents": audio_latents,
        "face_motion": face_motion,
        "vap_targets": vap_targets,
        "bc_timing_targets": bc_timing_targets,
        "voice_prompt_codes": voice_prompt_codes,
        "valid_mask": valid_mask,
        "valid_face_frames": valid_face_frames,
        "dataset_idx": dataset_idx
    }

def build_cached_loader(path: str | List[str], batch_size: int, shuffle: bool = True):
    if isinstance(path, str) and "," in path:
        path = [p.strip() for p in path.split(",")]
        
    dataset = CachedDataset(path)
    
    sampler = None
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle, drop_last=True
        )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        collate_fn=collate_cached_batch,
        num_workers=4,
        pin_memory=True,
        drop_last=True, # 모든 Rank가 동일한 배치 수를 갖도록 보장
    )
    return loader
