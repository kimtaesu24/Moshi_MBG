from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    "kyutai/DailyTalkContiguous",
    repo_type="dataset",
    local_dir="./daily-talk-contiguous"
)