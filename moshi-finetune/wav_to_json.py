import sphn
import json
from pathlib import Path
from glob import glob


import soundfile as sf
import numpy as np
from glob import glob
import os

def make_stereo_ami():
    bc_paths = glob("/home2/s20235100/data_process/AMI_corpus/dataset/*/bc_audio_with_silence_process_single/*.wav")
    ut_paths = glob("/home2/s20235100/data_process/AMI_corpus/dataset/*/ut_audio/*.wav")

    # Create dictionary for ut files with explicit matching
    ut_dict = {}
    for ut_path in ut_paths:
        ut_filename = os.path.basename(ut_path)  # e.g., "EN2001a_0_ut.wav"
        
        # Extract the bc equivalent name explicitly
        # Remove "_ut.wav" from the end and add ".wav"
        if ut_filename.endswith("_ut.wav"):
            bc_equivalent = ut_filename[:-7] + ".wav"  # Remove last 7 chars ("_ut.wav"), add ".wav"
            ut_dict[bc_equivalent] = ut_path

    # Process each bc file
    for bc_path in bc_paths:
        bc_filename = os.path.basename(bc_path)  # e.g., "EN2001a_0.wav"
        
        # Look for matching ut file
        if bc_filename in ut_dict:
            ut_path = ut_dict[bc_filename]
            
            print(f"Matching pair:")
            print(f"  BC: {bc_filename}")
            print(f"  UT: {os.path.basename(ut_path)}")
            
            # Read both audio files
            bc_audio, sr_bc = sf.read(bc_path)
            ut_audio, sr_ut = sf.read(ut_path)
            
            # Convert to mono if stereo
            if bc_audio.ndim > 1:
                bc_audio = bc_audio.mean(axis=1)
            if ut_audio.ndim > 1:
                ut_audio = ut_audio.mean(axis=1)
            
            print(f"  BC shape: {bc_audio.shape}, UT shape: {ut_audio.shape}")
            
            # Ensure same sample rate
            assert sr_bc == sr_ut, f"Sample rates don't match: {sr_bc} vs {sr_ut}"
            
            # Make them the same length (pad shorter one with zeros)
            max_len = max(len(bc_audio), len(ut_audio))
            if len(bc_audio) < max_len:
                bc_audio = np.pad(bc_audio, (0, max_len - len(bc_audio)))
            if len(ut_audio) < max_len:
                ut_audio = np.pad(ut_audio, (0, max_len - len(ut_audio)))
            
            print(f"  After padding - BC shape: {bc_audio.shape}, UT shape: {ut_audio.shape}")
            
            # Stack as stereo: left channel = bc, right channel = ut
            stereo_audio = np.stack([bc_audio, ut_audio], axis=1)
            
            # Save the stereo file
            output_dir = "./stereo_ami/data_stereo"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, bc_filename.replace(".wav", "_stereo.wav"))
            
            sf.write(output_path, stereo_audio, sr_bc)
            print(f"  Created: {os.path.basename(output_path)}\n")
        else:
            print(f"Warning: No matching ut file for {bc_filename}")

def make_json():
    paths = [str(f) for f in Path("./stereo_ami/data_stereo").glob("*.wav")]

    durations = sphn.durations(paths)

    with open("data.jsonl", "w") as fobj:
        for p, d in zip(paths, durations):
            if d is None:
                continue
            json.dump({"path": p, "duration": d}, fobj)
            fobj.write("\n")

if __name__ == "__main__":
    # make_stereo_ami()
    make_json()