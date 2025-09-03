import os
import subprocess
import sys
import shutil
from pathlib import Path

# setup.py will clone and install VibeVoice, copy voice files if they exist

def setup_vibevoice():
    repo_dir = "VibeVoice"
    original_dir = os.getcwd()
    
    # clone repo if needed
    if not os.path.exists(repo_dir):
        print("Cloning the VibeVoice repository...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/microsoft/VibeVoice.git"],
                check=True,
                capture_output=True,
                text=True
            )
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr}")
            sys.exit(1)
    else:
        print("Repository already exists. Skipping clone.")
    
    # install the package
    os.chdir(repo_dir)
    print(f"Changed directory to: {os.getcwd()}")
    
    print("Installing the VibeVoice package...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=True,
            capture_output=True,
            text=True
        )
        print("Package installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e.stderr}")
        sys.exit(1)
    
    # add to python path
    sys.path.insert(0, os.getcwd())
    
    # go back to original directory
    os.chdir(original_dir)
    print(f"Changed back to original directory: {os.getcwd()}")
    
    # copy voice files if they exist
    if os.path.exists("public/voices"):
        target_voices_dir = os.path.join(repo_dir, "demo", "voices")
        
        # clear existing voices and use only ours
        if os.path.exists(target_voices_dir):
            shutil.rmtree(target_voices_dir)
        os.makedirs(target_voices_dir)
        
        for file in os.listdir("public/voices"):
            if file.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                src = os.path.join("public/voices", file)
                dst = os.path.join(target_voices_dir, file)
                shutil.copy2(src, dst)
                print(f"Copied voice file: {file}")
    
    return repo_dir

def setup_voice_presets():
    # get voice files from vibevoice demo directory
    voices_dir = Path("VibeVoice/demo/voices")
    
    voice_presets = {}
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
    
    if voices_dir.exists():
        for audio_file in voices_dir.glob("*"):
            if audio_file.suffix.lower() in audio_extensions:
                name = audio_file.stem
                voice_presets[name] = str(audio_file)
    
    # if no voices found, create directory
    if not voice_presets and not voices_dir.exists():
        voices_dir.mkdir(parents=True, exist_ok=True)
    
    return dict(sorted(voice_presets.items()))