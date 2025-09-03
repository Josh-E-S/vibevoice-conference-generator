#!/bin/bash
# Script to copy voice files to VibeVoice demo directory

echo "Setting up voice files..."

# Create the target directory if it doesn't exist
mkdir -p VibeVoice/demo/voices

# Copy voice files from root voices directory if it exists
if [ -d "voices" ]; then
    echo "Copying voice files from voices/ to VibeVoice/demo/voices/"
    cp voices/*.mp3 VibeVoice/demo/voices/ 2>/dev/null
    cp voices/*.wav VibeVoice/demo/voices/ 2>/dev/null
    echo "Voice files copied successfully!"
else
    echo "No voices directory found in root"
fi

# List the voice files
echo "Available voice files:"
ls -la VibeVoice/demo/voices/