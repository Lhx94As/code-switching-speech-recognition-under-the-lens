import os
import librosa
import concurrent.futures
from tqdm import tqdm

def get_audio_duration(file_path):
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def calculate_total_duration(directory_path):
    # Get all WAV files
    audio_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} WAV files")
    
    # Process files in parallel
    total_duration = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for duration in tqdm(executor.map(get_audio_duration, audio_files), total=len(audio_files)):
            total_duration += duration
    
    # Convert to hours, minutes, seconds
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    
    return {
        "total_seconds": total_duration,
        "formatted_time": f"{hours}h {minutes}m {seconds}s",
        "file_count": len(audio_files)
    }

# Usage
if __name__ == "__main__":
    directory = "/home3/hexin/tts_finetune/CosyVoice/seame_ft_2/"  # Change this to your directory
    result = calculate_total_duration(directory)
    
    print(f"\nTotal duration of {result['file_count']} WAV files:")
    print(f"  {result['formatted_time']} ({result['total_seconds']:.2f} seconds)")