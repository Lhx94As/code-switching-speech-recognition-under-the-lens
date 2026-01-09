import os
import random
import json
from tqdm import tqdm

def preprocess_files(folder):
      wavscp = os.path.join(folder, "wav.scp")
      text = os.path.join(folder, "text")
      with open(wavscp, 'r') as f1:
            wave_data = f1.readlines()
      with open(text, 'r') as f2:
            trans_data = f2.readlines()
      audio_list = [audio_line.strip().split()[1] for audio_line in wave_data]
      identifier_list = [audio_line.strip().split()[0] for audio_line in wave_data]
      text_list = [' '.join(text_line.split()[1:]).lower().strip() for text_line in trans_data]
      return audio_list, text_list, identifier_list

def remove_special_tokens(text_list):
    text = [i.replace('<noise>', '') for i in text_list]
    text = [i.replace('<UNK>','') for i in text_list]
    return text


def extract_speaker_id_seame(identifier_list):
    spk_list = [x.split('-')[0] for x in identifier_list]
    return spk_list


def extract_speaker_id_asru(identifier_list):
    spk_list = [x.split('_')[0] for x in identifier_list]
    return spk_list

def extract_speaker_id_librispeech(identifier_list):
    spk_list = [x.split('-')[0] for x in identifier_list]
    return spk_list



def select_different_speakers(target_speaker_ids, prompt_speaker_ids):
    result_indices = []
    for target_id in tqdm(target_speaker_ids, desc="Finding different speakers", unit="ID"):
        different_speaker_indices = [
            idx for idx, spk_ in enumerate(prompt_speaker_ids) 
            if spk_ != target_id
        ]
        
        if different_speaker_indices:
            result_indices.append(random.choice(different_speaker_indices))
        else:
            result_indices.append(None)
    
    return result_indices

def create_mapped_lists(indices, list_a, list_b):

    # Create new empty lists
    list_a_prime = []
    list_b_prime = []
    
    for idx in indices:
        # Check if the index is valid
        if 0 <= idx < len(list_a) and 0 <= idx < len(list_b):
            list_a_prime.append(list_a[idx])
            list_b_prime.append(list_b[idx])
        else:
            # Handle out-of-bounds indices
            print(f"Warning: Index {idx} is out of bounds and will be skipped")
    return list_a_prime, list_b_prime

def save_pairs_to_json(list_a_prime, list_b_prime, output_file):
    # Ensure the lists are of equal length
    if len(list_a_prime) != len(list_b_prime):
        raise ValueError("The mapped lists must have the same length.")
    
    # Create a list of paired dictionaries
    pairs = [{"audio": a, "text": b} for a, b in zip(list_a_prime, list_b_prime)]
    
    # Write the list of pairs to a JSON file with indentation for readability
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=4)
    print(f"Prompts has been saved to {output_file}")

def load_pairs_from_json(input_file):
    with open(input_file, "r") as f:
        pairs = json.load(f)
    
    # Extract values into two separate lists
    list_a = [pair["audio"] for pair in pairs]
    list_b = [pair["text"] for pair in pairs]
    
    return list_a, list_b

def filter_short_texts(audios, texts, spks, min_words=5):

    if not len(texts) == len(audios) == len(spks):
        raise ValueError("Audio and text lists must have the same length")
    
    filtered_texts = []
    filtered_audios = []
    filtered_spks = []
    
    for text, audio, spk in zip(texts, audios, spks):
        # Count words by splitting on whitespace
        word_count = len(text.split(" "))
        
        if word_count >= min_words:
            filtered_texts.append(text)
            filtered_audios.append(audio)
            filtered_spks.append(spk)
    print(f"Filtering completed, ori: {len(audios)} filtered: {len(filtered_audios)}")
    
    return filtered_audios, filtered_texts, filtered_spks

def convert_to_text_file(dict_list, output_dir):
    output_wav = os.path.join(output_dir, "wav.scp")
    output_text = os.path.join(output_dir, "text")
    with open(output_wav, 'w') as f:
        for item in dict_list:
            f.write(f"{item['name']} {item['audio']}\n")
    with open(output_text, 'w') as f:
        for item in dict_list:
            f.write(f"{item['name']} {item['text']}\n")