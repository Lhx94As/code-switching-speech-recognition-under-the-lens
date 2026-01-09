import os
import json
import time
import logging
from argparse import ArgumentParser
from typing import List, Dict, Optional
from utils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CosyVoice")


def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        Parsed arguments object
    """
    parser = ArgumentParser(description="CosyVoice Batch TTS Processing")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, default="pretrained_models/CosyVoice2-0.5B",
                        help="Path to the pretrained CosyVoice model")
    parser.add_argument("--dataset", type=str, default='seame',
                        help="dataset")
    parser.add_argument("--data_path", type=str, default="/home3/hexin/whisper_train_data/train_org/",
                        help="Path to the pretrained CosyVoice model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a fine-tuned checkpoint")
    parser.add_argument("--prompt", type=str, default='seame',
                        help="dataset")
    parser.add_argument("--prompt_path", type=str, default="/home3/hexin/whisper_train_data/train_org/",
                        help="Path to the pretrained CosyVoice model")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated audio files")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger.info("Starting preprocessing")
    audio_list, text_list, identifier_list = preprocess_files(args.data_path)
    prompt_audio_list, prompt_text_list, prompt_identifier_list = preprocess_files(args.prompt_path)
    # filtered_audio, filtered_text, filtered_spk = filter_short_texts(audio_list, text_list, identifier_list, min_words=3)
    target_texts = text_list
    spk_list, prompt_spk_list = [], []
    if args.dataset == 'seame':
        spk_list = extract_speaker_id_seame(identifier_list)
    elif  args.dataset == 'asru':
        spk_list = extract_speaker_id_asru(identifier_list)
    else:
        raise ValueError('dataset only support seame or asru')
    
    if args.prompt == 'seame':
        prompt_spk_list = extract_speaker_id_seame(prompt_identifier_list)
    elif args.prompt == 'asru':
        prompt_spk_list = extract_speaker_id_asru(prompt_identifier_list)
    elif args.prompt == 'librispeech':
        prompt_spk_list = extract_speaker_id_librispeech(prompt_identifier_list)
    else:
        raise ValueError('dataset not supported')

    promtp_memory = f"prompt_t_{args.dataset}_p_{args.prompt}.json"
    if os.path.exists(promtp_memory):
        prompt_audio_list, prompt_text_list = load_pairs_from_json(promtp_memory)
        logging.info(f"loading prompt from existing path: {promtp_memory}")
    else:
        if args.dataset == args.prompt:
            prompt_spk_ids = select_different_speakers(spk_list, prompt_spk_list)
            prompt_audio_list, prompt_text_list = create_mapped_lists(prompt_spk_ids, prompt_audio_list, prompt_text_list)
            save_pairs_to_json(prompt_audio_list, prompt_text_list, promtp_memory)
        else:
            prompt_spk_ids = random.choices(range(len(prompt_spk_list)), k=len(spk_list))
            prompt_audio_list, prompt_text_list = create_mapped_lists(prompt_spk_ids, prompt_audio_list, prompt_text_list)
    # Initialize the model
    logging.info(f"num spks: {len(list(set(spk_list)))} num prompt spks: {len(list(set(prompt_spk_list)))}")
    

# syn_data_file = "/home3/hexin/tts_finetune/CosyVoice/seame_zeroshot/syn_data.json"
# prompt_audio_list, prompt_text_list = load_pairs_from_json(promtp_memory)