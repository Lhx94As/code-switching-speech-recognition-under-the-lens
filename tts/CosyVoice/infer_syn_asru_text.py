import sys
import os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, load_wav_16k
from transformers import AutoModel, AutoTokenizer
import torch
import torchaudio
import json
import time
import logging
from argparse import ArgumentParser
from typing import List, Dict, Optional
import glob
from utils import *
from summarize_info import *

# Global constants
SAMPLE_RATE = 16000
OUTPUT_PREFIX = 'output'

# Setup global logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CosyVoice")



def setup_model(model_path: str, checkpoint_path: Optional[str] = None):
    """
    Initialize the CosyVoice model and load a fine-tuned checkpoint if provided
    
    Args:
        model_path: Path to the pretrained model
        checkpoint_path: Optional path to a fine-tuned checkpoint
        
    Returns:
        Initialized CosyVoice model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize CosyVoice model
    cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False, device=device)
    
    # Load fine-tuned checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading fine-tuned checkpoint from {checkpoint_path}")
        ft_ckpt = torch.load(checkpoint_path)
        
        # Remove 'module.' prefix if it exists (common when trained with DataParallel)
        new_ft_ckpt = {k.replace("module.", ""): v for k, v in ft_ckpt.items()}
        
        cosyvoice.model.llm.load_state_dict(new_ft_ckpt, strict=True)
        logger.info('Successfully loaded fine-tuned checkpoint!')
    else:
        logger.info('Zero-shot inference!')
    
    return cosyvoice

def batch_inference(
    cosyvoice_model,
    prompt_speech_files: List[str],
    prompt_texts: List[str],
    target_texts: List[str],
    output_dir: str = "outputs"
):
    """
    Run batch inference with lists of prompt speeches, prompt texts, and target texts
    
    Args:
        cosyvoice_model: Initialized CosyVoice model
        prompt_speech_files: List of paths to prompt speech wav files
        prompt_texts: List of prompt text strings corresponding to prompt speeches
        target_texts: List of target text strings to synthesize
        output_dir: Directory to save generated audio files
    """
    os.makedirs(output_dir, exist_ok=True)

    syn_data = []

    for idx, (p_speech, p_text, t_text) in enumerate(zip(prompt_speech_files, prompt_texts, target_texts)):
        try:
            prompt_speech = load_wav(p_speech, 16000)
            identifier = os.path.split(p_speech)[-1].split('.')[0]
            
            for out_idx, output in enumerate(cosyvoice_model.inference_zero_shot(t_text, p_text, prompt_speech, stream=False)):
                file_name =  f"{identifier}_syn_{idx+1}_{out_idx+1}"
                output_path = os.path.join('/home3/hexin/tts_finetune/CosyVoice', output_dir, f"{file_name}.wav")
                torchaudio.save(
                    output_path, 
                    output['tts_speech'], 
                    cosyvoice_model.sample_rate
                )
                logger.info(f"Item {idx+1}/{len(prompt_speech_files)} Saved output to {output_path}")
                syn_data.append({
                    'name': file_name,
                    'audio': output_path,
                    'text': t_text,
                    })
                
        except Exception as e:
            logger.error(f"Error processing item {idx+1}: {str(e)}", exc_info=True)
    
    json_file_name = os.path.join('/home3/hexin/tts_finetune/CosyVoice', output_dir, "syn_data.json")
    with open(json_file_name, 'w', encoding="utf-8") as json_file:
        json.dump(syn_data, json_file, ensure_ascii=False, indent=4)
    convert_to_text_file(syn_data, output_dir)
    
    logger.info(f"Batch processing complete. Outputs saved to {output_dir}")

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
                        help="Path to data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a fine-tuned checkpoint")
    parser.add_argument("--prompt", type=str, default='seame',
                        help="prompt")
    parser.add_argument("--prompt_path", type=str, default="/home3/hexin/whisper_train_data/train_org/",
                        help="Path to prompt data")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated audio files")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger.info("Starting preprocessing")
    # audio_list, text_list, identifier_list = preprocess_files(args.data_path)
    prompt_audio_list, prompt_text_list, prompt_identifier_list = preprocess_files(args.prompt_path)
    # filtered_audio, filtered_text, filtered_spk = filter_short_texts(audio_list, text_list, identifier_list, min_words=3)
    data = []
    # with open("/home3/hexin/asru_data/llm_data/pure_Man_text_codeswitched_auto.jsonl", "r", encoding="utf-8") as f:
    #     for line in f:
    #         data.append(json.loads(line))
    # target_texts = [x['output'] for x in data]
    with open("/home3/hexin/asru_data/ds_prompts/deepseek_wo_exp_revised.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    target_texts = [' '.join(text_line.split(' ')[1:]).lower().strip('\n') for text_line in lines]
    identifier_list = [line.split(" ")[0].strip() for line in lines]

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

    promtp_memory = f"prompt_t_ds_wo_exp_{args.dataset}_p_{args.prompt}.json"
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
            save_pairs_to_json(prompt_audio_list, prompt_text_list, promtp_memory)
    # Initialize the model
    logging.info(f"num spks: {len(list(set(spk_list)))} num prompt spks: {len(list(set(prompt_spk_list)))}")
    logger.info("Preprocessing done, setup model")
    cosyvoice_model = setup_model(args.model_path, args.checkpoint)
    
    # Run batch inference
    batch_inference(
        cosyvoice_model,
        prompt_audio_list,
        prompt_text_list,
        target_texts,
        output_dir=args.output_dir
    )

    result = calculate_total_duration(args.output_dir)
    
    logging.info(f"\nTotal duration of {result['file_count']} WAV files:")
    logging.info(f"  {result['formatted_time']} ({result['total_seconds']:.2f} seconds)")

if __name__ == "__main__":
    main()