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
    cs_speech: List[str],
    cs_text: List[str],
    mono_text: List[str],
    output_dir: str = "outputs"
):
    """
    Run batch inference with lists of cs_speech, cs_text, and mono_text
    Args:
        cosyvoice_model: Initialized CosyVoice model
        cs_speech: List of paths to prompt speech wav files
        cs_text: List of prompt text strings corresponding to prompt speeches
        mono_text: List of target text strings to synthesize
        output_dir: Directory to save generated audio files
    """
    os.makedirs(output_dir, exist_ok=True)

    syn_data = []

    for idx, (p_speech, p_text, t_text) in enumerate(zip(cs_speech, cs_text, mono_text)):
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
                logger.info(f"Item {idx+1}/{len(cs_speech)} Saved output to {output_path}")
                syn_data.append({
                    'cs_speech': p_speech,
                    'cs_text': p_text,
                    'mono_speech': output_path,
                    'mono_text': t_text,
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
    parser.add_argument("--data_path", type=str, default="/home3/hexin/whisper_train_data/train_org/",
                        help="Path to data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a fine-tuned checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated audio files")
    
    return parser.parse_args()

def load_jsonl_data(jsonl_path: str):
    """
    Load data from a jsonl file and extract prompt_audio_list, prompt_text_list, and target_text_list.
    Args:
        jsonl_path: Path to the jsonl file
    Returns:
        prompt_audio_list: List of audio file paths (from 'audio' field)
        prompt_text_list: List of prompt texts (from 'input' field)
        target_text_list: List of target texts (from 'output' field)
    """
    prompt_audio_list = []
    prompt_text_list = []
    target_text_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompt_audio_list.append(item['audio'])
            prompt_text_list.append(item['input'])
            target_text_list.append(item['output'])
    return prompt_audio_list, prompt_text_list, target_text_list

def main():
    args = parse_arguments()
    logger.info("Starting preprocessing")

    # Load data from jsonl
    prompt_audio_list, prompt_text_list, target_text_list = load_jsonl_data(args.data_path)

    cosyvoice_model = setup_model(args.model_path, args.checkpoint)
    
    # Run batch inference
    batch_inference(
        cosyvoice_model,
        prompt_audio_list,
        prompt_text_list,
        target_text_list,
        output_dir=args.output_dir
    )

    result = calculate_total_duration(args.output_dir)
    
    logging.info(f"\nTotal duration of {result['file_count']} WAV files:")
    logging.info(f"  {result['formatted_time']} ({result['total_seconds']:.2f} seconds)")

if __name__ == "__main__":
    main()