import os
import glob
import torch
import torch.utils
import torch.utils.data
import collections
import argparse
from jiwer import wer as calculate_wer
from WhisperDataPre import WhisperDataset
from whisper.normalizers import EnglishTextNormalizer
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import logging
from convert_text import convert_text
from utils import *

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")

normalizer = EnglishTextNormalizer()


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        state = torch.load(fpath)
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        
        model_params = state.state_dict()

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(fpath, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    
    # new_state = WhisperForConditionalGeneration(config)  # Replace `config` with the appropriate model config
    new_state.load_state_dict(averaged_params)
    # new_state.state_dict() = averaged_params
    return new_state


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--test', type=str, default='/home3/hexin/espnet/egs2/asru/asr1/data/test/')
    parser.add_argument('--model', type=str, default="openai/whisper-small")
    parser.add_argument('--zeroshot', type=str2bool, default=False)
    parser.add_argument('--lang', type=str, default="zh")
    parser.add_argument('--save_dir', type=str, default="exp")
    # parser.add_argument('--average', type=str2bool, default=True)
    args = parser.parse_args()

    normalizer = EnglishTextNormalizer()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for single GPU, may need DDP for multiple GPUs

    test_path = args.test
    logging.info(f"Will evaluate on {test_path}")
    test_audio_list, test_text_list = preprocess_files(test_path)

    tokenizer = WhisperTokenizer.from_pretrained(args.model, language=args.lang, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model, language=args.lang, task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)

    save_dir = args.save_dir


    def evaluate(model, test_data, text_list):
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")
        with torch.no_grad():
                all_pred, all_gt = [], []
                for idx_, (feats, label_batch, labels) in enumerate(test_data):
                    generated_ids = model.generate(inputs=feats.to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    pred = normalizer(generated_text)
                    pred = pred if len(pred) > 0 else '<UNK>'
                    gt = normalizer(text_list[idx_])
                    gt = gt if len(gt) > 0 else '<UNK>'
                    logging.info(f"pred: {pred}, gt: {gt}")

                    all_pred.append(pred)
                    all_gt.append(gt)

                return calculate_wer(all_gt, all_pred)
    
    def evaluate_zero_shot(model, test_data, text_list):
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")
        with torch.no_grad():
                all_pred, all_gt = [], []
                for idx_, (feats, label_batch, labels) in enumerate(test_data):
                    generated_ids = model.generate(inputs=feats.to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    pred = convert_text(normalizer(generated_text))
                    pred = pred if len(pred) > 0 else '<UNK>'
                    gt = normalizer(text_list[idx_])
                    gt = gt if len(gt) > 0 else '<UNK>'
                    logging.info(f"pred: {pred}, gt: {gt}")

                    all_pred.append(pred)
                    all_gt.append(gt)

                return calculate_wer(all_gt, all_pred)
    
    # print("Dev data preparing...")
    logging.info("Test data preparing...")
    test_dataset = WhisperDataset(test_audio_list, test_text_list, feature_extractor, tokenizer)
    test_data = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=1,
                                            collate_fn=test_dataset.collate_whisper)
    
    # print(f"Training started, total epochs {args.epochs}")
    logging.info("Testing started...")
    if args.zeroshot is True:
        logging.info("zero-shot mode")
        model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
        model.eval()
        test_wer = evaluate_zero_shot(model, test_data, test_text_list)
        logging.info(f"evaluated using zero-shot, WER {test_wer}")
        # logging.info(f"evaluated using zero-shot, WER {test_wer} Eng WER: {test_wer} Man CER: {test_wer}")
    else:
        if os.path.isdir(save_dir):
            checkpoint_paths = glob.glob(f"{save_dir}/*.pt")
            num_models = len(checkpoint_paths)
            logging.info(f"Model averaging over {num_models} check points..")
            model = average_checkpoints(checkpoint_paths)
            model.eval()
            logging.info("Mode averaging done, evalute the model")
            test_wer = evaluate(model, test_data, test_text_list)
            torch.save(model, f"{save_dir}/averaged_model_wer_{test_wer:.3f}.pt")
            logging.info(f"Evaluation completed, averaged model with WER {test_wer}")
        elif os.path.isfile(save_dir):
            ckpt_ = save_dir
            logging.info(f"Evaluating with {ckpt_}")
            model = torch.load(ckpt_)
            model.eval()
            test_wer = evaluate(model, test_data, test_text_list)
            logging.info(f"Evaluation completed, single model with WER {test_wer}")
        else:
            print(f"Path does not exist or is invalid: {save_dir}.")




    






if __name__ == "__main__":
    main()





