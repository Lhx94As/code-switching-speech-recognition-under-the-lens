import os
import torch
import torch.utils
import torch.utils.data
import argparse
from jiwer import wer as calculate_wer
from WhisperDataPre import WhisperDataset
from whisper.normalizers import EnglishTextNormalizer
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import logging
from utils import preprocess_files, save_best_checkpoints


logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")

normalizer = EnglishTextNormalizer()

# def preprocess_files(folder):
#       wavscp = os.path.join(folder, "wav.scp")
#       text = os.path.join(folder, "text")
#       with open(wavscp, 'r') as f1:
#             wave_data = f1.readlines()
#       with open(text, 'r') as f2:
#             trans_data = f2.readlines()
#       audio_list = [audio_line.strip().split()[1] for audio_line in wave_data]
#       text_list = [' '.join(text_line.split()[1:]).lower().strip() for text_line in trans_data]
#       return audio_list, text_list


def main():
      parser = argparse.ArgumentParser(description='paras for making data')
      parser.add_argument('--train', type=str, default='train_data_folder')
      parser.add_argument('--dev', type=str, default='dev_data_folder')
      parser.add_argument('--dataset', type=str, default='asru')
      parser.add_argument('--model', type=str, default="openai/whisper-small")
      parser.add_argument('--epochs', type=int, default=5)
      parser.add_argument('--module', type=str, default='all')
      parser.add_argument('--lr', type=float, default=1e-6)
      parser.add_argument('--save_every', type=int, default=10000)
      parser.add_argument('--zeroshot', type=bool, default=False)
      parser.add_argument('--lang', type=str, default="zh")
      parser.add_argument('--batch', type=int, default=8)
      parser.add_argument('--accumulation', type=int, default=2)
      parser.add_argument('--save_dir', type=str, default="exp")
      args = parser.parse_args()

      normalizer = EnglishTextNormalizer()

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      # for single GPU, may need DDP for multiple GPUs

      train_path = args.train
      dev_path = args.dev
      train_audio_list, train_text_list = preprocess_files(train_path)
      dev_audio_list, dev_text_list = preprocess_files(dev_path)
      

      tokenizer = WhisperTokenizer.from_pretrained(args.model, language=args.lang, task="transcribe")
      processor = WhisperProcessor.from_pretrained(args.model, language=args.lang, task="transcribe")
      feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)

      model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)

      model.config.forced_decoder_ids = None
      model.config.suppress_tokens = []

      def evaluate(model, dev_data, text_list):
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")
            with torch.no_grad():
                  all_pred, all_gt = [], []
                  for idx_, (feats, label_batch, labels) in enumerate(dev_data):
                        generated_ids = model.generate(inputs=feats.to(device), forced_decoder_ids=forced_decoder_ids, max_new_tokens=150)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        pred = normalizer(generated_text)
                        pred = pred if len(pred) > 0 else '<UNK>'
                        gt = normalizer(text_list[idx_])
                        gt = gt if len(gt) > 0 else '<UNK>'

                        all_pred.append(pred)
                        all_gt.append(gt)

                  return calculate_wer(all_gt, all_pred)

      # print("Training data preparing...")
      logging.info("Training data preparing...")
      batch_size = args.batch
      train_dataset = WhisperDataset(train_audio_list, train_text_list, feature_extractor, tokenizer)
      train_data = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_whisper)
      
      # print("Dev data preparing...")
      logging.info("Dev data preparing...")
      dev_dataset = WhisperDataset(dev_audio_list, dev_text_list, feature_extractor, tokenizer)
      dev_data = torch.utils.data.DataLoader(dataset=dev_dataset, 
                                             batch_size=1,
                                             collate_fn=dev_dataset.collate_whisper)
      
      # print(f"Training started, total epochs {args.epochs}")
      logging.info(f"Training started, total epochs {args.epochs}")

      optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
      loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
      gradient_accumulation = args.accumulation
      save_every = args.save_every

      save_dir = args.save_dir
      os.system(f"mkdir -p {save_dir}")
      wer_list = []
      ckpt_records = None


      step, pre_step, loss = 0, 0, 0
      best_loss, best_wer = 100000, 100000
      wer_list.append(best_wer)

      if args.zeroshot:
            model.eval()
            dev_wer = evaluate(model, dev_data, dev_text_list)
            logging.info(f"zero shot performance: WER {dev_wer}")

      if args.module == 'encoder':
            for param in model.model.decoder.parameters():
                  param.requires_grad = False
            train_module = 'encoder'
      elif args.module == 'decoder':
            for param in model.model.encoder.parameters():
                  param.requires_grad = False
            train_module = 'decoder'
      elif args.module == 'all':
            train_module = 'all'
      else:
            raise ValueError(f"Invalid module specified: {args.module}. Expected 'encoder', 'decoder', or 'all'.")

      logging.info(f"Trainable module: {train_module}")

      trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
      logging.info(f"Model params: {trainable_params+non_trainable_params}\n #Non-trainable: {non_trainable_params}\n #Trainable: {trainable_params}")  


      model.train()
      for epoch in range(args.epochs):
            # print(f"Epoch: {epoch+1} starts")
            logging.info(f"Epoch: {epoch+1} starts")
            for _, (feats, label_batch, labels) in enumerate(train_data):

                  y_out = labels[:, 1:].to(device)

                  logits = model(input_features=feats.to(device), decoder_input_ids=label_batch.to(device)).logits
                  loss = loss_fn(logits.transpose(1,2), y_out)
                  
                  loss.backward()
                  step += 1

                  if (step+1) % gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                  if (step+1) % 500 == 0:
                        logging.info(f"Epoch: {epoch+1} ( step {step+1}): loss {loss:.3f}")

                  if (step+1) % save_every == 0:
                        model.eval()
                        dev_wer = evaluate(model, dev_data, dev_text_list)
                        logging.info(f"Evaluated at epoch: {epoch+1} ( step {step+1}): MER {dev_wer:.4f}")
                        model.train()
                        ckpt_records = save_best_checkpoints(model=model, 
                                                             step=step+1, 
                                                             error_rate=dev_wer, 
                                                             loss=loss, 
                                                             save_dir=save_dir, 
                                                             ckpt_records=ckpt_records)




if __name__ == "__main__":
    main()
