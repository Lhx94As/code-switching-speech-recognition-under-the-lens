import torch
import torchaudio
import whisper
from dataclasses import dataclass
import torch.nn.functional as F
from transformers import WhisperProcessor
from typing import Any, List, Dict, Union

def load_wave(wave_path) -> torch.Tensor:
      waveform, sr = torchaudio.load(wave_path, normalize=True)
      if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
      return waveform

class WhisperDataset(torch.utils.data.Dataset):
      def __init__(self, audio_list, text_list, featureprocessor, tokenizer, sample_rate=16000):
            super().__init__()
            self.audio_list = audio_list
            self.text_list = text_list
            self.sample_rate = sample_rate
            self.tokenizer = tokenizer
            self.feat_extractor = featureprocessor
            assert len(audio_list) == len(text_list), "audio and text not aligned"
      
      def __len__(self):
            return len(self.audio_list)

      def __getitem__(self, idx):
            audio_path = self.audio_list[idx]
            text = self.text_list[idx]

            item = {"audio": audio_path, "text": text}
            # audio
            audio = load_wave(audio_path).squeeze(0).numpy()
            feat = self.feat_extractor(audio, sampling_rate=16_000, return_tensors="pt")['input_features']
            item["mel"] = feat
            # text
            labels = self.tokenizer(text, max_length=1024, truncation=True).input_ids
            item["label"] = labels

            return item
      
      def collate_whisper(self, batch):

            feature_list = [{"input_features": item["mel"]} for item in batch]
            label_list = [{"input_ids": item["label"]} for item in batch]
            feats = self.feat_extractor.pad(feature_list, return_tensors="pt")
            labels_batch = self.tokenizer.pad(label_list, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            return feats['input_features'].squeeze(1), labels_batch["input_ids"][:,:-1], labels


class WhisperDatasetAdapter(torch.utils.data.Dataset):
      def __init__(self, audio_list, text_list, processor, sample_rate=16000):
            super().__init__()
            self.audio_list = audio_list
            self.text_list = text_list
            self.sample_rate = sample_rate
            self.processor = processor
            assert len(audio_list) == len(text_list), "audio and text not aligned"
      
      def __len__(self):
            return len(self.audio_list)

      def __getitem__(self, idx):
            audio_path = self.audio_list[idx]
            text = self.text_list[idx]

            # item = {"audio": audio_path, "text": text}
            # audio
            audio = load_wave(audio_path).squeeze(0).numpy()

            data = self.processor(audio=audio, sampling_rate=16_000, text=text)

            # feat = self.feat_extractor(audio, sampling_rate=16_000, return_tensors="pt")['input_features']
            # item["input_features"] = feat
            # # text
            # labels = self.tokenizer(text, max_length=1024, truncation=True).input_ids
            # item["labels"] = labels

            return data
      

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
      processor: Any

      def __call__(self, features):
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"][0]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                  labels = labels[:, 1:]

            batch["labels"] = labels

            return batch