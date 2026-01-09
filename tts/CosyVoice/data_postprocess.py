import os
from glob import glob

def convert_to_text_file(dict_list, output_file):
    with open(output_file, 'w') as f:
        for item in dict_list:
            f.write(f"{item['iden']} {item['content']}\n")

def combine_text_files(file1, file2, output_file):
    # Read contents from both files
    with open(file1, 'r') as f1:
        content1 = f1.readlines()
    
    with open(file2, 'r') as f2:
        content2 = f2.readlines()
    
    # Combine contents
    combined_content = content1 + content2
    
    # Write combined content to output file
    with open(output_file, 'w') as out_f:
        out_f.writelines(combined_content)


# folder = "/home3/hexin/tts_finetune/CosyVoice/seame_outputs"
# files = glob(folder + '/*.wav')
# print(len(files))
# ori_text_file = '/home3/hexin/asru_data/data_cs/text'
# ori_audio_file = '/home3/hexin/asru_data/data_cs/wav.scp'
syn_text_file = '/home3/hexin/asru_data/data_syn2/text'
syn_audio_file = '/home3/hexin/asru_data/data_syn2/wav.scp'
# combine_text_file = '/home3/hexin/asru_data/data_combine400/text'
# combine_audio_file = '/home3/hexin/asru_data/data_combine400/wav.scp'

# ori_text_file = '/home3/hexin/whisper_train_data/combine_seame2/text'
# ori_audio_file = '/home3/hexin/whisper_train_data/combine_seame2/wav.scp'
# syn_text_file = '/home3/hexin/whisper_train_data/syn_seame3/text'
# syn_audio_file = '/home3/hexin/whisper_train_data/syn_seame3/wav.scp'
# syn_text_file = '/home3/hexin/tts_finetune/CosyVoice/seame_missed/text'
# syn_audio_file = '/home3/hexin/tts_finetune/CosyVoice/seame_missed/wav.scp'
# combine_text_file = '/home3/hexin/whisper_train_data/combine_seame2/text'
# combine_audio_file = '/home3/hexin/whisper_train_data/combine_seame2/wav.scp'

temp_name = '/home3/hexin/wav_new.scp'
with open(syn_audio_file, 'r') as f:
      syn_audio_files = f.readlines()
# new_content = [x.replace('/seame_missed/', '/seame_ft_self/') for x in syn_audio_files]      
new_content = [x.replace(x.split(' ')[1], os.path.join('/home3/hexin/tts_finetune/CosyVoice/', x.split(' ')[1])) for x in syn_audio_files]

with open(temp_name, 'w') as f:
      for idx, i in enumerate(new_content):
          f.write(i)
        #   file_name = i.split(' ')[1].strip('\n')
        #   old_filename = syn_audio_files[idx].split(' ')[1].strip('\n')
        #   os.rename(old_filename, file_name)
# os.remove(syn_audio_file)
os.rename(temp_name, syn_audio_file)


# Example usage
# combine_text_files(ori_text_file, syn_text_file, combine_text_file)
# combine_text_files(ori_audio_file, syn_audio_file, combine_audio_file)