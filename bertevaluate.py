# This is an intent to use a Roberta model as a language model with Wav2Vec2
# but the initial results are worse than just using the wav2vec2 model

import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import os

DATASET_PATH = os.environ.get('DATASET_PATH') or './'


dataset = load_dataset(
    'csv', data_files={'test': 'test.csv'})
    
test_dataset = dataset['test']

processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-xlsr-catala-1/checkpoint-166000")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-large-xlsr-catala-1/checkpoint-166000")
model.to("cuda")


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(
        os.path.join(DATASET_PATH, batch["wav_filename"]))
    batch["speech"] = speech_array[0].numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)


def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def process_result(ground_truth, prediction):
    char_distance = levenshtein(ground_truth, prediction)
    char_length = len(ground_truth)
    word_distance = levenshtein(ground_truth.split(), prediction.split())
    word_length = len(ground_truth.split())
    wer = word_distance / word_length
    cer = char_distance / char_length

    wer = min(wer, 1.0)
    cer = min(cer, 1.0)

    result = {
        'original': ground_truth,
        'prediction': prediction,
        'word_distance': word_distance,
        'word_length': word_length,
        'wer': wer,
        'char_distance': char_distance,
        'char_length': char_length,
        'cer': cer
    }

    return result

from transformers import RobertaForMaskedLM, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("jordimas/julibert")
julibert = RobertaForMaskedLM.from_pretrained("jordimas/julibert").to("cuda")

print(julibert.config)
# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
       logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits


    decoded_results = []
    for logit in logits:
        pred_ids = torch.argmax(logit, dim=-1)
        mask = pred_ids.ge(1).unsqueeze(-1).expand(logit.size())
        vocab_size = logit.size()[-1]
        voice_prob = torch.nn.functional.softmax((torch.masked_select(logit, mask).view(-1,vocab_size)),dim=-1)
        print(voice_prob.size())
        gpt_input = torch.cat((torch.tensor([tokenizer.cls_token_id]).to("cuda"),pred_ids[pred_ids>0]), 0).unsqueeze(1)
        print(gpt_input.shape)
        gpt_prob = torch.nn.functional.softmax(julibert(gpt_input, labels=gpt_input).logits, dim=-1)
        print(gpt_prob.shape)
        bert_prob = gpt_prob.squeeze(1)[:voice_prob.size()[0],:vocab_size]
        print(bert_prob.size())
        comb_pred_ids = torch.argmax(bert_prob*voice_prob, dim=-1)
        decoded_results.append(processor.decode(comb_pred_ids))

    print(decoded_results)
    batch["pred_strings"] = decoded_results[0]

    processed = process_result(batch["transcript"],batch["pred_strings"])
    print(f"{processed['wer']},{batch['transcript']},{batch['pred_strings']}")
    batch['word_distance'] = processed['word_distance']
    batch['word_length'] = processed['word_length']
    batch['char_distance'] = processed['char_distance']
    batch['char_length'] = processed['char_length']
    return batch

result = test_dataset.map(evaluate)

word_distance_sum = 0
word_length_sum = 0
char_distance_sum = 0
char_length_sum = 0

for one in result:
    word_distance_sum += one['word_distance']
    word_length_sum += one['word_length']
    char_distance_sum += one['char_distance']
    char_length_sum += one['char_length']

total = f"TOTAL,{min(word_distance_sum/word_length_sum,1):.6f},{min(char_distance_sum/char_length_sum,1):.6f},,"
print(total)
# print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["transcript"])))

