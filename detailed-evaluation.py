import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import os

DATASET_PATH = os.environ.get('DATASET_PATH') or './'

processor = Wav2Vec2Processor.from_pretrained("wav2vec2-large-xlsr-catala")
model = Wav2Vec2ForCTC.from_pretrained("wav2vec2-large-xlsr-catala")
model.to("cuda")


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

SOURCE_DIR = 'age-gender-accent'

for root, dirnames, filenames in os.walk(SOURCE_DIR):
    for filename in filenames:
        dataset_filename = os.path.join(root, filename)
        dataset = load_dataset('csv', data_files={'test': dataset_filename})
    
        test_dataset = dataset['test']
        
        # Preprocessing the datasets.
        # We need to read the aduio files as arrays
        def speech_file_to_array_fn(batch):
            speech_array, sampling_rate = torchaudio.load(
                os.path.join(DATASET_PATH, batch["wav_filename"]))
            batch["speech"] = speech_array[0].numpy()
            return batch

        test_dataset = test_dataset.map(speech_file_to_array_fn)

        def evaluate(batch):
            inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

            with torch.no_grad():
                logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

            pred_ids = torch.argmax(logits, dim=-1)[0]
            batch["pred_strings"] = processor.decode(pred_ids)
            processed = process_result(batch["transcript"],batch["pred_strings"])
            batch['wer'] = processed['wer']
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

        with open(f"aga-results/{filename}", "w") as file:
            for one in result:
                file.write(f"{one['wer']},{one['transcript']},{one['pred_strings']}\n")
                word_distance_sum += one['word_distance']
                word_length_sum += one['word_length']
                char_distance_sum += one['char_distance']
                char_length_sum += one['char_length']

            total = f"TOTAL,{min(word_distance_sum/word_length_sum,1):.6f},{min(char_distance_sum/char_length_sum,1):.6f},,"
            file.write(total)

