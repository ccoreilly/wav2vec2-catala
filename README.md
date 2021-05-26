# Wav2Vec2 Català

Models de reconeixement automàtic de la parla Wav2Vec2 pel Català.

S'ha fet fine-tuning a partir de dos models base, el [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) i el [facebook/wav2vec2-large-100k-voxpopuli](https://huggingface.co/facebook/wav2vec2-large-100k-voxpopuli). Els podeu trobar a:
- [ccoreilly/wav2vec2-large-xlsr-catala](https://huggingface.co/ccoreilly/wav2vec2-large-xlsr-catala).
- [ccoreilly/wav2vec2-large-100k-voxpopuli-catala](https://huggingface.co/ccoreilly/wav2vec2-large-100k-voxpopuli-catala).

Fine-tuned Wav2Vec2 models for the Catalan language based on [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) and [facebook/wav2vec2-large-100k-voxpopuli](https://huggingface.co/facebook/wav2vec2-large-100k-voxpopuli)

You can find the models in the huggingface repository: 
- [ccoreilly/wav2vec2-large-xlsr-catala](https://huggingface.co/ccoreilly/wav2vec2-large-xlsr-catala).
- [ccoreilly/wav2vec2-large-100k-voxpopuli-catala](https://huggingface.co/ccoreilly/wav2vec2-large-100k-voxpopuli-catala).

## Datasets

- [Common Voice](https://huggingface.co/datasets/common_voice)
- [ParlamentParla](https://www.openslr.org/59/)

## WER

Avaluada en els següents datasets no vistos durant l'entrenament:

Word error rate was evaluated on the following datasets unseen by the model:

| Dataset | XLSR-53 | VoxPopuli |
| ------- | --- | --- |
| [Test split CV+ParlamentParla]((https://github.com/ccoreilly/wav2vec2-catala/blob/master/test-filtered.csv)) | 6,92% | 5.98% |
| [Google Crowsourced Corpus](https://www.openslr.org/69/) | 12,99% | 12,14% |
| Audiobook “La llegenda de Sant Jordi” | 13,23% | 12,02% |


Com que les dades de CommonVoice contenen metadades sobre l'edat, el gènere i la variant dialectal del parlant, podem avaluar el model segons aquests paràmetres. Desafortunadament, per alguna de les categories no hi ha prou dades com per considerar la mostra significativa, és per això que s'acompanya la taxa d'error amb la mida de la mostra.

| Edat | Mostra | XLSR-53 | VoxPopuli |
| ------- | --- | --- | --- |
| 10-19 | 64 | 7,96% | 8,54% |
| 20-29 | 330 | 7,52% | 6,10% |
| 30-39 | 377 | 5,65% | 4,55% |
| 40-49 | 611 | 6,37% | 6,17% |
| 50-59 | 438 | 5,75% | 5,30% |
| 60-69 | 166 | 4,82% | 4,20% |
| 70-79 | 37 | 5,81% | 5,33% |

| Accent | Mostra | XLSR-53 | VoxPopuli |
| ------- | --- | --- | --- |
| Balear | 64 | 5,84% | 5,11% |
| Central | 1202 | 5,98% | 5,37% |
| Nord-occidental | 140 | 6,60% | 5,77% |
| Septentrional | 75 | 5,11% | 5,58% |
| Valencià | 290 | 5,69% | 5,30% |

| Sexe | Mostra | XLSR-53 | VoxPopuli |
| ------- | --- | --- | --- |
| Femení | 749 | 5,57% | 4,95% |
| Masculí | 1280 | 6,65% | 5,98% |

## Com fer-lo servir / Usage

```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = load_dataset("common_voice", "ca", split="test[:2%]")

processor = Wav2Vec2Processor.from_pretrained("ccoreilly/wav2vec2-large-100k-voxpopuli-catala") 
model = Wav2Vec2ForCTC.from_pretrained("ccoreilly/wav2vec2-large-100k-voxpopuli-catala")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
	speech_array, sampling_rate = torchaudio.load(batch["path"])
	batch["speech"] = resampler(speech_array).squeeze().numpy()
	return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
	logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])
```