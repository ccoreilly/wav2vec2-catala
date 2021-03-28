# Wav2Vec2 Català

Un model de reconeixement automàtic de la parla Wav2Vec2 pel Català.

Podeu trobar el model al dipòsit de huggingface [ccoreilly/wav2vec2-large-xlsr-catala](https://huggingface.co/ccoreilly/wav2vec2-large-xlsr-catala).

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Catalan language using the [Common Voice](https://huggingface.co/datasets/common_voice) and [ParlamentParla](https://www.openslr.org/59/) datasets.

You can find the model in the huggingface repository [ccoreilly/wav2vec2-large-xlsr-catala](https://huggingface.co/ccoreilly/wav2vec2-large-xlsr-catala).


## WER

Avaluada en els següents datasets no vistos durant l'entrenament:

Word error rate was evaluated on the following datasets unseen by the model:

| Dataset | WER |
| ------- | --- |
| [Test split CV+ParlamentParla]((https://github.com/ccoreilly/wav2vec2-catala/blob/master/test.csv)) | 7.57% |
| [Google Crowsourced Corpus](https://www.openslr.org/69/) | 13.72% |
| Audiobook “La llegenda de Sant Jordi” | 13.23% | 


## Com fer-lo servir / Usage

```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

test_dataset = load_dataset("common_voice", "ca", split="test[:2%]")

processor = Wav2Vec2Processor.from_pretrained("ccoreilly/wav2vec2-large-xlsr-catala") 
model = Wav2Vec2ForCTC.from_pretrained("ccoreilly/wav2vec2-large-xlsr-catala")

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