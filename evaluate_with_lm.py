import math
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

from flashlight.lib.text.dictionary import create_word_dict, load_words
from flashlight.lib.text.decoder import (
    CriterionType,
    LexiconDecoderOptions,
    KenLM,
    SmearingMode,
    Trie,
    LexiconDecoder)
from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions


kenlm_args = {
    "kenlm_model_path": "cat.lm.bin",
    "lexicon_path": "cat.lexicon",  # Each in a new line: WORD W O R D
    "beam": 1000,
    "nbest": 1,
    "beam_threshold": 20,
    "lm_weight": 1.0,
    "word_score": 1.0,
    "sil_weight": 0
}


class KenLMDecoder(object):
    def __init__(self, kenlm_args, vocab_dict, blank="<pad>", silence="|", unk="<unk>"):

        self.vocab_size = len(vocab_dict)
        self.blank_token = (vocab_dict[blank])
        self.silence_token = vocab_dict[silence]
        self.unk_token = vocab_dict[unk]

        self.nbest = kenlm_args['nbest']

        if kenlm_args['lexicon_path']:
            vocab_keys = vocab_dict.keys()
            self.lexicon = load_words(kenlm_args['lexicon_path'])
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index(unk)

            self.lm = KenLM(kenlm_args['kenlm_model_path'], self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence_token)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)

                for spelling in spellings:
                    spelling_idxs = []
                    for token in spelling:
                        if token.upper() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.upper()])
                        elif token.lower() in vocab_keys:
                            spelling_idxs.append(vocab_dict[token.lower()])
                        else:
                            print("WARNING: The token", token,
                                  "not exist in your vocabulary, using <unk> token instead")
                            spelling_idxs.append(self.unk_token)
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(
                    vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                word_score=kenlm_args['word_score'],
                unk_score=-math.inf,
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence_token,
                self.blank_token,
                self.unk_word,
                [],
                False,
            )
        else:
            d = {w: [[w]] for w in vocab_dict.keys()}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(kenlm_args['kenlm_model_path'], self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=kenlm_args['beam'],
                beam_size_token=kenlm_args['beam_size_token'] if "beam_size_token" in kenlm_args else len(
                    vocab_dict),
                beam_threshold=kenlm_args['beam_threshold'],
                lm_weight=kenlm_args['lm_weight'],
                sil_score=kenlm_args['sil_weight'],
                log_add=False,
                criterion_type=CriterionType.CTC,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence_token, self.blank_token, []
            )

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank"""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank_token, idxs)
        return torch.LongTensor(list(idxs))

    def decode(self, emissions):
        B, T, N = emissions.size()
        # print(emissions.shape)
        tokens = []
        scores = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)
            nbest_results = results[: self.nbest]
            tokens_nbest = []
            scores_nbest = []
            for result in nbest_results:
                tokens_nbest.append(result.tokens)
                scores_nbest.append(result.score)
            tokens.append(tokens_nbest)
            scores.append(scores_nbest)

        token_array = np.array(tokens, dtype=object).transpose((1, 0, 2))
        scores_arrray = np.array(scores, dtype=object).transpose()
        return token_array, scores_arrray


processor = Wav2Vec2Processor.from_pretrained(
    "ccoreilly/wav2vec2-large-100k-voxpopuli-catala")
model = Wav2Vec2ForCTC.from_pretrained(
    "ccoreilly/wav2vec2-large-100k-voxpopuli-catala")
model.to("cuda")

vocab_dict = processor.tokenizer.get_vocab()
pad_token = processor.tokenizer.pad_token
silence_token = processor.tokenizer.word_delimiter_token
unk_token = processor.tokenizer.unk_token
kenlm = KenLMDecoder(kenlm_args, vocab_dict, blank=pad_token,
                     silence=silence_token, unk=unk_token)


DATASET_PATH = os.environ.get('DATASET_PATH') or './'


dataset = load_dataset(
    'csv', data_files={'test': 'test-filtered.csv'})

test_dataset = dataset['test']


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

# Preprocessing the datasets.
# We need to read the aduio files as arrays


def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000,
                       return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda"),
                       attention_mask=inputs.attention_mask.to("cuda")).logits

    logits = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    # get all candidates
    lm_tokens, lm_scores = kenlm.decode(logits.cpu().detach())
    # choise the best candidate
    pred_ids = lm_tokens[0][:]

    batch["pred_strings"] = processor.batch_decode(
        pred_ids)[0].lower().replace("-", " ")
    processed = process_result(batch["transcript"], batch["pred_strings"])
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
