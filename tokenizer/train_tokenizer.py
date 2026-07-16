from config import tokenizer_config
from pathlib import Path
from sentencepiece import SentencePieceTrainer

current_dir = Path(__file__).parent
root_dir = current_dir.parent
processed_file = root_dir / "data" / "processed" / "corpus_clean.txt"
output_dir =  root_dir / "artifacts" / "tokenizer"


config = tokenizer_config()

input_file = processed_file
model_prefix = str(output_dir / "tokenizer")
vocab_size = config.vocab_size
model_type = config.model_type
character_coverage = config.character_coverage
input_sentence_size = config.input_sentences_size

unk_token = config.unk_token
bos_token = config.bos_token
eos_token = config.eos_token
pad_token = config.pad_token

unk_id = config.unk_id
bos_id = config.bos_id
eos_id = config.eos_id
pad_id = config.pad_id

output_dir.mkdir(parents=True, exist_ok=True)

SentencePieceTrainer.train(
    input=str(input_file),
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage,
    input_sentence_size=input_sentence_size,
    unk_id=unk_id,
    bos_id=bos_id,
    eos_id=eos_id,
    pad_id=pad_id,
    unk_piece=unk_token,
    bos_piece=bos_token,
    eos_piece=eos_token,
    pad_piece=pad_token,
)

print("Tokenizer eğitimi tamamlandı.")
print(f"Model: {model_prefix}.model")
print(f"Vocab: {model_prefix}.vocab")