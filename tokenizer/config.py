from pydantic_settings import BaseSettings
from typing import List

class tokenizer_config(BaseSettings):
    vocab_size:int = 32000
    model_type:str = "bpe"
    character_coverage: float = 0.9999
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"    
    unk_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 3
    input_sentences_size:int = 10000000