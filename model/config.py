class OttomanLLMConfig:
    def __init__(self):
        self.vocab_size = 32000 #tokenizer kapasite
        self.d_model = 512 
        self.n_layers = 6 #transformers katman sayısı
        self.n_heads = 8 #attention heads
        self.d_ff = 2048 
        self.max_seq_len = 512
        self.dropout = 0.1
        self.norm_eps = 1e-5 #sifira bolmeyi engelleyen guvenlik payi
        self.rope_theta = 10000.0 #rope taban hızı