# OttomanLM

> Osmanlı dönemine özelleşmiş, sıfırdan PyTorch ile eğitilmiş "Modern Türkçe konuşan Tarihçi" bir dil modeli.

---

## Proje Hakkında

OttomanLM, yalnızca Osmanlı tarihi kaynaklarıyla eğitilmiş, soru-cevap odaklı küçük bir dil modelidir. **Projenin ana hedefi**, yapay zekânın Osmanlı İmparatorluğu dönemine ait tarihi bilgilere derinlemesine hakim olması, ancak kullanıcıyla iletişim kurarken eski metinlerin diliyle değil **Modern Türkçe** ile konuşmasıdır (Modelin dili modern, bilgisi tarihi olmalıdır). Model, Osmanlı padişahları, devlet yapısı, hukuk sistemi, ordu, ekonomi ve kültür gibi konularda Türkçe sorulara cevap verebilecek şekilde tasarlanmıştır.

Bu proje tamamen eğitim amaçlıdır. Sıfırdan mimari yazmak, özel tokenizer eğitmek ve domain-specific pretraining yapmak üzerine odaklanır.

---

## Mimari

| Parametre | Değer |
|---|---|
| Mimari | Decoder-only Transformer |
| Positional Encoding | RoPE (Rotary Position Embedding) |
| Normalization | Pre-norm (LayerNorm) |
| Activation | SwiGLU |
| Katman sayısı | 6 |
| Attention head | 8 |
| Embedding dim | 512 |
| FFN dim | 2048 |
| Context length | 512 token |
| Toplam parametre | ~50–70M |

Mimari modern GPT-2 tabanına RoPE, Pre-norm ve SwiGLU eklenmiş bir yapıdır. Llama mimarisine yakın, ancak çok daha küçük ölçekte.

---

## Veri Kaynakları

### Pretraining Verisi

| Kaynak | Format | İçerik |
|---|---|---|
| Wikipedia Türkçe | Düz metin | Osmanlı kategorileri (padişahlar, savaşlar, kurumlar...) |
| Wikisource Osmanlıca | Düz metin | Orijinal Osmanlıca metinlerin transkripsiyonları |
| İslam Ansiklopedisi (TDV) | PDF → metin | Osmanlı tarihi maddeleri |
| Osmanlı kronikleri | Düz metin | Naima Tarihi, Evliya Çelebi Seyahatnamesi, Peçevi Tarihi |
| HuggingFace Turkish Wikipedia | Dataset | Tarih filtrelenmiş Türkçe Wikipedia |
| CulturaX Türkçe | Dataset | mC4 + OSCAR temizlenmiş Türkçe veri |
| DergiPark Tarih Dergileri | Scraping | Açık erişimli tarih akademik makaleleri |
| TTK Yayınları | Scraping | Türk Tarih Kurumu akademik yayınları |

Hedef corpus boyutu: **500MB – 1GB** ham metin.

> **Not:** Veri dosyaları boyutları nedeniyle repo'ya dahil edilmemiştir. Aşağıdaki scriptlerle toplayabilirsiniz.

### Fine-tuning Verisi

500–2000 adet el yapımı veya GPT-4o yardımıyla üretilmiş soru-cevap çifti. Format:

```
<|user|> Osmanlı'da tımar sistemi neydi?
<|assistant|> Tımar sistemi, Osmanlı Devleti'nde toprağın...
```

---

## Proje Aşamaları

```
Aşama 1 — Veri toplama & temizleme
  ↓
Aşama 2 — Tokenizer eğitimi (SentencePiece BPE, ~32K vocab)
  ↓
Aşama 3 — Model mimarisi (PyTorch sıfırdan)
  ↓
Aşama 4 — Pretraining (causal LM, next-token prediction)
  ↓
Aşama 5 — Instruction fine-tuning (SFT, soru-cevap formatı)
```

---

## Kurulum

```bash
git clone https://github.com/kullanici/ottoman-lm.git
cd ottoman-lm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Gereksinimler

```
torch>=2.0
sentencepiece
wikipedia-api
requests
beautifulsoup4
tqdm
wandb
datasets
```

---

## Kullanım

### 1. Veri Toplama

Scriptleri öncelik sırasıyla çalıştırın:

```bash
# HuggingFace Türkçe Wikipedia (en büyük ve en kolay kaynak)
python scripts/download_hf_wiki_tr.py

# CulturaX streaming (hedef: 200 MB)
python scripts/download_hf_culturax_tr.py

# Akademik makaleler
python scripts/scrape_dergipark.py

# TTK yayınları
python scripts/scrape_ttk.py

# Wikipedia (doğrudan dump'tan filtreleme)
python scripts/scrape_wiki_dump_local.py

# TDV İslam Ansiklopedisi
python scripts/scrape_tdv_massive.py

# Tarih olayları
python scripts/scrape_tarihiolaylar.py
```

### 2. Korpus Temizleme

```bash
python scripts/clean_corpus.py
```

### 3. Tokenizer Eğitimi

```bash
python scripts/train_tokenizer.py \
  --input data/raw/ \
  --vocab_size 32000 \
  --output tokenizer/ottoman_bpe
```

### 4. Pretraining

```bash
python train.py \
  --data data/processed/corpus.bin \
  --tokenizer tokenizer/ottoman_bpe.model \
  --n_layers 6 \
  --n_heads 8 \
  --d_model 512 \
  --batch_size 32 \
  --lr 3e-4 \
  --max_steps 100000
```

### 5. Fine-tuning

```bash
python finetune.py \
  --checkpoint checkpoints/pretrain_final.pt \
  --data data/sft/qa_pairs.jsonl \
  --lr 3e-5 \
  --epochs 3
```

### 6. Inference

```bash
python chat.py --checkpoint checkpoints/sft_final.pt
```

```
Sen: Yavuz Sultan Selim Mısır'ı ne zaman fethetti?
Model: Yavuz Sultan Selim, 1517 yılında Ridaniye Savaşı'nı...
```

---

## Proje Yapısı

```
ottoman-lm/
│
├── data/
│   ├── raw/              # Ham scraped metinler (gitignore)
│   ├── processed/        # Tokenize edilmiş binary dosyalar (gitignore)
│   └── sft/              # Fine-tuning soru-cevap çiftleri (gitignore)
│
├── docs/
│   └── kaynaklar.md      # Detaylı veri kaynakları raporu
│
├── model/
│   ├── __init__.py
│   ├── attention.py      # MultiHeadAttention + RoPE
│   ├── feedforward.py    # SwiGLU FFN
│   ├── transformer.py    # TransformerBlock
│   └── gpt.py            # Ana model sınıfı
│
├── scripts/
│   ├── clean_corpus.py           # Korpus temizleme
│   ├── download_hf_wiki_tr.py    # HuggingFace Wikipedia indirme
│   ├── download_hf_culturax_tr.py # CulturaX streaming
│   ├── scrape_dergipark.py       # DergiPark makaleleri
│   ├── scrape_ttk.py             # TTK yayınları
│   ├── scrape_tdv.py             # TDV İslam Ansiklopedisi
│   ├── scrape_tdv_massive.py     # TDV genişletilmiş tarama
│   ├── scrape_tarihiolaylar.py   # Tarihiolaylar.com
│   ├── scrape_chronicles.py      # Osmanlı kronikleri
│   ├── scrape_gutenberg.py       # Project Gutenberg
│   ├── scrape_kadisicilleri.py   # Kadı sicilleri PDF
│   ├── scrape_kultur_portali.py  # Kültür portalı
│   ├── scrape_ottoman_hf.py      # HuggingFace Osmanlı
│   ├── scrape_wiki_dump_local.py # Wikipedia dump filtreleme
│   ├── scrape_wiki_history_filtered.py
│   ├── scrape_wikipedia.py       # Wikipedia scraping
│   ├── scrape_wikipedia_tr.py    # Wikipedia TR
│   ├── scrape_wikisource.py      # Wikisource
│   ├── extract_books.py          # Kitap çıkarma
│   └── generate_sft_qa_pairs.py  # SFT veri üretimi
│
├── tokenizer/
│   └── train_tokenizer.py  # Tokenizer eğitim scripti
│
├── checkpoints/          # Kaydedilen model ağırlıkları (gitignore)
├── train.py              # Pretraining scripti
├── finetune.py           # SFT scripti
├── chat.py               # Inference arayüzü
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Donanım

| Ortam | GPU | Tahmini süre (pretraining) |
|---|---|---|
| Google Colab Pro+ | A100 40GB | ~2–4 saat |
| RunPod / Lambda Labs | A100 / H100 | ~1–3 saat |
| Kaggle | T4 x2 | ~10–15 saat |
| Yerel (RTX 3090/4090) | 24GB | ~4–8 saat |

Mixed precision (bf16/fp16) aktif, gradient clipping 1.0, AdamW optimizer.

---

## Referanslar

- [nanoGPT — Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

---

## Lisans

MIT License — eğitim ve araştırma amaçlı serbesttir.
