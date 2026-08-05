"""
Sprint 1 — Tokenizer Doğrulama Scripti
--------------------------------------
Kontroller:
  1. Round-trip encode/decode tutarlılığı
  2. Special token ID doğrulaması
  3. Osmanlıca terim tokenizasyonu (tek token mu, parçalanıyor mu?)
  4. Fertility hesabı (ortalama kelime başı token sayısı)
  5. Vocabulary coverage (unique kelimeler içinde single-token olanların oranı)
  6. Corpus örneklemesi üzerinden özet istatistik
"""

import sys
import random
import logging
from pathlib import Path

import sentencepiece as spm

# ---------------------------------------------------------------------------
# Yol ayarları
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).parent
ROOT_DIR    = CURRENT_DIR.parent
MODEL_PATH  = ROOT_DIR / "artifacts" / "tokenizer" / "tokenizer.model"
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus_clean.txt"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Osmanlı tarihi ile ilgili test terimleri
# ---------------------------------------------------------------------------
OTTOMAN_TERMS = [
    # Unvanlar / kişiler
    "padişah", "sultan", "sadrazam", "vezir", "yeniçeri", "sipahi",
    "kadı", "müftü", "şeyhülislam", "paşa", "bey", "ağa",
    # Kurumlar / sistemler
    "divan", "tımar", "devşirme", "kapıkulu", "millet",
    "eyalet", "sancak", "kaza", "nahiye",
    # Olaylar / kavramlar
    "fetih", "cihad", "ferman", "kanun", "vakıf", "medrese",
    # Yerler
    "istanbul", "osmanlı", "anadolu", "rumeli", "hicaz",
    # Osmanlıca kökenli kelimeler
    "sefer", "hazine", "ordu", "donanma", "tersane",
]

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def load_corpus_sample(path: Path, n_lines: int = 50_000, seed: int = 42) -> list[str]:
    """Korpustan rastgele n_lines satır çek."""
    random.seed(seed)
    log.info(f"Korpus örnekleniyor: {path}  (hedef: {n_lines:,} satır)")
    all_lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)
    if len(all_lines) > n_lines:
        sample = random.sample(all_lines, n_lines)
    else:
        sample = all_lines
    log.info(f"Örneklenen satır sayısı: {len(sample):,}")
    return sample


def check_roundtrip(sp: spm.SentencePieceProcessor, texts: list[str], n: int = 200) -> float:
    """Encode → decode round-trip tutarlılığını kontrol et. Başarı oranını döndür."""
    sample = random.sample(texts, min(n, len(texts)))
    ok = 0
    for text in sample:
        encoded = sp.encode(text, out_type=int)
        decoded = sp.decode(encoded)
        if decoded.strip() == text.strip():
            ok += 1
    rate = ok / len(sample)
    return rate


def check_special_tokens(sp: spm.SentencePieceProcessor) -> None:
    """Special token ID'lerinin beklenenlerle eşleşip eşleşmediğini kontrol et."""
    expected = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
    }
    log.info("--- Special Token Kontrolleri ---")
    all_ok = True
    for token, expected_id in expected.items():
        actual_id = sp.piece_to_id(token)
        status = "✅" if actual_id == expected_id else "❌"
        log.info(f"  {status}  {token!r:12s} → beklenen ID={expected_id}, gerçek ID={actual_id}")
        if actual_id != expected_id:
            all_ok = False
    if all_ok:
        log.info("  Tüm special token'lar doğru.")
    else:
        log.warning("  UYARI: Bazı special token ID'leri yanlış!")


def check_ottoman_terms(sp: spm.SentencePieceProcessor) -> dict:
    """Her Osmanlıca terimi tokenize et, kaç parçaya bölündüğünü göster."""
    log.info("--- Osmanlıca Terim Tokenizasyonu ---")
    results = {}
    for term in OTTOMAN_TERMS:
        tokens = sp.encode(term, out_type=str)
        is_single = len(tokens) == 1
        marker = "✅" if is_single else f"⚠️ ({len(tokens)} parça)"
        log.info(f"  {marker:20s}  '{term}' → {tokens}")
        results[term] = tokens
    single_count = sum(1 for v in results.values() if len(v) == 1)
    log.info(
        f"\n  Tek token: {single_count}/{len(OTTOMAN_TERMS)} "
        f"({single_count/len(OTTOMAN_TERMS)*100:.1f}%)"
    )
    return results


def compute_fertility(sp: spm.SentencePieceProcessor, lines: list[str]) -> float:
    """
    Fertility = toplam token sayısı / toplam kelime sayısı.
    İdeal: 1.0 – 1.5.  2.0+ → vocab yetersiz.
    """
    total_tokens = 0
    total_words  = 0
    for line in lines:
        words = line.split()
        if not words:
            continue
        tokens = sp.encode(line, out_type=int)
        total_words  += len(words)
        total_tokens += len(tokens)
    if total_words == 0:
        return float("inf")
    return total_tokens / total_words


def compute_vocab_coverage(sp: spm.SentencePieceProcessor, lines: list[str]) -> float:
    """
    Unique kelimelerin kaçı tokenizer'da TEK token olarak temsil ediliyor?
    (Vocabulary Coverage)
    """
    unique_words: set[str] = set()
    for line in lines:
        unique_words.update(line.lower().split())

    single_token_words = 0
    for word in unique_words:
        tokens = sp.encode(word, out_type=int)
        if len(tokens) == 1:
            single_token_words += 1

    coverage = single_token_words / len(unique_words) if unique_words else 0.0
    return coverage, len(unique_words), single_token_words


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("OttomanLM — Tokenizer Doğrulama")
    log.info("=" * 60)

    # Model yükle
    if not MODEL_PATH.exists():
        log.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PATH))
    log.info(f"Model yüklendi: {MODEL_PATH}")
    log.info(f"Vocabulary boyutu: {sp.get_piece_size():,}")

    # 1) Special token kontrolü
    check_special_tokens(sp)

    # 2) Corpus örnekle
    if not CORPUS_PATH.exists():
        log.warning(f"Corpus bulunamadı: {CORPUS_PATH}. Osmanlı terimi ve fertility testleri sınırlı çalışacak.")
        lines = []
    else:
        lines = load_corpus_sample(CORPUS_PATH, n_lines=50_000)

    # 3) Round-trip testi
    if lines:
        log.info("--- Round-trip Encode/Decode Testi ---")
        rt_rate = check_roundtrip(sp, lines, n=500)
        log.info(f"  Round-trip başarı oranı: {rt_rate*100:.2f}%")
        if rt_rate < 0.95:
            log.warning("  UYARI: Round-trip başarı oranı düşük! Tokenizer'ı kontrol et.")
    else:
        log.warning("  Round-trip testi atlandı (corpus yok).")

    # 4) Osmanlıca terim kontrolü
    check_ottoman_terms(sp)

    # 5) Fertility
    if lines:
        log.info("--- Fertility Hesabı ---")
        fertility = compute_fertility(sp, lines)
        fertility_status = "✅ İdeal" if fertility <= 1.5 else ("⚠️ Kabul edilebilir" if fertility <= 2.0 else "❌ Yüksek — vocab yetersiz")
        log.info(f"  Fertility: {fertility:.4f}  {fertility_status}")
        log.info("  (İdeal: 1.0–1.5 | Kabul edilebilir: 1.5–2.0 | Sorunlu: 2.0+)")

    # 6) Vocabulary coverage
    if lines:
        log.info("--- Vocabulary Coverage ---")
        coverage, total_unique, single_tok = compute_vocab_coverage(sp, lines)
        log.info(f"  Unique kelime sayısı : {total_unique:,}")
        log.info(f"  Tek-token kelimeler  : {single_tok:,}")
        log.info(f"  Coverage             : {coverage*100:.2f}%")

    log.info("=" * 60)
    log.info("Doğrulama tamamlandı.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
