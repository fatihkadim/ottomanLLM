import torch
import sys
sys.path.insert(0, ".")

from model.rope import frequence_R, rotate_half, apply_rotary_emb

print("=" * 60)
print("RoPE Test")
print("=" * 60)

d_model = 512
n_heads = 8
d_k = d_model // n_heads  # 64

# TEST 1: frequence_R
print("\nTEST 1: frequence_R")
dummy = torch.randn(10, d_k)
inv_freq = frequence_R(dummy)
print(f"   inv_freq shape: {inv_freq.shape}  (beklenen: [{d_k // 2}])")
assert inv_freq.shape == (d_k // 2,), "frequence_R shape yanlis!"
assert inv_freq[0].item() == 1.0, "ilk frekans 1.0 olmali!"
print("   GECTI!")

# TEST 2: rotate_half (2D)
print("\nTEST 2: rotate_half (2D)")
x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
rotated = rotate_half(x)
expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0, -6.0, 5.0]])
print(f"   Girdi:    {x[0].tolist()}")
print(f"   Cikti:    {rotated[0].tolist()}")
assert torch.allclose(rotated, expected), "rotate_half yanlis!"
print("   GECTI!")

# TEST 3: rotate_half (4D)
print("\nTEST 3: rotate_half (4D)")
x_4d = torch.randn(2, 8, 10, 64)
rotated_4d = rotate_half(x_4d)
assert rotated_4d.shape == x_4d.shape, "4D shape bozuldu!"
print(f"   {x_4d.shape} -> {rotated_4d.shape}")
print("   GECTI!")

# TEST 4: apply_rotary_emb shape
print("\nTEST 4: apply_rotary_emb shape")
B, T = 2, 10
q = torch.randn(B, n_heads, T, d_k)
k = torch.randn(B, n_heads, T, d_k)
q_rot, k_rot = apply_rotary_emb(q, k)
print(f"   Q: {q.shape} -> {q_rot.shape}")
print(f"   K: {k.shape} -> {k_rot.shape}")
assert q_rot.shape == q.shape, "Q shape degisti!"
assert k_rot.shape == k.shape, "K shape degisti!"
print("   GECTI!")

# TEST 5: Goreceli pozisyon
print("\nTEST 5: Goreceli pozisyon - ayni mesafe = ayni score")
single_vec = torch.randn(1, 1, 1, d_k)
q_test = single_vec.expand(1, 1, 20, d_k).clone()
k_test = q_test.clone()
q_rot_test, k_rot_test = apply_rotary_emb(q_test, k_test)

score_2_5 = (q_rot_test[0, 0, 2] * k_rot_test[0, 0, 5]).sum()
score_10_13 = (q_rot_test[0, 0, 10] * k_rot_test[0, 0, 13]).sum()
print(f"   Score(poz2, poz5)   = {score_2_5.item():.4f}  (mesafe: 3)")
print(f"   Score(poz10, poz13) = {score_10_13.item():.4f}  (mesafe: 3)")
print(f"   Fark: {abs(score_2_5.item() - score_10_13.item()):.6f}")
assert torch.allclose(score_2_5, score_10_13, atol=1e-4), "Ayni mesafe farkli score verdi!"
print("   GECTI!")

# TEST 6: NaN / Inf
print("\nTEST 6: NaN / Inf kontrolu")
assert not torch.isnan(q_rot).any(), "Q'da NaN var!"
assert not torch.isinf(q_rot).any(), "Q'da Inf var!"
assert not torch.isnan(k_rot).any(), "K'da NaN var!"
assert not torch.isinf(k_rot).any(), "K'da Inf var!"
print("   GECTI!")

print("\n" + "=" * 60)
print("TUM TESTLER GECTI!")
print("=" * 60)
