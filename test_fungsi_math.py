import numpy as np
import math

# ==============================================================================
# UNIT TESTING FUNGSI MATEMATIKA C4.5 (Entropy & Info Gain)
# Tujuan: Memvalidasi pemahaman algoritma secara manual (White Box Testing)
# ==============================================================================

def hitung_entropy(labels):
    """
    Menghitung Entropy dari sekumpulan label.
    Rumus: -SUM(p * log2(p))
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0

    counts = np.unique(labels, return_counts=True)[1]
    probs = counts / n_labels
    
    entropy = 0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
            
    return entropy

def unit_test_entropy():
    print("🧪 MEMULAI UNIT TEST FUNGSI ENTROPY...")
    
    # KASUS 1: Data Homogen (Isinya sama semua -> Harusnya Entropy 0)
    # Misal: 5 data Hoax semua
    case_1 = [1, 1, 1, 1, 1]
    e1 = hitung_entropy(case_1)
    print(f"   [Test 1] Input: {case_1} -> Entropy: {e1}")
    assert e1 == 0, "❌ Gagal: Entropy data homogen harus 0"
    print("   ✅ Lulus.")

    # KASUS 2: Data Seimbang (50:50 -> Harusnya Entropy 1)
    # Misal: 2 Hoax, 2 Fakta
    case_2 = [1, 1, 0, 0]
    e2 = hitung_entropy(case_2)
    print(f"   [Test 2] Input: {case_2} -> Entropy: {e2}")
    assert e2 == 1.0, "❌ Gagal: Entropy data 50:50 harus 1"
    print("   ✅ Lulus.")
    
    # KASUS 3: Contoh Manual Laporan (Misal: 4 Hoax, 6 Fakta)
    # Rumus: -(0.4 log2 0.4) - (0.6 log2 0.6) ≈ 0.97
    case_3 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    e3 = hitung_entropy(case_3)
    print(f"   [Test 3] Input: 4 Pos, 6 Neg -> Entropy: {e3:.4f}")
    assert 0.96 < e3 < 0.98, "❌ Gagal: Perhitungan meleset dari toleransi."
    print("   ✅ Lulus.")

    print("\n🎉 SEMUA UNIT TEST ENTROPY BERHASIL!\n")

if __name__ == "__main__":
    unit_test_entropy()