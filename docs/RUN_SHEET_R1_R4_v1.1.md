# RUN SHEET v1.1 — R1–R4 TAM REJENERASYON KUYRUĞU (ASOC revizyon hattı)

**Sürüm:** v1.1 (2026-08-28) · **Üreten:** Kimi (Moonshot AI) · **Durum:** Denetim Turu-1 bulguları (Claude KB-03/KB-04, Codex F-03/F-05) işlendi
**Önceki sürüm:** `asoc_faz2/FAZ2_REHBER.md` v1.0 (Bölüm C, C1-C6) — **SUPERSEDED** (K-10 ile R1-R4'e genişledi; C2/C5 komutları hatalıydı, bu sürümde düzeltildi). v1.0 arşivde kalır, silinmez.

---

## 0. KANONİK PROTOKOL (her komutta açıkça sabitlenir — varsayılanlara GÜVENİLMEZ)

| Parametre | Değer | Kaynak |
|---|---|---|
| Tohumlar | 42 43 44 45 46 | makale §3.4 |
| Mimariler | MLP (hidden 256) MNIST; SimpleConvNet/WalshConvNet CIFAR | kod gerçeği (K-01/B1) |
| Epokalar (varsayılan rejim) | 10 MNIST / 20 CIFAR | makale §3.4 |
| Epokalar (eşit-bütçe rejimi) | 50 (tümü) | makale §4.2 |
| Walsh protokolü | α₀=0.5, code_dim=128, 50 epoka | makale §3.4 |
| EWC | λ=100, online, γ=1.0 | makale §3.4 (dikkat: betik varsayılanı λ=1000!) |
| SI | λ=1.0, damping ξ=0.1 | makale §3.4 |
| MAS | λ=1.0, num_samples=200 | makale §3.4 |
| LwF | λ=1.0, T=2.0 | betik varsayılanı (makale belirtmiyor — FAZ 3 onay paketi) |
| Softmax-Negotiation | α=0.2 | makale §3.4 |
| Sigmoid-Negotiation | α=0.7 | makale §3.4 |
| Optimizasyon | SGD momentum 0.9, batch 128, lr 0.01 | makale §3.4 |
| n_tasks | MNIST/CIFAR-10: 5 · CIFAR-100: **10 (HER ZAMAN açıkça yazılır — KB-02 dersi)** | makale §3.3 |

**Kurallar:**
1. Hiçbir komut betik varsayılanına bel bağlamaz; protokol tablosundaki her parametre komutta açıkça geçer.
2. Her kuyruk bloğundan ÖNCE preflight: ilgili betik için `--help` çıktısı bu run sheet'le karşılaştırılır (komut seti değiştiyse DUR, bana bildir).
3. Her koşu benzersiz dosya üretmeli; var olan sonucu ezen betik görülürse DUR, bana bildir (Codex F-05).
4. Kesinti olursa: var olan JSON'u olan (yöntem, veri kümesi, tohum) atlanır; kalanlar tamamlanır.
5. Sonuçlar `results/` altında toplanır; her faz sonunda `tar -czf faz2_RX_$(date +%Y%m%d).tar.gz results/` ile paketlenir ve bana ulaştırılır.

---

## R1 — REPLİKASYON (makaledeki varsayılan rejim; amaç: makale sayılarını yeniden üretmek)

### R1.1 Softmax çekirdek yöntemler (finetune, ewc, si, mas, lwf, negotiation)
```bash
python experiments/run_all_experiments.py --methods finetune ewc si mas lwf negotiation \
  --datasets split_mnist --n_tasks 5 --seeds 42 43 44 45 46 \
  --epochs_mnist 10 --batch_size 128 --lr 0.01 --optimizer sgd \
  --ewc_lambda 100 --ewc_mode online --si_lambda 1.0 --damping 0.1 \
  --mas_lambda 1.0 --mas_n_samples 200 --lwf_lambda 1.0 --lwf_temperature 2.0 \
  --negotiation_alpha 0.2 --verbose

python experiments/run_all_experiments.py --methods finetune ewc si mas lwf negotiation \
  --datasets split_cifar10 --n_tasks 5 --seeds 42 43 44 45 46 \
  --epochs_cifar 20 --batch_size 128 --lr 0.01 --optimizer sgd \
  --ewc_lambda 100 --ewc_mode online --si_lambda 1.0 --damping 0.1 \
  --mas_lambda 1.0 --mas_n_samples 200 --lwf_lambda 1.0 --lwf_temperature 2.0 \
  --negotiation_alpha 0.2 --verbose

python experiments/run_all_experiments.py --methods finetune ewc si mas lwf negotiation \
  --datasets split_cifar100 --n_tasks 10 --seeds 42 43 44 45 46 \
  --epochs_cifar 20 --batch_size 128 --lr 0.01 --optimizer sgd \
  --ewc_lambda 100 --ewc_mode online --si_lambda 1.0 --damping 0.1 \
  --mas_lambda 1.0 --mas_n_samples 200 --lwf_lambda 1.0 --lwf_temperature 2.0 \
  --negotiation_alpha 0.2 --verbose
```

### R1.2 Sigmoid varyantları (sigmoid finetune/ewc/si/negotiation + hybrid)
```bash
# NOT: once --help ile --methods seceneklerini dogrulayin (beklenen: sigmoid_finetune, sigmoid_ewc, sigmoid_si, sigmoid_negotiation, hybrid_negotiation ve 'all')
for ds in split_mnist split_cifar10; do
  for s in 42 43 44 45 46; do
    python experiments/run_sigmoid_comparison.py --dataset $ds --n_tasks 5 --methods all --seed $s --epochs 10   # mnist
  done
done
# CIFAR icin 20 epoka:
for s in 42 43 44 45 46; do
  python experiments/run_sigmoid_comparison.py --dataset split_cifar10 --n_tasks 5 --methods all --seed $s --epochs 20
  python experiments/run_sigmoid_comparison.py --dataset split_cifar100 --n_tasks 10 --methods all --seed $s --epochs 20
done
# MNIST icin 10 epoka notu: yukaridaki ilk dongu 10 epoka calistirir; split_mnist icin epochs=10 dogru.
```

### R1.3 Walsh Negotiation (makale protokolü — açıkça sabitli)
```bash
for s in 42 43 44 45 46; do
  python experiments/run_walsh_negotiation.py --dataset split_mnist   --n_tasks 5  --epochs 50 --alpha 0.5 --code_dim 128 --seed $s
  python experiments/run_walsh_negotiation.py --dataset split_cifar10 --n_tasks 5  --epochs 50 --alpha 0.5 --code_dim 128 --seed $s
  python experiments/run_walsh_negotiation.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --alpha 0.5 --code_dim 128 --seed $s
done
```
**Beklenen replikasyon ölçütü (5-tohum ort±ss, ddof=1):** MNIST ≈98.75±0.07/0.10±0.07; CIFAR-10 ≈90.11/1.71; CIFAR-100 ≈66.71/2.94. CIFAR-10 ss'leri ve MNIST BWT için F-01 notu: tam hassasiyetli yeni kayıtlar esas alınacak; farklılık çıkarsa DURmayın, kaydedin (FAZ 3'te tablolar yeni ham veriden üretilecek).

## R2 — EŞİT-BÜTÇE REJİMİ (tüm yöntemler 50 epoka)

### R2.1 Softmax çekirdek (50 epoka)
```bash
python experiments/run_all_experiments.py --methods finetune ewc si mas lwf negotiation \
  --datasets split_mnist --n_tasks 5 --seeds 42 43 44 45 46 \
  --epochs_mnist 50 --ewc_lambda 100 --ewc_mode online --si_lambda 1.0 --damping 0.1 \
  --mas_lambda 1.0 --mas_n_samples 200 --lwf_lambda 1.0 --lwf_temperature 2.0 \
  --negotiation_alpha 0.2 --verbose
python experiments/run_all_experiments.py --methods finetune ewc si mas lwf negotiation \
  --datasets split_cifar10 --n_tasks 5 --seeds 42 43 44 45 46 \
  --epochs_cifar 50 --ewc_lambda 100 --ewc_mode online --si_lambda 1.0 --damping 0.1 \
  --mas_lambda 1.0 --mas_n_samples 200 --lwf_lambda 1.0 --lwf_temperature 2.0 \
  --negotiation_alpha 0.2 --verbose
python experiments/run_all_experiments.py --methods finetune ewc si mas lwf negotiation \
  --datasets split_cifar100 --n_tasks 10 --seeds 42 43 44 45 46 \
  --epochs_cifar 50 --ewc_lambda 100 --ewc_mode online --si_lambda 1.0 --damping 0.1 \
  --mas_lambda 1.0 --mas_n_samples 200 --lwf_lambda 1.0 --lwf_temperature 2.0 \
  --negotiation_alpha 0.2 --verbose
```

### R2.2 Joint referans (v1.1 betiği — batch-interleaved)
```bash
for s in 42 43 44 45 46; do
  python experiments/run_joint_training.py --dataset split_mnist   --n_tasks 5  --epochs 50 --seed $s
  python experiments/run_joint_training.py --dataset split_cifar10 --n_tasks 5  --epochs 50 --seed $s
  python experiments/run_joint_training.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --seed $s
done
```

## R3 — YENİ KANIT DENeyLERİ

### R3.1 EWC λ-taraması @50 epoka (çöküşün sınırı; online mod dahil — Claude öneri 6)
```bash
python experiments/ewc_hyperparam_search.py --dataset split_cifar10  --n_tasks 5  --epochs 50 --lambda_values 1 10 50 100 500 --ewc_modes online --seeds 42 43 44 45 46
python experiments/ewc_hyperparam_search.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --lambda_values 1 10 50 100 500 --ewc_modes online --seeds 42 43 44 45 46
```
(Süre baskısında: önce seed 42 + λ ∈ {1, 100, 500} ön-tarama; tam ızgara sonra.)

### R3.2 Sigmoid-SI kanıt paketi (Codex F-02 — nedensel iddia sınanmadan yazılmayacak)
1. Önce bana dönün: `src/baselines/sigmoid_si.py` için kanonik-SI hizalama + tek-minibatch güncelleme-eşliği + non-finite tanılama yamasını ben yazacağım (onay paketiyle).
2. Ardından: gradyan-norm eğrileri + kırpma ablasyonu koşuları (yama sonrası komutlar v1.2 run sheet'e girecek).

### R3.3 Bileşen ablasyonları
```bash
# Kod boyutu (CIFAR-100, 5 tohum)
for cd in 64 256; do
  for s in 42 43 44 45 46; do
    python experiments/run_walsh_negotiation.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --alpha 0.5 --code_dim $cd --seed $s
  done
done
# alpha0 duyarlılık (CIFAR-10) — dogru bayrak: --alpha_values (KB-04)
for s in 42 43 44 45 46; do
  python experiments/alpha_search_negotiation.py --dataset split_cifar10 --n_tasks 5 --epochs 20 --alpha_values 0.3,0.5,0.7 --variants softmax --seed $s
done
```

### R3.4 (Aday, kullanıcı onayına sunulacak) Küçük-bellekli ER tabanı (Claude T10-öneri 4)
Kapsam savunması için 200-örneklik tek bir Experience Replay tabanı. Depoda ER eğiticisi yok; eklenirse v1.2 run sheet'te.

## R4 — STABİLİZE CIFAR-10 (ayarlı rejim; makale §4.1.4 replikasyonu)
```bash
python experiments/run_all_experiments.py --methods finetune ewc si mas lwf negotiation \
  --datasets split_cifar10 --n_tasks 5 --seeds 42 43 44 45 46 \
  --epochs_cifar 20 --lr 0.0025 --weight_decay 0.0005 \
  --ewc_lambda 50 --ewc_mode online --si_lambda 1.0 --damping 0.1 \
  --mas_lambda 1.0 --mas_n_samples 200 --lwf_lambda 1.0 --lwf_temperature 2.0 \
  --negotiation_alpha 0.5 --verbose
```
Dikkat: run_all_experiments.py'nin multistep scheduler + gradyan kırpma bayrağı OLUP OLMADIĞI preflight'ta denetlenecek; yoksa bu rejim betik düzeyinde desteklenmiyor demektir → DUR, bana bildir (R4 için betik yaması gerekir).

## Maliyet ve izleme notları (RTX 4070 Laptop 8 GB)
- Walsh CIFAR-100 50-epoka koşusu ≈ 8-9 dk/tohum (geçmiş log zaman damgalarından). R1+R2 kaba tahmin: 2-3 gün aralıklı GPU.
- Uzun kuyruklar öncesi `nvidia-smi` ile VRAM izleyin; bellek hatası = DUR + bana bildir (batch 128 protokol; değiştirilmez, birlikte karar veririz).
- Süre ölçümü: her koşu `time` ile; joint v1.1 zaten `train_time_sec` yazıyor.

**Kayıt kuralı:** Bu run sheet'in her bloğu çalıştırıldığında günceye yeni kayıt (tarih+saat+sonuç dosya sayısı+anomali notu).
