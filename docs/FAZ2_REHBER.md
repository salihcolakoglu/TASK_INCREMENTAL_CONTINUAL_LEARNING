# FAZ 2 — ORTAM KURULUMU VE DENEY KUYRUĞU REHBERİ

**Hazırlayan:** Kimi (Yapay Zekâ Araştırma Asistanı) · **Tarih:** 2026-08-28
**Hedef makine:** theBeast — Windows 11 Pro, RTX 4070 8 GB, sürücü 595.79 (CUDA 13.2), WSL 2.7.12
**Hedef:** Makalenin eksik deneylerini, makaleyle birebir aynı kod/protokolle tamamlamak ve ham sonuçları JSON olarak toplamak.

---

## BÖLÜM A — Ortam kurulumu (WSL2 + Ubuntu)

WSL kuruldu, Ubuntu iniyor. Ubuntu ilk açılışta kullanıcı adı/parola isteyecek. Ardından Ubuntu terminalinde sırayla:

### A1. GPU geçişini doğrula
```bash
nvidia-smi
```
RTX 4070 görünmeli. **WSL içine ayrıca NVIDIA sürücüsü KURULMAZ** (Windows sürücüsü paylaşılır). Görünmezse: Windows tarafında `wsl --update` + yeniden başlatma.

### A2. Miniconda kur
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### A3. Depoyu ve ortamı kur
```bash
sudo apt update && sudo apt install -y git
mkdir -p ~/cl && cd ~/cl
git clone https://github.com/salihcolakoglu/TASK_INCREMENTAL_CONTINUAL_LEARNING.git
cd TASK_INCREMENTAL_CONTINUAL_LEARNING

conda create -n cl python=3.10 -y
conda activate cl

# README'nin sabitlediği sürümler (PyTorch 2.5.1 + CUDA 12.1; sürücü 13.2 geriye uyumlu)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install avalanche-lib==0.6.0
pip install numpy tqdm matplotlib seaborn scikit-learn pandas tabulate tensorboard
pip install -r requirements.txt   # kalanlar icin (wandb opsiyonel; offline calisir)
```

### A4. Doğrulama (3 dakika)
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python test_setup.py
python experiments/run_finetune.py --dataset split_mnist --epochs 1 --seed 42
```
Beklenen: `2.5.1+cu121 True NVIDIA GeForce RTX 4070` ve 1 epokalık sorunsuz koşu.

> **VRAM notu:** 8 GB, batch 128 ile SimpleConvNet/WalshConvNet için fazlasıyla yeterli. Bellek hatası olursa `--batch_size 64` denenebilir (ancak makale protokolü 128 — değiştirirseniz sonuç JSON'una not düşülür; varsayılanı koruyun).

---

## BÖLÜM B — Kritik karar (deneylere başlamadan ÖNCE)

Depo incelemesinde saptanan **kod–makale uyuşmazlıkları** (ayrıntı: bu belgenin sonundaki Ek 1):

1. **Mimari:** Makale CIFAR için "ResNet-18" diyor; kod `SimpleConvNet`/`WalshConvNet` (küçük özel CNN). MNIST için makale "400-400 MLP" diyor; kod varsayılanı `hidden_size=256`.
2. **Şekil 3** "4 seeds" diyor; metin 5 tohum.
3. **CIFAR-100 seed 43'ün ilk koşusu** (57.79%) başarısız olup 9 dk sonra tekrarlanmış; makale tekrarlanan koşuyu kullanmış (ham veriyle birebir doğruladım: 66.71±0.84 ancak böyle tutuyor). Bu dışlama makalede belgelenmemiş.

**Karar seçenekleri:**
- **(Önerilen) B1:** Makale metni gerçek mimariyi yazacak şekilde düzeltilir (FAZ 3'te); FAZ 2 deneyleri mevcut kodla, makale protokolüyle (varsayılan parametreler) koşulur. Makaledeki tüm sayılar tek kod tabanında tutarlı kalır.
- **B2:** Koda ResNet-18 eklenir ve TÜM ana deneyler (tabanlar dahil) yeniden koşulur — makalenin tüm sayıları değişir, süre ciddi uzar. **Önerilmez** (tez takvimi riski).

Bu rehber B1 varsayımıyla yazılmıştır.

---

## BÖLÜM C — Deney kuyruğu (öncelik sırasıyla)

Tüm koşular repo kökünden (`~/cl/TASK_INCREMENTAL_CONTINUAL_LEARNING`) ve `conda activate cl` ile. Tohumlar hep `42 43 44 45 46`.

### C1. FAZ 2.1 — Joint Training üst sınırı (ZORUNLU) — ~1-2 saat
Önce bu rehberle gelen betiği depoya kopyalayın: `run_joint_training.py` → `experiments/` altına.
```bash
for ds in split_mnist split_cifar10; do
  for s in 42 43 44 45 46; do
    python experiments/run_joint_training.py --dataset $ds --n_tasks 5 --epochs 50 --seed $s
  done
done
for s in 42 43 44 45 46; do
  python experiments/run_joint_training.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --seed $s
done
```
Çıktı: `results/joint_training/joint_<ds>_seed<s>.json` (15 dosya).

### C2. FAZ 2.2 — Tam matris: LwF + MAS tüm veri kümeleri, iki rejim (ZORUNLU)
Varsayılan rejim (MNIST 10 ep., CIFAR 20 ep.) ve eşit-hesaplama rejimi (50 ep.):
```bash
# Varsayilan rejim
python experiments/run_all_experiments.py --methods lwf mas --datasets split_mnist --epochs 10 --seeds 42 43 44 45 46
python experiments/run_all_experiments.py --methods lwf mas --datasets split_cifar10 --epochs 20 --seeds 42 43 44 45 46
python experiments/run_all_experiments.py --methods lwf mas --datasets split_cifar100 --n_tasks 10 --epochs 20 --seeds 42 43 44 45 46
# Esit-hesaplama rejimi (50 epoka; LwF + MAS + tablolarin eksik kalanlari)
python experiments/run_all_experiments.py --methods lwf mas --datasets split_mnist --epochs 50 --seeds 42 43 44 45 46
python experiments/run_all_experiments.py --methods lwf mas --datasets split_cifar10 --epochs 50 --seeds 42 43 44 45 46
python experiments/run_all_experiments.py --methods lwf mas --datasets split_cifar100 --n_tasks 10 --epochs 50 --seeds 42 43 44 45 46
```
> Not: `run_all_experiments.py` parametre adlarını betikteki `--help` ile teyit edin (`--n_tasks`, `--epochs`, `--seeds` mevcut; metod adları `lwf`, `mas`).

### C3. FAZ 2.3 — EWC λ-taraması @ 50 epoka (çöküşün sınırı)
Repo hazır betik içeriyor:
```bash
python experiments/ewc_hyperparam_search.py --dataset split_cifar10 --epochs 50 --lambda_values 1 10 50 100 500 --seeds 42 43 44 45 46
python experiments/ewc_hyperparam_search.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --lambda_values 1 10 50 100 500 --seeds 42 43 44 45 46
```
(Süre baskısı olursa tek tohum 42 + iki uç λ ön-taraması yeterli; nihai için 5 tohum.)

### C4. FAZ 2.4 — Sigmoid-SI kanıt paketi
- Mevcut: `run_sigmoid_comparison.py` / `quick_sigmoid_test.py` — NaN anındaki gradyan normlarını kaydedecek şekilde çalıştırın; gradyan kırpma (norm 1.0) ile bir "kurtarma" denemesi:
```bash
python experiments/run_sigmoid_comparison.py --dataset split_cifar10 --seed 42   # mevcut protokol
```
Kurtarma ablasyonu için SI eğiticisine geçici `--grad_clip` seçeneği gerekebilir — o aşamada bana dönün, yamayı ben yazarım.

### C5. FAZ 2.5 — Bileşen ablasyonları
- Kod boyutu: `python experiments/run_walsh_negotiation.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --code_dim 64` ve `--code_dim 256` (5 tohum).
- α₀ duyarlılığı: repo `alpha_search_negotiation.py` içeriyor; `--alpha 0.3 0.5 0.7` ızgarası, CIFAR-10, 5 tohum.

### C6. FAZ 2.6 — Hesap maliyeti sağlaması
Joint betiği `train_time_sec` alanını zaten yazıyor. Diğer yöntemler için: her koşunun kabuk süresini `time` ile ölçüp not edin yeterli (örn. `time python experiments/run_si.py ...`). İleri düzey: `torch.profiler` FLOP sayımı — istenirse ayrı betik yazarım.

---

## BÖLÜM D — Sonuçların bana ulaştırılması

Kuyruklar bittikçe (paket paket de olur):
```bash
cd ~/cl/TASK_INCREMENTAL_CONTINUAL_LEARNING
tar -czf faz2_sonuclar_$(date +%Y%m%d).tar.gz results/
```
Dosyayı Windows tarafına kopyalayın (`/mnt/c/...` üzerinden, örn. çalışma klasörüme) ya da bana yolunu söyleyin. Ben:
1. Her tabloyu ham JSON'dan yeniden hesaplayıp makale tablolarıyla çaprazlarım (Walsh için bunu zaten yaptım — Ek 1),
2. Holm düzeltmeli p-değerleri + Hedges g etki büyüklüklerini hesaplarım,
3. Güncellenmiş tabloları LaTeX'e işlerim.

---

## Ek 1 — Ham veri doğrulaması (28 Ağustos 2026, bu oturumda yapıldı)

Reprodüksiyon paketindeki (`reproducibility_package.tar.gz`, 68 dosya) ham Walsh JSON'larından makale değerlerini birebir yeniden hesapladım:

| Veri kümesi | Makale (acc / forg / BWT) | Ham veriden (5 dosya) | Durum |
|---|---|---|---|
| MNIST | 98.75±0.07 / 0.10±0.07 / −0.03±0.08 | 98.75±0.07 / 0.10±0.07 / −0.04±0.08 | ✓ (BWT yuvarlama farkı) |
| CIFAR-10 | 90.11±0.78 / 1.71±0.44 / −1.53±0.47 | 90.11±0.79 / 1.71±0.47 / −1.53±0.51 | ✓ (std'de ±0.02-0.04 fark — ondalık hassasiyeti) |
| CIFAR-100 | 66.71±0.84 / 2.94±1.05 / −2.67±1.09 | 66.71±0.84 / 2.94±1.06 / −2.67±1.09 (başarısız ilk seed-43 koşusu dışlanınca) | ✓ |

**Saptanan kod–makale farkları (FAZ 3 gündemi):**
1. Mimari beyanı (ResNet-18/400-400) ↔ kod (SimpleConvNet/WalshConvNet, hidden 256) — B1/B2 kararı yukarıda.
2. Esneklik formülü: kod `α ← α · 1/(2α−α²)` (çarpımsal güncelleme); makale yalnız `p(α)=1/(2α−α²)` yazıyor — FAZ 3'te güncellemenin çarpımsal olduğu açıkça yazılacak (içerik tutarlı, ifade eksik).
3. seed-43 CIFAR-100 ilk koşusunun dışlanması makalede belgelenmemiş — FAZ 3'te şeffaflık notu.
4. Şekil 3 başlığı "4 seeds" — FAZ 2'de şekil ham veriden yeniden üretilecek.
