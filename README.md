# Deep Photo Style Transfer — Mimari Değişiklikli Sürüm

Luan et al. (CVPR 2017\) "Deep Photo Style Transfer" makalesinin mimari değişiklikler içeren bir yeniden uygulamasıdır. Orijinal makaledeki VGG-19 \+ Gram matrisi \+ DilatedNet üçlüsü yerine modern alternatifler (EfficientNet-B4, AdaIN, SegFormer-B2) kullanılır.

## İçindekiler

1. [Proje Özeti](#proje-özeti)  
2. [Mimari Farklılıklar](#mimari-farklılıklar)  
3. [Kurulum](#kurulum)  
4. [Veri Seti Hazırlığı](#veri-seti-hazırlığı)  
5. [Kodun Çalışması — Detaylı Anlatım](#kodun-çalışması--detaylı-anlatım)  
6. [Hücre Yapısı](#hücre-yapısı)  
7. [Hiperparametreler](#hiperparametreler)  
8. [Deneyler ve Sonuçlar](#deneyler-ve-sonuçlar)  
9. [Bilimsel Bulgular](#bilimsel-bulgular)  
10. [Kaynakça](#kaynakça)

---

## Proje Özeti

Fotorealistik stil transferi; bir referans görselin (örn. gün batımı sahili) renk, ton ve atmosferik karakterini, içerik görselinin (örn. şehir manzarası) yapısını ve nesne kimliğini koruyarak aktarır.

Bu proje üç temel kaybı birleştirir:

L\_total \= α · L\_content \+ γ · L\_style(AdaIN) \+ λ · L\_matting

- **L\_content**: Çıktının nesne/yapı bilgisini korur (EfficientNet ara katman MSE).  
- **L\_style (AdaIN)**: Çıktının renk dağılımını stil görseline yaklaştırır.  
- **L\_matting (Levin Laplacian)**: Piksellerin lokal komşulukla tutarlı kalmasını zorlar — fotorealistliği korur.

## Mimari Farklılıklar

| Bileşen | Orijinal (Luan 2017\) | Bu Proje |
| :---- | :---- | :---- |
| Backbone | VGG-19 | EfficientNet-B4 (torchvision) |
| Stil kaybı | Gram matrisi | AdaIN (Adaptive Instance Normalization) |
| Segmentasyon | DilatedNet | SegFormer-B2 (transformers) |
| Regularizasyon | Matting Laplacian | Matting Laplacian (aynı) |

## Kurulum

### Gereken Ortam

- Google Colab Pro (GPU önerilir; ücretsiz Colab'da da çalışır ama yavaş)  
- Google Drive hesabı (veri seti için)  
- Python 3.10+

### Bağımlılıklar

Hücre 2 otomatik kurar:

torch, torchvision      \# backbone ve optimizasyon

transformers==4.41.2    \# SegFormer için

timm==1.0.3             \# EfficientNet alternatifi

scipy==1.11.4           \# sparse matting Laplacian

lpips==0.1.4            \# algısal mesafe metriği

scikit-image==0.22.0    \# SSIM hesabı

numpy\<2                 \# binary uyumluluk için zorunlu

Not: NumPy 2.x binary uyumsuzluklara yol açar. Hücre 2 sonrası mutlaka `Runtime → Restart runtime` yapın.

## Veri Seti Hazırlığı

### MIT-Adobe FiveK (Kaggle Versiyonu)

1. Kaggle'da `MIT Adobe FiveK` arayın (https://www.kaggle.com/datasets/thbdh5765/mit-adobe-5k-dataset)  
2. Dataset'i Drive'a yükleyin: `MyDrive/FiveK/`  
3. Beklenen yapı:

MyDrive/FiveK/

├── training/

│   ├── INPUT\_IMAGES/    ← content görselleri (kullanılır)

│   └── GT\_IMAGES/       ← uzman rötuşları (kullanılmaz)

├── testing/

└── validation/

### Stil Görseli

Veri setinin DIŞINDAN seçilir, manuel olarak yüklenir:

/content/style\_reference.jpg

İdeal stil görseli özellikleri:

- 2-3 baskın semantic bölge (gökyüzü \+ deniz/yer)  
- Belirgin renk paleti (gün batımı, sis, ay ışığı gibi)  
- En az 512px uzun kenar

---

## Kodun Çalışması — Detaylı Anlatım

Bu bölüm her hücrenin **ne yaptığını**, **neden gerekli olduğunu** ve **hangi sırayla çalıştırılması gerektiğini** açıklar. Hücreler iki kategoriye ayrılır: **kurulum hücreleri** (bir kez çalıştırılır) ve **deney hücreleri** (her yeni content/stil değişikliğinde tekrarlanır).

### Aşama A — İlk Kurulum (Bir Kez Yapılır)

Bu aşama yeni bir Colab oturumunun başında veya proje ilk kez çalıştırıldığında yapılır. Yaklaşık 15-25 dakika sürer.

**Sıralama:**

Hücre 1 → Hücre 2 → \[Restart Runtime\] → Hücre 1 → Hücre 2 → Hücre 3 → ... → Hücre 6

#### Hücre 1 — Drive Bağlantısı ve Veri Kopyalama

- Google Drive'ı Colab'a bağlar (`drive.mount`)  
- Drive'daki `FiveK/training/INPUT_IMAGES` klasörünü yerel `/content/FiveK/input` konumuna kopyalar  
- Yerel disk Drive'dan çok daha hızlı I/O sağlar; optimizasyon adımlarında bu kritiktir  
- Eğer yerel klasör zaten doluysa kopyalamayı atlar

**Süre:** İlk çalıştırmada 5-15 dk, sonraki çalıştırmalarda \<5 sn.

#### Hücre 2 — Kütüphane Kurulumu

- `pip install` ile gerekli paketleri yükler  
- En kritik: `numpy<2` zorlanır (NumPy 2.x torchvision ile binary uyumsuz)  
- Bu hücreden sonra `Runtime → Restart runtime` ZORUNLUDUR; aksi takdirde Hücre 3'te `ValueError: numpy.dtype size changed` hatası alınır

**Süre:** \~1 dakika \+ restart süresi.

#### Hücre 3 — Import ve Cihaz Kontrolü

- Tüm kütüphaneleri import eder  
- GPU varsa onu seçer, yoksa CPU uyarısı verir  
- CPU üzerinde optimizasyon çok yavaş (300 adım \= \~1 saat); GPU şart

**Süre:** \<5 sn.

#### Hücre 4 — Hiperparametreler

- Tüm sabitler (yollar, ağırlıklar, adım sayısı vs.) burada toplanır  
- Diğer hücreler bu sabitleri okur  
- Sadece bu hücreyi değiştirerek tüm davranışı kontrol edebilirsiniz

**Süre:** \<1 sn.

#### Hücre 5 — Klasör Doğrulama

- `INPUT_FOLDER` var mı kontrol eder  
- `OUTPUT_FOLDER` (`/content/outputs`) yoksa oluşturur  
- Hücre 1 hatalı çalışmışsa burada hata fırlatılır

**Süre:** \<1 sn.

#### Hücre 6 — Veri Filtreleme

- Tüm INPUT\_IMAGES dosyalarını tarar  
- Bozuk, çok küçük (256px altı) veya aşırı dikdörtgen (aspect ratio \> 3\) olanları eler  
- İlk `DATASET_LIMIT` (varsayılan 1000\) geçerli dosyayı `valid_files` listesinde toplar  
- Bu liste sonraki tüm hücrelerde content adayı havuzu olarak kullanılır

**Süre:** \~30 sn (300 dosya için).

### Aşama B — Görsel Seçimi ve Hazırlığı

Bu aşama her yeni content/stil çifti için yapılır. Yaklaşık 2-12 dk.

**Sıralama:**

Stil görselini /content/style\_reference.jpg olarak yükle

     ↓

Hücre 7 (içerik+stil yükle, görselleştir)

     ↓

\[Opsiyonel\] Hücre 7.5 (en iyi içerik adaylarını sırala)

     ↓

\[7.5 kullanıldıysa\] Hücre 7 (seçilen yeni içerikle TEKRAR)

#### Hücre 7 — Görsel Yükleme ve Transform

- Content görselini `valid_files[0]` (veya manuel seçilen yol) üzerinden yükler  
- Stil görselini `/content/style_reference.jpg` üzerinden yükler  
- Her ikisini uzun kenar 512px, kısa kenar 32'nin katı olacak şekilde yeniden boyutlandırır  
- ImageNet mean/std ile normalize ederek `content_norm` ve `style_norm` tensörlerini oluşturur  
- Content ve stil görsellerini yan yana matplotlib ile gösterir

**Süre:** \<5 sn.

#### Hücre 7.5 — Semantik Benzerlik ile İçerik Seçimi (OPSİYONEL)

Bu hücre stil görselinizle **en uygun** content adayını otomatik bulur.

**Nasıl çalışır:**

1. SegFormer-B2 ile stil görselinin segmentasyon etiketlerini hesaplar  
2. `RANKING_SAMPLE_SIZE` kadar content adayının segmentasyon dağılımını çıkarır  
3. Stil ile her aday arasında **histogram kesişimi skoru** hesaplar (0-1 arası)  
4. En yüksek skorlu ilk 10 adayı listeler ve 7 tanesini görsel olarak gösterir

**Skor yorumu:**

- 0.0-0.2 → tamamen farklı sahne tipleri  
- 0.2-0.4 → kısmi örtüşme (1-2 ortak büyük bölge)  
- 0.4-0.6 → iyi örtüşme  
- 0.6+ → çok iyi örtüşme

**Süre:** 1-10 dk (RANKING\_SAMPLE\_SIZE değerine göre).

#### NEDEN Hücre 7.5 SONRASI Hücre 7'yi TEKRAR ÇALIŞTIRMAK GEREKİR

Bu kritik bir noktadır ve sıklıkla atlanır. Açıklayalım:

**Akış adım adım:**

1. **İlk çalıştırma:** Hücre 7 çalışır → `content_path_example = valid_files[0]` varsayılan olarak ayarlanır → Bu ilk görsel `content_tensor` ve `content_np` değişkenlerine yüklenir.  
     
2. **Sıralama:** Hücre 7.5 çalışır → 300 aday taranır, skor sıralaması üretilir (`scores` listesi). Bu hücre KENDİSİ `content_tensor`'ı GÜNCELLEMEZ — sadece skorları hesaplar ve gösterir.  
     
3. **Karar:** Kullanıcı `scores[0][0]` (en yüksek skorlu adayın yolu) veya beğendiği başka bir adayı görsel inceleme sonrası seçer.  
     
4. **Yeni içeriği yüklemek:** Seçilen yolun `content_tensor`'a aktarılması için Hücre 7'nin **YENİDEN** çalıştırılması gerekir. Hücre 7'nin başında şu satırı güncelleyin:  
     
   content\_path\_example \= scores\[0\]\[0\]   \# 7.5'in en iyi adayı  
     
   \# veya:  
     
   content\_path\_example \= scores\[3\]\[0\]   \# 4\. sıradaki aday  
     
   \# veya elle:  
     
   content\_path\_example \= "/content/FiveK/input/a4378-\_DGW0272\_N1.5.JPG"  
     
5. **Sonuç:** Hücre 7 tekrar çalışınca artık `content_tensor` yeni görseli içerir; Hücre 8 ve sonrası bu güncellenmiş tensör üzerinde çalışır.

**Eğer Hücre 7'yi tekrar çalıştırmazsanız:** Hücre 7.5 sadece bir öneri listesi üretir; gerçek `content_tensor` hâlâ ilk yüklediğiniz görseldir. Sonraki tüm optimizasyon eski içerik üzerinde çalışır. Bu sessiz bir bug — hata mesajı yok ama yanlış sonuç alırsınız.

**Özet:** Hücre 7.5 "ne yüklemem gerek?" sorusuna cevap verir. Hücre 7 "o görseli yükler". İkisini sırasıyla çalıştırmak şarttır.

### Aşama C — Model Yükleme ve Fonksiyon Tanımları

Bu hücreler oturumda BİR KEZ çalıştırılır. Değişkenler ve modeller bellekte kaldığı sürece her yeni içerik için tekrar gerekmez.

**Sıralama:**

Hücre 8 → Hücre 9 → Hücre 10 → (Hücre 11\) → (Hücre 12 sonra) → Hücre 14

#### Hücre 8 — EfficientNet-B4 Yükleme

- torchvision'dan EfficientNet-B4 (ImageNet pretrained) yükler  
- `eval()` moduna alır, gradient kapatır (sadece özellik çıkarımı yapar)  
- Test girdisiyle features\[3\] ve features\[5\]'in çıktı boyutlarını yazdırır

**Süre:** \~10 sn (ilk yüklemede ağırlık indirilir).

#### Hücre 9-12 — Kayıp Fonksiyonu Tanımları

- Hücre 9: AdaIN stil kaybı fonksiyonu (maske destekli)  
- Hücre 10: İçerik kaybı fonksiyonu  
- Hücre 12: Matting Laplacian kaybı fonksiyonu  
- Hücre 14: Toplam kayıp fonksiyonu (üç kaybı birleştirir)

Bu hücreler sadece fonksiyon tanımlar; bellekte kalırlar.

**Süre:** Toplam \<2 sn.

### Aşama D — Görsele Özel Hesaplar

Bu hücreler **her yeni content/stil çifti** için yeniden çalıştırılmalıdır çünkü çıktıları o spesifik görsele bağımlıdır.

#### Hücre 11 — Matting Laplacian Matrisi

- Content görselinin piksellerinden 3×3 pencerelerle Levin Laplacian'ını hesaplar  
- scipy.sparse formatında üretir, sonra torch sparse tensora çevirir  
- 512×352 piksel görsel için 4.479.716 sıfırdan farklı değer içeren seyrek matris (180.224×180.224)  
- **İçerik değişince YENİDEN hesaplanmalı** çünkü matris değerleri o görselin piksellerinden türetilir

**Süre:** 15-30 sn.

#### Hücre 13 — SegFormer Maskeleri

- SegFormer-B2 (ADE20K) ile hem content hem stil görselinin segmentasyon maskelerini üretir  
- Maskeler AdaIN stil kaybında bölge bazlı eşleştirme için kullanılır  
- **İçerik veya stil değişince YENİDEN hesaplanmalı**

**Süre:** \~30 sn (ilk çalıştırmada model indirilir, sonraki çalıştırmalarda \~5 sn).

### Aşama E — Ana Optimizasyon ve Değerlendirme

**Sıralama:**

Hücre 15 → Hücre 16 → Hücre 17

#### Hücre 15 — Stil Transferi Optimizasyonu

- Çıktıyı content'in kopyası olarak başlatır (`requires_grad=True`)  
- LBFGS optimizer ile `NUM_STEPS` (varsayılan 400\) adım çalıştırır  
- Her adımda toplam kaybı geri yayılım ile günceller  
- LBFGS sparse gradient için "contiguous düzeltmesi" uygulanır  
- Her LOG\_EVERY (50) adımda kayıp değerlerini yazdırır

**Süre:** 3-6 dk.

#### Hücre 16 — Görselleştirme

- Content, stil ve çıktıyı 1×3 grid ile matplotlib'de gösterir  
- Görsel inceleme için kritik

**Süre:** \<5 sn.

#### Hücre 17 — Metrikler ve Kaydet

- SSIM (skimage) hesaplar — yapısal benzerlik (0-1, content vs çıktı)  
- LPIPS (AlexNet tabanlı) hesaplar — algısal mesafe  
- Çıktıyı PNG olarak `/content/outputs/output_base.png` kaydeder

**Süre:** \~15 sn.

### Aşama F — Karşılaştırma Deneyleri (OPSİYONEL)

Bu hücreler ana sonuç elde edildikten sonra çalıştırılır. Toplamda 30-60 dk sürer.

#### Hücre 18 — Lambda Ablasyonu

- `LAMBDA_ABLATION_VALUES` listesindeki her λ için ayrı optimizasyon  
- Tipik değerler: \[1, 100, 10000, 1000000\]  
- 1×N grid görselleştirme (Luan et al. Figure 3 tarzı)

**Süre:** 15-30 dk.

#### Hücre 19 — Backbone Karşılaştırması

- EfficientNet-B4 (mevcut) vs VGG-19 (klasik)  
- İki koşulda: (1) eşit hiperparametre, (2) her backbone'a özgü gamma  
- İki tablo \+ 1×3 görsel çıktısı

**Süre:** 10-15 dk.

#### Hücre 20 — Stil Kaybı Karşılaştırması

- AdaIN vs Gram matrisi (her ikisi de EfficientNet ile)  
- Hücre 19'a benzer iki tablo yaklaşımı uygulanabilir

**Süre:** 8-10 dk.

#### Hücre 21 — Tüm Sonuçları Kaydet

- Ana çıktı, ablasyon görselleri, backbone ve stil-kaybı karşılaştırma çıktılarını PNG olarak `/content/outputs/` altına yazar

**Süre:** \~10 sn.

### Tipik İlk Çalıştırma Sırası

Hücre 1 → 2 → \[Restart Runtime\] → 1 → 2 → 3 → 4 → 5 → 6

                                                      ↓

                   Stil görselini upload et

                                                      ↓

                                                   Hücre 7

                                                      ↓

                                          \\\[Op\\\] Hücre 7.5

                                                      ↓

                          \\\[Op\\\] Hücre 7 (yeni içerikle)

                                                      ↓

                                              Hücre 8 → 9 → 10

                                                      ↓

                                                   Hücre 11

                                                      ↓

                                              Hücre 12 → 13 → 14

                                                      ↓

                                              Hücre 15 → 16 → 17

                                                      ↓

                          \\\[Op\\\] Hücre 18 → 19 → 20 → 21

### Tipik İkinci Çalıştırma Sırası (İçerik Değişince)

Aynı oturumda farklı bir content denemek istediğinizde:

Hücre 7 (yeni content\_path\_example ile)

     ↓

Hücre 11 (Matting Laplacian yeniden)

     ↓

Hücre 13 (Segmentasyon maskeleri yeniden)

     ↓

Hücre 15 → 16 → 17

Hücre 1-6, 8-10, 12, 14 bellekteyken atlanır.

### Hangi Hücrelerin Yeniden Çalıştırılması Gerekir?

| Değişen Şey | Yeniden Çalıştırılacak Hücreler |
| :---- | :---- |
| Stil görseli | 7 → 13 → 15 → 16 → 17 |
| Content görseli | 7 → 11 → 13 → 15 → 16 → 17 |
| `GAMMA_STYLE` veya `LAMBDA_MATTING` | 4 → 15 → 16 → 17 |
| `NUM_STEPS` | 4 → 15 → 16 → 17 |
| Backbone | 19 (kendi içinde her şeyi yapar) |
| Lambda aralığı | 4 → 18 |
| Yeni oturum | Tümü (1'den itibaren) |

---

## Hücre Yapısı

| \# | İçerik | Süre |
| :---- | :---- | :---- |
| 1 | Drive bağlantısı \+ veri kopyalama | 5-15 dk |
| 2 | pip kurulumları | \~1 dk |
| 3 | Import \+ GPU kontrolü | \<5 sn |
| 4 | Hiperparametreler | \<1 sn |
| 5 | Klasör doğrulama | \<1 sn |
| 6 | Veri filtreleme | \~30 sn |
| 7 | Görsel yükleme \+ transform | \<5 sn |
| 7.5 | Stille semantik benzerlik sıralaması (opsiyonel) | 1-10 dk |
| 8 | EfficientNet-B4 yükleme | \~10 sn |
| 9-10 | AdaIN ve içerik kayıp testleri | \<1 sn |
| 11 | Matting Laplacian hesabı | 15-30 sn |
| 12 | Matting kaybı testi | \<1 sn |
| 13 | SegFormer \+ maskeler | \~30 sn |
| 14 | Toplam kayıp fonksiyonu | \<1 sn |
| 15 | Ana optimizasyon (400 adım LBFGS) | 3-6 dk |
| 16 | Görselleştirme | \<5 sn |
| 17 | SSIM \+ LPIPS \+ kaydet | \~15 sn |
| 18 | λ ablasyonu (4-5 değer) | 15-30 dk |
| 19 | Backbone karşılaştırması | 10-15 dk |
| 20 | Stil kaybı karşılaştırması | 8-10 dk |
| 21 | Tüm sonuçları kaydet | \~10 sn |

## Hiperparametreler

Tüm sabitler **Hücre 4'te** toplanmıştır. Önemli olanlar:

\# Kayıp ağırlıkları (en kritik)

ALPHA\_CONTENT  \= 1.0       \# içerik korumasının ağırlığı

GAMMA\_STYLE    \= 300       \# stil aktarımının ağırlığı

LAMBDA\_MATTING \= 5000      \# fotorealizm kısıtının ağırlığı

\# Görüntü işleme

MAX\_SIZE       \= 512       \# uzun kenar maksimum piksel

SIZE\_MULTIPLE  \= 32        \# CNN uyumu için boyut çarpanı

DATASET\_LIMIT  \= 1000      \# filtrelenecek aday sayısı

\# Optimizasyon

NUM\_STEPS      \= 400       \# LBFGS adım sayısı

LBFGS\_LR       \= 1.0       \# öğrenme oranı

\# Katman seçimleri

CONTENT\_LAYER\_INDICES \= \[3, 5\]

STYLE\_LAYER\_INDICES   \= \[1, 3, 5, 7\]

### Hiperparametre Tuning Kılavuzu

| Sonuç | Öneri |
| :---- | :---- |
| Stil az aktarılmış (SSIM \> 0.95) | `GAMMA_STYLE` ↑ veya `LAMBDA_MATTING` ↓ |
| Renk artefaktı (rainbow) | `LAMBDA_MATTING` ↑ |
| Content kayboldu | `ALPHA_CONTENT` ↑ veya `GAMMA_STYLE` ↓ |
| Yavaş yakınsama | `NUM_STEPS` ↑ |

`γ : λ` oranı yaklaşık **1:15-20** ideal.

## Deneyler ve Sonuçlar

### Ana Deney

Tek bir content-stil çifti üzerinde stil transferi:

- SSIM (content vs çıktı): yapısal benzerlik (1'e yakın \= aynı)  
- LPIPS (content vs çıktı): algısal mesafe (0'a yakın \= aynı)

Hedef aralık: SSIM ∈ \[0.78, 0.87\], LPIPS ∈ \[0.30, 0.45\].

### Lambda Ablasyonu (Hücre 18\)

λ ∈ \[1, 100, 10000, 1000000\] için aynı içerik/stil çifti. Luan et al. Figure 3 tarzı 1×4 grid görselleştirmesi.

- Düşük λ: agresif stil, rainbow artefakt  
- Orta λ: dengeli, fotorealistik  
- Yüksek λ: stil ezilir, content kalır

### Backbone Karşılaştırması (Hücre 19\)

EfficientNet-B4 vs VGG-19, iki koşulda:

1. Eşit hiperparametre (gamma=300)  
2. Adil karşılaştırma (her backbone'a özgü gamma)

İki tablo \+ 1×3 görsel.

### Stil Kaybı Karşılaştırması (Hücre 20\)

AdaIN vs Gram matrisi, aynı backbone (EfficientNet) ile.

## Bilimsel Bulgular

### 1\. Lambda parametresinin etkisi

Matting Laplacian ağırlığı (λ), fotorealizm-stilizasyon trade-off'unu doğrudan kontrol eder. λ=10⁴ civarı Luan et al.'in bulgularıyla tutarlıdır.

### 2\. Backbone mimarisinin stil duyarlılığı

VGG-19'un orta katman aktivasyonları, EfficientNet-B4'ünkilere göre istatistiksel olarak daha küçük ölçeklidir. Bu, aynı `gamma` ile eşitsiz stil aktarımına yol açar. Adil karşılaştırma için backbone'a özgü `gamma` ayarı zorunludur.

### 3\. Stil kaybı yönteminin etkisi

AdaIN (mean+std eşleştirme) ve Gram matrisi (kanal korelasyonu) ayrı ölçek profillerine sahip kayıplar üretir. AdaIN renk transferi yönünde, Gram doku transferi yönünde eğilimlidir.

### 4\. Metriklerin görsel kaliteyi tam yansıtmadığı durum

LPIPS yüksekliği "stilizasyon başarısı" anlamına gelmez — renk artefaktları da LPIPS'i yükseltir. Görsel inceleme her zaman metriklerle birlikte yapılmalıdır.

## Bilinen Kısıtlamalar

- Tek görsel üzerinde optimizasyon yapılır (per-image LBFGS), gerçek zamanlı çıkarım yok.  
- Matting Laplacian hesabı O(H·W·9²) bellek tüketir; 1024px üzeri görseller için ek bellek gerekir.  
- Stil ve içerik görsellerinin semantic benzerliği düşükse aktarım zayıf olur (Hücre 7.5 önerilir).

## Çıktı Dosyaları

Hücre 21 sonrası `/content/outputs/` altında:

- `output_main.png` — ana stil transferi sonucu  
- `output_lambda_X.png` — λ ablasyonu sonuçları  
- `output_vgg19.png` — VGG-19 ile sonuç  
- `output_gram.png` — Gram matrisi ile sonuç  
- `content.png`, `style_reference.png` — referans görseller

