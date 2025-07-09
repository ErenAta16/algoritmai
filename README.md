# 🤖 Akıllı Algoritma Öneri Sistemi
## 🎯 Proje Amacı

Bu proje, makine öğrenmesi dünyasındaki en büyük zorluklardan birini çözmek için geliştirilmiştir: **doğru algoritma seçimi**. Kullanıcıların proje gereksinimlerine göre 229 farklı ML algoritması arasından en uygun olanları otomatik olarak öneren akıllı bir karar destek sistemidir.

## 🚀 Sistem Özellikleri

- **229 ML Algoritması** kapsamlı veri havuzu
- **16 Değerlendirme Kriteri** ile çok boyutlu analiz
- **Ward Clustering** ile 10 akıllı algoritma grubu
- **Otomatik Öneri Sistemi** - kullanıcı kriterlerine göre algoritma önerisi
- **Gerçek Zamanlı Analiz** - anında sonuç alma
- **Kapsamlı Görselleştirme** - 15 detaylı grafik ve rapor

## 📊 Veri Seti İçeriği

### Ana Veri Seti: `Algoritma_Veri_Seti.xlsx`
**Boyut:** 229 algoritma × 17 özellik

| Özellik Kategorisi | Özellikler | Açıklama |
|-------------------|------------|-----------|
| **Kategorik Kriterler** | Öğrenim Türü (7 seçenek) | Denetimli, Denetimsiz, Derin Öğrenme, vb. |
| | Kullanım Alanı (71 seçenek) | Görüntü İşleme, NLP, Sınıflandırma, vb. |
| | Karmaşıklık Düzeyi (5 seçenek) | Düşük → Çok Yüksek |
| | Veri Büyüklüğü (5 seçenek) | Küçük → Çok Büyük |
| | Donanım Gereksinimi (9 seçenek) | Düşük → Yüksek |
| **Sayısal Kriterler** | Popülerlik Skor (0.0-1.0) | Algoritmanın yaygınlık derecesi |
| | Aşırı Öğrenme Eğilimi Skor | Overfitting riski |
| | Fine-tune Gerekliliği Skor | İnce ayar ihtiyacı |

## 🏗️ Sistem Mimarisi

### 1. Veri İşleme Pipeline
```
Veri Seti (229×17) → One-Hot Encoding → Standardization → PCA (%95 varyans) → 110 bileşen
```

### 2. Kümeleme Analizi
```python
Ward Clustering (n_clusters=10)
├── Küme 0: Klasik Algoritmalar (37 algoritma)
├── Küme 1: Gelişmiş Denetimli (65 algoritma)  
├── Küme 2: Denetimsiz Öğrenme (31 algoritma)
├── Küme 3: Karar Ağaçları (46 algoritma)
├── Küme 4: Boosting Algoritmaları (29 algoritma)
└── ...
```


## 📁 Proje Yapısı

```
algoritma_yedek/
├── 📁 algorithms/
│   ├── analyze_dataset.py          # Veri seti analizi
│   ├── clustering_analysis.py      # Ana kümeleme analizi
│   ├── Veri_seti.csv              # CSV formatında veri
│   ├── Veri_seti.xlsx             # Excel formatında veri
│   ├── requirements.txt           # Python bağımlılıkları
│   └── README.md                  # Bu dosya
├── 📁 models/
│   ├── kmeans_model.joblib        # K-means modeli (33.2 KB)
│   └── ward_model.joblib          # Ward modeli (9.1 KB)
├── 📁 visualizations/
│   ├── 📁 clustering/             # Kümeleme görselleri (8 dosya)
│   ├── 📁 metrics/                # Metrik görselleri (7 dosya)
│   └── 📁 analysis/               # Analiz görselleri
├── Algoritma_Veri_Seti.xlsx       # Ana veri seti
└── cluster_characteristics.json   # Küme karakteristikleri
```

##