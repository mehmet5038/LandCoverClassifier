# LandCoverClassifier

Bu repo, uydu ve hava görüntülerinden baskın arazi türünü sınıflandırmak için geliştirilen derin öğrenme ve geleneksel makine öğrenmesi tabanlı bir sistemin kodunu içerir.

## Proje Açıklaması

Bu proje, uydu görüntülerinden **şehir, orman, tarım, plaj, dağ ve çöl** gibi arazi türlerini otomatik olarak sınıflandırmayı amaçlamaktadır. İki farklı yöntem uygulanmış ve karşılaştırılmıştır:

- **Yöntem 1:** Convolutional Neural Networks (CNN) ile sınıflandırma
- **Yöntem 2:** Elle çıkarılan özelliklerle geleneksel makine öğrenmesi algoritmaları (Random Forest, Gradient Boosting vb.)

Bu proje, **TOBB ETÜ Bilgisayar Mühendisliği** bölümü, **YAP 470** dersi içindir.

### Yöntem 1 – CNN

**Notebooklar:**
- `train_cnn_uc_merced.ipynb` – UC Merced veri seti ile CNN eğitimi  
- `train_cnn_skyview.ipynb` – SkyView veri seti ile CNN eğitimi  
- `test_cnn.ipynb` – CNN modellerinin test edilmesi, karışıklık matrisi ve metrik analizi

---

### Yöntem 2 – Makine Öğrenmesi

- **Özellikler:** HOG, LBP, HSV histogram, GLCM
- **Boyut azaltma:** PCA
- **Modeller:** Random Forest, Gradient Boosting, MLP
- Grid Search ile hiperparametre optimizasyonu

**Notebooklar:**
- `train_ml_uc_merced.ipynb` – UC Merced veri seti ile ML eğitimi  
- `train_ml_skyview.ipynb` – SkyView veri seti ile ML eğitimi  
- `test_ml.ipynb` – ML modellerinin test edilmesi ve karşılaştırılması