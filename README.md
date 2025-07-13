# LandCoverClassifier

Bu repo, uydu ve hava görüntülerinden baskın arazi türünü sınıflandırmak için geliştirilen derin öğrenme ve geleneksel makine öğrenmesi tabanlı bir sistemin kodunu içerir.

## Proje Açıklaması

Bu proje, uydu görüntülerinden **şehir, orman, tarım, plaj, dağ ve çöl** gibi arazi türlerini otomatik olarak sınıflandırmayı amaçlamaktadır. İki farklı yöntem uygulanmış ve karşılaştırılmıştır:

- **Yöntem 1:** Convolutional Neural Networks (CNN) ile sınıflandırma
- **Yöntem 2:** Elle çıkarılan özelliklerle geleneksel makine öğrenmesi algoritmaları (Random Forest, Gradient Boosting vb.)

Bu proje, **TOBB ETÜ Bilgisayar Mühendisliği** bölümü, **YAP 470** dersi içindir.

### Yöntem 1 | Convolutional Neural Network (CNN) ile Arazi Türü Sınıflandırması:

Model, piksel değerleri normalize edilmiş ve augmentasyon uygulanmış uydu görüntülerini girdi olarak alır. Ardından konvolüsyonel katmanlar üzerinden öznitelik çıkarımı yaparak her görüntüyü ait olduğu arazi sınıfına göre sınıflandırır.

#### Jupyter Notebookları:

- **`train_cnn.ipynb`:** 
  - CNN modelinin oluşturulup eğitildiği kısımdır.
  - Eğitim sırasında `ImageDataGenerator` ile veri artırma yapılır.
  - Eğitim ve doğrulama doğruluğu, kaybı grafikle izlenir.
  - Erken durdurma (EarlyStopping) ile eğitim kontrol edilir.
  - Eğitim sonunda model `.keras` formatında kaydedilir.

- **`test_model.ipynb`:** 
  - Eğitilen modelin doğrulama, test verisi üzerindeki başarımını değerlendirir.
  - Karışıklık matrisi ve sınıf bazlı metrikler (`precision`, `recall`, `F1-score`) hesaplanır.
  - Rastgele seçilen örnek görseller üzerinde gerçek ve tahmin karşılaştırması yapılır.