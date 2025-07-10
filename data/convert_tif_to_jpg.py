import os
from PIL import Image

# 🔁 Kaynak ve hedef klasör yolları
source_root = "data/sample_ucmerced_tif"  # BURAYI KENDİ YOLUNA GÖRE DÜZENLE
target_root = "data/sample_ucmerced"

# 🌍 Kullanılacak sınıflar
classes = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral",
    "denseresidential", "forest", "freeway", "golfcourse", "harbor", "intersection",
    "mediumresidential", "mobilehomepark", "overpass", "parkinglot", "river",
    "runway", "sparseresidential", "storagetanks", "tenniscourt"
]

# 📁 Hedef klasörleri oluştur
os.makedirs(target_root, exist_ok=True)

for cls in classes:
    src_dir = os.path.join(source_root, cls)
    tgt_dir = os.path.join(target_root, cls)
    os.makedirs(tgt_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(src_dir):
        if filename.endswith(".tif"):
            src_path = os.path.join(src_dir, filename)
            img = Image.open(src_path).convert("RGB")
            out_name = filename.replace(".tif", ".jpg")
            out_path = os.path.join(tgt_dir, out_name)
            img.save(out_path, "JPEG")
            count += 1

            if count >= 20:  # Her sınıftan sadece 20 örnek
                break

print("Dönüştürme tamamlandı.")
