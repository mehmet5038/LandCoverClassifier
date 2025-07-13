import os
from PIL import Image

source = "data/uc_merced"
target = "data/uc_merced_converted"

classes = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral",
    "denseresidential", "forest", "freeway", "golfcourse", "harbor", "intersection",
    "mediumresidential", "mobilehomepark", "overpass", "parkinglot", "river",
    "runway", "sparseresidential", "storagetanks", "tenniscourt"
]

os.makedirs(target, exist_ok=True)

for cls in classes:
    src_dir = os.path.join(source, cls)
    tgt_dir = os.path.join(target, cls)
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

            if count >= 100:
                break
            
print("Dönüştürme tamamlandı.")