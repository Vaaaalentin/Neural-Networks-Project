import os
import shutil
from pathlib import Path
import kagglehub

chemin_download = kagglehub.dataset_download(
    "gpiosenka/sports-classification"
)
source_dir = Path(chemin_download)

selected_sports = [
    "football",
    "basketball",
    "tennis",
    "swimming",
    "ski jumping",
    "track bicycle",
    "surfing",
    "bowling",
    "rowing",
    "judo",
]

target_dir = (Path("data") / "sports_10").resolve()

for split in ["train", "valid", "test"]:
  for sport in selected_sports:
    src = source_dir / split / sport
    dst = target_dir / split / sport
    if src.exists():
      shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
      print(f"{sport} is not in the dataset")