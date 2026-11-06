from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MOVIELENS_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}
AMAZON_MOVIES_5CORE_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
    "reviews_Movies_and_TV_5.json.gz"
)


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination} already exists")
        return destination
    response = requests.get(url, stream=True, timeout=(30, 180))
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with tmp.open("wb") as f, tqdm(total=total or None, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return destination


def download_dataset(name: str, raw_dir: str | Path = "data/raw") -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if name in MOVIELENS_URLS:
        archive = download_file(MOVIELENS_URLS[name], raw_dir / f"{name}.zip")
        target = raw_dir / name
        if not target.exists():
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(raw_dir)
        return target
    if name == "amazon-movies-tv":
        archive = download_file(AMAZON_MOVIES_5CORE_URL, raw_dir / "reviews_Movies_and_TV_5.json.gz")
        return archive
    if name == "netflix":
        raise ValueError(
            "Netflix Prize files are not downloaded automatically. Obtain the source files "
            "under the dataset provider's terms and place combined_data_*.txt in data/raw/netflix/."
        )
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public CA-MF-RecNAS source datasets")
    parser.add_argument("dataset", choices=[*MOVIELENS_URLS, "amazon-movies-tv", "netflix"])
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    print(download_dataset(args.dataset, args.raw_dir))


if __name__ == "__main__":
    main()
