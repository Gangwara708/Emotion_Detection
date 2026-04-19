"""
Dataset and pre-trained model downloader.

Usage:
    python download_dataset.py                  # FER2013 via Kaggle API
    python download_dataset.py --ag_models      # Age/Gender Caffe models only
    python download_dataset.py --all            # Everything

FER2013 on Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
Age/Gender models: Gil Levi & Tal Hassner (2015) — publicly available.

Kaggle API setup (one-time):
  1. Go to kaggle.com → Account → Create API Token
  2. Place kaggle.json in ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import shutil


# ── Age / Gender Caffe model URLs (publicly hosted mirrors) ───────────────────

AG_MODELS = {
    'age_deploy.prototxt': (
        'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/'
        'master/age_net_definitions/deploy.prototxt'
    ),
    'age_net.caffemodel': (
        'https://www.dropbox.com/s/xfeytas6iij1gbl/age_net.caffemodel?dl=1'
    ),
    'gender_deploy.prototxt': (
        'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/'
        'master/gender_net_definitions/deploy.prototxt'
    ),
    'gender_net.caffemodel': (
        'https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1'
    ),
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
        print(f'\r  [{bar}] {pct:.1f}%', end='', flush=True)


def download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    print(f'Downloading {os.path.basename(dest)} ...')
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
    except Exception as e:
        print(f'\n[ERROR] Failed to download {url}: {e}')
        return False
    return True


def download_ag_models(model_dir: str = 'models'):
    print('\n── Age / Gender Caffe Models ─────────────────────────────')
    for fname, url in AG_MODELS.items():
        dest = os.path.join(model_dir, fname)
        if os.path.exists(dest):
            print(f'  [SKIP] {fname} already exists.')
            continue
        download_file(url, dest)
    print('Age/Gender models ready.\n')


def download_fer2013_kaggle(data_dir: str = 'data'):
    """Download FER2013 using the Kaggle API."""
    print('\n── FER2013 Dataset via Kaggle API ────────────────────────')
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print('[ERROR] Kaggle credentials not found.')
        print_manual_instructions()
        return

    try:
        import kaggle
        os.makedirs(data_dir, exist_ok=True)
        print('Downloading FER2013 from Kaggle ...')
        kaggle.api.dataset_download_files(
            'msambare/fer2013',
            path=data_dir,
            unzip=True,
            quiet=False,
        )
        print(f'\nFER2013 downloaded to {data_dir}/')

        # Rename CSV if needed
        for fname in os.listdir(data_dir):
            if fname.endswith('.csv') and fname != 'fer2013.csv':
                src = os.path.join(data_dir, fname)
                dst = os.path.join(data_dir, 'fer2013.csv')
                shutil.move(src, dst)
                print(f'Renamed {fname} → fer2013.csv')
                break

    except ImportError:
        print('[ERROR] kaggle package not installed. Run: pip install kaggle')
    except Exception as e:
        print(f'[ERROR] {e}')
        print_manual_instructions()


def print_manual_instructions():
    print("""
Manual Download Instructions:
──────────────────────────────
1. Visit https://www.kaggle.com/datasets/msambare/fer2013
2. Sign in and click "Download"
3. Extract and place fer2013.csv inside the  data/  folder:

       Emotion_Detection/
       └── data/
           └── fer2013.csv

Alternative (directory format):
  Extract so you get:
       data/
       ├── train/
       │   ├── angry/
       │   ├── disgust/
       │   ├── fear/
       │   ├── happy/
       │   ├── neutral/
       │   ├── sad/
       │   └── surprise/
       └── test/
           └── ... (same structure)

Then run training:
    python -m src.train --data data/fer2013.csv
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ag_models', action='store_true',
                   help='Download Age/Gender Caffe models only')
    p.add_argument('--all', action='store_true',
                   help='Download everything (FER2013 + Age/Gender models)')
    p.add_argument('--data_dir',  type=str, default='data')
    p.add_argument('--model_dir', type=str, default='models')
    args = p.parse_args()

    if args.ag_models or args.all:
        download_ag_models(args.model_dir)

    if not args.ag_models or args.all:
        download_fer2013_kaggle(args.data_dir)

    if not args.ag_models and not args.all:
        print_manual_instructions()


if __name__ == '__main__':
    main()
