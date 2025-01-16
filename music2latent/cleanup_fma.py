
import multiprocessing as mp
from pathlib import Path
from .train import fma_path
import soundfile as sf
import numpy as np
from tqdm import tqdm

def try_open(p):
    try:
        data, sr = sf.read(p)
        # invalid data
        if np.any(np.isnan(data)):
            return None
        # too-short
        if len(data) < sr:
            return None
        return p
    except sf.LibsndfileError:
        return None

def main():
    fma_files = sorted(fma_path.glob('*/*mp3'))
    pbar = tqdm(fma_files)
    with mp.Pool(6) as pool, open('fma_good_files.txt','w') as f:
        for result in pool.imap_unordered(try_open, fma_files):
            if result is not None:
                f.write(f"{result}\n")
            pbar.update()


if __name__=='__main__':
    main()