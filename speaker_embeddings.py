import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import torchaudio
from torchaudio.functional import resample
import numpy as np
from tqdm import tqdm
from preprocessing.ProsodicConditionExtractor import ProsodicConditionExtractor


def process_wav(in_path, out_path):
    wav, sr = torchaudio.load(in_path)
    wav = resample(wav, sr, 16000).squeeze()
    
    prosodic_condition = pros_cond_ext.extract_condition_from_reference_wave(wav, already_normalized=True).cpu()

    np.save(out_path.with_suffix(".npy"), prosodic_condition.squeeze().numpy())
    return out_path


def preprocess_dataset(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    # executor = ProcessPoolExecutor(max_workers=cpu_count())
    print(f"Extracting features for {args.in_dir}")
    for in_path in tqdm(args.in_dir.rglob("*.wav")):
        relative_path = in_path.relative_to(args.in_dir)
        out_path = args.out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        a = process_wav(in_path, out_path)
        # futures.append(executor.submit(process_wav, in_path, out_path))

    # results = [future.result() for future in tqdm(futures)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mel-spectrograms for an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    args = parser.parse_args()
    pros_cond_ext = ProsodicConditionExtractor(sr=22050, device='cuda')
    preprocess_dataset(args)
