from typing import List

from joblib import parallel_backend, Parallel, delayed
from tqdm import tqdm

from model import word_frame, words_frame, cleanup_columns
import pandas as pd

accepted_words_path = 'data/input/accepted_words.txt'
possible_words_path = 'data/input/possible_words.txt'
accepted_words_ohe_path = 'data/processed/accepted_words_ohe.csv'


def read_file(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [line.strip() for line in f.readlines()]


def write_ohe_for_words(input_path: str, output_path: str) -> None:
    with parallel_backend('threading', n_jobs=-1):
        rows = Parallel()(delayed(word_frame)(word) for word in tqdm(read_file(input_path)))
        df = cleanup_columns(words_frame(rows))
        df.to_csv(output_path)


def train_model() -> None:
    # write_ohe_for_words(accepted_words_path, accepted_words_ohe_path)
    accepted_words_ohe = pd.read_csv(accepted_words_ohe_path)

    for column in accepted_words_ohe.columns:
        print(f'{column}: {accepted_words_ohe[column].sum()}')


if __name__ == '__main__':
    train_model()
