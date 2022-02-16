import string
from functools import reduce
from typing import List, Dict

import pandas as pd
from pandas import DataFrame


def alphabets() -> List[str]:
    return list(string.ascii_lowercase)


def positional_alphabet_columns(word_length: int = 5) -> List[str]:
    return [f'{letter}_{position}' for letter in alphabets() for position in range(1, word_length + 1)]


def letter_positions_in_word(word: str) -> List[str]:
    return [f'{letter}_{position + 1}' for position, letter in enumerate(word)]


def word_position_count_as_row(word: str) -> Dict[str, int]:
    letter_positions = letter_positions_in_word(word)

    return {
        column_name: 1 if column_name in letter_positions else 0
        for column_name in positional_alphabet_columns(len(word))
    }


def word_frame(word: str) -> DataFrame:
    return pd.DataFrame(word_position_count_as_row(word), index=[word])


def words_frame(word_frames: List[DataFrame]) -> DataFrame:
    return pd.concat(word_frames)


def cleanup_columns(words_ohe: DataFrame) -> DataFrame:
    return reduce(
        lambda acc, col: acc.drop(col, axis=1),
        filter(lambda col: words_ohe[col].sum() == 0, words_ohe.columns),
        words_ohe
    )
