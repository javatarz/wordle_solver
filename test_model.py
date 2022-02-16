from typing import Dict, List

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from model import alphabets, positional_alphabet_columns, letter_positions_in_word, word_position_count_as_row, \
    word_frame, words_frame, cleanup_columns

expected_positional_alphabet_columns = [
    'a_1', 'a_2', 'a_3', 'a_4', 'a_5',
    'b_1', 'b_2', 'b_3', 'b_4', 'b_5',
    'c_1', 'c_2', 'c_3', 'c_4', 'c_5',
    'd_1', 'd_2', 'd_3', 'd_4', 'd_5',
    'e_1', 'e_2', 'e_3', 'e_4', 'e_5',
    'f_1', 'f_2', 'f_3', 'f_4', 'f_5',
    'g_1', 'g_2', 'g_3', 'g_4', 'g_5',
    'h_1', 'h_2', 'h_3', 'h_4', 'h_5',
    'i_1', 'i_2', 'i_3', 'i_4', 'i_5',
    'j_1', 'j_2', 'j_3', 'j_4', 'j_5',
    'k_1', 'k_2', 'k_3', 'k_4', 'k_5',
    'l_1', 'l_2', 'l_3', 'l_4', 'l_5',
    'm_1', 'm_2', 'm_3', 'm_4', 'm_5',
    'n_1', 'n_2', 'n_3', 'n_4', 'n_5',
    'o_1', 'o_2', 'o_3', 'o_4', 'o_5',
    'p_1', 'p_2', 'p_3', 'p_4', 'p_5',
    'q_1', 'q_2', 'q_3', 'q_4', 'q_5',
    'r_1', 'r_2', 'r_3', 'r_4', 'r_5',
    's_1', 's_2', 's_3', 's_4', 's_5',
    't_1', 't_2', 't_3', 't_4', 't_5',
    'u_1', 'u_2', 'u_3', 'u_4', 'u_5',
    'v_1', 'v_2', 'v_3', 'v_4', 'v_5',
    'w_1', 'w_2', 'w_3', 'w_4', 'w_5',
    'x_1', 'x_2', 'x_3', 'x_4', 'x_5',
    'y_1', 'y_2', 'y_3', 'y_4', 'y_5',
    'z_1', 'z_2', 'z_3', 'z_4', 'z_5'
]
expected_ohe_apple = {
    'a_1': 1, 'a_2': 0, 'a_3': 0, 'a_4': 0, 'a_5': 0,
    'b_1': 0, 'b_2': 0, 'b_3': 0, 'b_4': 0, 'b_5': 0,
    'c_1': 0, 'c_2': 0, 'c_3': 0, 'c_4': 0, 'c_5': 0,
    'd_1': 0, 'd_2': 0, 'd_3': 0, 'd_4': 0, 'd_5': 0,
    'e_1': 0, 'e_2': 0, 'e_3': 0, 'e_4': 0, 'e_5': 1,
    'f_1': 0, 'f_2': 0, 'f_3': 0, 'f_4': 0, 'f_5': 0,
    'g_1': 0, 'g_2': 0, 'g_3': 0, 'g_4': 0, 'g_5': 0,
    'h_1': 0, 'h_2': 0, 'h_3': 0, 'h_4': 0, 'h_5': 0,
    'i_1': 0, 'i_2': 0, 'i_3': 0, 'i_4': 0, 'i_5': 0,
    'j_1': 0, 'j_2': 0, 'j_3': 0, 'j_4': 0, 'j_5': 0,
    'k_1': 0, 'k_2': 0, 'k_3': 0, 'k_4': 0, 'k_5': 0,
    'l_1': 0, 'l_2': 0, 'l_3': 0, 'l_4': 1, 'l_5': 0,
    'm_1': 0, 'm_2': 0, 'm_3': 0, 'm_4': 0, 'm_5': 0,
    'n_1': 0, 'n_2': 0, 'n_3': 0, 'n_4': 0, 'n_5': 0,
    'o_1': 0, 'o_2': 0, 'o_3': 0, 'o_4': 0, 'o_5': 0,
    'p_1': 0, 'p_2': 1, 'p_3': 1, 'p_4': 0, 'p_5': 0,
    'q_1': 0, 'q_2': 0, 'q_3': 0, 'q_4': 0, 'q_5': 0,
    'r_1': 0, 'r_2': 0, 'r_3': 0, 'r_4': 0, 'r_5': 0,
    's_1': 0, 's_2': 0, 's_3': 0, 's_4': 0, 's_5': 0,
    't_1': 0, 't_2': 0, 't_3': 0, 't_4': 0, 't_5': 0,
    'u_1': 0, 'u_2': 0, 'u_3': 0, 'u_4': 0, 'u_5': 0,
    'v_1': 0, 'v_2': 0, 'v_3': 0, 'v_4': 0, 'v_5': 0,
    'w_1': 0, 'w_2': 0, 'w_3': 0, 'w_4': 0, 'w_5': 0,
    'x_1': 0, 'x_2': 0, 'x_3': 0, 'x_4': 0, 'x_5': 0,
    'y_1': 0, 'y_2': 0, 'y_3': 0, 'y_4': 0, 'y_5': 0,
    'z_1': 0, 'z_2': 0, 'z_3': 0, 'z_4': 0, 'z_5': 0
}
expected_ohe_straw = {
    'a_1': 0, 'a_2': 0, 'a_3': 0, 'a_4': 1, 'a_5': 0,
    'b_1': 0, 'b_2': 0, 'b_3': 0, 'b_4': 0, 'b_5': 0,
    'c_1': 0, 'c_2': 0, 'c_3': 0, 'c_4': 0, 'c_5': 0,
    'd_1': 0, 'd_2': 0, 'd_3': 0, 'd_4': 0, 'd_5': 0,
    'e_1': 0, 'e_2': 0, 'e_3': 0, 'e_4': 0, 'e_5': 0,
    'f_1': 0, 'f_2': 0, 'f_3': 0, 'f_4': 0, 'f_5': 0,
    'g_1': 0, 'g_2': 0, 'g_3': 0, 'g_4': 0, 'g_5': 0,
    'h_1': 0, 'h_2': 0, 'h_3': 0, 'h_4': 0, 'h_5': 0,
    'i_1': 0, 'i_2': 0, 'i_3': 0, 'i_4': 0, 'i_5': 0,
    'j_1': 0, 'j_2': 0, 'j_3': 0, 'j_4': 0, 'j_5': 0,
    'k_1': 0, 'k_2': 0, 'k_3': 0, 'k_4': 0, 'k_5': 0,
    'l_1': 0, 'l_2': 0, 'l_3': 0, 'l_4': 0, 'l_5': 0,
    'm_1': 0, 'm_2': 0, 'm_3': 0, 'm_4': 0, 'm_5': 0,
    'n_1': 0, 'n_2': 0, 'n_3': 0, 'n_4': 0, 'n_5': 0,
    'o_1': 0, 'o_2': 0, 'o_3': 0, 'o_4': 0, 'o_5': 0,
    'p_1': 0, 'p_2': 0, 'p_3': 0, 'p_4': 0, 'p_5': 0,
    'q_1': 0, 'q_2': 0, 'q_3': 0, 'q_4': 0, 'q_5': 0,
    'r_1': 0, 'r_2': 0, 'r_3': 1, 'r_4': 0, 'r_5': 0,
    's_1': 1, 's_2': 0, 's_3': 0, 's_4': 0, 's_5': 0,
    't_1': 0, 't_2': 1, 't_3': 0, 't_4': 0, 't_5': 0,
    'u_1': 0, 'u_2': 0, 'u_3': 0, 'u_4': 0, 'u_5': 0,
    'v_1': 0, 'v_2': 0, 'v_3': 0, 'v_4': 0, 'v_5': 0,
    'w_1': 0, 'w_2': 0, 'w_3': 0, 'w_4': 0, 'w_5': 1,
    'x_1': 0, 'x_2': 0, 'x_3': 0, 'x_4': 0, 'x_5': 0,
    'y_1': 0, 'y_2': 0, 'y_3': 0, 'y_4': 0, 'y_5': 0,
    'z_1': 0, 'z_2': 0, 'z_3': 0, 'z_4': 0, 'z_5': 0
}


def test_alphabets() -> None:
    expected = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
        'v', 'w', 'x', 'y', 'z'
    ]

    assert alphabets() == expected


def test_positional_alphabet_columns() -> None:
    assert positional_alphabet_columns() == expected_positional_alphabet_columns


@pytest.mark.parametrize('word, expected', [
    ('apple', ['a_1', 'p_2', 'p_3', 'l_4', 'e_5']),
    ('straw', ['s_1', 't_2', 'r_3', 'a_4', 'w_5']),
])
def test_letter_positions_in_word(word: str, expected: List[str]) -> None:
    assert letter_positions_in_word(word) == expected


@pytest.mark.parametrize('word, expected', [
    ('apple', expected_ohe_apple),
    ('straw', expected_ohe_straw),
])
def test_word_position_count_as_row(word: str, expected: Dict[str, int]) -> None:
    assert word_position_count_as_row(word) == expected


@pytest.mark.parametrize('word, ohe', [
    ('apple', expected_ohe_apple),
    ('straw', expected_ohe_straw),
])
def test_word_frame(word: str, ohe: Dict[str, int]) -> None:
    expected = pd.DataFrame(ohe, index=[word])

    assert_frame_equal(word_frame(word), expected)


def test_concat_words() -> None:
    apple = pd.DataFrame(expected_ohe_apple, index=['apple'])
    straw = pd.DataFrame(expected_ohe_straw, index=['straw'])

    actual = words_frame([apple, straw])

    assert list(actual.columns) == expected_positional_alphabet_columns
    assert actual.shape[0] == 2


def test_cleanup_columns() -> None:
    apple = pd.DataFrame(expected_ohe_apple, index=['apple'])
    straw = pd.DataFrame(expected_ohe_straw, index=['straw'])
    input_df = words_frame([apple, straw])
    expected_data = {
        'a_1': [1, 0], 'p_2': [1, 0], 'p_3': [1, 0], 'l_4': [1, 0], 'e_5': [1, 0],
        's_1': [0, 1], 't_2': [0, 1], 'r_3': [0, 1], 'a_4': [0, 1], 'w_5': [0, 1]
    }
    expected = pd.DataFrame(expected_data, index=['apple', 'straw']).sort_index(axis=1)

    actual = cleanup_columns(input_df)

    assert_frame_equal(actual, expected)
