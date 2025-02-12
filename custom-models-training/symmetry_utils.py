from typing import Tuple

import torch
import torch.nn as nn


def get_key_query_matrix(key_matrix: nn.Linear,
                         query_matrix: nn.Linear,
                         num_attention_heads: int,
                         attention_head_size: int,
                         hidden_size: int) -> torch.Tensor:
    """
    Calculate the Matrix M (the matrix multiplication of the key and query matrices)

    :param key_matrix: The key matrix (torch.nn.Linear)
    :param query_matrix: The query matrix (torch.nn.Linear)
    :param num_attention_heads: Number of attention heads
    :param attention_head_size: Size of each attention head
    :param hidden_size: Embedding size
    :return: The matrix M
    """
    key_matrix_head = key_matrix.weight.view(num_attention_heads, attention_head_size, hidden_size).transpose(
        -1, -2)
    query_matrix_head = query_matrix.weight.view(num_attention_heads, attention_head_size,
                                                 hidden_size).transpose(-1, -2)
    M = torch.matmul(query_matrix_head, key_matrix_head.transpose(-1, -2))
    return M


def to_symmetric(M: torch.Tensor) -> torch.Tensor:
    """
    Convert a matrix to a symmetric matrix
    :param M: The matrix to convert
    :return: The symmetric matrix
    """
    return 0.5 * (M + M.transpose(-1, -2))


def to_asymmetric(M: torch.Tensor) -> torch.Tensor:
    """
    Convert a matrix to an asymmetric matrix
    :param M: The matrix to convert
    :return: The skew symmetric matrix
    """
    return 0.5 * (M - M.transpose(-1, -2))


def to_symmetric_asymmetric_fast(key_matrix: nn.Linear,
                                 query_matrix: nn.Linear,
                                 num_attention_heads: int,
                                 attention_head_size: int,
                                 hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the symmetric and skew symmetric scores of the key and query matrices for each head
    param key_matrix: The key matrix (torch.nn.Linear)
    :param query_matrix: The query matrix (torch.nn.Linear)
    :param num_attention_heads: Number of attention heads
    :param attention_head_size: Size of each attention head
    :param hidden_size: Embedding size
    :return: A tuple of the symmetric and skew symmetric scores
    """
    key = key_matrix.weight.data.view(num_attention_heads, attention_head_size, hidden_size)
    key_t = key.transpose(-1, -2)
    query = query_matrix.weight.data.view(num_attention_heads, attention_head_size, hidden_size)
    query_t = query.transpose(-1, -2)
    A = torch.matmul(query, query_t)
    B = torch.matmul(key, key_t)
    C = torch.matmul(key, query_t)
    S = .5 * (1 + (torch.einsum('hij,hji->h', C, C)) / torch.einsum('hij,hji->h', A, B))
    N = .5 * (1 - (torch.einsum('hij,hji->h', C, C)) / torch.einsum('hij,hji->h', A, B))

    return S, N
