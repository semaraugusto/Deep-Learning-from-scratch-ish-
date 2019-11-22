import torch
import time


def basic_matmul(A, B):
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    assert a_cols == b_rows
    C = torch.zeros(a_rows, b_cols)
    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(a_cols):
                C[i, j] += A[i, k] * B[k, j]

    return C


def elementwise_matmul(A, B):
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    C = torch.zeros(a_rows, b_cols)
    assert a_cols == b_rows
    for i in range(a_rows):
        for j in range(b_cols):
            C[i, j] = (A[i, :] * B[:, j]).sum()

    return C


def broadcast_matmul(A, B):
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    C = torch.zeros(a_rows, b_cols)
    assert a_cols == b_rows
    for i in range(a_rows):
        C[i] = (A[i, ..., None] * B).sum(dim=0)

    return C


def einsum_matmul(A, B):
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    assert a_cols == b_rows
    return torch.einsum('ik,kj->ij', A,B)


A = torch.randn(100, 1000)
B = torch.randn(1000, 100)

start = time.time()
correct = torch.matmul(A, B)
diff_torch = time.time() - start
print(f'ELAPSED TIME TORCH {diff_torch}')

start = time.time()
C = basic_matmul(A, B)
diff_basic = time.time() - start
assert torch.allclose(C, correct, rtol=1e-3, atol=1e-5)
print(f'ELAPSED TIME BASIC MATMUL {diff_basic}')

start = time.time()
C = elementwise_matmul(A, B)
diff_elementwise = time.time() - start
assert torch.allclose(C, correct, rtol=1e-3, atol=1e-5)
print(f'ELAPSED TIME ELEMENTWISE MATMUL {diff_elementwise}')

start = time.time()
C = broadcast_matmul(A, B)
diff_broadcast = time.time() - start
assert torch.allclose(C, correct, rtol=1e-3, atol=1e-5)
print(f'ELAPSED TIME BROADCAST MATMUL {diff_broadcast}')

start = time.time()
C = einsum_matmul(A, B)
diff_einsum = time.time() - start
assert torch.allclose(C, correct, rtol=1e-3, atol=1e-5)
print(f'ELAPSED TIME EINSUM MATMUL {diff_einsum}')
