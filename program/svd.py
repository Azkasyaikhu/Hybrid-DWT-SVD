import numpy as np

def apply_svd(matrix):
    """
    Melakukan SVD pada matriks input (misalnya subband LL3).
    Mengembalikan matriks U, S (singular values dalam bentuk 1D array), dan Vt.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt

def reconstruct_svd(U, S, Vt):
    """
    Melakukan rekonstruksi matriks dari komponen U, S, dan Vt (ISVD).
    """
    S_matrix = np.diag(S)
    reconstructed_matrix = np.dot(U, np.dot(S_matrix, Vt))
    return reconstructed_matrix