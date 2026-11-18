import pywt
import numpy as np

def perform_dwt_level3(image):
    """
    Melakukan dekomposisi DWT hingga level 3 menggunakan wavelet 'haar'.
    Mengembalikan koefisien level 3 [LL3, (LH3, HL3, HH3), ...].
    """
    # Level 1
    coeffs1 = pywt.wavedec2(image, 'haar', level=1)
    # Level 2 dari LL1
    coeffs2 = pywt.wavedec2(coeffs1[0], 'haar', level=1)
    # Level 3 dari LL2
    coeffs3 = pywt.wavedec2(coeffs2[0], 'haar', level=1)
    
    # coeffs3[0] adalah subband LL3
    return coeffs3

def perform_idwt_level3(coeffs3, coeffs2_orig, coeffs1_orig):
    """
    Melakukan rekonstruksi Inverse DWT (IDWT) dari level 3 kembali ke citra semula.
    """
    # Rekonstruksi dari level 3 ke LL2
    reconstructed_ll2 = pywt.waverec2(coeffs3, 'haar')
    
    # Rekonstruksi ke level berikutnya (jika diperlukan secara utuh, 
    # atau gunakan pywt.waverec2 langsung dengan struktur koefisien lengkap)
    pass