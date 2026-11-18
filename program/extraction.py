import numpy as np
import pywt
import cv2
from program.svd import apply_svd

def extract_watermark(watermarked_image, extraction_data, alpha=0.01):
    """
    Proses ekstraksi watermark dari watermarked image.
    """
    U_host = extraction_data['U_host']
    Vt_host = extraction_data['Vt_host']
    S_host = extraction_data['S_host']
    U_w = extraction_data['U_w']
    Vt_w = extraction_data['Vt_w']
    
    # 1. DWT Level 3 pada watermarked image (atau citra ber-attack)
    coeffs = pywt.wavedec2(watermarked_image, 'haar', level=3)
    LL3_marked = coeffs[0]
    
    # 2. SVD pada LL3_marked
    U_m, S_m, Vt_m = apply_svd(LL3_marked)
    
    # 3. Ekstraksi singular values watermark (S_w_extracted)
    # S_marked = S_host + alpha * S_w  =>  S_w = (S_marked - S_host) / alpha
    # Pastikan ukuran S_m dan S_host sama
    min_len = min(len(S_m), len(S_host))
    S_m_trim = S_m[:min_len]
    S_host_trim = S_host[:min_len]
    
    S_w_extracted = (S_m_trim - S_host_trim) / alpha
    
    # Sesuaikan ukuran dengan U_w dan Vt_w yang disimpan saat embedding
    w_dim = U_w.shape[0]
    if len(S_w_extracted) < w_dim:
        # Padding dengan zeros jika kurang
        S_w_padded = np.zeros(w_dim)
        S_w_padded[:len(S_w_extracted)] = S_w_extracted
        S_w_extracted = S_w_padded
    else:
        S_w_extracted = S_w_extracted[:w_dim]
    
    # 4. Rekonstruksi citra watermark asli
    S_w_matrix = np.diag(S_w_extracted)
    extracted_watermark_float = np.dot(U_w, np.dot(S_w_matrix, Vt_w))
    
    # Normalisasi ke rentang 0-255 dan konversi ke uint8
    extracted_watermark = cv2.normalize(extracted_watermark_float, None, 0, 255, cv2.NORM_MINMAX)
    extracted_watermark = extracted_watermark.astype(np.uint8)
    
    return extracted_watermark