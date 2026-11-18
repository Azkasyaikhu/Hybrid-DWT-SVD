import numpy as np
import pywt
from program.svd import apply_svd, reconstruct_svd

def embed_watermark(host_image, watermark_image, alpha=0.01):
    """
    Proses embedding Hybrid DWT-SVD pada subband LL3.
    """
    # 1. DWT Level 3 pada host image
    coeffs = pywt.wavedec2(host_image, 'haar', level=3)
    LL3 = coeffs[0]  # Indeks 0 adalah subband LL3
    
    # 2. SVD pada subband LL3 host
    U_host, S_host, Vt_host = apply_svd(LL3)
    
    # 3. SVD pada watermark (resize watermark agar sesuai dengan dimensi S_host jika perlu)
    import cv2
    watermark_resized = cv2.resize(watermark_image, (S_host.shape[0], S_host.shape[0]))
    U_w, S_w, Vt_w = apply_svd(watermark_resized.astype(float))
    
    # 4. Modifikasi Singular Values Host dengan Watermark (S_host + alpha * S_w)
    S_marked = S_host + (alpha * S_w)
    
    # 5. Rekonstruksi LL3 baru menggunakan S_marked, U_host, dan Vt_host
    LL3_marked = reconstruct_svd(U_host, S_marked, Vt_host)
    
    # Masukkan kembali LL3_marked ke dalam koefisien DWT
    coeffs_marked = list(coeffs)
    coeffs_marked[0] = LL3_marked
    
    # 6. Inverse DWT (IDWT) untuk menghasilkan watermarked image
    watermarked_image = pywt.waverec2(coeffs_marked, 'haar')
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
    
    # Simpan komponen U_host, Vt_host, dan S_w untuk proses ekstraksi (jika diperlukan dalam skema non-blind)
    extraction_data = {
        'U_host': U_host,
        'Vt_host': Vt_host,
        'S_host': S_host,
        'U_w': U_w,
        'Vt_w': Vt_w
    }
    
    return watermarked_image, extraction_data