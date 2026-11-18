import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_psnr(original, watermarked):
    """Menghitung nilai Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, watermarked):
    """Menghitung nilai Structural Similarity Index (SSIM)."""
    score, _ = ssim(original, watermarked, full=True)
    return score

def calculate_nc(watermark_orig, watermark_extracted):
    """Menghitung Normalized Cross-Correlation (NC) untuk kemiripan watermark."""
    w_orig = watermark_orig.astype(np.float64).flatten()
    w_ext = watermark_extracted.astype(np.float64).flatten()
    
    numerator = np.sum(w_orig * w_ext)
    denominator = np.sqrt(np.sum(w_orig ** 2) * np.sum(w_ext ** 2))
    
    if denominator == 0:
        return 0.0
        
    nc = numerator / denominator
    return nc