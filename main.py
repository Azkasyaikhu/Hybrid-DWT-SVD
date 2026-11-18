import os
import cv2
from program.preprocessing import load_and_resize_image
from program.embedding import embed_watermark
from program.extraction import extract_watermark
from program.evaluasi import calculate_psnr, calculate_ssim, calculate_nc

def main():
    # Buat direktori output jika belum ada
    os.makedirs("hasil/watermarked", exist_ok=True)
    os.makedirs("hasil/attacked", exist_ok=True)
    os.makedirs("hasil/extracted", exist_ok=True)
    
    # Path dataset (sesuaikan dengan file yang ingin diuji)
    host_path = "dataset/gambar_standar/lena_gray_512.tif"
    watermark_path = "dataset/watermark/logo.jpeg"
    
    print("[INFO] Memuat dan memproses dataset...")
    host_img = load_and_resize_image(host_path, target_size=(512, 512), grayscale=True)
    watermark_img = load_and_resize_image(watermark_path, target_size=(512, 512), grayscale=True)
    
    # Parameter alpha awal
    alpha = 0.01
    
    print(f"[INFO] Melakukan embedding watermark dengan alpha = {alpha}...")
    watermarked_img, extraction_data = embed_watermark(host_img, watermark_img, alpha=alpha)
    
    # Simpan hasil watermarked
    cv2.imwrite("hasil/watermarked/lena_watermarked.png", watermarked_img)
    
    # Evaluasi Imperceptibility (Tanpa Serangan)
    psnr_val = calculate_psnr(host_img, watermarked_img)
    ssim_val = calculate_ssim(host_img, watermarked_img)
    print(f"=== HASIL EVALUASI TANPA SERANGAN ===")
    print(f"PSNR : {psnr_val:.4f} dB")
    print(f"SSIM : {ssim_val:.4f}")
    
    print("[INFO] Melakukan ekstraksi watermark...")
    extracted_w = extract_watermark(watermarked_img, extraction_data, alpha=alpha)
    
    # Simpan hasil ekstraksi
    cv2.imwrite("hasil/extracted/extracted_logo.png", extracted_w)
    
    # Samakan ukuran citra hasil ekstraksi dengan watermark asli jika berbeda
    if extracted_w.shape != watermark_img.shape:
        extracted_w = cv2.resize(extracted_w, (watermark_img.shape[1], watermark_img.shape[0]))

    nc_val = calculate_nc(watermark_img, extracted_w)
    print(f"NC (Normalized Cross-Correlation) : {nc_val:.4f}")
    print("[INFO] Project selesai dieksekusi dengan sukses!")

if __name__ == "__main__":
    main()