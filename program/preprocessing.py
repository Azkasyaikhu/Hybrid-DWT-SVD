import cv2
import os

def load_and_resize_image(image_path, target_size=(512, 512), grayscale=True):
    """
    Membaca gambar dari path, mengubah ke grayscale (opsional), 
    dan melakukan resize ke ukuran target (default 512x512).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File tidak ditemukan di path: {image_path}")
    
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
    if img is None:
        raise ValueError(f"Gagal mendekode gambar: {image_path}")
        
    img_resized = cv2.resize(img, target_size)
    return img_resized