import cv2
import matplotlib.pyplot as plt
from skimage import exposure
import numpy as np
import os

def plot_image_and_hist(image, title, ax_img, ax_hist):
    """Fungsi pembantu untuk menampilkan citra dan histogramnya pada subplot"""
    # Menampilkan citra
    ax_img.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax_img.set_title(title)
    ax_img.axis('off')
    
    # Menampilkan histogram
    # Menggunakan ravel() untuk mengubah matriks 2D menjadi array 1D
    ax_hist.hist(image.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    ax_hist.set_title(f"Histogram: {title}")
    ax_hist.set_xlim([0, 256])

def manual_histogram_equalization(image):
    """
    Melakukan histogram equalization secara manual tanpa cv2.equalizeHist.
    """
    # 1. Hitung histogram (frekuensi kemunculan tiap intensitas 0-255)
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    
    # 2. Hitung probabilitas (PDF) dan CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()
    
    # 3. Normalisasi CDF 
    # Masking nilai 0 agar tidak ikut dihitung dalam normalisasi
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Rumus transformasi s_k = round((CDF(v) - CDF_min) / (M*N - CDF_min) * (L-1))
    # L = 256 (derajat keabuan)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # Kembalikan nilai yang di-mask menjadi 0 dan bulatkan ke integer
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 4. Petakan nilai intensitas citra asli ke nilai baru berdasarkan CDF
    image_equalized = cdf_final[image]
    
    return image_equalized

def match_histogram_library(source_image, target_image):
    """
    Melakukan histogram specification menggunakan library scikit-image.
    Menyamakan distribusi histogram source_image dengan target_image.
    """
    # match_histograms mengembalikan citra float, kita ubah kembali ke uint8
    matched = exposure.match_histograms(source_image, target_image)
    return matched.astype('uint8')

def main():
    # ==========================================
    # Persiapan Folder Output
    # ==========================================
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' berhasil dibuat.")

    # 1. Load citra grayscale bebas (Pastikan file gambar ada di direktori yang sama)
    img_source = cv2.imread('source.jpg', cv2.IMREAD_GRAYSCALE)
    img_target = cv2.imread('target.jpg', cv2.IMREAD_GRAYSCALE)

    if img_source is None or img_target is None:
        print("Error: Citra tidak ditemukan. Pastikan 'source.jpg' dan 'target.jpg' ada.")
        return

    # 2. Implementasi Histogram Equalization Manual
    img_eq_manual = manual_histogram_equalization(img_source)

    # 3. Implementasi Histogram Equalization Library (OpenCV)
    img_eq_cv2 = cv2.equalizeHist(img_source)

    # 4. Implementasi Histogram Specification
    img_matched = match_histogram_library(img_source, img_target)

    # ==========================================
    # Menyimpan Citra Hasil ke Folder Output
    # ==========================================
    cv2.imwrite(os.path.join(output_dir, '1_source_image.jpg'), img_source)
    cv2.imwrite(os.path.join(output_dir, '2_eq_manual.jpg'), img_eq_manual)
    cv2.imwrite(os.path.join(output_dir, '3_eq_opencv.jpg'), img_eq_cv2)
    cv2.imwrite(os.path.join(output_dir, '4_target_image.jpg'), img_target)
    cv2.imwrite(os.path.join(output_dir, '5_matched_specification.jpg'), img_matched)
    print(f"Semua citra hasil berhasil disimpan di dalam folder '{output_dir}'.")

    # ==========================================
    # 5. Visualisasi Seluruh Hasil dan Histogram
    # ==========================================
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    fig.tight_layout(pad=5.0)

    # Baris 1: Citra Awal (Source)
    plot_image_and_hist(img_source, "Citra Awal (Source)", axes[0, 0], axes[0, 1])

    # Baris 2: Equalization Manual
    plot_image_and_hist(img_eq_manual, "Equalization (Manual)", axes[1, 0], axes[1, 1])

    # Baris 3: Equalization Library
    plot_image_and_hist(img_eq_cv2, "Equalization (OpenCV)", axes[2, 0], axes[2, 1])

    # Baris 4: Citra Target (Untuk Specification)
    plot_image_and_hist(img_target, "Citra Target", axes[3, 0], axes[3, 1])

    # Baris 5: Hasil Histogram Specification
    plot_image_and_hist(img_matched, "Hasil Specification (Matched)", axes[4, 0], axes[4, 1])

    # Menyimpan juga plot gabungan ke folder output
    fig.savefig(os.path.join(output_dir, '6_plot_histogram_lengkap.png'), dpi=300)
    print(f"Plot histogram gabungan berhasil disimpan sebagai '6_plot_histogram_lengkap.png'.")

    plt.show()

if __name__ == "__main__":
    main()