import cv2
import matplotlib.pyplot as plt
from utils import plot_image_and_hist
from equalization import manual_histogram_equalization
from specification import match_histogram_library

def main():
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

    plt.show()

if __name__ == "__main__":
    main()