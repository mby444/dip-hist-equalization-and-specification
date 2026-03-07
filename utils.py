import matplotlib.pyplot as plt
import cv2

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