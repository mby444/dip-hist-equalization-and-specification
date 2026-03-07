from skimage import exposure

def match_histogram_library(source_image, target_image):
    """
    Melakukan histogram specification menggunakan library scikit-image.
    Menyamakan distribusi histogram source_image dengan target_image.
    """
    # match_histograms mengembalikan citra float, kita ubah kembali ke uint8
    matched = exposure.match_histograms(source_image, target_image)
    return matched.astype('uint8')