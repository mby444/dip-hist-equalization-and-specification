import numpy as np

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