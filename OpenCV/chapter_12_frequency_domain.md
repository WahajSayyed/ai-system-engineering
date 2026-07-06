# Chapter 12: Frequency Domain Processing

> *"Every image is a symphony of frequencies. Low frequencies are the melody — the smooth gradients of tone and color. High frequencies are the percussion — sharp edges, fine textures, noise. The Fourier Transform hands you the conductor's score."*

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Explain what the 2D DFT represents and why it matters for image processing
2. Compute the DFT with both NumPy (`np.fft.fft2`) and OpenCV (`cv2.dft`)
3. Visualize the magnitude spectrum correctly using log-scaling and frequency shift
4. Understand which image features correspond to which frequencies
5. Design and apply ideal, Gaussian, and Butterworth low-pass and high-pass filters
6. Explain the convolution theorem and apply it to understand spatial filters
7. Understand and avoid ringing artifacts caused by ideal rectangular filters
8. Apply notch filters to remove periodic noise patterns
9. Perform frequency-domain sharpening using high-boost filters
10. Choose between spatial-domain and frequency-domain filtering for a given task

---

## 12.1 The Frequency Perspective on Images

### 12.1.1 What the DFT Tells You

Every image can be exactly decomposed into a sum of 2D sinusoidal waves of different frequencies, directions, amplitudes, and phases. The **2D Discrete Fourier Transform (DFT)** performs this decomposition:

```
F(k, l) = Σᵢ₌₀ᴺ⁻¹ Σⱼ₌₀ᴹ⁻¹ f(i, j) · exp(-2πi(ki/N + lj/M))
```

The complex output `F(k, l)` represents a sinusoidal wave with:
- **Frequency:** how many cycles fit across the image width (k) and height (l)
- **Amplitude:** `|F(k,l)|` — how strongly that wave is present
- **Phase:** `angle(F(k,l))` — the spatial offset of that wave

The key insight for image processing:

| Frequency Region | Corresponds to |
|-----------------|----------------|
| DC (k=0, l=0, center of shifted spectrum) | Mean brightness, global illumination |
| Low frequencies (near center) | Smooth regions, gradual gradients, large structures |
| Mid frequencies | Medium-scale textures, object interiors |
| High frequencies (far from center) | Sharp edges, fine textures, noise |

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def build_frequency_intuition():
    """
    Create images that show what different frequency components look like.
    """
    N = 256
    x = np.arange(N)
    Y, X = np.mgrid[0:N, 0:N]

    images = {}

    # Pure low-frequency wave: 2 cycles across N pixels
    w_low = np.sin(2 * np.pi * 2 * x / N)
    images["Low freq (2 cycles)"] = ((np.outer(w_low, w_low) + 1) / 2 * 255).astype(np.uint8)

    # Pure high-frequency wave: 30 cycles
    w_high = np.sin(2 * np.pi * 30 * x / N)
    images["High freq (30 cycles)"] = ((np.outer(w_high, w_high) + 1) / 2 * 255).astype(np.uint8)

    # Diagonal sinusoid (direction in frequency domain)
    diag = np.sin(2 * np.pi * 15 * (X + Y) / N)
    images["Diagonal sinusoid"] = ((diag + 1) / 2 * 255).astype(np.uint8)

    # Natural image (sum of many frequencies)
    natural = sum(
        amp * np.sin(2 * np.pi * f * (X + Y) / N)
        for f, amp in [(2, 1.0), (5, 0.7), (10, 0.4), (20, 0.2), (40, 0.1)]
    )
    mn, mx = natural.min(), natural.max()
    images["Natural-like"] = ((natural - mn) / (mx - mn) * 255).astype(np.uint8)

    # Show each image and its spectrum
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, (title, img) in enumerate(images.items()):
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(title, fontsize=9); axes[0, col].axis("off")

        fshift = np.fft.fftshift(np.fft.fft2(img.astype(float)))
        mag = 20 * np.log(np.abs(fshift) + 1)
        axes[1, col].imshow(mag, cmap="hot")
        axes[1, col].set_title("Spectrum (log magnitude)", fontsize=9)
        axes[1, col].axis("off")

    plt.suptitle("Spatial Domain vs Frequency Domain: Building Intuition", fontsize=12)
    plt.tight_layout(); plt.show()


build_frequency_intuition()
```

---

## 12.2 Computing the 2D DFT

### 12.2.1 Using NumPy: np.fft.fft2

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_spectrum_numpy(img_gray):
    """
    Compute the 2D DFT of a grayscale image using NumPy.

    Returns the shifted spectrum (complex), log magnitude, and phase.
    """
    # Step 1: Forward 2D FFT
    f = np.fft.fft2(img_gray.astype(np.float64))
    # f has shape (H, W), dtype=complex128

    # Step 2: Shift DC component from top-left corner to image center
    # Without shift: DC at [0,0]; After shift: DC at [H/2, W/2]
    fshift = np.fft.fftshift(f)

    # Step 3: Log magnitude spectrum for display
    # Raw magnitude spans many orders of magnitude → log compresses range
    magnitude = 20 * np.log(np.abs(fshift) + 1)   # +1 avoids log(0)

    # Step 4: Phase spectrum
    phase = np.angle(fshift)   # range: [-π, +π]

    return fshift, magnitude, phase


# Load test image
img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 200, -1)
    cv2.rectangle(img, (40, 40), (100, 100), 150, -1)
    noise = np.random.normal(0, 8, img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

fshift, mag_spectrum, phase_spectrum = compute_spectrum_numpy(img)

print(f"Image shape: {img.shape}")
print(f"DFT shape:   {fshift.shape}")
print(f"DFT dtype:   {fshift.dtype}")

# DC component is at center after fftshift
cy, cx = img.shape[0] // 2, img.shape[1] // 2
print(f"DC component magnitude: {np.abs(fshift[cy, cx]):.1f}")
print(f"  = sum of all pixel values = {img.sum():.1f} ✓")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Spatial domain (original)"); axes[0].axis("off")

axes[1].imshow(mag_spectrum, cmap="hot")
axes[1].set_title("Magnitude spectrum\n(log scale, DC at center)"); axes[1].axis("off")

axes[2].imshow(phase_spectrum, cmap="bwr", vmin=-np.pi, vmax=np.pi)
axes[2].set_title("Phase spectrum\n[-π, +π]"); axes[2].axis("off")

plt.tight_layout(); plt.show()
```

### 12.2.2 Using OpenCV: cv2.dft

OpenCV stores the complex DFT in a `(H, W, 2)` float32 array where channel 0 = real part and channel 1 = imaginary part:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_spectrum_opencv(img_gray):
    """
    Compute the 2D DFT using cv2.dft.
    Returns shifted DFT (H,W,2) float32 and log magnitude.
    """
    img_f32 = img_gray.astype(np.float32)

    # cv2.dft requires float32; DFT_COMPLEX_OUTPUT returns both real and imag channels
    dft = cv2.dft(img_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # dft shape: (H, W, 2) — channel 0=real, channel 1=imaginary

    # Shift DC to center
    dft_shift = np.fft.fftshift(dft)

    # Magnitude using cv2.magnitude (sqrt of sum of squares)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    mag_log   = 20 * np.log(magnitude + 1)

    return dft_shift, mag_log


def idft_opencv(dft_shift):
    """
    Compute the inverse DFT using cv2.idft.
    Input: shifted DFT (H, W, 2) float32
    Output: spatial image (H, W) float32
    """
    # Un-shift: put DC back at top-left corner
    dft_ishift = np.fft.ifftshift(dft_shift)

    # cv2.idft with DFT_SCALE normalizes by 1/N, DFT_REAL_OUTPUT gives real result
    img_back = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return img_back


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 200, -1)

dft_shift, mag_log = compute_spectrum_opencv(img)

# Round-trip fidelity test
reconstructed = idft_opencv(dft_shift)
reconstructed_u8 = np.clip(reconstructed, 0, 255).astype(np.uint8)
diff = np.abs(img.astype(float) - reconstructed_u8.astype(float))
print(f"Round-trip max error: {diff.max():.4f}  (should be near 0)")

# Compare NumPy and OpenCV magnitudes
_, mag_numpy, _ = compute_spectrum_numpy(img)
print(f"NumPy vs OpenCV max diff: {np.abs(mag_log - mag_numpy).max():.4f}")
```

### 12.2.3 Optimal Padding for Speed

FFT algorithms run fastest when image dimensions are products of small primes (2, 3, 5):

```python
import cv2
import numpy as np
import time


def benchmark_dft_sizes(img_gray):
    """Show speedup from padding to optimal DFT size."""
    rows, cols = img_gray.shape

    # Find optimal dimensions
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)

    print(f"Original size: {rows} × {cols}")
    print(f"Optimal size:  {nrows} × {ncols}")
    print(f"Padding:       {nrows-rows} rows, {ncols-cols} cols")

    # Time at original size
    img_f32 = img_gray.astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(20):
        cv2.dft(img_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    t_orig = (time.perf_counter() - t0) / 20 * 1000

    # Pad and time at optimal size
    img_padded = cv2.copyMakeBorder(
        img_gray, 0, nrows - rows, 0, ncols - cols,
        cv2.BORDER_CONSTANT, value=0
    )
    img_pad_f32 = img_padded.astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(20):
        cv2.dft(img_pad_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    t_opt = (time.perf_counter() - t0) / 20 * 1000

    print(f"\nDFT timing at {rows}×{cols}:    {t_orig:.2f} ms")
    print(f"DFT timing at {nrows}×{ncols}: {t_opt:.2f} ms")
    print(f"Speedup: {t_orig / t_opt:.1f}×")
    print(f"\nIMPORTANT: After IDFT on padded image, crop back:")
    print(f"  result = idft_result[:rows, :cols]")


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (342, 548), dtype=np.uint8)

benchmark_dft_sizes(img)
```

---

## 12.3 Reading the Magnitude Spectrum

The spectrum encodes essential information about image structure. Learning to read it is a key diagnostic skill:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def spectrum_guide():
    """
    Create images to demonstrate the spectrum reading rules.
    """
    N = 256
    Y, X = np.mgrid[0:N, 0:N]

    examples = {}

    # 1. Constant → only DC spike at center
    examples["Constant\n(only DC)"] = np.full((N, N), 128, dtype=np.uint8)

    # 2. Horizontal stripes → vertical line in spectrum
    stripes = ((np.sin(2 * np.pi * 8 * np.arange(N) / N) > 0) * 255).astype(np.uint8)
    examples["Horizontal stripes\n(→ vertical spec line)"] = np.tile(stripes.reshape(-1, 1), (1, N))

    # 3. Diagonal stripes → diagonal line in spectrum
    diag = ((np.sin(2 * np.pi * 8 * (X + Y) / N) > 0) * 255).astype(np.uint8)
    examples["Diagonal stripes\n(→ diagonal spec line)"] = diag

    # 4. Random noise → uniform spectrum
    rng = np.random.default_rng(42)
    examples["White noise\n(uniform spectrum)"] = rng.integers(0, 256, (N, N), dtype=np.uint8)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col, (title, img) in enumerate(examples.items()):
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(title, fontsize=9); axes[0, col].axis("off")

        fshift = np.fft.fftshift(np.fft.fft2(img.astype(float)))
        mag = 20 * np.log(np.abs(fshift) + 1)
        axes[1, col].imshow(mag, cmap="hot")
        cy, cx = N // 2, N // 2
        axes[1, col].axhline(cy, color="cyan", alpha=0.4, linewidth=0.8)
        axes[1, col].axvline(cx, color="cyan", alpha=0.4, linewidth=0.8)
        axes[1, col].set_title("Spectrum", fontsize=9); axes[1, col].axis("off")

    plt.suptitle("Spectrum Reading Guide", fontsize=12)
    plt.tight_layout(); plt.show()

    print("Spectrum Reading Rules:")
    print("  Bright center (DC)          → dominant mean brightness")
    print("  Energy along H-axis         → vertical edges/features")
    print("  Energy along V-axis         → horizontal edges/features")
    print("  Bright isolated off-center  → periodic patterns (fabric, screens)")
    print("  Energy spreading outward    → fine texture, sharp edges")
    print("  Uniform energy spread       → noise or white-noise-like image")


spectrum_guide()
```

---

## 12.4 The Convolution Theorem

The most important DFT property for image processing:

**Spatial convolution ↔ Frequency multiplication**

```
f * h  ⟺  F · H
```

This means a spatial filter with kernel `h` is equivalent to multiplying the DFT of `f` by `H = DFT(h)`. For large kernels, this can be faster than direct convolution.

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def demonstrate_convolution_theorem(img_gray):
    """Verify: spatial conv == frequency multiplication."""
    h, w = img_gray.shape
    sigma = 8.0
    ksize = int(6 * sigma) | 1

    # Method 1: Spatial Gaussian blur
    result_spatial = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)

    # Method 2: Frequency-domain multiplication
    # Build the Gaussian kernel and pad to image size
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel_2d = kernel_1d @ kernel_1d.T  # outer product → 2D kernel

    # Pad kernel to image size (center at (0,0) for circular convolution)
    kernel_padded = np.zeros((h, w), dtype=np.float64)
    kernel_padded[:ksize, :ksize] = kernel_2d

    # Compute DFTs
    F_image  = np.fft.fft2(img_gray.astype(np.float64))
    F_kernel = np.fft.fft2(kernel_padded)

    # Multiply in frequency domain
    G = F_image * F_kernel

    # Inverse DFT + fix circular shift
    result_freq = np.real(np.fft.ifft2(G))
    result_freq = np.roll(np.roll(result_freq, -ksize//2, axis=0), -ksize//2, axis=1)
    result_freq = np.clip(result_freq, 0, 255).astype(np.uint8)

    # Compare (border regions will differ due to edge handling differences)
    interior = img_gray[ksize:-ksize, ksize:-ksize]
    diff_interior = np.abs(
        result_spatial[ksize:-ksize, ksize:-ksize].astype(int) -
        result_freq[ksize:-ksize, ksize:-ksize].astype(int)
    )
    print(f"Convolution theorem: max diff (interior) = {diff_interior.max()}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(img_gray, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(result_spatial, cmap="gray"); axes[1].set_title("Spatial GaussianBlur"); axes[1].axis("off")
    axes[2].imshow(result_freq, cmap="gray"); axes[2].set_title("Frequency multiplication"); axes[2].axis("off")
    plt.suptitle("Convolution Theorem Verification", fontsize=12)
    plt.tight_layout(); plt.show()


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 200, -1)

demonstrate_convolution_theorem(img)
```

---

## 12.5 Frequency Domain Filtering Framework

All frequency domain filters share this workflow:

```python
import numpy as np
import cv2


def apply_frequency_filter(img_gray: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a frequency domain filter H to a grayscale image.

    Parameters
    ----------
    img_gray : np.ndarray (H_img, W_img) uint8
    H : np.ndarray (H_img, W_img) float  — filter (DC at center after fftshift)

    Returns
    -------
    filtered : np.ndarray (H_img, W_img) uint8

    Notes
    -----
    H must have DC at the center (as produced by fftshift convention).
    For a low-pass filter: H=1 at center, decaying outward.
    For a high-pass filter: H=0 at center, increasing outward.
    """
    # Forward DFT → shift DC to center
    f = np.fft.fft2(img_gray.astype(np.float64))
    fshift = np.fft.fftshift(f)

    # Apply filter (element-wise multiplication in frequency domain)
    filtered_shift = fshift * H

    # Un-shift → Inverse DFT → take real part
    filtered_back = np.fft.ifftshift(filtered_shift)
    result = np.real(np.fft.ifft2(filtered_back))

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Filter design functions ────────────────────────────────────────────────────

def make_distance_map(H_img: int, W_img: int) -> np.ndarray:
    """Build a distance-from-center map for filter design."""
    Y, X = np.mgrid[-H_img//2 : H_img//2, -W_img//2 : W_img//2]
    return np.sqrt(X**2 + Y**2)


def ideal_lpf(H_img: int, W_img: int, cutoff: float) -> np.ndarray:
    """
    Ideal (sharp) low-pass filter.
    H=1 inside cutoff, 0 outside.
    WARNING: Produces ringing (Gibbs phenomenon) due to sinc kernel in spatial domain.
    Avoid in practice — use Gaussian or Butterworth instead.
    """
    return (make_distance_map(H_img, W_img) <= cutoff).astype(np.float64)


def gaussian_lpf(H_img: int, W_img: int, sigma: float) -> np.ndarray:
    """
    Gaussian low-pass filter.
    H(d) = exp(-d² / 2σ²)
    Smooth rolloff: no ringing artifacts. Recommended for most uses.
    sigma controls the cutoff bandwidth (larger sigma = more frequencies pass).
    """
    d = make_distance_map(H_img, W_img)
    return np.exp(-d**2 / (2 * sigma**2))


def butterworth_lpf(H_img: int, W_img: int, cutoff: float, order: int = 2) -> np.ndarray:
    """
    Butterworth low-pass filter.
    H(d) = 1 / (1 + (d/D₀)^(2n))
    Steeper rolloff than Gaussian; higher order = sharper = more ringing.
    order=1: very smooth; order=2: good balance; order≥5: near-ideal with some ringing.
    """
    d = make_distance_map(H_img, W_img)
    return 1.0 / (1.0 + (d / (cutoff + 1e-10)) ** (2 * order))


def gaussian_hpf(H_img: int, W_img: int, sigma: float) -> np.ndarray:
    """Gaussian high-pass filter = 1 - Gaussian LPF."""
    return 1.0 - gaussian_lpf(H_img, W_img, sigma)


def butterworth_hpf(H_img: int, W_img: int, cutoff: float, order: int = 2) -> np.ndarray:
    """Butterworth high-pass filter = 1 - Butterworth LPF."""
    return 1.0 - butterworth_lpf(H_img, W_img, cutoff, order)


def band_pass_filter(H_img: int, W_img: int, low_cut: float, high_cut: float,
                     filter_type: str = "gaussian") -> np.ndarray:
    """
    Band-pass filter: pass frequencies between low_cut and high_cut.
    Computed as: LPF(high_cut) - LPF(low_cut)
    """
    if filter_type == "gaussian":
        return gaussian_lpf(H_img, W_img, high_cut) - gaussian_lpf(H_img, W_img, low_cut)
    return butterworth_lpf(H_img, W_img, high_cut) - butterworth_lpf(H_img, W_img, low_cut)


def high_boost_filter(H_img: int, W_img: int, sigma: float, boost: float = 1.5) -> np.ndarray:
    """
    High-boost (sharpening) filter.
    H = 1 + boost * HPF = 1 + boost * (1 - LPF)
    boost=0: identity; boost=1: unsharp mask; boost>1: stronger sharpening.
    """
    hpf = gaussian_hpf(H_img, W_img, sigma)
    return 1.0 + boost * hpf
```

### 12.5.1 Low-Pass Filter Comparison: Ringing vs Smooth

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compare_lpf_filters(img_gray, cutoff=30):
    """Compare ideal, Gaussian, and Butterworth LPF on the same image."""
    h, w = img_gray.shape

    filters = {
        "Ideal LPF\n(AVOID: ringing!)": ideal_lpf(h, w, cutoff),
        f"Gaussian LPF\n(σ={cutoff}, recommended)": gaussian_lpf(h, w, cutoff),
        f"Butterworth n=2\n(D₀={cutoff})": butterworth_lpf(h, w, cutoff, order=2),
        f"Butterworth n=8\n(D₀={cutoff}, near-ideal)": butterworth_lpf(h, w, cutoff, order=8),
    }

    results = {name: apply_frequency_filter(img_gray, H) for name, H in filters.items()}

    # Show filter shapes and results
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    for col, (name, H) in enumerate(filters.items()):
        # Row 0: filter in frequency domain
        axes[0, col].imshow(H, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(name, fontsize=9); axes[0, col].axis("off")

        # Row 1: filtered image
        axes[1, col].imshow(results[name], cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title("Filtered"); axes[1, col].axis("off")

        # Row 2: 1D profile to show ringing
        row = results[name][h // 2, :]
        row_orig = img_gray[h // 2, :]
        axes[2, col].plot(row_orig, 'k--', linewidth=1, alpha=0.5, label="Original")
        axes[2, col].plot(row, linewidth=1.5, label="Filtered")
        axes[2, col].set_title("Horizontal profile"); axes[2, col].legend(fontsize=7)
        axes[2, col].grid(alpha=0.3)

    plt.suptitle(f"LPF Comparison (cutoff={cutoff}): Notice Ideal Filter Ringing!",
                  fontsize=12)
    plt.tight_layout(); plt.show()

    # Quantify ringing: variance near a sharp edge
    for name, result in results.items():
        # Sample a strip near an edge region
        strip = result[h//2 - 5 : h//2 + 5, :20]
        ringing_measure = strip.astype(float).std()
        print(f"  {name.split(chr(10))[0]:<30}: edge-strip std={ringing_measure:.2f}")


# Test on sharp-edged image (ringing shows most on hard edges)
test_img = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(test_img, (60, 60), (196, 196), 220, -1)

compare_lpf_filters(test_img, cutoff=30)

print("\n⚠️  KEY LESSON: Ideal (rectangular) filters cause Gibbs/ringing artifacts.")
print("   Always use Gaussian or Butterworth filters in practice.")
```

### 12.5.2 High-Pass Filters

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def compare_hpf_filters(img_gray):
    """Compare HPF types and show connection to edge detection."""
    h, w = img_gray.shape

    hpf_gauss_tight  = gaussian_hpf(h, w, sigma=15)
    hpf_gauss_broad  = gaussian_hpf(h, w, sigma=40)
    hpf_butter_n2    = butterworth_hpf(h, w, cutoff=20, order=2)

    result_gt = apply_frequency_filter(img_gray, hpf_gauss_tight)
    result_gb = apply_frequency_filter(img_gray, hpf_gauss_broad)
    result_bw = apply_frequency_filter(img_gray, hpf_butter_n2)

    # Spatial Canny for comparison
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.0)
    edges_canny = cv2.Canny(blurred, 50, 150)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, im, title in zip(axes,
        [img_gray, result_gt, result_gb, result_bw, edges_canny],
        ["Original",
         "Gaussian HPF (σ=15)\ntight cutoff",
         "Gaussian HPF (σ=40)\nbroad cutoff",
         "Butterworth HPF\n(D₀=20, n=2)",
         "Canny edge\n(spatial, for comparison)"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9); ax.axis("off")
    plt.suptitle("HPF = Frequency-Domain Edge Detection", fontsize=12)
    plt.tight_layout(); plt.show()

    print("Key insight: HPF in frequency domain = edge detection in spatial domain.")
    print("  → Confirms that edges ARE the high-frequency content of images.")


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 200, -1)
    cv2.rectangle(img, (40, 40), (100, 100), 150, -1)

compare_hpf_filters(img)
```

---

## 12.6 Notch Filters: Removing Periodic Noise

Periodic noise (from electrical interference, scanner lines, or screen photography) appears as isolated bright spots in the frequency spectrum. Notch filters surgically block those frequencies:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_notch_filter(H_img: int, W_img: int,
                       notch_centers: list,
                       notch_radius: float,
                       filter_type: str = "gaussian") -> np.ndarray:
    """
    Create a notch filter that blocks specific frequency locations.

    Parameters
    ----------
    H_img, W_img : int   Image dimensions
    notch_centers : list of (u, v) tuples
        Frequency offsets from DC center.
        Conjugate pairs (u,v) and (-u,-v) are handled automatically.
    notch_radius : float   Radius of each notch (frequency units)
    filter_type : str     'gaussian' (smooth, recommended) or 'ideal' (sharp, avoid)

    Returns
    -------
    H : np.ndarray (H_img, W_img) float  — notch filter (1=pass, 0=block)
    """
    H = np.ones((H_img, W_img), dtype=np.float64)
    cy, cx = H_img // 2, W_img // 2
    Y, X   = np.mgrid[0:H_img, 0:W_img]

    for (u, v) in notch_centers:
        # Block both (u,v) and conjugate (-u,-v) for Hermitian symmetry
        for du, dv in [(u, v), (-u, -v)]:
            d = np.sqrt((X - (cx + du))**2 + (Y - (cy + dv))**2)
            if filter_type == "gaussian":
                H *= 1.0 - np.exp(-d**2 / (2 * notch_radius**2))
            else:
                H *= (d > notch_radius).astype(float)

    return H


def create_moire_image(img_clean, freq1=(20, 5), freq2=(-5, 20),
                        amplitude=35):
    """Add two-frequency moiré pattern to simulate screen photography."""
    h, w = img_clean.shape
    Y, X = np.mgrid[0:h, 0:w]
    u1, v1 = freq1
    u2, v2 = freq2
    moire = (amplitude * np.sin(2 * np.pi * (u1*X/w + v1*Y/h)) +
             amplitude * 0.7 * np.sin(2 * np.pi * (u2*X/w + v2*Y/h)))
    return np.clip(img_clean.astype(float) + moire, 0, 255).astype(np.uint8)


# ── Demo ──────────────────────────────────────────────────────────────────────
img_clean = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img_clean is None:
    img_clean = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img_clean, (128, 128), 80, 200, -1)
    cv2.rectangle(img_clean, (30, 160), (100, 230), 150, -1)

# Resize to 256×256 for clean demo
img_clean = cv2.resize(img_clean, (256, 256))

# Add moiré
moire_freq1 = (20, 5)
moire_freq2 = (-5, 20)
img_moire = create_moire_image(img_clean, moire_freq1, moire_freq2)

h, w = img_moire.shape

# Build notch filter for known frequencies
notch_H = make_notch_filter(
    h, w,
    notch_centers=[moire_freq1, moire_freq2],
    notch_radius=8,
    filter_type="gaussian"
)

# Apply notch filter
img_restored = apply_frequency_filter(img_moire, notch_H)

# Compute spectra for visualization
def get_log_spectrum(img):
    fshift = np.fft.fftshift(np.fft.fft2(img.astype(float)))
    return 20 * np.log(np.abs(fshift) + 1)

spec_clean   = get_log_spectrum(img_clean)
spec_moire   = get_log_spectrum(img_moire)
spec_restored= get_log_spectrum(img_restored)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# Row 0: images
axes[0,0].imshow(img_clean,    cmap="gray"); axes[0,0].set_title("Clean original"); axes[0,0].axis("off")
axes[0,1].imshow(img_moire,    cmap="gray"); axes[0,1].set_title("With moiré noise"); axes[0,1].axis("off")
axes[0,2].imshow(notch_H,      cmap="gray"); axes[0,2].set_title("Notch filter\n(dark = blocked)"); axes[0,2].axis("off")
axes[0,3].imshow(img_restored, cmap="gray"); axes[0,3].set_title("Restored"); axes[0,3].axis("off")
# Row 1: spectra
axes[1,0].imshow(spec_clean,    cmap="hot"); axes[1,0].set_title("Clean spectrum"); axes[1,0].axis("off")
axes[1,1].imshow(spec_moire,    cmap="hot"); axes[1,1].set_title("Moiré spectrum\n(bright = noise spikes)"); axes[1,1].axis("off")
axes[1,2].axis("off")
axes[1,3].imshow(spec_restored, cmap="hot"); axes[1,3].set_title("Restored spectrum\n(spikes removed)"); axes[1,3].axis("off")

plt.suptitle("Notch Filtering: Removing Periodic Moiré Noise", fontsize=12)
plt.tight_layout(); plt.show()

# PSNR comparison
def psnr(ref, test):
    mse = np.mean((ref.astype(float) - test.astype(float))**2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)

print(f"PSNR before denoising: {psnr(img_clean, img_moire):.1f} dB")
print(f"PSNR after  denoising: {psnr(img_clean, img_restored):.1f} dB")
```

---

## 12.7 Frequency Domain Sharpening

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def sharpen_frequency_domain(img_gray, sigma=40, boost=2.0):
    """
    Sharpen using high-boost filter in frequency domain.
    H = 1 + boost * (1 - Gaussian_LPF)
    Equivalent to: output = img + boost * (img - blurred_img)
    """
    h, w = img_gray.shape
    H = high_boost_filter(h, w, sigma=sigma, boost=boost)
    return apply_frequency_filter(img_gray, H)


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 200, -1)
    cv2.rectangle(img, (40, 40), (100, 100), 150, -1)
    img = cv2.GaussianBlur(img, (7, 7), 2)   # pre-blur to make sharpening visible

h, w = img.shape

# Spatial unsharp mask for comparison
blurred_sp  = cv2.GaussianBlur(img, (21, 21), 5.0)
unsharp_sp  = np.clip(img.astype(float) * 2.5 - blurred_sp.astype(float) * 1.5, 0, 255).astype(np.uint8)

# Frequency domain
sharp_gentle = sharpen_frequency_domain(img, sigma=40, boost=1.0)
sharp_strong = sharpen_frequency_domain(img, sigma=40, boost=3.0)

# Show filter profiles
H_gentle = high_boost_filter(h, w, sigma=40, boost=1.0)
H_strong = high_boost_filter(h, w, sigma=40, boost=3.0)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
# Row 0: filter profiles
for col, (H, title) in enumerate([
    (gaussian_lpf(h, w, 40),  "Gaussian LPF σ=40\n(reference)"),
    (H_gentle,                 "High-boost boost=1\n(H=1 + 1·HPF)"),
    (H_strong,                 "High-boost boost=3\n(H=1 + 3·HPF)"),
    (np.ones((h, w)),          "Identity\n(H=1 everywhere)"),
]):
    center_row = H[h // 2, :]
    axes[0, col].plot(center_row, color="steelblue", linewidth=2)
    axes[0, col].set_ylim([0, max(4.0, H.max() * 1.1)])
    axes[0, col].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0, col].set_title(title, fontsize=9); axes[0, col].grid(alpha=0.3)

# Row 1: results
for col, (im, title) in enumerate([
    (img,          "Original (slightly blurred)"),
    (unsharp_sp,   "Spatial unsharp mask"),
    (sharp_gentle, "Freq high-boost (A=1)"),
    (sharp_strong, "Freq high-boost (A=3)"),
]):
    axes[1, col].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[1, col].set_title(title, fontsize=9); axes[1, col].axis("off")

plt.suptitle("High-Boost Frequency Sharpening", fontsize=12)
plt.tight_layout(); plt.show()
```

---

## 12.8 Visualizing Spatial Filters in the Frequency Domain

One of the most illuminating uses of the DFT: visualize the frequency response of any spatial kernel:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt


def kernel_frequency_response(kernel: np.ndarray,
                               img_size: tuple = (256, 256)) -> np.ndarray:
    """
    Compute the frequency response magnitude of a spatial filter kernel.
    Pads kernel to img_size and computes DFT.
    Returns normalized magnitude in [0, 1].
    """
    h, w = img_size
    kh, kw = kernel.shape

    # Pad kernel to img_size (zero-pad, kernel at top-left for correct phase)
    k_padded = np.zeros((h, w), dtype=np.float64)
    k_padded[:kh, :kw] = kernel

    # DFT → shift → magnitude
    K_shift = np.fft.fftshift(np.fft.fft2(k_padded))
    K_mag   = np.abs(K_shift)

    # Normalize to [0, 1]
    return K_mag / (K_mag.max() + 1e-12)


# Compare frequency responses of classic spatial filters
kernels = {
    "3×3 Box (avg)":
        np.ones((3, 3)) / 9.0,
    "7×7 Box (avg)":
        np.ones((7, 7)) / 49.0,
    "Gaussian σ=2":
        (lambda k, s: (k @ k.T))(cv2.getGaussianKernel(13, 2), 2),
    "Laplacian\n(edge detect)":
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64),
    "Sharpening\n(5-center)":
        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float64),
    "Sobel X\n(vertical edges)":
        np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64),
}

fig, axes = plt.subplots(2, len(kernels), figsize=(18, 6))
for col, (name, kernel) in enumerate(kernels.items()):
    H_mag = kernel_frequency_response(kernel)

    # 2D frequency response
    axes[0, col].imshow(H_mag, cmap="hot", vmin=0, vmax=1)
    axes[0, col].set_title(name, fontsize=8); axes[0, col].axis("off")

    # 1D radial cross-section
    center_row = H_mag[128, :]
    axes[1, col].plot(center_row, color="steelblue", linewidth=1.5)
    axes[1, col].set_ylim([0, 1.1])
    axes[1, col].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="-3 dB")
    axes[1, col].set_title("Radial profile", fontsize=8)
    axes[1, col].grid(alpha=0.3)

    # Classify filter type
    dc_val     = H_mag[128, 128]   # center (DC)
    corner_val = H_mag[0, 0]       # corner (highest freq)
    ftype = ("Low-pass" if dc_val > 0.7 and corner_val < 0.3 else
             "High-pass" if dc_val < 0.3 and corner_val > 0.3 else
             "Band-pass / other")
    print(f"{name.split(chr(10))[0]:<20}: DC={dc_val:.3f}, HF={corner_val:.3f} → {ftype}")

plt.suptitle("Frequency Response of Spatial Filters", fontsize=12)
plt.tight_layout(); plt.show()
```

---

## 12.9 When to Use Frequency Domain vs Spatial Domain

```python
import numpy as np
import cv2
import time


def performance_crossover(img_gray):
    """
    Find where spatial and frequency domain filtering cross over in speed.
    For small kernels: spatial wins. For large kernels: may tie or lose to freq.
    """
    h, w = img_gray.shape
    results = []

    for sigma in [0.5, 1, 2, 3, 5, 8, 12, 20, 30]:
        ksize = int(6 * sigma) | 1

        # Spatial: cv2.GaussianBlur (highly optimized — SIMD, multi-thread)
        t0 = time.perf_counter()
        for _ in range(30):
            cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)
        t_spatial = (time.perf_counter() - t0) / 30 * 1000

        # Frequency domain: full DFT pipeline
        H = gaussian_lpf(h, w, sigma * (h / (2 * np.pi)))
        t0 = time.perf_counter()
        for _ in range(30):
            apply_frequency_filter(img_gray, H)
        t_freq = (time.perf_counter() - t0) / 30 * 1000

        winner = "Spatial ✓" if t_spatial < t_freq else "Frequency ✓"
        results.append((sigma, ksize, t_spatial, t_freq, winner))
        print(f"  σ={sigma:5.1f} k={ksize:3d}: Spatial={t_spatial:6.2f}ms  "
              f"Freq={t_freq:6.2f}ms  → {winner}")

    return results


img = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

print(f"Performance on {img.shape[1]}×{img.shape[0]} image:")
print("─" * 65)
results = performance_crossover(img)

print("\n── Decision guide ──")
print("  Use SPATIAL domain (cv2 functions) when:")
print("    • Kernel size ≤ 50 px (almost always faster)")
print("    • Real-time processing required")
print("    • Non-linear operation (median, bilateral, morphological)")
print("    • Direct OpenCV function available (GaussianBlur, bilateralFilter)")
print("  Use FREQUENCY domain when:")
print("    • Removing periodic noise (notch filters — no spatial equivalent)")
print("    • Designing filters from known frequency specs")
print("    • Analyzing what a spatial filter does frequency-wise")
print("    • Very large kernels on large images (DFT = O(N² log N), conv = O(N² k²))")
```

---

## 12.10 Complete Project: Periodic Noise Removal System

A complete, robust pipeline that automatically detects and removes periodic noise:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class NoiseFrequency:
    """A detected periodic noise frequency."""
    u: int       # horizontal offset from DC center
    v: int       # vertical offset from DC center
    magnitude: float   # strength (log-magnitude at this frequency)


def find_noise_frequencies(img_gray: np.ndarray,
                            percentile: float = 99.5,
                            min_offset: int = 10) -> List[NoiseFrequency]:
    """
    Automatically detect periodic noise frequencies in the spectrum.

    Strategy:
    1. Compute log-magnitude spectrum
    2. Estimate expected "smooth" background with Gaussian blur
    3. Residual = spectrum - background (shows anomalous peaks)
    4. Threshold the residual at given percentile
    5. Return detected peaks (excluding DC neighborhood)

    Parameters
    ----------
    percentile : float  Threshold at this percentile of residual
    min_offset : int    Minimum distance from DC to be considered noise

    Returns
    -------
    list of NoiseFrequency, sorted by magnitude descending
    """
    h, w = img_gray.shape

    # Compute spectrum
    fshift = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float64)))
    mag = np.log(np.abs(fshift) + 1).astype(np.float32)

    # Estimate background via Gaussian smoothing
    bg = cv2.GaussianBlur(mag, (51, 51), 15)

    # Residual: where signal exceeds background
    residual = (mag - bg).astype(np.float32)

    # Threshold
    thresh = np.percentile(residual, percentile)
    peak_mask = residual > thresh

    # Exclude DC neighborhood
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[0:h, 0:w]
    dc_region = np.sqrt((X - cx)**2 + (Y - cy)**2) < min_offset
    peak_mask[dc_region] = False

    # Extract peak locations
    peak_ys, peak_xs = np.where(peak_mask)
    noises = []
    seen = set()

    for py, px in zip(peak_ys.tolist(), peak_xs.tolist()):
        u, v = int(px - cx), int(py - cy)
        # Deduplicate conjugate pairs
        key = tuple(sorted([(u, v), (-u, -v)]))
        if key not in seen:
            seen.add(key)
            noises.append(NoiseFrequency(u=u, v=v,
                                          magnitude=float(residual[py, px])))

    noises.sort(key=lambda x: x.magnitude, reverse=True)
    return noises


def auto_denoise_periodic(img_gray: np.ndarray,
                           max_notches: int = 10,
                           notch_radius: float = 10.0,
                           noise_percentile: float = 99.5) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Automatically detect and remove periodic noise using adaptive notch filtering.

    Parameters
    ----------
    max_notches : int     Maximum number of noise frequencies to remove
    notch_radius : float  Notch radius (frequency units)
    noise_percentile : float  Threshold for noise detection

    Returns
    -------
    restored : np.ndarray (H, W) uint8
    notch_H  : np.ndarray (H, W) float — filter applied
    noises   : list of NoiseFrequency
    """
    h, w = img_gray.shape

    # Detect noise frequencies
    noises = find_noise_frequencies(img_gray, noise_percentile)

    print(f"Detected {len(noises)} potential noise frequencies:")
    for i, nf in enumerate(noises[:max_notches]):
        print(f"  [{i}] offset=({nf.u:+d},{nf.v:+d}), magnitude={nf.magnitude:.3f}")

    if not noises:
        return img_gray, np.ones((h, w)), []

    # Build notch filter for the top-N detections
    top_noises = noises[:max_notches]
    centers = [(nf.u, nf.v) for nf in top_noises]
    notch_H = make_notch_filter(h, w, centers, notch_radius, "gaussian")

    # Apply
    restored = apply_frequency_filter(img_gray, notch_H)

    return restored, notch_H, top_noises


# ── Full pipeline demo ────────────────────────────────────────────────────────
img_clean = cv2.imread("sample.jpg", cv2.IMREAD_GRAYSCALE)
if img_clean is None:
    img_clean = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img_clean, (128, 128), 80, 200, -1)
    cv2.rectangle(img_clean, (30, 160), (100, 230), 150, -1)
img_clean = cv2.resize(img_clean, (256, 256))

# Simulate three noise sources
img_noisy = create_moire_image(img_clean, (20, 5), (-5, 20), amplitude=30)
# Add a third noise frequency
h, w = img_noisy.shape
Y, X = np.mgrid[0:h, 0:w]
img_noisy = np.clip(
    img_noisy.astype(float) +
    20 * np.sin(2 * np.pi * 35 * Y / h),   # strong horizontal stripes
    0, 255
).astype(np.uint8)

# Run auto-denoising
print("─" * 50)
print("Running auto-denoise pipeline:")
restored, notch_H, detected = auto_denoise_periodic(img_noisy,
                                                      max_notches=6,
                                                      notch_radius=8)

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0,0].imshow(img_clean,    cmap="gray"); axes[0,0].set_title("Clean"); axes[0,0].axis("off")
axes[0,1].imshow(img_noisy,    cmap="gray"); axes[0,1].set_title("Noisy (3 frequencies)"); axes[0,1].axis("off")
axes[0,2].imshow(restored,     cmap="gray"); axes[0,2].set_title(f"Restored ({len(detected)} notches applied)"); axes[0,2].axis("off")

spec_n = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_noisy.astype(float)))) + 1)
spec_r = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(restored.astype(float)))) + 1)
axes[1,0].imshow(spec_n, cmap="hot"); axes[1,0].set_title("Noisy spectrum"); axes[1,0].axis("off")
axes[1,1].imshow(notch_H * 255, cmap="gray"); axes[1,1].set_title("Notch filter"); axes[1,1].axis("off")
axes[1,2].imshow(spec_r, cmap="hot"); axes[1,2].set_title("Restored spectrum"); axes[1,2].axis("off")

plt.suptitle("Auto Periodic Noise Removal", fontsize=12)
plt.tight_layout(); plt.show()

def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float))**2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)

print(f"\nPSNR before: {psnr(img_clean, img_noisy):.1f} dB")
print(f"PSNR after:  {psnr(img_clean, restored):.1f} dB")
```

---

## 12.11 Summary

**The Fourier Transform** decomposes an image into 2D sinusoidal components. Low frequencies (near DC, spectrum center) represent smooth gradients and large-scale structure. High frequencies (spectrum periphery) represent sharp edges, fine textures, and noise.

**Computing the DFT:**
- NumPy: `np.fft.fft2` → `np.fft.fftshift` → visualize `20*log(|F|+1)`
- OpenCV: `cv2.dft(img_f32, DFT_COMPLEX_OUTPUT)` → `fftshift` → `cv2.magnitude`
- Always pad to `cv2.getOptimalDFTSize` for 3–4× speedup

**Filter design (DC at center convention):**
- **Ideal LPF**: rectangular, sharp — produces ringing. **Avoid.**
- **Gaussian LPF**: `H = exp(-d²/2σ²)` — smooth rolloff, no artifacts. Recommended.
- **Butterworth**: `H = 1/(1+(d/D₀)^2n)` — steeper rolloff; mild ringing at high orders
- **HPF** = `1 - LPF` — frequency-domain edge detection
- **Notch**: block isolated frequency spots (conjugate pairs) — removes periodic noise
- **High-boost**: `H = 1 + A·HPF` — sharpening

**Convolution theorem**: spatial convolution ↔ frequency multiplication. Use to understand and design spatial filters.

**Ringing**: caused by ideal (rectangular) filters. Always use smooth filters (Gaussian/Butterworth).

**When to choose frequency domain**: large kernels, periodic noise removal, filter design from frequency specs, filter analysis.

---

## 12.12 Exercises

### Warm-up

**12.1** Compute the DFT of a 256×256 grayscale image using both `np.fft.fft2` and `cv2.dft`. Display magnitude spectra from both (log-scaled). Show the phase spectrum alongside. Verify that the DC component equals the sum of all pixel values.

**12.2** Create four synthetic images: (a) constant, (b) horizontal stripes at 8 cycles, (c) diagonal stripes, (d) random noise. Compute and display the DFT spectrum of each. For each, annotate where you expect energy in the spectrum and verify.

**12.3** Demonstrate optimal DFT padding: for a 342×548 image, compare DFT computation time at the original size vs `cv2.getOptimalDFTSize` padded size. Run 50 iterations each and report the speedup. Then verify that the padded result, after cropping back to 342×548, matches the unpadded result.

### Core

**12.4** Apply all four LPF types (ideal, Gaussian, Butterworth n=2, Butterworth n=8) to a binary image with sharp edges (white rectangle on black background). For a horizontal profile through an edge, plot all four filtered profiles on the same axes. Quantify ringing as the standard deviation of the signal in the uniform background regions 10–30 pixels from the edge.

**12.5** Build `kernel_frequency_analysis(kernel)` that: (a) computes the 2D frequency response, (b) plots the response as a 2D heatmap and 1D radial profile, (c) automatically classifies the filter as low-pass / high-pass / band-pass / all-pass by measuring H at DC, mid, and high frequencies, (d) estimates the -3 dB cutoff radius. Apply to: box blur, Gaussian, Laplacian, sharpening kernel, and one Sobel direction.

**12.6** Implement `add_periodic_noise(img, frequencies)` that adds multiple sinusoidal noise patterns. Then implement `remove_periodic_noise(noisy_img, frequencies)` using notch filters. Test with 1, 3, and 5 noise frequencies. Report PSNR improvement for each case.

**12.7** Implement frequency-domain Wiener deconvolution: `H_wiener(u,v) = H*(u,v) / (|H|² + λ)` where H is the DFT of a known blur kernel and λ is regularization. Demonstrate on a box-blurred image. Plot PSNR vs λ for λ ∈ [1e-4, 1e-1] and find the optimal λ.

### Challenge

**12.8** Implement **phase correlation image registration**: given two images differing by a known translation, compute the cross-power spectrum `(F₁ · F₂*) / |F₁ · F₂*|`, take the IDFT, and find the peak location (= translation vector). Test on translations of 10, 25, and 50 pixels in both x and y. Report estimated vs true translation error.

**12.9** Build a **texture synthesis tool** via random phase: (a) compute DFT of a reference texture, (b) generate new textures by keeping `|F_ref|` but replacing the phase with uniform random phases, (c) compute IDFT. Generate 8 realizations and show they have the same frequency content but different spatial arrangement. Use `cv2.compareHist` on the magnitude spectra to verify similarity.

**12.10** Implement a **frequency-domain watermark**: embed a mark by modifying a few mid-frequency DFT coefficients by a fixed small amount ε. Implement `embed(img, mark_id)` and `detect(img, mark_id) → confidence_score`. Test robustness: does the mark survive (a) Gaussian blur σ=2, (b) 5% JPEG-style quantization, (c) rotation by 2°, (d) cropping 10% of the image? Report confidence scores after each attack.

---

## Further Reading

- **OpenCV DFT tutorial:** https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html
- **Gonzalez & Woods, "Digital Image Processing" (4th ed., 2018)** — Chapter 4: Filtering in the Frequency Domain. The definitive textbook treatment.
- **NumPy FFT documentation:** https://numpy.org/doc/stable/reference/routines.fft.html
- **Bracewell, R.N. (2000). "The Fourier Transform and its Applications" (3rd ed.)** — the mathematics reference
- **Szeliski, R. (2022). "Computer Vision: Algorithms and Applications" §3.4** — frequency-domain filtering in the CV context
- **Oppenheim & Schafer, "Discrete-Time Signal Processing" (3rd ed.)** — foundational signal processing, Chapter 7 on filter design
