"""
Noise shaping & decimation — visualize quantization noise in the frequency domain.
Companion code for the blog article (slug: noise-shaping-python).

- Pure numpy for the DSP core; matplotlib for plots; seaborn only for cosmetics.
- Deterministic: a fixed RNG seed gives identical figures/numbers on any OS / Colab.
- Bilingual plot labels: LANG = "en" (portable / Colab / Medium / LinkedIn) or "ja" (blog figures).

Run:
    pip install numpy matplotlib seaborn            # + matplotlib-fontja if LANG="ja"
    python noise_shaping.py
Writes PNGs into ./figures/ and prints the SNR/ENOB table.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(context="notebook", style="whitegrid")  # cosmetic only; safe to drop
except ImportError:
    pass

# ---- plot language ----
LANG = os.environ.get("PLOT_LANG", "en")   # "en": Colab/英語（Medium・LinkedIn 流用）。"ja": ブログ用日本語図。環境変数 PLOT_LANG でも切替可

if LANG == "ja":
    try:
        import matplotlib_fontja  # noqa: F401  (pip install matplotlib-fontja / 日本語フォント同梱・Python 3.12+ 対応)
    except ImportError:
        import matplotlib
        matplotlib.rcParams["font.family"] = ["IPAexGothic", "Noto Sans CJK JP", "sans-serif"]
        # 豆腐（□）が出るなら: pip install matplotlib-fontja  （または WSL で sudo apt install fonts-noto-cjk）
    import matplotlib
    matplotlib.rcParams["axes.unicode_minus"] = False  # 日本語フォントでのマイナス記号対策

# Headless (bare WSL shell, no display)? Uncomment to avoid opening windows:
#   import matplotlib; matplotlib.use("Agg")
# In Colab / Jupyter the inline backend shows the figures automatically.

os.makedirs("figures", exist_ok=True)

# ---- reproducibility ----
SEED = 12345
rng = np.random.default_rng(SEED)

# ---- parameters ----
N      = 1 << 16          # samples (power of two -> clean FFT)
NBITS  = 4               # quantizer resolution (deliberately low)
OSR    = 64              # oversampling ratio
FS     = 1.0             # normalized sample rate
CYCLES = 101             # integer, prime-ish -> coherent tone, little FFT leakage
F0     = CYCLES / N * FS # input tone, inside the band of interest
AMP    = 0.5             # amplitude (fraction of full scale) -> headroom for the modulator
BAND   = FS / (2 * OSR)  # band of interest: DC .. fs/(2*OSR)

n = np.arange(N)
x = AMP * np.sin(2 * np.pi * F0 * n)        # full scale is +-1

# ---- uniform quantizer (step DELTA) + TPDF dither ----
DELTA = 2.0 / (2 ** NBITS)

def quantize(u):
    return np.clip(DELTA * np.round(u / DELTA), -1.0, 1.0 - DELTA)

# Triangular (TPDF) dither decorrelates the quantization error -> clean, textbook spectra
dither = (rng.random(N) - rng.random(N)) * DELTA

# (a) direct quantization, no shaping
y_direct = quantize(x + dither)

# (b) 1st-order sigma-delta (error feedback): Y = X + (1 - z^-1) E, so NTF(z) = 1 - z^-1
y_sd = np.empty(N)
e = 0.0
for i in range(N):
    w = x[i] - e                 # subtract the previous error (NO dither on the input path)
    v = quantize(w + dither[i])  # dither the quantizer -> the dither is noise-shaped too
    e = v - w                    # this step's error (dither + quantization error)
    y_sd[i] = v

# ---- windowed-sinc low-pass + decimate by OSR (numpy only) ----
def lowpass_fir(cutoff, numtaps):
    m = np.arange(numtaps) - (numtaps - 1) / 2.0
    h = 2 * cutoff * np.sinc(2 * cutoff * m) * np.hamming(numtaps)
    return h / h.sum()           # unity DC gain

h = lowpass_fir(BAND, 255)

def decimate(sig):
    return np.convolve(sig, h, mode="same")[::OSR]

y_dec = decimate(y_sd)           # decimated shaped output (fs -> fs/OSR)

# ---- spectra (one-sided PSD, dB relative to a full-scale sine) ----
def psd_db(sig):
    w = np.hanning(len(sig))
    w = w / np.sqrt(np.mean(w ** 2))        # keep power
    X = np.fft.rfft(sig * w) / len(sig)
    p = 2.0 * np.abs(X) ** 2                 # one-sided
    f = np.fft.rfftfreq(len(sig), d=1.0)
    return f, 10 * np.log10(p / 0.5 + 1e-20) # 0.5 = power of a full-scale sine

# ---- in-band SNR / ENOB ----
def snr_enob(sig, f0, band):
    w = np.hanning(len(sig)); w /= np.sqrt(np.mean(w ** 2))
    P = np.abs(np.fft.rfft(sig * w)) ** 2
    f = np.fft.rfftfreq(len(sig), d=1.0)
    k0 = int(np.argmin(np.abs(f - f0)))
    sig_bins = slice(max(k0 - 3, 1), k0 + 4)     # tone + leakage
    inband = f <= band
    inband[0] = False                            # drop DC
    noise = inband.copy(); noise[sig_bins] = False
    snr = 10 * np.log10(P[sig_bins].sum() / P[noise].sum())
    return snr, (snr - 1.76) / 6.02

snr_direct, enob_direct = snr_enob(y_direct, F0, BAND)
snr_sd,     enob_sd     = snr_enob(y_sd,     F0, BAND)
snr_dec,    enob_dec    = snr_enob(y_dec,    F0 * OSR, 0.5)   # decimated: tone at F0*OSR, full band

print(f"{'stage':<28}{'in-band SNR [dB]':>18}{'ENOB [bit]':>14}")
print(f"{'direct quantization':<28}{snr_direct:>18.1f}{enob_direct:>14.1f}")
print(f"{'sigma-delta (shaped)':<28}{snr_sd:>18.1f}{enob_sd:>14.1f}")
print(f"{'after decimation':<28}{snr_dec:>18.1f}{enob_dec:>14.1f}")

# ---- labels (en / ja) ----
JA = (LANG == "ja")
def L(en, ja):
    return ja if JA else en

# ---- Figure 1: time domain (sine vs 4-bit quantized) ----
seg = slice(0, 400)
plt.figure(figsize=(8, 3.2))
plt.plot(n[seg], x[seg], label=L("input sine", "入力正弦波"), lw=1.5)
plt.step(n[seg], y_direct[seg], where="mid",
         label=L(f"{NBITS}-bit quantized", f"{NBITS}ビット量子化"), lw=1.0)
plt.xlabel(L("sample", "サンプル")); plt.ylabel(L("amplitude", "振幅"))
plt.legend(loc="upper right")
plt.title(L("Time domain: quantization staircase", "時間領域：量子化の階段"))
plt.tight_layout()
plt.savefig("figures/fig1_timedomain.png", dpi=130); plt.show()

# ---- Figure 2: spectra, direct vs shaped (+ NTF theory overlay) ----
f, Pdir = psd_db(y_direct)
_, Psd  = psd_db(y_sd)
w_ntf = 2 * np.pi * f / FS
ntf_db = 10 * np.log10(4 * np.sin(w_ntf / 2) ** 2 + 1e-20)  # |1 - e^{-jw}|^2 = 4 sin^2(w/2)
hi = f > 0.05                                               # 高域（シェーピング雑音が支配的）で
ntf_offset = float(np.median(Psd[hi] - ntf_db[hi]))         # 理論曲線を実測の床へ自動フィット
plt.figure(figsize=(8, 4))
plt.semilogx(f[1:], Pdir[1:], label=L("direct quantization", "直接量子化"), lw=1.0, alpha=0.8)
plt.semilogx(f[1:], Psd[1:],  label=L("sigma-delta (shaped)", "シグマデルタ（シェーピング）"), lw=1.0, alpha=0.8)
plt.semilogx(f[1:], ntf_db[1:] + ntf_offset, "--", color="k",
             label=r"$|NTF|^2=(2\sin(\pi f/f_s))^2$ " + L("(theory)", "（理論）"), lw=1.2)
plt.axvline(BAND, color="r", ls=":", lw=1.0, label=L("band edge fs/(2*OSR)", "帯域端 fs/(2*OSR)"))
plt.xlabel(L("frequency [x fs]", "周波数 [×fs]"))
plt.ylabel(L("PSD [dB rel. full-scale sine]", "PSD [dB, フルスケール正弦波基準]"))
plt.ylim(-140, 0); plt.legend(loc="lower right", fontsize=8)
plt.title(L("Noise moves out of band — the theory curve predicts the shape",
            "ノイズが帯域外へ移る — 理論曲線が形を予言する"))
plt.tight_layout(); plt.savefig("figures/fig2_spectra.png", dpi=130); plt.show()

# ---- Figure 3: decimated spectrum ----
fd, Pdec = psd_db(y_dec)
plt.figure(figsize=(8, 3.6))
plt.plot(fd, Pdec, lw=1.0)
plt.xlabel(L("frequency [x fs_decimated]", "周波数 [×fs（間引き後）]"))
plt.ylabel(L("PSD [dB rel. full-scale sine]", "PSD [dB, フルスケール正弦波基準]"))
plt.ylim(-140, 0)
plt.title(L(f"After decimation: low in-band floor (ENOB ~ {enob_dec:.1f} bit from {NBITS}-bit)",
            f"デシメーション後：帯域内の床が低い（{NBITS}ビット→ENOB 約{enob_dec:.1f}ビット）"))
plt.tight_layout(); plt.savefig("figures/fig3_decimated.png", dpi=130); plt.show()

# ---- Figure 4: in-band SNR vs OSR (measured vs 9 dB/octave) ----
osr_list = np.array([8, 16, 32, 64, 128, 256])
snr_meas = [snr_enob(y_sd, F0, FS / (2 * o))[0] for o in osr_list]
ref = snr_meas[0] + 30 * np.log10(osr_list / osr_list[0])   # 1st order: 9 dB/octave
plt.figure(figsize=(7, 4))
plt.semilogx(osr_list, snr_meas, "o-", base=2, label=L("measured", "実測"))
plt.semilogx(osr_list, ref, "--", base=2, label=L("9 dB / octave (1st order)", "9 dB/オクターブ（1次）"))
plt.xlabel(L("oversampling ratio (OSR)", "オーバーサンプリング比 (OSR)"))
plt.ylabel(L("in-band SNR [dB]", "帯域内SNR [dB]"))
plt.legend()
plt.title(L("1st-order shaping: ~9 dB per OSR octave", "1次シェーピング：OSR 1オクターブごとに約9 dB"))
plt.tight_layout(); plt.savefig("figures/fig4_snr_vs_osr.png", dpi=130); plt.show()

print("\nfigures written to ./figures/")
