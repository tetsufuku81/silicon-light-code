"""
Oversampling vs. noise shaping — how a faster clock buys resolution.
Companion code for the blog article (slug: oversampling-python). Sequel to noise-shaping-python.

Spine of the figures:
  - Plain oversampling (white quantization noise, no shaping) buys only
    10*log10(OSR) dB of in-band SNR = 3.01 dB/octave = 0.5 bit/octave.
  - 1st-order noise shaping turns the same clock (OSR) into
    9 dB/octave = 1.5 bit/octave, because the in-band floor is tilted down.
  So the "spare speed" of a fast ADC only becomes resolution *through* shaping.

- Pure numpy for the DSP core; matplotlib for plots; seaborn only for cosmetics.
- Deterministic: a fixed RNG seed gives identical figures/numbers on any OS / Colab.
- Bilingual plot labels: LANG = "en" (portable / Colab / Medium / LinkedIn) or "ja" (blog figures).

Run:
    pip install numpy matplotlib seaborn            # + matplotlib-fontja if LANG="ja"
    python oversampling.py
Writes PNGs into ./figures/ and prints the SNR/ENOB-vs-OSR table.
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

os.makedirs("figures", exist_ok=True)

# ---- reproducibility ----
SEED = 12345
rng = np.random.default_rng(SEED)

# ---- parameters ----
N      = 1 << 16          # samples (power of two -> clean FFT)
NBITS  = 4               # quantizer resolution (deliberately low)
FS     = 1.0             # normalized sample rate
CYCLES = 101             # integer, prime-ish -> coherent tone, little FFT leakage
F0     = CYCLES / N * FS # input tone, kept well inside the narrowest band of interest
AMP    = 0.5             # amplitude (fraction of full scale) -> headroom for the modulator

n = np.arange(N)
x = AMP * np.sin(2 * np.pi * F0 * n)        # full scale is +-1

# ---- uniform quantizer (step DELTA) + TPDF dither ----
DELTA = 2.0 / (2 ** NBITS)

def quantize(u):
    return np.clip(DELTA * np.round(u / DELTA), -1.0, 1.0 - DELTA)

# Triangular (TPDF) dither decorrelates the quantization error -> white, textbook floor
dither = (rng.random(N) - rng.random(N)) * DELTA

# (a) plain quantization, NO shaping -> white in-band noise. This is the "oversampling only" case:
#     we never change the signal, only how wide a band we later keep (fs/(2*OSR)).
y_plain = quantize(x + dither)

# (b) 1st-order sigma-delta (error feedback): Y = X + (1 - z^-1) E, so NTF(z) = 1 - z^-1
y_sd = np.empty(N)
e = 0.0
for i in range(N):
    w = x[i] - e                 # subtract the previous error (NO dither on the input path)
    v = quantize(w + dither[i])  # dither the quantizer -> the dither is noise-shaped too
    e = v - w                    # this step's error (dither + quantization error)
    y_sd[i] = v

# ---- in-band SNR / ENOB over an arbitrary band [0, band] ----
# Pre-compute the windowed spectrum once; "oversampling" is just choosing a narrower band.
_win = np.hanning(N); _win /= np.sqrt(np.mean(_win ** 2))   # keep power
_f   = np.fft.rfftfreq(N, d=1.0 / FS)

def power_spectrum(sig):
    return np.abs(np.fft.rfft(sig * _win)) ** 2

P_plain = power_spectrum(y_plain)
P_sd    = power_spectrum(y_sd)

def snr_enob(P, f0, band):
    k0 = int(np.argmin(np.abs(_f - f0)))
    sig_bins = slice(max(k0 - 3, 1), k0 + 4)      # tone + spectral leakage
    inband = _f <= band
    inband[0] = False                             # drop DC
    noise = inband.copy(); noise[sig_bins] = False
    snr = 10 * np.log10(P[sig_bins].sum() / P[noise].sum())
    return snr, (snr - 1.76) / 6.02               # ENOB = (SNR - 1.76) / 6.02

# ---- sweep OSR: in-band SNR for plain oversampling vs 1st-order shaping ----
osr_list = np.array([4, 8, 16, 32, 64, 128, 256])
snr_plain = np.array([snr_enob(P_plain, F0, FS / (2 * o))[0] for o in osr_list])
snr_shape = np.array([snr_enob(P_sd,    F0, FS / (2 * o))[0] for o in osr_list])

# theory slopes anchored at the first OSR:
#   plain oversampling: 10*log10(OSR) -> 3.01 dB/oct = 0.5 bit/oct
#   1st-order shaping : 30*log10(OSR) -> 9.03 dB/oct = 1.5 bit/oct
ref_plain = snr_plain[0] + 10 * np.log10(osr_list / osr_list[0])
ref_shape = snr_shape[0] + 30 * np.log10(osr_list / osr_list[0])

print(f"{'OSR':>6}{'plain SNR':>12}{'plain ENOB':>12}{'shaped SNR':>12}{'shaped ENOB':>13}")
for i, o in enumerate(osr_list):
    print(f"{o:>6}{snr_plain[i]:>12.1f}{(snr_plain[i]-1.76)/6.02:>12.1f}"
          f"{snr_shape[i]:>12.1f}{(snr_shape[i]-1.76)/6.02:>13.1f}")

# ---- labels (en / ja) ----
JA = (LANG == "ja")
def L(en, ja):
    return ja if JA else en

# ---- Figure 1: the same noise, a narrower band ----
# Plain quantization has a FLAT (white) floor: narrowing the band to fs/(2*OSR) removes
# only the slice outside the band -> 3 dB per octave. Shaping tilts the floor down toward
# DC, so the same narrowing removes far more -> 9 dB per octave. Shown with band edges.
def psd_db(P):
    p = 2.0 * P / N ** 2                           # one-sided
    return 10 * np.log10(p / 0.5 + 1e-20)          # dB rel. full-scale sine (0.5 = its power)

plt.figure(figsize=(8, 4))
plt.semilogx(_f[1:], psd_db(P_plain)[1:], lw=1.0, alpha=0.85,
             label=L("plain quantization (white floor)", "直接量子化（平らなノイズフロア）"))
plt.semilogx(_f[1:], psd_db(P_sd)[1:], lw=1.0, alpha=0.85,
             label=L("1st-order shaping (tilted floor)", "1次シェーピング（傾いたノイズフロア）"))
for o, c in zip((16, 64, 256), ("#888", "#aa4444", "#cc8800")):
    plt.axvline(FS / (2 * o), color=c, ls=":", lw=1.0,
                label=L(f"band edge, OSR={o}", f"帯域端 OSR={o}"))
plt.xlabel(L("frequency [x fs]", "周波数 [×fs]"))
plt.ylabel(L("PSD [dB rel. full-scale sine]", "PSD [dB, フルスケール正弦波基準]"))
plt.ylim(-140, 0); plt.legend(loc="lower right", fontsize=8)
plt.title(L("Same total noise — narrowing the band pays off far more when the floor is tilted",
            "総ノイズは同じ — ノイズフロアが傾いているほど、帯域を狭めた見返りが大きい"))
plt.tight_layout(); plt.savefig("figures/fig1_band_narrowing.png", dpi=130); plt.show()

# ---- Figure 2 (core): in-band SNR vs OSR, plain vs shaped ----
plt.figure(figsize=(7.5, 4.5))
plt.semilogx(osr_list, snr_plain, "o-", base=2, color="#1f77b4",
             label=L("oversampling only (measured)", "オーバーサンプリングのみ（実測）"))
plt.semilogx(osr_list, ref_plain, "--", base=2, color="#1f77b4", alpha=0.6,
             label=L("3 dB/oct = 0.5 bit/oct", "3 dB/oct = 0.5 bit/oct"))
plt.semilogx(osr_list, snr_shape, "s-", base=2, color="#d62728",
             label=L("+ 1st-order shaping (measured)", "＋1次シェーピング（実測）"))
plt.semilogx(osr_list, ref_shape, "--", base=2, color="#d62728", alpha=0.6,
             label=L("9 dB/oct = 1.5 bit/oct", "9 dB/oct = 1.5 bit/oct"))
plt.xlabel(L("oversampling ratio (OSR)", "オーバーサンプリング比 (OSR)"))
plt.ylabel(L("in-band SNR [dB]", "帯域内SNR [dB]"))
plt.legend(loc="upper left", fontsize=8)
plt.title(L("The spare clock becomes resolution only through shaping",
            "余ったクロックは、シェーピングを通して初めて分解能になる"))
plt.tight_layout(); plt.savefig("figures/fig2_snr_vs_osr.png", dpi=130); plt.show()

# ---- Figure 3: the same story in effective bits (ENOB gained vs OSR) ----
enob_plain = (snr_plain - 1.76) / 6.02 - (snr_plain[0] - 1.76) / 6.02
enob_shape = (snr_shape - 1.76) / 6.02 - (snr_shape[0] - 1.76) / 6.02
octaves = np.log2(osr_list / osr_list[0])
plt.figure(figsize=(7.5, 4))
plt.plot(octaves, enob_plain, "o-", color="#1f77b4",
         label=L("oversampling only (~0.5 bit/oct)", "オーバーサンプリングのみ（約0.5 bit/oct）"))
plt.plot(octaves, enob_shape, "s-", color="#d62728",
         label=L("+ 1st-order shaping (~1.5 bit/oct)", "＋1次シェーピング（約1.5 bit/oct）"))
plt.xlabel(L("OSR doublings (octaves)", "OSRの倍化（オクターブ）"))
plt.ylabel(L("effective bits gained", "得られた実効ビット"))
plt.legend(loc="upper left", fontsize=9)
plt.title(L("Trading speed for resolution: 0.5 vs 1.5 bit per octave",
            "速さを分解能に替える：1オクターブあたり 0.5 対 1.5 ビット"))
plt.tight_layout(); plt.savefig("figures/fig3_bits_vs_osr.png", dpi=130); plt.show()

print("\nfigures written to ./figures/")
