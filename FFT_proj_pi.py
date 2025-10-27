import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# ----------------- Settings -----------------
PORT = "/dev/ttyACM0"
BAUD = 1000000
N = 1024
HEADER = b'\xCD\xAB'
VREF = 3.3
FS = 18860.0
HISTORY_LEN = 100
TRACK_LEN = 200
FRAME_SIZE = 2 + N*2
MIN_FREQ = 5.0
EXCLUDE_HZ = 50
WINDOW_TYPE = "blackmanharris"   # "hann" or "blackmanharris"

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ----------------- Initialize Figures -----------------
plt.ion()
fig_fft, ax_fft = plt.subplots()
ax_fft.set_title("FFT Spectrum")
ax_fft.set_xlabel("Frequency [Hz]")
ax_fft.set_ylabel("Magnitude [dB]")

# ----------------- Helper Functions -----------------
def get_window(name, N):
    if name == "blackmanharris":
        return np.blackmanharris(N)
    elif name in ["hann", "hanning"]:
        return np.hanning(N)
    else:
        return np.ones(N)

def parabolic_interpolation(mag_db, idx, df):
    if idx <= 0 or idx >= len(mag_db)-1:
        return 0.0
    alpha, beta, gamma = mag_db[idx-1], mag_db[idx], mag_db[idx+1]
    return 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma) * df

def precise_mag(mag_db, idx):
    if 0 < idx < len(mag_db)-1:
        alpha, beta, gamma = mag_db[idx-1], mag_db[idx], mag_db[idx+1]
        return beta - 0.25 * (alpha - gamma)**2 / (alpha - 2*beta + gamma)
    return mag_db[idx]

def compute_fft(volt):
    volt = volt - np.mean(volt)
    window = get_window(WINDOW_TYPE, N)
    window_power = np.sum(window**2)/N
    fft_vals = np.fft.rfft(volt * window)
    mag = (2.0 / (N * np.sqrt(window_power))) * np.abs(fft_vals)
    mag_db = 20 * np.log10(mag + 1e-15)
    freq_axis = np.fft.rfftfreq(N, 1/FS)
    return freq_axis, mag, mag_db

def compute_thd(freq_axis, mag, mag_db, main_freq, main_amp):
    harmonic_mags = []
    df = FS / N
    for h in [2, 3, 4, 5]:
        target = h * main_freq
        if target >= FS/2:
            continue
        idx = np.argmin(np.abs(freq_axis - target))
        harm_db = precise_mag(mag_db, idx)
        harm_amp = 10**(harm_db/20)
        harmonic_mags.append(harm_amp)
    if not harmonic_mags:
        return 0.0
    return np.sqrt(np.sum(np.array(harmonic_mags)**2)) / main_amp

# ----------------- Main Loop -----------------
while True:
    try:
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
            continue

        adc_vals = np.frombuffer(frame[2:], dtype=np.uint16)
        volt = adc_vals * VREF / 4095.0

        # --- FFT ---
        freq_axis, mag, mag_db = compute_fft(volt)
        df = FS / N
        mask = freq_axis > MIN_FREQ
        mag_db_valid = mag_db[mask]
        mag_valid = mag[mask]
        freq_axis_valid = freq_axis[mask]

        main_idx = np.argmax(mag_db_valid)
        main_freq = freq_axis_valid[main_idx] + parabolic_interpolation(mag_db_valid, main_idx, df)
        main_amp = mag_valid[main_idx]

        # --- THD and RMS ---
        thd = compute_thd(freq_axis_valid, mag_valid, mag_db_valid, main_freq, main_amp)
        rms = np.sqrt(np.mean(volt**2))

        # --- Coherence Check ---
        cycles = main_freq * N / FS
        coherence_ok = abs(cycles - round(cycles)) < 0.1

        # --- Plot ---
        ax_fft.cla()
        ax_fft.plot(freq_axis_valid, mag_db_valid, color='navy')
        ax_fft.scatter(main_freq, 20*np.log10(main_amp), color='green', label=f"Main {main_freq:.1f}Hz")
        ax_fft.set_title(f"FFT | Main {main_freq:.2f} Hz | THD={thd*100:.2f}% | RMS={rms:.3f} V")
        if not coherence_ok:
            ax_fft.set_title(ax_fft.get_title() + " ⚠️ Incoherent Sampling")
        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("Magnitude [dB]")
        ax_fft.grid(True)
        ax_fft.legend()
        plt.pause(0.001)

    except Exception as e:
        print("Frame skipped:", e)
        continue
