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
PEAK_MIN_DB = -40
EXCLUDE_HZ = 50  # exclude ±50 Hz around main tone when searching secondary

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ----------------- Initialize Windows -----------------
plt.ion()
fig_time, ax_time = plt.subplots()
fig_time.canvas.manager.set_window_title("Time Domain Signal")

fig_fft_text, (ax_fft, ax_text) = plt.subplots(2,1, figsize=(10,8), constrained_layout=True)
fig_fft_text.canvas.manager.set_window_title("FFT Spectrum + Info")
ax_text.axis("off")

fig_spec, ax_spec = plt.subplots()
fig_spec.canvas.manager.set_window_title("Spectrogram")

fig_track, ax_track = plt.subplots()
fig_track.canvas.manager.set_window_title("Frequency Tracking")

# ----------------- Buffers -----------------
spec_history = deque(np.zeros(N//2), maxlen=HISTORY_LEN)
freq_history = deque([0]*TRACK_LEN, maxlen=TRACK_LEN)
time_history = deque(np.linspace(-TRACK_LEN, 0, TRACK_LEN), maxlen=TRACK_LEN)

# ----------------- Helper Functions -----------------
def smooth(data, w=5):
    if len(data) < w: return data
    return np.convolve(data, np.ones(w)/w, mode='same')

def parabolic_interpolation(mag_db, idx, df):
    if idx <= 0 or idx >= len(mag_db)-1:
        return 0.0
    alpha, beta, gamma = mag_db[idx-1], mag_db[idx], mag_db[idx+1]
    return 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma) * df

def precise_mag(mag_db, idx):
    """Refine magnitude using parabolic fit around a bin."""
    if 0 < idx < len(mag_db)-1:
        alpha, beta, gamma = mag_db[idx-1], mag_db[idx], mag_db[idx+1]
        return beta - 0.25 * (alpha - gamma) * (alpha - gamma) / (alpha - 2*beta + gamma)
    return mag_db[idx]

def find_peaks(volt):
    volt = volt - np.mean(volt)
    window = np.hanning(N)
    fft_vals = np.fft.rfft(volt * window)
    mag = np.abs(fft_vals) * 2 / (np.sum(window)/2)
    mag_db = 20 * np.log10(mag + 1e-12)
    freq_axis = np.fft.rfftfreq(N, 1/FS)

    # Noise floor suppression
    noise_floor = np.percentile(mag, 90) * 1.5
    mag_db[mag < noise_floor] = -120

    # Detect main peak
    mask = freq_axis > MIN_FREQ
    mag_db_valid = mag_db[mask]
    mag_valid = mag[mask]
    freq_axis_valid = freq_axis[mask]

    df = FS / N
    main_idx = np.argmax(mag_db_valid)
    main_freq = freq_axis_valid[main_idx] + parabolic_interpolation(mag_db_valid, main_idx, df)
    main_amp = mag_valid[main_idx]

    # Exclude ±EXCLUDE_HZ around main tone
    mag_copy = mag_db_valid.copy()
    exclude_bins = int(EXCLUDE_HZ / df)
    start, end = max(main_idx - exclude_bins, 0), min(main_idx + exclude_bins + 1, len(mag_copy))
    mag_copy[start:end] = -np.inf

    # Find secondary peaks
    secondary_peaks = []
    for i in range(1, len(mag_copy)-1):
        if mag_copy[i] > PEAK_MIN_DB and mag_copy[i] > mag_copy[i-1] and mag_copy[i] > mag_copy[i+1]:
            secondary_peaks.append((freq_axis_valid[i], mag_valid[i]))

    # RMS and THD
    rms = np.sqrt(np.mean(volt**2))
    harmonic_mags = []
    for h in [2,3,4]:
        target = h * main_freq
        if target < FS/2:
            idx = np.argmin(np.abs(freq_axis_valid - target))
            harm_mag_db = precise_mag(mag_db_valid, idx)
            harm_amp = 10**(harm_mag_db/20)
            harmonic_mags.append(harm_amp)
    thd = np.sqrt(np.sum(np.array(harmonic_mags)**2)) / main_amp if harmonic_mags else 0.0

    # SNR (main vs next strongest)
    snr_db = np.nan
    if secondary_peaks:
        snr_db = 20*np.log10(main_amp/(secondary_peaks[0][1]+1e-12))

    return freq_axis, mag_db, main_freq, main_amp, secondary_peaks, rms, thd, snr_db

# ----------------- Main Loop -----------------
while True:
    try:
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
            continue

        adc_vals = np.frombuffer(frame[2:], dtype=np.uint16)
        volt = adc_vals * VREF / 4095.0

        freq_axis, mag_db, main_freq, main_amp, secondary_peaks, rms, thd, snr_db = find_peaks(volt)

        # --- Time Domain ---
        ax_time.cla()
        ax_time.plot(np.arange(N)/FS, volt, color='black')
        ax_time.set_title("Time Domain")
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Voltage [V]")
        ax_time.grid(True)
        fig_time.canvas.draw_idle()

        # --- FFT Spectrum ---
        ax_fft.cla()
        ax_fft.plot(freq_axis, mag_db, color='purple')
        ax_fft.scatter(main_freq, 20*np.log10(main_amp), color='green', s=60, label="Main Peak")
        for f,a in secondary_peaks:
            ax_fft.scatter(f, 20*np.log10(a), color='orange', s=50)
            ax_fft.text(f, 20*np.log10(a)+3, f"{f:.1f}Hz", fontsize=9)
        ax_fft.set_title("FFT Spectrum")
        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("Magnitude [dB]")
        ax_fft.grid(True)
        fig_fft_text.canvas.draw_idle()

        # --- Text Info Panel ---
        ax_text.cla()
        ax_text.axis("off")
        bg = plt.Rectangle((0,0),1,1, transform=ax_text.transAxes, color='lightyellow', alpha=0.5)
        ax_text.add_patch(bg)

        thd_color = 'red' if thd*100 > 5 else 'darkgreen'
        snr_color = 'orange' if (not np.isnan(snr_db) and snr_db < 20) else 'darkgreen'

        lines = [
            ("Main Frequency", f"{main_freq:.2f} Hz", 'darkblue'),
            ("Amplitude", f"{main_amp:.4f} V", 'darkblue'),
            ("RMS", f"{rms:.4f} V", 'darkblue'),
            ("THD", f"{thd*100:.2f} %", thd_color),
            ("SNR", f"{snr_db:.2f} dB" if not np.isnan(snr_db) else "N/A", snr_color)
        ]

        for i,(f,a) in enumerate(secondary_peaks):
            lines.append((f"Secondary {i+1}", f"{f:.2f} Hz | {a:.4f} V", 'darkorange'))

        for i, (label, value, color) in enumerate(lines):
            ax_text.text(0.05, 0.95 - 0.12*i, f"{label}: {value}",
                         fontsize=12, family='monospace', color=color, va='top')

        fig_fft_text.canvas.draw_idle()

        # --- Spectrogram ---
        spec_history.append(mag_db[:N//2])
        spec_array = np.array(spec_history).T
        ax_spec.cla()
        ax_spec.imshow(spec_array, aspect='auto', origin='lower',
                       extent=[0,HISTORY_LEN,0,FS/2],
                       cmap='inferno', vmin=-100, vmax=0)
        ax_spec.set_title("Spectrogram")
        ax_spec.set_xlabel("Frame Index")
        ax_spec.set_ylabel("Frequency [Hz]")
        fig_spec.canvas.draw_idle()

        # --- Frequency Tracking ---
        freq_history.append(main_freq)
        time_history.append(time_history[-1]+1)
        ax_track.cla()
        ax_track.plot(list(time_history), smooth(freq_history,5), color='teal')
        ax_track.set_title("Main Frequency Tracking")
        ax_track.set_xlabel("Frame Index")
        ax_track.set_ylabel("Frequency [Hz]")
        ax_track.grid(True)
        fig_track.canvas.draw_idle()

        plt.pause(0.001)

    except Exception as e:
        print("Frame skipped:", e)
        continue
