import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import deque
import time
import csv

# ----------------- Settings -----------------
PORT = "/dev/ttyACM0"
BAUD = 1000000
N = 1024
HEADER = b'\xCD\xAB'
VREF = 3.3
FS = 18860.0
HISTORY_LEN = 80
TRACK_LEN = 150
FRAME_SIZE = 2 + N * 2
MIN_FREQ = 5.0
PEAK_MIN_DB = -40
EXCLUDE_BINS = 5

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ----------------- Figures -----------------
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.canvas.manager.set_window_title("Real-Time FFT Analyzer with CSV Save")

ax_time, ax_fft, ax_spec, ax_track = axs.flatten()

# --- Time domain ---
time_line, = ax_time.plot([], [], color='black')
ax_time.set_title("Time Domain")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Voltage [V]")
ax_time.grid(True)

# --- FFT Spectrum ---
fft_line, = ax_fft.plot([], [], color='purple')
peak_marker, = ax_fft.plot([], [], 'go', markersize=8)
sec_peaks_plot, = ax_fft.plot([], [], 'ro', markersize=5)
ax_fft.set_title("FFT Spectrum")
ax_fft.set_xlabel("Frequency [Hz]")
ax_fft.set_ylabel("Magnitude [dB]")
ax_fft.grid(True)

# --- Spectrogram ---
spec_img = ax_spec.imshow(np.zeros((N//2, HISTORY_LEN)),
                          aspect='auto', origin='lower',
                          extent=[0, HISTORY_LEN, 0, FS/2],
                          cmap='inferno', vmin=-100, vmax=0)
ax_spec.set_title("Spectrogram")
ax_spec.set_xlabel("Frame Index")
ax_spec.set_ylabel("Frequency [Hz]")

# --- Frequency Tracking ---
track_line, = ax_track.plot([], [], color='teal')
ax_track.set_title("Main Frequency Tracking")
ax_track.set_xlabel("Frame")
ax_track.set_ylabel("Frequency [Hz]")
ax_track.grid(True)

# --- CSV Save Button ---
button_ax = plt.axes([0.83, 0.02, 0.13, 0.05])
save_button = Button(button_ax, '💾 Save CSV', color='lightgray', hovercolor='lightgreen')

# ----------------- Buffers -----------------
spec_history = deque(np.zeros(N//2), maxlen=HISTORY_LEN)
freq_history = deque([0]*TRACK_LEN, maxlen=TRACK_LEN)
time_history = deque(np.linspace(-TRACK_LEN, 0, TRACK_LEN), maxlen=TRACK_LEN)
prev_time = time.time()
frame_count = 0
fps_display = 0

# ----------------- Data Window -----------------
fig_info, ax_info = plt.subplots(figsize=(5, 3))
fig_info.canvas.manager.set_window_title("Live FFT Data")
ax_info.axis("off")
info_text = ax_info.text(0.05, 0.95, "", fontsize=12, va="top", family="monospace")

# ----------------- Helper Functions -----------------
def smooth(data, w=5):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='same')

def parabolic_interpolation(mag_db, idx, df):
    if idx <= 0 or idx >= len(mag_db)-1:
        return 0.0
    alpha, beta, gamma = mag_db[idx-1], mag_db[idx], mag_db[idx+1]
    return 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma) * df

def find_peaks(volt):
    volt = volt - np.mean(volt)
    window = np.hanning(N)
    fft_vals = np.fft.rfft(volt * window)
    mag = np.abs(fft_vals) * 2 / (N * (np.sum(window)/N))
    mag_db = 20 * np.log10(mag + 1e-12)
    freq_axis = np.fft.rfftfreq(N, 1/FS)

    mask = freq_axis > MIN_FREQ
    mag_db_valid = mag_db[mask]
    mag_valid = mag[mask]
    freq_axis_valid = freq_axis[mask]

    df = FS / N
    main_idx = np.argmax(mag_db_valid)
    main_freq = freq_axis_valid[main_idx] + parabolic_interpolation(mag_db_valid, main_idx, df)
    main_amp = mag_valid[main_idx]

    mag_copy = mag_db_valid.copy()
    start, end = max(main_idx - EXCLUDE_BINS, 0), min(main_idx + EXCLUDE_BINS + 1, len(mag_copy))
    mag_copy[start:end] = -np.inf

    secondary_peaks = []
    for i in range(1, len(mag_copy)-1):
        if mag_copy[i] > PEAK_MIN_DB and mag_copy[i] > mag_copy[i-1] and mag_copy[i] > mag_copy[i+1]:
            secondary_peaks.append((freq_axis_valid[i], mag_valid[i]))

    rms = np.sqrt(np.mean(volt**2))
    snr_db = np.nan
    if secondary_peaks:
        snr_db = 20 * np.log10(main_amp / (secondary_peaks[0][1] + 1e-12))

    return freq_axis, mag_db, main_freq, main_amp, secondary_peaks, rms, snr_db

# ----------------- CSV Save -----------------
def save_to_csv(event):
    filename = f"fft_snapshot_{int(time.time())}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency (Hz)", "Magnitude (dB)"])
        for f_hz, mag_db_val in zip(current_freq_axis, current_mag_db):
            writer.writerow([f_hz, mag_db_val])
    print(f"✅ Saved FFT data to {filename}")

save_button.on_clicked(save_to_csv)

# ----------------- Main Loop -----------------
current_freq_axis, current_mag_db = np.zeros(N//2+1), np.zeros(N//2+1)

while True:
    try:
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
            continue

        adc_vals = np.frombuffer(frame[2:], dtype=np.uint16)
        volt = adc_vals * VREF / 4095.0

        freq_axis, mag_db, main_freq, main_amp, secondary_peaks, rms, snr_db = find_peaks(volt)
        current_freq_axis, current_mag_db = freq_axis, mag_db

        # --- Update plots ---
        time_line.set_data(np.arange(N)/FS, volt)
        ax_time.relim(); ax_time.autoscale_view()

        fft_line.set_data(freq_axis, mag_db)
        peak_marker.set_data([main_freq], [20*np.log10(main_amp)])
        if secondary_peaks:
            sec_peaks_plot.set_data([p[0] for p in secondary_peaks],
                                    [20*np.log10(p[1]) for p in secondary_peaks])
        else:
            sec_peaks_plot.set_data([], [])
        ax_fft.relim(); ax_fft.autoscale_view()

        spec_history.append(mag_db[:N//2])
        spec_array = np.array(spec_history).T
        spec_img.set_data(spec_array)
        spec_img.set_clim(np.percentile(spec_array, [5, 95]))

        freq_history.append(main_freq)
        time_history.append(time_history[-1] + 1)
        track_line.set_data(time_history, smooth(freq_history, 5))
        ax_track.relim(); ax_track.autoscale_view()

        # --- Info window update ---
        sec_freq = secondary_peaks[0][0] if secondary_peaks else 0
        sec_amp = secondary_peaks[0][1] if secondary_peaks else 0
        info_text.set_text(
            f"Main Frequency : {main_freq:8.2f} Hz\n"
            f"Main Amplitude : {main_amp:8.4f} V\n"
            f"2nd Frequency  : {sec_freq:8.2f} Hz\n"
            f"2nd Amplitude  : {sec_amp:8.4f} V\n"
            f"SNR (est)      : {snr_db:8.2f} dB\n"
            f"RMS Voltage    : {rms:8.4f} V"
        )
        fig_info.canvas.draw_idle()

        # --- FPS update ---
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1.0:
            fps_display = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now
            fig.suptitle(f"Main: {main_freq:.1f} Hz | SNR: {snr_db:.1f} dB | FPS: {fps_display:.1f}")

        plt.pause(0.001)

    except KeyboardInterrupt:
        print("🛑 Exiting...")
        break
    except Exception as e:
        print("⚠️ Frame skipped:", e)
        continue
