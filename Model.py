import numpy as np
import threading
import time
import pylsl  # LSL library
from pylsl import StreamInlet
import sys
from queue import Queue  # Use the thread-safe Queue
from scipy.signal import filtfilt
from scipy.fft import fft, fftfreq
import pickle
from scipy import signal

import json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt


import socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# serverAddressPort = ("127.0.0.1", 1880)

# sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock_recv.bind(("127.0.0.1", 5056))

from collections import deque

class EEGModel:
    def __init__(self, num_channels=8, samp_freq=250, window_size_second=4, band_pass = (2,40),stream_name="SomSom", status_queue=None, command_queue=None):
        self.num_channels = num_channels
        self.window_size = samp_freq * window_size_second
        self.eeg_data = np.zeros((self.num_channels, self.window_size))  # Real-time data buffer
        self.main_channel_roll = np.zeros((self.num_channels, self.window_size))  # For rolling samples
        self.stream_name = stream_name  # Name of the LSL stream to connect to
        self.status_queue = status_queue  # Queue for updating connection status
        self.queue1 = Queue(maxsize=10000)  # For storing LSL data, limited to prevent overflow
        self.queue2 = Queue(maxsize=10000)  # For storing rolled EEG data
        self.queue3 = Queue(maxsize=10000)  # For storing filtered EEG
        self.queue4 = Queue(maxsize=10000)  # For storing FFT data
        self.queue5 =  Queue(maxsize=10000)
        self.command_queue = command_queue
        self.running = False
        self.samp_freq = samp_freq
        self.command = "Non"
        self.EEG_epoch = np.zeros((num_channels, samp_freq))
        nyq = 0.5 * samp_freq
        self.b, self.a = signal.butter(4,[band_pass[0]/nyq,band_pass[1]/nyq],'bandpass')

        # --- Feature/extraction knobs ---
        self.use_channels = None         # e.g., [O1_idx, Oz_idx, O2_idx]; None => use all
        self.alpha_band = (10.0, 12.0)
        self.snr_bw   = 0.7              # +/- Hz around targets/harmonics
        self.noise_bw = 0.7              # width of each side noise band
        self.noise_off= 1.5              # offset from target to start the noise band
        self.max_harm = 3                # up to 3rd harmonic (as available)

        # --- Smoothing & thresholds (unsupervised) ---
        self.ema = 0.12                  # exponential smoothing for features
        # Eyes-closed detector thresholds (robust z-scores)
        self.alpha_z_on  = 0.0         # enter eyes-closed if z >= on
        self.alpha_z_off = 0.0          # leave eyes-closed if z <  off (hysteresis)
        # SSVEP detector thresholds (robust z-scores)
        self.snr_z_thr   = 1.00          # must exceed this
        self.snr_margin  = 0.30          # winner - runner-up >= margin
        # Rolling history for robust baselines (median/MAD)
        self.hist_len = 200
        self.alpha_hist = deque(maxlen=self.hist_len)
        self.snr6_hist  = deque(maxlen=self.hist_len)
        self.snr15_hist = deque(maxlen=self.hist_len)
        # Smoothed features
        self.alpha_ema = None
        self.snr6_ema  = None
        self.snr15_ema = None
        # States
        self._eyes_closed = False        # sticky with hysteresis
        self._last_ec_label = "Non"

         # --- MQTT (optional) ---
        self.use_mqtt    = False
        self.mqtt_client = None
        self.mqtt_topic  = "fibot/movement"
        self.mqtt_qos    = 0
        self.mqtt_retain = False

        self.sleep_time = 0.05  # Sleep time for threads

    def start_streaming(self):
        """Start the LSL data stream and other processing functions."""
        self.running = True

        # Start only essential threads to manage memory usage
        threading.Thread(target=self.data_from_lsl, daemon=True).start()
        threading.Thread(target=self.rolling_samples, daemon=True).start()
        threading.Thread(target=self.filtering_windowed_data, daemon=True).start()
        threading.Thread(target=self.fft_process, daemon=True).start()
        threading.Thread(target=self.send_command, daemon=True).start()
        # threading.Thread(target=self.recv_from_unity, daemon=True).start()
        
    def stop_streaming(self):
        """Stop the LSL data stream."""
        self.running = False

    def data_from_lsl(self):
        """Receive data from an LSL stream and update the model in real-time."""
        while self.running:
            streams = pylsl.resolve_streams()
            for stream in streams:
                if stream.name() == self.stream_name:
                    inlet = StreamInlet(stream)
                    if self.status_queue:
                        self.status_queue.put((True, self.stream_name))
                    while self.running:
                        sample, _ = inlet.pull_chunk()
                        if sample:
                            data = np.array(sample).T
                            # print(data.shape)
                            if data.shape[0] == self.num_channels:
                                self.EEG_epoch = np.roll(self.EEG_epoch, -data.shape[1], axis=1)
                                self.EEG_epoch[:, -data.shape[1]:] = data
                                if not self.queue1.full():
                                    self.queue1.put(self.EEG_epoch)  # Add data if queue1 has space

                        time.sleep(self.sleep_time)

    def rolling_samples(self):
        """Continuously roll and update the main EEG data buffer."""
        while self.running:
            if not self.queue1.empty():
                try:
                    data = self.queue1.get()  # Get data from queue1
                    shift = data.shape[1]
                    # Roll and replace data within fixed memory
                    self.main_channel_roll = np.roll(self.main_channel_roll, -shift, axis=1)
                    self.main_channel_roll[:, -shift:] = data
                    if not self.queue2.full():
                        self.queue2.put(self.main_channel_roll)
                except Exception as e:
                    print(f"Rolling samples error: {e}")

            time.sleep(self.sleep_time)

    def filtering_windowed_data(self):
        while self.running:
            if not self.queue2.empty():
                data = self.queue2.get()
                filtered_data = filtfilt(self.b, self.a, data)
                if not self.queue3.full():
                    self.queue3.put(filtered_data)

            time.sleep(self.sleep_time)

    def fft_process(self):
        """Compute FFT on filtered data at a limited frequency to manage memory usage."""
        while self.running:
            if not self.queue3.empty():
                filtered_data = self.queue3.get()
                if filtered_data.shape[1] >= self.window_size:
                    yf = fft(filtered_data) 
                    power_spectrum = np.abs(yf) ** 2
                    if not self.queue4.full():
                        self.queue4.put(power_spectrum[:])  # Keep only up to 40Hz for efficiency

                     #####Additional code of your preprocess###########
                     # power_spectrum: same shape after FFT | we keep only positive freqs
                    ps = power_spectrum[:, : self.window_size // 2]
                    xf = fftfreq(self.window_size, 1 / self.samp_freq)[: self.window_size // 2]
                    ps = ps[[9,10], :]
                    # Channel-average (robust alternative: median across channels)
                    ps_avg = ps.mean(axis=0)  # or: np.median(ps, axis=0)
                    
                    # --- Features ---
                    # Eyes-closed: alpha power 8–12 Hz
                    alpha = self._bandpower(ps_avg, xf, *self.alpha_band)

                    # SSVEP: multi-harmonic SNRs for 6 Hz and 15 Hz
                    snr6  = self._snr_multiharm(ps_avg, xf, f0=6.0,
                                                bw=self.snr_bw, noise_bw=self.noise_bw,
                                                noise_off=self.noise_off, max_harm=self.max_harm)
                    snr15 = self._snr_multiharm(ps_avg, xf, f0=15.0,
                                                bw=self.snr_bw, noise_bw=self.noise_bw,
                                                noise_off=self.noise_off, max_harm=self.max_harm)

                    # Smooth features a bit over time
                    self.alpha_ema = self._ema(self.alpha_ema, alpha, self.ema)
                    self.snr6_ema  = self._ema(self.snr6_ema,  snr6,  self.ema)
                    self.snr15_ema = self._ema(self.snr15_ema, snr15, self.ema)

                    alpha_sm = self.alpha_ema
                    snr6_sm  = self.snr6_ema
                    snr15_sm = self.snr15_ema

                    # --- Robust z-scores (adaptive, no calibration file needed) ---
                    za  = self._robust_z(alpha_sm, self.alpha_hist)
                    z6  = self._robust_z(snr6_sm,  self.snr6_hist)
                    z15 = self._robust_z(snr15_sm, self.snr15_hist)

                    print(np.mean(za), self._eyes_closed)

                    # =========================
                    # MODEL 1: EyesClosed vs Non
                    # =========================
                    if self._eyes_closed:
                        # stay closed until alpha falls below the lower threshold (hysteresis)
                        if za < self.alpha_z_off:
                            self._eyes_closed = False
                    else:
                        # enter closed when alpha exceeds the upper threshold
                        if za >= self.alpha_z_on:
                            self._eyes_closed = True

                    ec_label = "EyesClosed" if self._eyes_closed else "Non"
                    # (Optional) one-shot transition when reopening eyes
                    # if self._last_ec_label == "EyesClosed" and ec_label == "Non":
                    #     try: self.command_queue.put("EyesOpened")
                    #     except: pass
                    self._last_ec_label = ec_label

                    # ==============================
                    # MODEL 2: 6 Hz vs 15 Hz vs Non
                    # ==============================
                    # If eyes are clearly closed, you may want to force SSVEP = Non
                    if self._eyes_closed:
                        ssvep_label = "Non"
                    else:
                        # pick the stronger SSVEP (by z-score), require margin and absolute level
                        best = max(z6, z15)
                        which = "6Hz" if z6 >= z15 else "15Hz"
                        other = z15 if which == "6Hz" else z6
                        if (best >= self.snr_z_thr) and ((best - other) >= self.snr_margin):
                            ssvep_label = which
                        else:
                            ssvep_label = "Non"

                    # --- Emit results (choose your own output format) ---
                    try:
                        # Option A: send two messages
                        # self.command_queue.put({"EC": ec_label})           # Model 1
                        # self.command_queue.put({"SSVEP": ssvep_label})     # Model 2
                        # Option B (if your downstream expects a single string):
                        self.command_queue.put(f"{ec_label}|{ssvep_label}")
                    except:
                        pass
                               
            time.sleep(self.sleep_time)  # Reduce FFT frequency to conserve memory and processing


    # def send_command(self):
    #     while self.running:
    #         if not self.command_queue.empty():
    #             send = self.command_queue.get()
    #             sock.sendto(str.encode(send), serverAddressPort)
    #         time.sleep(self.sleep_time)


    def send_command(self):
        """Send commands via MQTT if enabled (preferred), else UDP fallback."""
        while self.running:
            try:
                if not self.command_queue.empty():
                    msg = self.command_queue.get()

                    # --- Prefer MQTT if configured ---
                    if self.use_mqtt and self.mqtt_client is not None:
                        
                        # print(f"MQTT publish: {msg}")
                        # Make sure we're connected; if not, try a quick reconnect
                        if not self.mqtt_client.is_connected():
                            try:
                                self.mqtt_client.reconnect()
                            except Exception:
                                pass  # fall through to UDP fallback

                        if self.mqtt_client.is_connected():
                            ec, ssvep = msg.split("|", 1)
                            if ec == "EyesClosed":
                                mqtt_msg = "x"
                            else:
                                mqtt_msg = "w"
                                # if ssvep == "6Hz":
                                #     mqtt_msg = "w"
                                # elif ssvep == "15Hz":
                                #     mqtt_msg = "s"

                            # raw string publish
                            self.mqtt_client.publish(
                                self.mqtt_topic,
                                mqtt_msg if isinstance(mqtt_msg, (str, bytes, bytearray)) else str(mqtt_msg),
                                qos=self.mqtt_qos,
                                retain=self.mqtt_retain
                            )
                            # continue to next loop tick
                            time.sleep(self.sleep_time)
                            continue

            except Exception as e:
                print(f"send_command error: {e}")

            time.sleep(self.sleep_time)


                
    ##### Helper functions for SSVEP feature extraction #####

    def enable_mqtt(self,
                    host="127.0.0.1",
                    port=1883,
                    topic="eeg/command",
                    client_id="eeg-publisher-1",
                    username=None,
                    password=None,
                    tls=False):
        """Call this once to turn on MQTT publishing."""
        self.mqtt_topic = topic
        self.use_mqtt = True

        self.mqtt_client = mqtt.Client(client_id=client_id, clean_session=True, transport="tcp")
        if username and password:
            self.mqtt_client.username_pw_set(username, password)
        if tls:
            import ssl
            self.mqtt_client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)

        # Optional: LWT to indicate this publisher went offline unexpectedly
        self.mqtt_client.will_set(f"{topic}/status", payload="offline", qos=1, retain=True)

        # Lightweight callbacks (debug/robustness)
        def _on_connect(client, userdata, flags, rc, properties=None):
            if rc == mqtt.MQTT_ERR_SUCCESS or rc == 0:
                # mark online (retained)
                client.publish(f"{topic}/status", "online", qos=1, retain=True)

        def _on_disconnect(client, userdata, rc, properties=None):
            # Optionally log or try reconnect inside send loop
            pass

        self.mqtt_client.on_connect = _on_connect
        self.mqtt_client.on_disconnect = _on_disconnect
        # Connect and start background loop
        self.mqtt_client.connect(host, port, keepalive=60)
        self.mqtt_client.loop_start()

        time.sleep(1.0)  # Give some time to connect

    def _ema(self, prev, x, a):
        return x if prev is None else (1.0 - a) * prev + a * x

    def _robust_z(self, x, hist, eps=1e-9):
        """Robust z-score using running median/MAD from history (which includes x)."""
        hist.append(float(x))
        arr = np.asarray(hist, dtype=float)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + eps
        return (x - med) / (1.4826 * mad + eps)

    def _band_idx(self, xf, lo, hi):
        return np.where((xf >= lo) & (xf <= hi))[0]

    def _bandpower(self, ps_avg, xf, lo, hi):
        idx = self._band_idx(xf, lo, hi)
        if idx.size == 0: return 0.0
        return float(ps_avg[idx].mean())

    def _snr_multiharm(self, ps_avg, xf, f0, bw=0.7, noise_bw=0.7, noise_off=1.5, max_harm=3):
        """Sum power in ±bw around each harmonic / sum power in two side noise bands."""
        nyq = xf[-1]
        num = 0.0; den = 0.0
        for k in range(1, max_harm + 1):
            fk = k * f0
            if fk + bw > nyq: break
            # target band
            tidx = self._band_idx(xf, fk - bw, fk + bw)
            num += ps_avg[tidx].sum()
            # side noise bands
            lidx = self._band_idx(xf, fk - noise_off - noise_bw, fk - noise_off)
            ridx = self._band_idx(xf, fk + noise_off, fk + noise_off + noise_bw)
            # fallback if empty (low freq edges)
            if lidx.size == 0 and ridx.size == 0:
                lidx = self._band_idx(xf, max(0.0, fk - 2.5*bw), fk - 1.5*bw)
                ridx = self._band_idx(xf, fk + 1.5*bw, fk + 2.5*bw)
            den += ps_avg[lidx].sum() + ps_avg[ridx].sum()
        return num / max(den, 1e-9)
