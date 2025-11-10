#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Realtime SSVEP Pipeline (LSL + Pygame + FBCCA/eCCA)
# Frequencies: 6, 14, 20 Hz
# Threads: EEGInlet / Stimulus / Backend
# Modes: 'C' calibrate (4x1.2 s), 'G' game, ESC quit

import sys, time, math, threading, queue, collections, numpy as np
import pygame
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock
from scipy.signal import butter, filtfilt, iirnotch

# ---------------- Config ----------------
FREQS = [6.0, 14.0, 20.0]
LABELS = {0: '6 Hz', 1: '14 Hz', 2: '20 Hz'}
HARMONICS = [1, 2, 3]
BANDS = [(6, 25), (12, 40), (18, 60)]
BAND_WEIGHTS = np.array([1.0, 0.6, 0.4])
NOTCH_HZ = 50.0
WINDOW_SEC = 1.2
STRIDE_SEC = 0.3
CAL_TRIALS_PER_CLASS = 4
CAL_REST_SEC = 0.8
FS_FALLBACK = 250.0
RING_SEC = 20.0
CHANNELS_TO_USE = None  # e.g., [0,1,2] to select channels
# ---------------------------------------

def design_filters(fs):
    def bp(low, high, fs, order=4):
        ny = 0.5 * fs
        b, a = butter(order, [low/ny, high/ny], btype='bandpass')
        return b, a
    bp_filters = [bp(lo, hi, fs) for (lo, hi) in BANDS]
    Q = 30.0
    b_notch, a_notch = iirnotch(NOTCH_HZ/(0.5*fs), Q)
    return bp_filters, (b_notch, a_notch)

def apply_filters(x, filters):
    (b_notch, a_notch), bp_filters = filters[1], filters[0]
    xn = filtfilt(b_notch, a_notch, x, axis=0, method='gust')
    bands = []
    for (b, a) in bp_filters:
        bands.append(filtfilt(b, a, xn, axis=0, method='gust'))
    return bands

def make_refs(freqs, harmonics, fs, n):
    t = np.arange(n) / fs
    refs = []
    for f in freqs:
        waves = []
        for k in harmonics:
            waves.append(np.sin(2*np.pi*k*f*t))
            waves.append(np.cos(2*np.pi*k*f*t))
        refs.append(np.stack(waves, axis=0))
    return np.array(refs)

def cca_corr(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Ux, Sx, VTx = np.linalg.svd(Xc, full_matrices=False)
    Uy, Sy, VTy = np.linalg.svd(Yc, full_matrices=False)
    K = Ux.T @ Uy
    s = np.linalg.svd(K, compute_uv=False)
    return s[0] if s.size else 0.0

def fbcca_score(epoch, fs, filters, refs_per_band):
    bands = apply_filters(epoch, filters)
    scores = np.zeros(len(FREQS))
    for bi, xb in enumerate(bands):
        for fi in range(len(FREQS)):
            Y = refs_per_band[bi][fi].T
            scores[fi] += BAND_WEIGHTS[bi] * cca_corr(xb, Y)
    return scores

def ecca_score(epoch, fs, filters, refs_per_band, templates):
    bands = apply_filters(epoch, filters)
    scores = np.zeros(len(FREQS))
    for bi, xb in enumerate(bands):
        for fi in range(len(FREQS)):
            Y_ref = refs_per_band[bi][fi].T
            Tmpl = templates[fi]
            s1 = cca_corr(xb, Y_ref)
            s2 = cca_corr(xb, Tmpl)
            s3 = cca_corr(Tmpl, Y_ref)
            scores[fi] += BAND_WEIGHTS[bi] * (s1 + s2 + s3)
    return scores

class EEGInlet(threading.Thread):
    def __init__(self, ring_sec=RING_SEC):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.fs = None
        self.nch = None
        self.buffer = None
        self.lock = threading.Lock()
    def run(self):
        print('[EEGInlet] Resolving EEG stream...')
        streams = resolve_stream('type', 'EEG', timeout=10)
        if not streams:
            print('[EEGInlet] No EEG LSL stream found. Exiting thread.')
            return
        inlet = StreamInlet(streams[0], max_buflen=RING_SEC*2, recover=True)
        info = inlet.info()
        fs = info.nominal_srate()
        nch = info.channel_count()
        if fs <= 0: fs = FS_FALLBACK
        self.fs, self.nch = fs, nch
        cap = int(RING_SEC * fs)
        self.buffer = collections.deque(maxlen=cap)
        print(f'[EEGInlet] Connected: fs={fs:.1f} Hz, nch={nch}')
        while not self.stop_flag.is_set():
            samples, timestamps = inlet.pull_chunk(timeout=0.1)
            if timestamps:
                with self.lock:
                    for ts, s in zip(timestamps, samples):
                        self.buffer.append((ts, np.asarray(s, dtype=float)))
    def get_window(self, t_end, sec):
        if self.buffer is None: return None, None
        with self.lock:
            arr_ts = np.array([ts for ts, _ in self.buffer], dtype=float)
            if arr_ts.size == 0: return None, None
            fs = self.fs or FS_FALLBACK
            n = int(round(sec * fs))
            idx = np.where(arr_ts <= t_end)[0]
            if idx.size < n: return None, None
            end_i = idx[-1]; beg_i = end_i - n + 1
            if beg_i < 0: return None, None
            X = np.stack([self.buffer[i][1] for i in range(beg_i, end_i+1)], axis=0)
            if CHANNELS_TO_USE is not None: X = X[:, CHANNELS_TO_USE]
            return X, (beg_i, end_i)
    def stop(self):
        self.stop_flag.set()

class MarkerBus(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'ssvep_markers')
        self.outlet = StreamOutlet(info)
    def push(self, text):
        ts = local_clock()
        self.outlet.push_sample([text], ts)
        self.q.put((ts, text))
    def run(self):
        while not self.stop_flag.is_set():
            time.sleep(0.1)
    def stop(self):
        self.stop_flag.set()

class Backend(threading.Thread):
    def __init__(self, inlet, markers):
        super().__init__(daemon=True)
        self.inlet = inlet
        self.markers = markers
        self.stop_flag = threading.Event()
        self.mode = 'idle'
        self.cal_data = {i: [] for i in range(len(FREQS))}
        self.templates = None
        self.last_decisions = collections.deque(maxlen=3)
        self.filters = None
        self.refs_per_band = None
    def _prepare_refs_filters(self):
        fs = self.inlet.fs or FS_FALLBACK
        n = int(round(WINDOW_SEC * fs))
        refs = make_refs(FREQS, HARMONICS, fs, n)
        refs_per_band = [refs for _ in range(len(BANDS))]
        bp_filters, notch = design_filters(fs)
        self.filters = (bp_filters, notch)
        self.refs_per_band = refs_per_band
    def run(self):
        while not self.stop_flag.is_set() and (self.inlet.fs is None):
            time.sleep(0.05)
        if self.stop_flag.is_set(): return
        self._prepare_refs_filters()
        print('[Backend] Ready. Press C to calibrate, G to start game.')
        while not self.stop_flag.is_set():
            try: ts, m = self.markers.q.get(timeout=0.05)
            except queue.Empty: continue
            if m.startswith('CAL_START_'):
                self.mode = 'calibrating'
                cls = int(m.split('_')[-1])
                self.current_class = cls
            elif m == 'CAL_END':
                X, _ = self.inlet.get_window(ts, WINDOW_SEC)
                if X is not None: self.cal_data[self.current_class].append(X)
                self.current_class = None
            elif m == 'CAL_DONE':
                self._build_templates(); self.mode = 'idle'
                print('[Backend] Calibration complete. Templates built for eCCA.')
            elif m == 'GAME_ON':
                self.mode = 'online'; print('[Backend] Online decoding ON.')
            elif m == 'GAME_OFF':
                self.mode = 'idle'; print('[Backend] Online decoding OFF.')
            elif m == 'TICK':
                if self.mode == 'online': self._online_step(ts)
    def _build_templates(self):
        self.templates = []
        any_class = next((k for k in self.cal_data if len(self.cal_data[k]) > 0), None)
        zero_shape = None
        if any_class is not None: zero_shape = np.array(self.cal_data[any_class][0]).shape
        for i in range(len(FREQS)):
            if len(self.cal_data[i]) == 0:
                if zero_shape is None:
                    zero_shape = (int(round((self.inlet.fs or FS_FALLBACK)*WINDOW_SEC)), self.inlet.nch or 1)
                self.templates.append(np.zeros(zero_shape))
            else:
                self.templates.append(np.mean(np.stack(self.cal_data[i], axis=0), axis=0))
    def _online_step(self, t_now):
        fs = self.inlet.fs or FS_FALLBACK
        X, _ = self.inlet.get_window(t_now, WINDOW_SEC)
        if X is None: return
        if self.templates is None:
            scores = fbcca_score(X, fs, (self.filters[0], self.filters[1]), self.refs_per_band)
        else:
            scores = ecca_score(X, fs, (self.filters[0], self.filters[1]), self.refs_per_band, self.templates)
        fi = int(np.argmax(scores))
        self.last_decisions.append(fi)
        if len(self.last_decisions) == self.last_decisions.maxlen:
            vote = np.bincount(np.array(self.last_decisions)).argmax()
            top2 = np.sort(scores)[-2:]
            margin = top2[-1] - top2[-2] if top2.size == 2 else 0.0
            if margin > 0.1:
                print(f'[DECISION] {LABELS[vote]}  (scores={np.round(scores,3)})')
    def stop(self):
        self.stop_flag.set()

class Stimulus(threading.Thread):
    def __init__(self, markers, backend):
        super().__init__(daemon=True)
        self.markers = markers
        self.backend = backend
        self.stop_flag = threading.Event()
    def run(self):
        if pygame is None:
            print('[Stimulus] pygame not installed. pip install pygame')
            return
        pygame.init()
        screen = pygame.display.set_mode((1000, 600))
        pygame.display.set_caption('SSVEP 6 | 14 | 20 Hz  (C=Calibrate, G=Game, ESC=Quit)')
        clock = pygame.time.Clock()
        rects = [
            pygame.Rect(120, 200, 200, 200),
            pygame.Rect(400, 200, 200, 200),
            pygame.Rect(680, 200, 200, 200),
        ]
        last_tick = local_clock()
        game_on = False
        cal_schedule = None
        cal_active = False
        cal_idx = -1
        while not self.stop_flag.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.stop_flag.set()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self.stop_flag.set()
                    elif event.key == pygame.K_c:
                        cal_schedule = self._build_cal_schedule(); cal_active = True; cal_idx = -1
                        print('[Stimulus] Calibration scheduled.')
                    elif event.key == pygame.K_g:
                        game_on = not game_on; self.markers.push('GAME_ON' if game_on else 'GAME_OFF')
            now = local_clock()
            if (now - last_tick) >= STRIDE_SEC:
                last_tick = now; self.markers.push('TICK')
            screen.fill((25, 25, 25))
            if cal_active and cal_schedule:
                if cal_idx == -1 or cal_schedule[cal_idx]['end'] <= now:
                    cal_idx += 1
                    if cal_idx >= len(cal_schedule):
                        cal_active = False; cal_schedule = None; self.markers.push('CAL_DONE')
                        print('[Stimulus] Calibration finished.')
                    else:
                        entry = cal_schedule[cal_idx]
                        if entry['type'] == 'stim': self.markers.push(f'CAL_START_{entry['cls']}')
                        elif entry['type'] == 'rest': self.markers.push('CAL_END')
                if cal_active and cal_schedule and cal_schedule[cal_idx]['type'] == 'stim':
                    cls = cal_schedule[cal_idx]['cls']
                    self._draw_flickers(screen, rects, now, highlight=cls)
                else:
                    self._draw_flickers(screen, rects, now, highlight=None)
            else:
                self._draw_flickers(screen, rects, now, highlight=None)
            self._draw_text(screen, 'C: calibrate (4x1.2s each)   G: toggle game   ESC: quit', 18, (255,255,255), (20, 20))
            if game_on: self._draw_text(screen, 'ONLINE DECODING: ON', 22, (80,255,80), (20, 50))
            pygame.display.flip(); clock.tick(120)
        pygame.quit()
    def _build_cal_schedule(self):
        # Randomized 3-class x N trials schedule, each 1.2s stim + 0.8s rest
        blocks = []
        now = local_clock()
        order = np.tile(np.arange(3), CAL_TRIALS_PER_CLASS)
        np.random.shuffle(order)
        t = now + 1.0
        for cls in order:
            blocks.append({'type': 'stim', 'cls': int(cls), 'beg': t, 'end': t + WINDOW_SEC}); t += WINDOW_SEC
            blocks.append({'type': 'rest', 'beg': t, 'end': t + CAL_REST_SEC}); t += CAL_REST_SEC
        return blocks
    def _draw_flickers(self, screen, rects, now, highlight=None):
        for i, rect in enumerate(rects):
            f = FREQS[i]
            lum = 0.5*(1.0 + math.sin(2*math.pi*f*now))
            c = int(40 + 215*lum)
            col = (c, c, c)
            pygame.draw.rect(screen, col, rect)
            pygame.draw.rect(screen, (100, 100, 100), rect, 4)
            if highlight == i: pygame.draw.rect(screen, (0, 200, 255), rect, 8)
            txt = f'{LABELS[i]}'
            self._draw_text(screen, txt, 24, (10,10,10) if c>128 else (230,230,230), (rect.x+50, rect.y+85))
    def _draw_text(self, screen, text, size, color, pos):
        font = pygame.font.SysFont(None, size)
        img = font.render(text, True, color)
        screen.blit(img, pos)
    def stop(self):
        self.stop_flag.set()

def main():
    print('=== Realtime SSVEP Pipeline (6/14/20 Hz) ===')
    print('Requirements: pip install pylsl pygame numpy scipy')
    inlet = EEGInlet(); markers = MarkerBus(); backend = Backend(inlet, markers); stim = Stimulus(markers, backend)
    inlet.start(); markers.start(); backend.start(); stim.start()
    try:
        while stim.is_alive(): time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stim.stop(); backend.stop(); markers.stop(); inlet.stop(); print('Shutting down...')

if __name__ == '__main__':
    main()