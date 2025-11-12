#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SSVEP UI (static layout, responsive fullscreen)
# - Define layout in BASE coordinates; app scales & centers for any window/fullscreen.
# - Modes: FIT (letterbox), FILL (crop), STRETCH (distort).

import json, math, sys, os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame
from pylsl import StreamInfo, StreamOutlet, local_clock

# ---------------- User configuration (EDIT HERE) ----------------
START_FULLSCREEN = False
WINDOW_SIZE = (1200, 700)         # Base windowed size
BG_COLOR = (25, 25, 25)

# Base design space (all rects defined in these coordinates)
BASE_SIZE = (1200, 700)

# How to scale from BASE_SIZE to the actual window: 'FIT' | 'FILL' | 'STRETCH'
LAYOUT_MODE = 'FIT'

# Calibration & timing
CAL_TRIALS_PER_CLASS = 4
WINDOW_SEC = 1.2
REST_SEC = 0.8
STRIDE_SEC = 0.3

# Define your blocks in BASE coordinates: [x, y, w, h]
BLOCKS = [
    { 'label': '6 Hz',  'freq': 6.0,  'rect': [0, 50, 220, 220] },
    # { 'label': '15 Hz', 'freq': 15.0, 'rect': [490, 480, 220, 220] },
    { 'label': '15 Hz', 'freq': 15.0, 'rect': [980, 50, 220, 220] },
]
# ----------------------------------------------------------------

@dataclass
class Block:
    label: str
    freq: float
    rect: Tuple[int, int, int, int]  # base coords
    def as_rect(self):
        return pygame.Rect(*self.rect)

def compute_transform(win_w, win_h, base_w, base_h, mode):
    sx = win_w / base_w
    sy = win_h / base_h
    mode = (mode or 'FIT').upper()
    if mode == 'FIT':
        s = min(sx, sy)
        sx = sy = s
        ox = (win_w - base_w * s) * 0.5
        oy = (win_h - base_h * s) * 0.5
    elif mode == 'FILL':
        s = max(sx, sy)
        sx = sy = s
        ox = (win_w - base_w * s) * 0.5
        oy = (win_h - base_h * s) * 0.5
    else:  # STRETCH
        ox = 0.0; oy = 0.0
    return sx, sy, ox, oy

def transform_rect(r, sx, sy, ox, oy):
    x = int(round(r.x * sx + ox))
    y = int(round(r.y * sy + oy))
    w = int(round(r.w * sx))
    h = int(round(r.h * sy))
    return pygame.Rect(x, y, max(1, w), max(1, h))

class MarkerBus:
    def __init__(self):
        info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'ssvep_ui_markers')
        self.outlet = StreamOutlet(info)
    def push(self, text):
        ts = local_clock()
        self.outlet.push_sample([text], ts)
        print(f'[MARKER {ts:.3f}] {text}')

class UI:
    def __init__(self, blocks: List[Block]):
        TARGET_DISPLAY = 1          # 0 = primary, 1 = second monitor, etc.
        os.environ["SDL_VIDEO_FULLSCREEN_DISPLAY"] = str(TARGET_DISPLAY)
        pygame.init()
        self.fullscreen = START_FULLSCREEN
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        size = (pygame.display.Info().current_w, pygame.display.Info().current_h) if self.fullscreen else WINDOW_SIZE
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption(f'SSVEP UI (static, {LAYOUT_MODE})')
        self.clock = pygame.time.Clock()
        self.font_med = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 18)

        self.blocks = blocks
        self.sync_on = True
        self.sync_t0 = local_clock()
        self.game_on = False
        self.last_tick = local_clock()
        self.markers = MarkerBus()

        self.cal_schedule = None
        self.cal_active = False
        self.cal_idx = -1

        self.base_w, self.base_h = BASE_SIZE

    def build_cal_schedule(self):
        now = local_clock()
        order = np.tile(np.arange(len(self.blocks)), CAL_TRIALS_PER_CLASS)
        np.random.shuffle(order)
        t = now + 1.0
        plan = []
        for cls in order:
            plan.append({'kind':'stim','cls':int(cls),'beg':t,'end':t+WINDOW_SEC}); t += WINDOW_SEC
            plan.append({'kind':'rest','cls':None,'beg':t,'end':t+REST_SEC}); t += REST_SEC
        return plan

    def draw_text_center(self, text, center_xy, color):
        img = self.font_med.render(text, True, color)
        rect = img.get_rect(center=center_xy)
        self.screen.blit(img, rect.topleft)

    def draw_blocks(self, now):
        win_w, win_h = self.screen.get_size()
        sx, sy, ox, oy = compute_transform(win_w, win_h, self.base_w, self.base_h, LAYOUT_MODE)
        for idx, b in enumerate(self.blocks):
            br = b.as_rect()
            rr = transform_rect(br, sx, sy, ox, oy)
            t_ref = (now - self.sync_t0) if self.sync_on else now
            lum = 0.5 * (1.0 + math.sin(2*math.pi*b.freq*t_ref))
            c = int(40 + 215*lum)
            col = (c, c, c)
            pygame.draw.rect(self.screen, col, rr)
            pygame.draw.rect(self.screen, (100,100,100), rr, 3)
            if self.cal_active and self.cal_schedule and self.cal_schedule[self.cal_idx]['kind'] == 'stim' and self.cal_schedule[self.cal_idx]['cls'] == idx:
                pygame.draw.rect(self.screen, (0,200,255), rr, 6)
            fg = (10,10,10) if c>128 else (235,235,235)
            # self.draw_text_center(b.label, (rr.centerx, rr.centery), fg)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    elif event.key == pygame.K_f:
                        self.fullscreen = not self.fullscreen
                        flags = pygame.FULLSCREEN if self.fullscreen else 0
                        size = (pygame.display.Info().current_w, pygame.display.Info().current_h) if self.fullscreen else WINDOW_SIZE
                        self.screen = pygame.display.set_mode(size, flags)
                    elif event.key == pygame.K_s:
                        self.sync_on = not self.sync_on
                        if self.sync_on: self.sync_t0 = local_clock()
                        print(f"[SYNC] Phase-lock: {'ON' if self.sync_on else 'OFF'}")
                    elif event.key == pygame.K_g:
                        self.game_on = not self.game_on
                        self.markers.push('GAME_ON' if self.game_on else 'GAME_OFF')
                    elif event.key == pygame.K_c:
                        self.cal_schedule = self.build_cal_schedule(); self.cal_active = True; self.cal_idx = -1
                        if self.sync_on: self.sync_t0 = local_clock()
                        print('[CAL] Calibration scheduled.')

            now = local_clock()
            if (now - self.last_tick) >= STRIDE_SEC:
                self.last_tick = now; self.markers.push('TICK')

            self.screen.fill(BG_COLOR)

            if self.cal_active and self.cal_schedule is not None:
                if self.cal_idx == -1 or self.cal_schedule[self.cal_idx]['end'] <= now:
                    self.cal_idx += 1
                    if self.cal_idx >= len(self.cal_schedule):
                        self.cal_active = False; self.cal_schedule = None
                        self.markers.push('CAL_DONE'); print('[CAL] Calibration finished.')
                    else:
                        blk = self.cal_schedule[self.cal_idx]
                        if blk['kind'] == 'stim':
                            if self.sync_on: self.sync_t0 = blk['beg']
                            cls = blk['cls']
                            self.markers.push(f'CAL_START_{cls}')
                            self.markers.push(f'CAL_META_FREQ_{self.blocks[cls].freq:.2f}')
                        else:
                            self.markers.push('CAL_END')

            self.draw_blocks(now)

            hud = f"C: calibrate ({CAL_TRIALS_PER_CLASS}x{WINDOW_SEC:.1f}s)  G: game  S: sync  F: fullscreen  ESC: quit  [{LAYOUT_MODE}]"
            self.screen.blit(self.font_small.render(hud, True, (255,255,255)), (16,16))
            if self.game_on:
                self.screen.blit(self.font_small.render('ONLINE (GAME): ON', True, (80,255,80)), (16, 40))

            pygame.display.flip(); self.clock.tick(120)

        pygame.quit()

def load_blocks_from_json(path: str) -> List[Block]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return [Block(label=str(d['label']), freq=float(d['freq']), rect=tuple(d['rect'])) for d in raw]

def main():
    if len(sys.argv) > 1:
        blocks = load_blocks_from_json(sys.argv[1])
        print(f'[CONFIG] Loaded layout from {sys.argv[1]} in BASE space {BASE_SIZE}')
    else:
        blocks = [Block(label=b['label'], freq=float(b['freq']), rect=tuple(b['rect'])) for b in BLOCKS]
        print(f'[CONFIG] Using BLOCKS from source in BASE space {BASE_SIZE}')
    ui = UI(blocks); ui.run()

if __name__ == '__main__':
    main()