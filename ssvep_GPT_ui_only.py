#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SSVEP UI (static layout) - set block positions/sizes BEFORE running.
# - No runtime editing. Reads from constants below or optional JSON file.
# - Emits LSL markers for calibration/game and a periodic TICK.

import json
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame
from pylsl import StreamInfo, StreamOutlet, local_clock

# ---------------- User configuration (edit here) ----------------
START_FULLSCREEN = False          # True to start fullscreen
SCREEN_SIZE = (1200, 700)         # used if START_FULLSCREEN is False
BG_COLOR = (25, 25, 25)

# Calibration & timing
CAL_TRIALS_PER_CLASS = 4          # number of trials per block
WINDOW_SEC = 1.2                  # stimulus duration per trial
REST_SEC = 0.8                    # rest between trials
STRIDE_SEC = 0.3                  # period for TICK markers

# Define your blocks here: label, frequency (Hz), rect = [x, y, w, h]
BLOCKS = [
    { 'label': '7 Hz',  'freq': 7.0,  'rect': [120, 240, 220, 220] },
    { 'label': '15 Hz', 'freq': 15.0, 'rect': [460, 240, 220, 220] },
    { 'label': '20 Hz', 'freq': 20.0, 'rect': [800, 240, 220, 220] },
]

@dataclass
class Block:
    label: str
    freq: float
    rect: Tuple[int, int, int, int]  # x,y,w,h
    def to_rect(self) -> pygame.Rect:
        x, y, w, h = self.rect
        return pygame.Rect(int(x), int(y), int(w), int(h))

class MarkerBus:
    def __init__(self):
        info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'ssvep_ui_markers')
        self.outlet = StreamOutlet(info)
    def push(self, text: str):
        ts = local_clock()
        self.outlet.push_sample([text], ts)
        print('[MARKER %.3f] %s' % (ts, text))

class UI:
    def __init__(self, blocks: List[Block]):
        pygame.init()
        self.fullscreen = START_FULLSCREEN
        size = (pygame.display.Info().current_w, pygame.display.Info().current_h) if self.fullscreen else SCREEN_SIZE
        flags = pygame.FULLSCREEN if self.fullscreen else 0
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption('SSVEP UI - static layout')
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont(None, 18)
        self.font_med = pygame.font.SysFont(None, 24)

        self.blocks = blocks
        self.sync_on = True          # phase-locked luminance timing
        self.sync_t0 = local_clock()
        self.game_on = False
        self.last_tick = local_clock()
        self.markers = MarkerBus()

        self.cal_schedule = None     # list of dicts with keys: kind, cls, beg, end
        self.cal_active = False
        self.cal_idx = -1

    # -------------- Calibration schedule --------------
    def build_cal_schedule(self):
        now = local_clock()
        order = np.tile(np.arange(len(self.blocks)), CAL_TRIALS_PER_CLASS)
        np.random.shuffle(order)
        t = now + 1.0
        plan = []
        for cls in order:
            plan.append({'kind': 'stim', 'cls': int(cls), 'beg': t, 'end': t + WINDOW_SEC}); t += WINDOW_SEC
            plan.append({'kind': 'rest', 'cls': None, 'beg': t, 'end': t + REST_SEC}); t += REST_SEC
        return plan

    # -------------- Drawing --------------
    def draw_text(self, text, size, color, pos):
        font = pygame.font.SysFont(None, size)
        img = font.render(text, True, color)
        self.screen.blit(img, pos)

    def draw_blocks(self, now: float, highlight_cls: Optional[int] = None):
        for i, b in enumerate(self.blocks):
            r = b.to_rect()
            t_ref = (now - self.sync_t0) if self.sync_on else now
            lum = 0.5 * (1.0 + math.sin(2 * math.pi * b.freq * t_ref))
            c = int(40 + 215 * lum)
            col = (c, c, c)
            pygame.draw.rect(self.screen, col, r)
            pygame.draw.rect(self.screen, (100, 100, 100), r, 3)
            if highlight_cls is not None and i == highlight_cls:
                pygame.draw.rect(self.screen, (0, 200, 255), r, 6)
            # fg = (10, 10, 10) if c > 128 else (235, 235, 235)
            # text_pos = (r.x + r.w // 2 - 28, r.y + r.h // 2 - 12)
            # self.draw_text('%s' % b.label, 24, fg, text_pos)

    # -------------- Main loop --------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_f:
                        # Toggle fullscreen for convenience (layout is static)
                        self.fullscreen = not self.fullscreen
                        size = (pygame.display.Info().current_w, pygame.display.Info().current_h) if self.fullscreen else SCREEN_SIZE
                        flags = pygame.FULLSCREEN if self.fullscreen else 0
                        self.screen = pygame.display.set_mode(size, flags)
                    elif event.key == pygame.K_s:
                        # toggle phase-lock
                        self.sync_on = not self.sync_on
                        if self.sync_on:
                            self.sync_t0 = local_clock()
                        print('[SYNC] Phase-lock: %s' % ('ON' if self.sync_on else 'OFF'))
                    elif event.key == pygame.K_g:
                        # Toggle GAME markers
                        self.game_on = not self.game_on
                        self.markers.push('GAME_ON' if self.game_on else 'GAME_OFF')
                    elif event.key == pygame.K_c:
                        # Start calibration run
                        self.cal_schedule = self.build_cal_schedule()
                        self.cal_active = True
                        self.cal_idx = -1
                        if self.sync_on:
                            self.sync_t0 = local_clock()
                        print('[CAL] Calibration scheduled.')

            now = local_clock()
            # Emit TICK for backends
            if (now - self.last_tick) >= STRIDE_SEC:
                self.last_tick = now
                self.markers.push('TICK')

            # Update calibration state and draw
            self.screen.fill(BG_COLOR)
            if self.cal_active and self.cal_schedule is not None:
                if self.cal_idx == -1 or self.cal_schedule[self.cal_idx]['end'] <= now:
                    self.cal_idx += 1
                    if self.cal_idx >= len(self.cal_schedule):
                        self.cal_active = False
                        self.cal_schedule = None
                        self.markers.push('CAL_DONE')
                        print('[CAL] Calibration finished.')
                    else:
                        blk = self.cal_schedule[self.cal_idx]
                        if blk['kind'] == 'stim':
                            if self.sync_on:
                                self.sync_t0 = blk['beg']
                            cls = blk['cls']
                            self.markers.push('CAL_START_%d' % cls)
                            self.markers.push('CAL_META_FREQ_%.2f' % self.blocks[cls].freq)
                        else:
                            self.markers.push('CAL_END')
                if self.cal_active and self.cal_schedule and self.cal_schedule[self.cal_idx]['kind'] == 'stim':
                    cls = self.cal_schedule[self.cal_idx]['cls']
                    self.draw_blocks(now, highlight_cls=cls)
                else:
                    self.draw_blocks(now, highlight_cls=None)
            else:
                self.draw_blocks(now, highlight_cls=None)

            # HUD
            hud1 = 'C: calibrate (%dx%.1fs)  G: game  S: sync  F: fullscreen  ESC: quit' % (CAL_TRIALS_PER_CLASS, WINDOW_SEC)
            self.draw_text(hud1, 18, (255, 255, 255), (16, 16))
            if self.game_on:
                self.draw_text('ONLINE (GAME): ON', 22, (80, 255, 80), (16, 40))

            pygame.display.flip()
            self.clock.tick(120)

        pygame.quit()

def load_blocks_from_json(path: str) -> List[Block]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    blocks = []
    for d in raw:
        label = d['label']
        freq = float(d['freq'])
        x, y, w, h = d['rect']
        blocks.append(Block(label=label, freq=freq, rect=(int(x), int(y), int(w), int(h))))
    return blocks

def main():
    # Build block list from constants or JSON override
    if len(sys.argv) > 1:
        blocks = load_blocks_from_json(sys.argv[1])
        print('[CONFIG] Loaded layout from %s' % sys.argv[1])
    else:
        blocks = [Block(label=b['label'], freq=float(b['freq']), rect=tuple(b['rect'])) for b in BLOCKS]
        print('[CONFIG] Using BLOCKS from source file.')

    ui = UI(blocks)
    ui.run()

if __name__ == '__main__':
    main()