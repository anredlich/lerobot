#!/usr/bin/env python
"""
LeRobot Dataset Splitter GUI
=============================

A self-contained GUI tool for splitting LeRobot v2.1 episodes into
multiple episodes with different task labels.

Supports splitting multiple source episodes in one run, combining
all the resulting segments into a single output dataset.

Usage:
    python dataset_splitter.py
    python dataset_splitter.py --source /path/to/dataset

Safety:
    - The output path is BLOCKED from being the same as (or inside) the input path.
    - A confirmation dialog is shown before any write operation.
    - The input dataset is only ever opened for reading.

Requirements:
    - Python 3.10+
    - tkinter  (ships with most Python installations)
    - pandas + pyarrow  (pip install pandas pyarrow)
    - ffmpeg on PATH     (for video trimming)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("dataset_splitter")


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskSegment:
    start_frame: int
    end_frame: int
    task: str

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame

    def __post_init__(self):
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {self.start_frame}")
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) must be > start_frame ({self.start_frame})"
            )


@dataclass
class _NewEpisode:
    new_episode_index: int
    source_episode_index: int
    source_start_frame: int
    source_end_frame: int
    task: str
    length: int = field(init=False)

    def __post_init__(self):
        self.length = self.source_end_frame - self.source_start_frame


# ═══════════════════════════════════════════════════════════════════════════
# Splitter engine  (read-only on source, writes only to output)
# ═══════════════════════════════════════════════════════════════════════════

class EpisodeSplitter:
    """Splits LeRobot episodes and writes a complete new dataset."""

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _save_json(data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        entries: list[dict] = []
        if path.exists():
            with open(path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        entries.append(json.loads(stripped))
        return entries

    @staticmethod
    def _save_jsonl(data: list[dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    # --- construction ------------------------------------------------------

    def __init__(self, source_root: Path, output_root: Path,
                 ffmpeg_path: str = "ffmpeg"):
        self.source_root = Path(source_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.ffmpeg_path = ffmpeg_path

        if not self.source_root.exists():
            raise FileNotFoundError(f"Source not found: {self.source_root}")

        # ── SAFETY: never allow output == source or output inside source ──
        try:
            self.output_root.relative_to(self.source_root)
            raise ValueError(
                "Output path is inside the source dataset. "
                "Choose a different output path to protect your data."
            )
        except ValueError as exc:
            if "protect your data" in str(exc):
                raise
        if self.output_root == self.source_root:
            raise ValueError("Output path must differ from source path.")

        meta = self.source_root / "meta"
        self.info = self._load_json(meta / "info.json")
        self.source_tasks = {
            e["task"]: e["task_index"]
            for e in self._load_jsonl(meta / "tasks.jsonl")
        }
        self.source_episodes = {
            e["episode_index"]: e
            for e in self._load_jsonl(meta / "episodes.jsonl")
        }

        self.new_tasks: dict[str, int] = {}
        self.new_episodes: list[_NewEpisode] = []
        self._next_ep = 0

        self.fps = self.info.get("fps", 30)
        self.chunks_size = self.info.get("chunks_size", 1000)
        self.video_keys = [
            k for k, v in self.info.get("features", {}).items()
            if v.get("dtype") == "video"
        ]

    # --- path helpers ------------------------------------------------------

    def _chunk(self, ep: int) -> int:
        return ep // self.chunks_size

    def _parquet(self, root: Path, ep: int) -> Path:
        return root / "data" / f"chunk-{self._chunk(ep):03d}" / f"episode_{ep:06d}.parquet"

    def _video(self, root: Path, ep: int, key: str) -> Path:
        return root / "videos" / f"chunk-{self._chunk(ep):03d}" / key / f"episode_{ep:06d}.mp4"

    # --- public API --------------------------------------------------------

    def _ensure_task(self, task: str) -> int:
        if task not in self.new_tasks:
            self.new_tasks[task] = len(self.new_tasks)
        return self.new_tasks[task]

    def split_episode(self, episode_index: int,
                      segments: list[TaskSegment]) -> list[int]:
        if episode_index not in self.source_episodes:
            raise ValueError(f"Episode {episode_index} not in source")
        ep_len = self.source_episodes[episode_index]["length"]
        for s in segments:
            if s.end_frame > ep_len:
                raise ValueError(
                    f"end_frame {s.end_frame} > episode length {ep_len}")

        indices: list[int] = []
        for s in sorted(segments, key=lambda x: x.start_frame):
            self._ensure_task(s.task)
            ne = _NewEpisode(self._next_ep, episode_index,
                             s.start_frame, s.end_frame, s.task)
            self.new_episodes.append(ne)
            indices.append(self._next_ep)
            self._next_ep += 1
        return indices

    def copy_episode_unchanged(self, episode_index: int) -> int:
        ep = self.source_episodes[episode_index]
        task = ep["tasks"][0] if ep.get("tasks") else "unknown"
        self._ensure_task(task)
        ne = _NewEpisode(self._next_ep, episode_index, 0, ep["length"], task)
        self.new_episodes.append(ne)
        idx = self._next_ep
        self._next_ep += 1
        return idx

    # --- write -------------------------------------------------------------

    def save(self, skip_videos: bool = False,
             progress_cb=None):
        """Write the new dataset.  *progress_cb(current, total, message)*."""
        if not self.new_episodes:
            raise ValueError("Nothing to save – add segments first.")

        total_steps = len(self.new_episodes) + 2          # +metadata +reindex
        done = 0

        def _tick(msg: str):
            nonlocal done
            done += 1
            if progress_cb:
                progress_cb(done, total_steps, msg)
            log.info(msg)

        self.output_root.mkdir(parents=True, exist_ok=True)

        all_stats: dict[int, dict] = {}
        total_frames = 0

        for ne in self.new_episodes:
            stats = self._write_parquet(ne)
            all_stats[ne.new_episode_index] = stats
            if not skip_videos:
                for vk in self.video_keys:
                    self._write_video(ne, vk)
            total_frames += ne.length
            _tick(f"Episode {ne.new_episode_index} written "
                  f"(src ep {ne.source_episode_index}, "
                  f"frames {ne.source_start_frame}-{ne.source_end_frame})")

        self._reindex_globals()
        _tick("Global indices updated")

        self._write_metadata(total_frames, all_stats)
        _tick("Metadata saved")

    # --- internal write helpers (only touch output_root) -------------------

    def _write_parquet(self, ne: _NewEpisode) -> dict:
        if not HAS_PANDAS:
            raise RuntimeError("pandas + pyarrow required (pip install pandas pyarrow)")
        src = self._parquet(self.source_root, ne.source_episode_index)
        dst = self._parquet(self.output_root, ne.new_episode_index)
        dst.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src)
        full = ne.source_start_frame == 0 and ne.length == len(df)
        df = df.copy() if full else df.iloc[ne.source_start_frame:ne.source_end_frame].copy()

        df["episode_index"] = ne.new_episode_index
        df["frame_index"] = np.arange(len(df))
        df["task_index"] = self.new_tasks[ne.task]
        df["index"] = np.arange(len(df))
        df["timestamp"] = df["frame_index"] / self.fps
        df.to_parquet(dst, index=False)

        return self._stats(df)

    @staticmethod
    def _stats(df: "pd.DataFrame") -> dict:
        """Compute per-column stats in LeRobot v2.1 format.

        Produces per-dimension lists for min/max/mean/std and a count field,
        matching the format used in ``episodes_stats.jsonl``.
        """
        out: dict = {}
        for col in df.columns:
            try:
                vals = df[col].values
                if hasattr(vals[0], "__len__"):
                    # Multi-dimensional column (e.g. action, observation.state)
                    s = np.stack(vals)
                    ndim = s.ndim
                    if ndim == 2:
                        # Shape (N, D) – per-dimension stats as flat lists
                        out[col] = dict(
                            min=np.min(s, axis=0).tolist(),
                            max=np.max(s, axis=0).tolist(),
                            mean=np.mean(s, axis=0).tolist(),
                            std=np.std(s, axis=0).tolist(),
                            count=[len(s)],
                        )
                    elif ndim >= 3:
                        # Image-like: shape (N, C, H, W) or similar
                        out[col] = dict(
                            min=np.min(s, axis=0).tolist(),
                            max=np.max(s, axis=0).tolist(),
                            mean=np.mean(s, axis=0).tolist(),
                            std=np.std(s, axis=0).tolist(),
                            count=[s.shape[0]],
                        )
                    else:
                        out[col] = dict(
                            min=[float(np.min(s))],
                            max=[float(np.max(s))],
                            mean=[float(np.mean(s))],
                            std=[float(np.std(s))],
                            count=[len(s)],
                        )
                elif np.issubdtype(type(vals[0]), np.number):
                    # Scalar column (timestamp, frame_index, etc.)
                    out[col] = dict(
                        min=[float(np.min(vals))],
                        max=[float(np.max(vals))],
                        mean=[float(np.mean(vals))],
                        std=[float(np.std(vals))],
                        count=[len(vals)],
                    )
            except Exception:
                pass
        return out

    def _write_video(self, ne: _NewEpisode, key: str):
        src = self._video(self.source_root, ne.source_episode_index, key)
        if not src.exists():
            return
        dst = self._video(self.output_root, ne.new_episode_index, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        t0 = ne.source_start_frame / self.fps
        dur = ne.length / self.fps
        subprocess.run([
            self.ffmpeg_path, "-y",
            "-i", str(src),
            "-ss", f"{t0:.6f}",
            "-t", f"{dur:.6f}",
            "-vf", "setpts=PTS-STARTPTS",
            "-c:v", "libsvtav1",
            "-pix_fmt", "yuv420p",
            "-g", "2",
            "-crf", "30",
            "-svtav1-params", "fast-decode=0",
            "-r", str(self.fps),
            "-frames:v", str(ne.length),
            "-an",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ], capture_output=True, check=True)

    def _reindex_globals(self):
        if not HAS_PANDAS:
            return
        gi = 0
        for ne in self.new_episodes:
            p = self._parquet(self.output_root, ne.new_episode_index)
            df = pd.read_parquet(p)
            df["index"] = np.arange(gi, gi + len(df))
            df.to_parquet(p, index=False)
            gi += len(df)

    def _write_metadata(self, total_frames: int, ep_stats: dict):
        meta = self.output_root / "meta"

        info = self.info.copy()
        total_videos = len(self.new_episodes) * len(self.video_keys)
        info.update(total_episodes=len(self.new_episodes),
                    total_tasks=len(self.new_tasks),
                    total_frames=total_frames,
                    total_videos=total_videos,
                    total_chunks=self._chunk(len(self.new_episodes) - 1) + 1,
                    splits={"train": f"0:{len(self.new_episodes)}"})
        self._save_json(info, meta / "info.json")

        self._save_jsonl([
            {"task_index": i, "task": t}
            for t, i in sorted(self.new_tasks.items(), key=lambda x: x[1])
        ], meta / "tasks.jsonl")

        self._save_jsonl([
            {"episode_index": ne.new_episode_index,
             "tasks": [ne.task], "length": ne.length}
            for ne in self.new_episodes
        ], meta / "episodes.jsonl")

        # Write episodes_stats.jsonl (one JSON line per episode)
        self._save_jsonl([
            {"episode_index": idx, "stats": st}
            for idx, st in sorted(ep_stats.items())
        ], meta / "episodes_stats.jsonl")


# ═══════════════════════════════════════════════════════════════════════════
# Safety helpers
# ═══════════════════════════════════════════════════════════════════════════

def paths_overlap(src: str, dst: str) -> bool:
    """True when *dst* equals or is inside *src* (or vice-versa)."""
    if not src or not dst:
        return False
    a, b = Path(src).resolve(), Path(dst).resolve()
    if a == b:
        return True
    try:
        b.relative_to(a)
        return True
    except ValueError:
        pass
    try:
        a.relative_to(b)
        return True
    except ValueError:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════════

# Colour palette — industrial / utilitarian with warm accents
_BG        = "#1e1e2e"
_BG_ALT    = "#262637"
_FG        = "#cdd6f4"
_FG_DIM    = "#7f849c"
_ACCENT    = "#f9e2af"
_GREEN     = "#a6e3a1"
_RED       = "#f38ba8"
_BLUE      = "#89b4fa"
_SURFACE   = "#313244"
_OVERLAY   = "#45475a"
_BORDER    = "#585b70"


class DatasetSplitterApp:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, initial_source: str = ""):
        # ── Multi-episode plan ──
        # plan[episode_index] = [{"start": int, "end": int, "task": str}, ...]
        # "end" is inclusive (last frame)
        self.plan: dict[int, list[dict]] = {}

        self.source_info: dict | None = None
        self.source_episodes: dict | None = None

        self.root = tk.Tk()
        self.root.title("LeRobot Dataset Splitter")
        self.root.configure(bg=_BG)
        self.root.minsize(820, 720)
        self.root.geometry("860x780")

        # Shared tk style
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=_BG, foreground=_FG,
                        fieldbackground=_SURFACE, bordercolor=_BORDER,
                        insertcolor=_FG)
        style.configure("TLabel",   background=_BG, foreground=_FG)
        style.configure("TFrame",   background=_BG)
        style.configure("Dim.TLabel", foreground=_FG_DIM)
        style.configure("Accent.TLabel", foreground=_ACCENT)
        style.configure("Green.TLabel",  foreground=_GREEN)
        style.configure("Red.TLabel",    foreground=_RED)
        style.configure("TEntry",   fieldbackground=_SURFACE,
                        foreground=_FG, insertcolor=_FG, padding=(4, 6), font=("Consolas", 11))
        style.configure("Treeview", background=_SURFACE,
                        foreground=_FG, fieldbackground=_SURFACE,
                        rowheight=26)
        style.configure("Treeview.Heading",
                        background=_OVERLAY, foreground=_ACCENT)
        style.map("Treeview",
                  background=[("selected", _OVERLAY)],
                  foreground=[("selected", _ACCENT)])

        self._build_ui()

        if initial_source:
            self.source_var.set(initial_source)
            self._load_source()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=10, pady=4)
        root = self.root

        # ── Title ─────────────────────────────────────────────────────
        title_frame = ttk.Frame(root)
        title_frame.pack(fill=tk.X, **pad)
        ttk.Label(title_frame, text="LEROBOT DATASET SPLITTER",
                  font=("Consolas", 15, "bold"),
                  style="Accent.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame, text="  ·  split episodes by task",
                  style="Dim.TLabel").pack(side=tk.LEFT)

        # ── Paths section ─────────────────────────────────────────────
        path_frame = ttk.LabelFrame(root, text=" Paths ",
                                     padding=8)
        path_frame.pack(fill=tk.X, **pad)

        # Source
        ttk.Label(path_frame, text="Input dataset:").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.source_var = tk.StringVar()
        src_entry = tk.Entry(path_frame, textvariable=self.source_var,
                     width=52, bg=_SURFACE, fg=_FG,
                     insertbackground=_FG, font=("Consolas", 11),
                     relief=tk.FLAT, highlightthickness=1,
                     highlightbackground=_BORDER)
        src_entry.grid(row=0, column=1, sticky=tk.EW, padx=4)
        tk.Button(path_frame, text="Browse…", bg=_SURFACE, fg=_FG,
                  activebackground=_OVERLAY, activeforeground=_ACCENT,
                  relief=tk.FLAT, command=self._browse_source).grid(
            row=0, column=2, padx=2)
        tk.Button(path_frame, text="Load", bg=_SURFACE, fg=_GREEN,
                  activebackground=_OVERLAY, activeforeground=_GREEN,
                  relief=tk.FLAT, command=self._load_source).grid(
            row=0, column=3, padx=2)

        # Output
        ttk.Label(path_frame, text="Output dataset:").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        self.output_var = tk.StringVar()
        tk.Entry(path_frame, textvariable=self.output_var,
                width=52, bg=_SURFACE, fg=_FG,
                insertbackground=_FG, font=("Consolas", 11),
                relief=tk.FLAT, highlightthickness=1,
                highlightbackground=_BORDER).grid(row=1, column=1, sticky=tk.EW, padx=4)
        tk.Button(path_frame, text="Browse…", bg=_SURFACE, fg=_FG,
                  activebackground=_OVERLAY, activeforeground=_ACCENT,
                  relief=tk.FLAT, command=self._browse_output).grid(
            row=1, column=2, padx=2)

        path_frame.columnconfigure(1, weight=1)

        # Source info label
        self.source_info_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(path_frame, textvariable=self.source_info_var,
                  style="Dim.TLabel").grid(
            row=2, column=0, columnspan=4, sticky=tk.W, pady=(4, 0))

        # ── Episode + Segment entry ───────────────────────────────────
        entry_frame = ttk.LabelFrame(root, text=" Add Segment ", padding=8)
        entry_frame.pack(fill=tk.X, **pad)

        # Row 0: Episode selector
        ttk.Label(entry_frame, text="Episode:").grid(
            row=0, column=0, sticky=tk.W)
        self.episode_var = tk.StringVar(value="0")
        ep_spin = ttk.Spinbox(entry_frame, textvariable=self.episode_var,
                              from_=0, to=9999, width=8)
        ep_spin.grid(row=0, column=1, padx=4, sticky=tk.W)

        self.ep_info_var = tk.StringVar(value="")
        ttk.Label(entry_frame, textvariable=self.ep_info_var,
                  style="Dim.TLabel").grid(
            row=0, column=2, columnspan=5, sticky=tk.W, padx=10)

        self.episode_var.trace_add("write", lambda *_: self._update_ep_info())

        # Row 1: Start, End, Task, Add button
        ttk.Label(entry_frame, text="Start frame:").grid(
            row=1, column=0, sticky=tk.W, pady=(4, 0))
        self.start_var = tk.StringVar(value="0")
        ttk.Entry(entry_frame, textvariable=self.start_var,
                  width=10).grid(row=1, column=1, padx=4, pady=(4, 0))

        ttk.Label(entry_frame, text="Last frame:").grid(
            row=1, column=2, sticky=tk.W, pady=(4, 0))
        self.end_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.end_var,
                  width=10).grid(row=1, column=3, padx=4, pady=(4, 0))

        ttk.Label(entry_frame, text="Task:").grid(
            row=1, column=4, sticky=tk.W, pady=(4, 0))
        self.task_var = tk.StringVar()
        ttk.Entry(entry_frame, textvariable=self.task_var,
                  width=28).grid(row=1, column=5, padx=4, sticky=tk.EW, pady=(4, 0))

        tk.Button(entry_frame, text="  Add Segment  ", bg=_SURFACE, fg=_GREEN,
                  activebackground=_OVERLAY, activeforeground=_GREEN,
                  relief=tk.FLAT, font=("Consolas", 10, "bold"),
                  command=self._add_segment).grid(
            row=1, column=6, padx=(8, 0), pady=(4, 0))

        entry_frame.columnconfigure(5, weight=1)

        # ── Split plan table ──────────────────────────────────────────
        table_frame = ttk.Frame(root)
        table_frame.pack(fill=tk.BOTH, expand=True, **pad)

        cols = ("src_ep", "start", "end", "frames", "task")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                 height=10)
        self.tree.heading("src_ep", text="Src Episode")
        self.tree.heading("start",  text="Start Frame")
        self.tree.heading("end",    text="Last Frame")
        self.tree.heading("frames", text="Frames")
        self.tree.heading("task",   text="Task Label")
        self.tree.column("src_ep", width=90,  anchor=tk.CENTER)
        self.tree.column("start",  width=90,  anchor=tk.CENTER)
        self.tree.column("end",    width=90,  anchor=tk.CENTER)
        self.tree.column("frames", width=70,  anchor=tk.CENTER)
        self.tree.column("task",   width=300, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL,
                           command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Table buttons ─────────────────────────────────────────────
        tbl_btn = ttk.Frame(root)
        tbl_btn.pack(fill=tk.X, **pad)
        tk.Button(tbl_btn, text="Remove Selected", bg=_SURFACE, fg=_RED,
                  activebackground=_OVERLAY, activeforeground=_RED,
                  relief=tk.FLAT, command=self._remove_segment).pack(
            side=tk.LEFT, padx=4)
        tk.Button(tbl_btn, text="Clear All", bg=_SURFACE, fg=_FG_DIM,
                  activebackground=_OVERLAY, activeforeground=_FG,
                  relief=tk.FLAT, command=self._clear_plan).pack(
            side=tk.LEFT, padx=4)

        self.total_var = tk.StringVar(value="0 episodes · 0 segments · 0 frames")
        ttk.Label(tbl_btn, textvariable=self.total_var,
                  style="Dim.TLabel").pack(side=tk.RIGHT)

        # ── Action bar ────────────────────────────────────────────────
        act_frame = ttk.Frame(root)
        act_frame.pack(fill=tk.X, padx=10, pady=(2, 8))

        self.split_btn = tk.Button(
            act_frame,
            text="   ▶  RUN SPLIT   ",
            bg=_ACCENT, fg=_BG,
            activebackground="#f5c97e", activeforeground=_BG,
            font=("Consolas", 12, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._run_split,
        )
        self.split_btn.pack(side=tk.RIGHT, padx=4)

        # Save / load keyframes
        tk.Button(act_frame, text="Save all keyframes…",
                  bg=_SURFACE, fg=_BLUE,
                  activebackground=_OVERLAY, activeforeground=_BLUE,
                  relief=tk.FLAT, command=self._save_all_keyframes).pack(
            side=tk.LEFT, padx=4)
        tk.Button(act_frame, text="Load keyframes…",
                  bg=_SURFACE, fg=_BLUE,
                  activebackground=_OVERLAY, activeforeground=_BLUE,
                  relief=tk.FLAT, command=self._load_keyframes).pack(
            side=tk.LEFT, padx=4)

        # ── Status / progress bar ─────────────────────────────────────
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var,
                  style="Dim.TLabel").pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(status_frame, length=220,
                                         mode="determinate")
        self.progress.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_source(self):
        d = filedialog.askdirectory(title="Select input dataset folder")
        if d:
            self.source_var.set(d)
            self._load_source()

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_var.set(d)

    # ------------------------------------------------------------------
    # Load source dataset metadata
    # ------------------------------------------------------------------

    def _load_source(self):
        src = self.source_var.get().strip()
        if not src:
            return
        meta = Path(src) / "meta"
        if not meta.exists():
            messagebox.showerror("Error",
                                 f"No 'meta' folder found in:\n{src}")
            return

        try:
            self.source_info = EpisodeSplitter._load_json(meta / "info.json")
            eps = EpisodeSplitter._load_jsonl(meta / "episodes.jsonl")
            self.source_episodes = {e["episode_index"]: e for e in eps}

            n_eps = len(self.source_episodes)
            n_frames = sum(e["length"] for e in self.source_episodes.values())
            fps = self.source_info.get("fps", "?")
            robot = self.source_info.get("robot_type", "?")
            self.source_info_var.set(
                f"Loaded ✓  ·  {n_eps} episodes  ·  {n_frames} frames  ·  "
                f"{fps} fps  ·  robot: {robot}"
            )
            # Suggest output path
            if not self.output_var.get():
                self.output_var.set(src.rstrip("/\\") + "_split")

            self._update_ep_info()
            log.info("Source loaded: %s (%d episodes)", src, n_eps)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load dataset:\n{exc}")

    def _update_ep_info(self):
        if not self.source_episodes:
            return
        try:
            idx = int(self.episode_var.get())
        except ValueError:
            self.ep_info_var.set("")
            return
        ep = self.source_episodes.get(idx)
        if ep:
            tasks = ", ".join(ep.get("tasks", []))
            n_segs = len(self.plan.get(idx, []))
            plan_str = f"  ·  {n_segs} segment(s) in plan" if n_segs else ""
            self.ep_info_var.set(
                f"{ep['length']} frames  ·  task: {tasks or '—'}{plan_str}")
        else:
            self.ep_info_var.set("episode not found")

    # ------------------------------------------------------------------
    # Flat view of the plan (for the table)
    # ------------------------------------------------------------------

    def _flat_plan(self) -> list[tuple[int, dict]]:
        """Return all segments as (episode_index, segment_dict) sorted by
        episode then start frame."""
        items = []
        for ep_idx in sorted(self.plan):
            for seg in sorted(self.plan[ep_idx], key=lambda s: s["start"]):
                items.append((ep_idx, seg))
        return items

    # ------------------------------------------------------------------
    # Segment management
    # ------------------------------------------------------------------

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        flat = self._flat_plan()
        total_f = 0
        for ep_idx, s in flat:
            f = s["end"] - s["start"] + 1
            total_f += f
            self.tree.insert("", tk.END,
                             values=(ep_idx, s["start"], s["end"], f, s["task"]))
        n_eps = len(self.plan)
        n_segs = len(flat)
        self.total_var.set(
            f"{n_eps} episode{'s' if n_eps != 1 else ''} · "
            f"{n_segs} segment{'s' if n_segs != 1 else ''} · "
            f"{total_f} frames"
        )
        self._update_ep_info()

    def _add_segment(self):
        try:
            start = int(self.start_var.get())
            end = int(self.end_var.get())
        except ValueError:
            messagebox.showwarning("Input error",
                                   "Start and End must be integers.")
            return
        task = self.task_var.get().strip()
        if not task:
            messagebox.showwarning("Input error", "Task label cannot be empty.")
            return
        if end < start:
            messagebox.showwarning("Input error",
                                   "Last frame must be >= Start frame.")
            return

        try:
            ep_idx = int(self.episode_var.get())
        except ValueError:
            messagebox.showwarning("Input error", "Invalid episode index.")
            return

        # Check against episode length
        if self.source_episodes:
            ep = self.source_episodes.get(ep_idx)
            if not ep:
                messagebox.showwarning("Input error",
                                       f"Episode {ep_idx} not found in source dataset.")
                return
            if end >= ep["length"]:
                messagebox.showwarning(
                    "Input error",
                    f"Last frame ({end}) exceeds max frame ({ep['length'] - 1}).")
                return

        # Check for overlaps within the same episode
        existing = self.plan.get(ep_idx, [])
        for s in existing:
            if start <= s["end"] and end >= s["start"]:
                messagebox.showwarning(
                    "Overlap",
                    f"Frames {start}-{end} overlap with existing segment "
                    f"{s['start']}-{s['end']} ('{s['task']}') in episode {ep_idx}."
                )
                return

        # Add to plan
        if ep_idx not in self.plan:
            self.plan[ep_idx] = []
        self.plan[ep_idx].append({"start": start, "end": end, "task": task})
        self._refresh_table()

        # Auto-advance start to this segment's end
        self.start_var.set(str(end + 1))
        self.end_var.set("")
        self.task_var.set("")

    def _remove_segment(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        flat = self._flat_plan()
        if idx < len(flat):
            ep_idx, seg = flat[idx]
            self.plan[ep_idx].remove(seg)
            if not self.plan[ep_idx]:
                del self.plan[ep_idx]
            self._refresh_table()

    def _clear_plan(self):
        if self.plan and messagebox.askyesno(
                "Confirm", "Remove all segments from all episodes?"):
            self.plan.clear()
            self._refresh_table()

    # ------------------------------------------------------------------
    # Keyframes I/O
    # ------------------------------------------------------------------

    def _save_all_keyframes(self):
        """Save keyframes for ALL episodes in the plan to a single file."""
        if not self.plan:
            messagebox.showinfo("Nothing to save",
                                "No segments in the plan. Add segments first.")
            return

        dataset_name = Path(self.source_var.get()).name if self.source_var.get() else "dataset"
        initial_dir = Path(self.source_var.get()).parent if self.source_var.get() else None

        path = filedialog.asksaveasfilename(
            title="Save all keyframes",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialdir=initial_dir,
            initialfile=f"keyframes_{dataset_name}.json",
        )
        if not path:
            return

        episodes = []
        for ep_idx in sorted(self.plan):
            segments = self.plan[ep_idx]
            if not segments:
                continue

            ep_length = 0
            if self.source_episodes:
                ep = self.source_episodes.get(ep_idx)
                if ep:
                    ep_length = ep["length"]

            episodes.append({
                "episode_index": ep_idx,
                "episode_length": ep_length,
                "keyframes": [
                    {"frame": s["start"], "end_frame": s["end"] + 1, "task": s["task"]}
                    for s in sorted(segments, key=lambda x: x["start"])
                ],
            })

        data = {
            "repo_id": dataset_name,
            "episodes": episodes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.status_var.set(
            f"Keyframes saved → {Path(path).name} "
            f"({len(episodes)} episodes)")

    def _load_keyframes(self):
        """Load keyframes from one or more files.  Supports both formats:
        - Multi-episode: {"episodes": [{"episode_index":..., "keyframes":...}, ...]}
        - Single-episode (legacy): {"episode_index":..., "keyframes":...}
        """
        initial_dir = Path(self.source_var.get()).parent if self.source_var.get() else None
        paths = filedialog.askopenfilenames(
            title="Load keyframes (select one or more files)",
            filetypes=[("JSON", "*.json")],
            initialdir=initial_dir,
        )
        if not paths:
            return

        total_loaded = 0
        errors = []

        def _parse_episode_entry(entry: dict) -> tuple[int, list[dict]]:
            """Parse a single episode's keyframes into (ep_idx, segments)."""
            ep_idx = entry.get("episode_index", 0)
            ep_len = entry.get("episode_length", 0)
            kfs = entry.get("keyframes", [])
            segs = []
            for i, kf in enumerate(sorted(kfs, key=lambda k: k["frame"])):
                start = kf["frame"]
                if "end_frame" in kf:
                    end = kf["end_frame"] - 1
                elif i < len(kfs) - 1:
                    end = sorted(kfs, key=lambda k: k["frame"])[i + 1]["frame"]
                else:
                    end = ep_len
                segs.append({"start": start, "end": end, "task": kf["task"]})
            return ep_idx, segs

        for path in paths:
            try:
                with open(path) as f:
                    data = json.load(f)

                # Multi-episode format
                if "episodes" in data:
                    for entry in data["episodes"]:
                        ep_idx, segs = _parse_episode_entry(entry)
                        if segs:
                            self.plan[ep_idx] = segs
                            total_loaded += len(segs)
                            self.episode_var.set(str(ep_idx))
                # Single-episode format (legacy)
                else:
                    ep_idx, segs = _parse_episode_entry(data)
                    if segs:
                        self.plan[ep_idx] = segs
                        total_loaded += len(segs)
                        self.episode_var.set(str(ep_idx))

            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")

        self._refresh_table()

        if errors:
            messagebox.showerror("Errors loading keyframes",
                                 "\n".join(errors))
        if total_loaded:
            n_files = len(paths) - len(errors)
            self.status_var.set(
                f"Loaded {total_loaded} segments from {n_files} file(s)")

    # ------------------------------------------------------------------
    # Run the split
    # ------------------------------------------------------------------

    def _run_split(self):
        # Validate inputs
        src = self.source_var.get().strip()
        dst = self.output_var.get().strip()

        if not src:
            messagebox.showerror("Error", "Set an input dataset path.")
            return
        if not dst:
            messagebox.showerror("Error", "Set an output dataset path.")
            return
        if not self.plan:
            messagebox.showerror("Error", "Add at least one segment.")
            return

        # ── SAFETY: paths must not overlap ────────────────────────────
        if paths_overlap(src, dst):
            messagebox.showerror(
                "⛔  Output path rejected",
                "The output path is the same as (or inside) the input dataset.\n\n"
                "To protect your data, the output must be a DIFFERENT location.\n\n"
                "Please choose another output folder."
            )
            return

        if Path(dst).exists() and any(Path(dst).iterdir()):
            if not messagebox.askyesno(
                "Output exists",
                f"The folder already contains files:\n{dst}\n\n"
                "Files may be overwritten.  Continue?"
            ):
                return

        # Build summary for confirmation
        flat = self._flat_plan()
        n_src_eps = len(self.plan)
        n_segs = len(flat)
        total_f = sum(s["end"] - s["start"] + 1 for _, s in flat)
        ep_list = ", ".join(str(e) for e in sorted(self.plan))

        if not messagebox.askyesno(
            "Confirm split",
            f"Split {n_src_eps} source episode(s) [{ep_list}] into "
            f"{n_segs} new episode(s) ({total_f} total frames)?\n\n"
            f"Input (READ ONLY):\n  {src}\n\n"
            f"Output (WRITE):\n  {dst}"
        ):
            return

        # Disable button, run in thread
        self.split_btn.config(state=tk.DISABLED, text="  Splitting…  ")
        self.progress["value"] = 0

        def do_split():
            try:
                splitter = EpisodeSplitter(src, dst)

                # Process each source episode in order
                for ep_idx in sorted(self.plan):
                    segments = self.plan[ep_idx]
                    segs = [
                        TaskSegment(s["start"], s["end"] + 1, s["task"])
                        for s in segments
                    ]
                    splitter.split_episode(ep_idx, segs)

                splitter.save(
                    progress_cb=lambda cur, tot, msg: self.root.after(
                        0, self._on_progress, cur, tot, msg
                    )
                )
                self.root.after(0, self._on_done, None)
            except Exception as exc:
                self.root.after(0, self._on_done, exc)

        threading.Thread(target=do_split, daemon=True).start()

    def _on_progress(self, current: int, total: int, msg: str):
        pct = int(100 * current / max(total, 1))
        self.progress["value"] = pct
        self.status_var.set(msg)

    def _on_done(self, error: Exception | None):
        self.split_btn.config(state=tk.NORMAL,
                              text="   ▶  RUN SPLIT   ")
        self.progress["value"] = 100 if error is None else 0
        if error:
            messagebox.showerror("Split failed", str(error))
            self.status_var.set(f"FAILED: {error}")
        else:
            dst = self.output_var.get()
            flat = self._flat_plan()
            messagebox.showinfo(
                "Split complete ✓",
                f"New dataset with {len(flat)} episodes saved to:\n{dst}")
            self.status_var.set("Split complete ✓")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="LeRobot Dataset Splitter GUI")
    p.add_argument("--source", type=str, default="",
                   help="Pre-fill input dataset path")
    args = p.parse_args()

    app = DatasetSplitterApp(initial_source=args.source)
    app.run()


if __name__ == "__main__":
    main()
