#!/usr/bin/env python
"""
LeRobot Dataset Merger GUI
============================

A self-contained GUI tool for combining multiple LeRobot v2.1 datasets
into a single dataset.  All episodes from every source dataset are
appended sequentially into the output, with episode indices, global
indices, task mappings, and metadata rebuilt from scratch.

Videos are copied directly (no re-encoding) since whole episodes are
transferred unchanged.

Usage:
    python dataset_merger.py
    python dataset_merger.py --sources /path/to/ds1 /path/to/ds2

Safety:
    - Output path is BLOCKED from overlapping any source dataset.
    - A confirmation dialog is shown before any write operation.
    - Source datasets are only ever opened for reading.

Requirements:
    - Python 3.10+
    - tkinter  (ships with most Python installations)
    - pandas + pyarrow  (pip install pandas pyarrow)
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
log = logging.getLogger("dataset_merger")


# ═══════════════════════════════════════════════════════════════════════════
# JSON / JSONL helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    if path.exists():
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    entries.append(json.loads(stripped))
    return entries


def _save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Source dataset descriptor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SourceDataset:
    """Lightweight descriptor loaded from a source dataset's metadata."""
    root: Path
    info: dict
    episodes: list[dict]          # from episodes.jsonl
    tasks: dict[str, int]         # task_name → task_index in source
    episode_stats: dict[int, dict]  # ep_index → stats dict

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def fps(self) -> int:
        return self.info.get("fps", 30)

    @property
    def chunks_size(self) -> int:
        return self.info.get("chunks_size", 1000)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        return sum(e["length"] for e in self.episodes)

    @property
    def video_keys(self) -> list[str]:
        return [
            k for k, v in self.info.get("features", {}).items()
            if v.get("dtype") == "video"
        ]

    @staticmethod
    def load(root: Path) -> "SourceDataset":
        root = Path(root).resolve()
        meta = root / "meta"
        if not meta.exists():
            raise FileNotFoundError(f"No 'meta' folder in: {root}")

        info = _load_json(meta / "info.json")
        episodes = _load_jsonl(meta / "episodes.jsonl")
        tasks_list = _load_jsonl(meta / "tasks.jsonl")
        tasks = {e["task"]: e["task_index"] for e in tasks_list}

        # Load per-episode stats
        episode_stats: dict[int, dict] = {}
        stats_path = meta / "episodes_stats.jsonl"
        if stats_path.exists():
            for entry in _load_jsonl(stats_path):
                episode_stats[entry["episode_index"]] = entry.get("stats", {})

        return SourceDataset(
            root=root, info=info, episodes=episodes,
            tasks=tasks, episode_stats=episode_stats,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Merger engine
# ═══════════════════════════════════════════════════════════════════════════

class DatasetMerger:
    """Merges multiple LeRobot v2.1 datasets into one."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root).resolve()
        self.sources: list[SourceDataset] = []
        self.new_tasks: dict[str, int] = {}
        self._next_ep = 0

    def add_source(self, source: SourceDataset):
        """Register a source dataset for merging."""
        # Validate output doesn't overlap source
        if _paths_overlap(str(source.root), str(self.output_root)):
            raise ValueError(
                f"Output path overlaps source '{source.name}'. "
                "Choose a different output path."
            )
        self.sources.append(source)

    def _ensure_task(self, task: str) -> int:
        if task not in self.new_tasks:
            self.new_tasks[task] = len(self.new_tasks)
        return self.new_tasks[task]

    def _chunk(self, ep: int, chunks_size: int) -> int:
        return ep // chunks_size

    def _parquet(self, root: Path, ep: int, chunks_size: int) -> Path:
        return root / "data" / f"chunk-{self._chunk(ep, chunks_size):03d}" / f"episode_{ep:06d}.parquet"

    def _video(self, root: Path, ep: int, key: str, chunks_size: int) -> Path:
        return root / "videos" / f"chunk-{self._chunk(ep, chunks_size):03d}" / key / f"episode_{ep:06d}.mp4"

    @staticmethod
    def _stats(df: "pd.DataFrame") -> dict:
        """Compute per-column stats in LeRobot v2.1 format."""
        out: dict = {}
        for col in df.columns:
            try:
                vals = df[col].values
                if hasattr(vals[0], "__len__"):
                    s = np.stack(vals)
                    ndim = s.ndim
                    if ndim == 2:
                        out[col] = dict(
                            min=np.min(s, axis=0).tolist(),
                            max=np.max(s, axis=0).tolist(),
                            mean=np.mean(s, axis=0).tolist(),
                            std=np.std(s, axis=0).tolist(),
                            count=[len(s)],
                        )
                    elif ndim >= 3:
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

    def save(self, progress_cb=None):
        """Write the merged dataset."""
        if not self.sources:
            raise ValueError("No source datasets added.")
        if not HAS_PANDAS:
            raise RuntimeError("pandas + pyarrow required")

        # Validate compatibility
        ref = self.sources[0]
        for s in self.sources[1:]:
            if s.fps != ref.fps:
                raise ValueError(
                    f"FPS mismatch: '{ref.name}' has {ref.fps} fps, "
                    f"'{s.name}' has {s.fps} fps.")
            if s.video_keys != ref.video_keys:
                raise ValueError(
                    f"Video key mismatch between '{ref.name}' and '{s.name}'.\n"
                    f"  {ref.name}: {ref.video_keys}\n"
                    f"  {s.name}: {s.video_keys}")

        fps = ref.fps
        chunks_size = ref.chunks_size
        video_keys = ref.video_keys

        # Count total work
        total_episodes = sum(s.n_episodes for s in self.sources)
        total_steps = total_episodes + 2  # +reindex +metadata
        done = 0

        def _tick(msg: str):
            nonlocal done
            done += 1
            if progress_cb:
                progress_cb(done, total_steps, msg)
            log.info(msg)

        self.output_root.mkdir(parents=True, exist_ok=True)

        # ── Episode data ──
        all_stats: dict[int, dict] = {}
        episode_records: list[dict] = []
        total_frames = 0

        for src in self.sources:
            for ep_entry in src.episodes:
                src_ep = ep_entry["episode_index"]
                new_ep = self._next_ep
                ep_len = ep_entry["length"]
                ep_tasks = ep_entry.get("tasks", ["unknown"])

                # Register tasks
                for t in ep_tasks:
                    self._ensure_task(t)

                # ── Copy parquet ──
                src_pq = self._parquet(src.root, src_ep, src.chunks_size)
                dst_pq = self._parquet(self.output_root, new_ep, chunks_size)
                dst_pq.parent.mkdir(parents=True, exist_ok=True)

                df = pd.read_parquet(src_pq)
                df["episode_index"] = new_ep
                df["frame_index"] = np.arange(len(df))
                df["timestamp"] = df["frame_index"] / fps
                # Remap task_index to new unified task mapping
                if "task_index" in df.columns:
                    # Build source→new task index mapping for this source
                    src_task_map = {}
                    for task_name, src_idx in src.tasks.items():
                        src_task_map[src_idx] = self._ensure_task(task_name)
                    df["task_index"] = df["task_index"].map(src_task_map)
                else:
                    df["task_index"] = self._ensure_task(ep_tasks[0])
                df["index"] = np.arange(len(df))  # placeholder, reindexed later
                df.to_parquet(dst_pq, index=False)

                # ── Copy videos (no re-encoding) ──
                for vk in video_keys:
                    src_vid = self._video(src.root, src_ep, vk, src.chunks_size)
                    if src_vid.exists():
                        dst_vid = self._video(self.output_root, new_ep, vk, chunks_size)
                        dst_vid.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_vid, dst_vid)

                # ── Stats ──
                if src_ep in src.episode_stats:
                    all_stats[new_ep] = src.episode_stats[src_ep]
                else:
                    # Recompute stats from the parquet we just wrote
                    all_stats[new_ep] = self._stats(df)

                episode_records.append({
                    "episode_index": new_ep,
                    "tasks": ep_tasks,
                    "length": ep_len,
                })

                total_frames += ep_len
                self._next_ep += 1

                _tick(f"Episode {new_ep} ← {src.name} ep {src_ep} "
                      f"({ep_len} frames)")

        # ── Reindex global indices ──
        gi = 0
        for ep_rec in episode_records:
            p = self._parquet(self.output_root, ep_rec["episode_index"], chunks_size)
            df = pd.read_parquet(p)
            df["index"] = np.arange(gi, gi + len(df))
            df.to_parquet(p, index=False)
            gi += len(df)
        _tick("Global indices updated")

        # ── Write metadata ──
        meta = self.output_root / "meta"

        # Use the first source's info as the base
        info = ref.info.copy()
        total_videos = self._next_ep * len(video_keys)
        info.update(
            total_episodes=self._next_ep,
            total_tasks=len(self.new_tasks),
            total_frames=total_frames,
            total_videos=total_videos,
            total_chunks=self._chunk(self._next_ep - 1, chunks_size) + 1,
            splits={"train": f"0:{self._next_ep}"},
        )
        _save_json(info, meta / "info.json")

        _save_jsonl([
            {"task_index": i, "task": t}
            for t, i in sorted(self.new_tasks.items(), key=lambda x: x[1])
        ], meta / "tasks.jsonl")

        _save_jsonl(episode_records, meta / "episodes.jsonl")

        _save_jsonl([
            {"episode_index": idx, "stats": st}
            for idx, st in sorted(all_stats.items())
        ], meta / "episodes_stats.jsonl")

        _tick("Metadata saved")


# ═══════════════════════════════════════════════════════════════════════════
# Safety helpers
# ═══════════════════════════════════════════════════════════════════════════

def _paths_overlap(src: str, dst: str) -> bool:
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


class DatasetMergerApp:

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self):
        self.sources: list[SourceDataset] = []

        self.root = tk.Tk()
        self.root.title("LeRobot Dataset Merger")
        self.root.configure(bg=_BG)
        self.root.minsize(820, 560)
        self.root.geometry("860x620")

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
                        foreground=_FG, insertcolor=_FG, padding=(4, 6),
                        font=("Consolas", 11))
        style.configure("Treeview", background=_SURFACE,
                        foreground=_FG, fieldbackground=_SURFACE,
                        rowheight=26)
        style.configure("Treeview.Heading",
                        background=_OVERLAY, foreground=_ACCENT)
        style.map("Treeview",
                  background=[("selected", _OVERLAY)],
                  foreground=[("selected", _ACCENT)])

        self._build_ui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=10, pady=4)
        root = self.root

        # ── Title ─────────────────────────────────────────────────────
        title_frame = ttk.Frame(root)
        title_frame.pack(fill=tk.X, **pad)
        ttk.Label(title_frame, text="LEROBOT DATASET MERGER",
                  font=("Consolas", 15, "bold"),
                  style="Accent.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame, text="  ·  combine datasets",
                  style="Dim.TLabel").pack(side=tk.LEFT)

        # ── Add source section ────────────────────────────────────────
        add_frame = ttk.LabelFrame(root, text=" Source Datasets ", padding=8)
        add_frame.pack(fill=tk.X, **pad)

        tk.Button(add_frame, text="  Add Dataset(s)…  ",
                  bg=_SURFACE, fg=_GREEN,
                  activebackground=_OVERLAY, activeforeground=_GREEN,
                  relief=tk.FLAT, font=("Consolas", 10, "bold"),
                  command=self._add_sources).pack(side=tk.LEFT, padx=4)

        tk.Button(add_frame, text="Move Up",
                  bg=_SURFACE, fg=_BLUE,
                  activebackground=_OVERLAY, activeforeground=_BLUE,
                  relief=tk.FLAT, command=self._move_up).pack(
            side=tk.LEFT, padx=4)
        tk.Button(add_frame, text="Move Down",
                  bg=_SURFACE, fg=_BLUE,
                  activebackground=_OVERLAY, activeforeground=_BLUE,
                  relief=tk.FLAT, command=self._move_down).pack(
            side=tk.LEFT, padx=4)
        tk.Button(add_frame, text="Remove Selected",
                  bg=_SURFACE, fg=_RED,
                  activebackground=_OVERLAY, activeforeground=_RED,
                  relief=tk.FLAT, command=self._remove_source).pack(
            side=tk.LEFT, padx=4)

        # ── Source datasets table ─────────────────────────────────────
        table_frame = ttk.Frame(root)
        table_frame.pack(fill=tk.BOTH, expand=True, **pad)

        cols = ("order", "name", "episodes", "frames", "fps", "tasks", "path")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                 height=10)
        self.tree.heading("order",    text="#")
        self.tree.heading("name",     text="Dataset Name")
        self.tree.heading("episodes", text="Episodes")
        self.tree.heading("frames",   text="Frames")
        self.tree.heading("fps",      text="FPS")
        self.tree.heading("tasks",    text="Tasks")
        self.tree.heading("path",     text="Path")
        self.tree.column("order",    width=40,  anchor=tk.CENTER)
        self.tree.column("name",     width=180, anchor=tk.W)
        self.tree.column("episodes", width=70,  anchor=tk.CENTER)
        self.tree.column("frames",   width=70,  anchor=tk.CENTER)
        self.tree.column("fps",      width=50,  anchor=tk.CENTER)
        self.tree.column("tasks",    width=120, anchor=tk.W)
        self.tree.column("path",     width=300, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL,
                           command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Summary ───────────────────────────────────────────────────
        summary_frame = ttk.Frame(root)
        summary_frame.pack(fill=tk.X, **pad)
        self.summary_var = tk.StringVar(
            value="0 datasets · 0 episodes · 0 frames")
        ttk.Label(summary_frame, textvariable=self.summary_var,
                  style="Dim.TLabel").pack(side=tk.RIGHT)

        # ── Output path ───────────────────────────────────────────────
        out_frame = ttk.LabelFrame(root, text=" Output ", padding=8)
        out_frame.pack(fill=tk.X, **pad)

        ttk.Label(out_frame, text="Output dataset:").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.output_var = tk.StringVar()
        tk.Entry(out_frame, textvariable=self.output_var,
                 width=56, bg=_SURFACE, fg=_FG,
                 insertbackground=_FG, font=("Consolas", 11),
                 relief=tk.FLAT, highlightthickness=1,
                 highlightbackground=_BORDER).grid(
            row=0, column=1, sticky=tk.EW, padx=4)
        tk.Button(out_frame, text="Browse…", bg=_SURFACE, fg=_FG,
                  activebackground=_OVERLAY, activeforeground=_ACCENT,
                  relief=tk.FLAT, command=self._browse_output).grid(
            row=0, column=2, padx=2)
        out_frame.columnconfigure(1, weight=1)

        # ── Action bar ────────────────────────────────────────────────
        act_frame = ttk.Frame(root)
        act_frame.pack(fill=tk.X, padx=10, pady=(2, 8))

        self.merge_btn = tk.Button(
            act_frame,
            text="   ▶  RUN MERGE   ",
            bg=_ACCENT, fg=_BG,
            activebackground="#f5c97e", activeforeground=_BG,
            font=("Consolas", 12, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._run_merge,
        )
        self.merge_btn.pack(side=tk.RIGHT, padx=4)

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
    # Default dataset directory
    # ------------------------------------------------------------------

    @staticmethod
    def _default_dataset_dir() -> str | None:
        """Return the default dataset directory if it exists."""
        candidates = [
            Path.home() / "lerobot" / "lerobot" / "scripts" / "dataset",
            Path.home() / "lerobot" / "scripts" / "dataset",
        ]
        for c in candidates:
            if c.is_dir():
                return str(c)
        return None

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_output(self):
        d = filedialog.askdirectory(
            title="Select output folder",
            initialdir=self._default_dataset_dir())
        if d:
            self.output_var.set(d)

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _try_add_dataset(self, path: Path) -> bool:
        """Try to load and add a single dataset path.
        Returns True if added successfully."""
        resolved = path.resolve()
        if any(s.root == resolved for s in self.sources):
            messagebox.showwarning(
                "Duplicate",
                f"'{resolved.name}' is already in the list.")
            return False
        try:
            src = SourceDataset.load(resolved)
            self.sources.append(src)
            self._refresh_table()
            self.status_var.set(f"Added: {src.name}")

            # Auto-suggest output if first source
            if len(self.sources) == 1 and not self.output_var.get():
                self.output_var.set(
                    str(resolved.parent / (resolved.name + "_merged")))
            return True
        except Exception as exc:
            messagebox.showerror("Error",
                                 f"Failed to load dataset:\n{exc}")
            return False

    def _add_sources(self):
        """Let user pick one dataset folder at a time."""
        initial = self._default_dataset_dir()
        while True:
            d = filedialog.askdirectory(
                title="Open a dataset folder and click OK  (Cancel when done)",
                initialdir=initial)
            if not d:
                break

            selected = Path(d)

            # Check it's actually a dataset
            if not (selected / "meta").is_dir():
                messagebox.showerror(
                    "Not a dataset",
                    f"No 'meta' folder found in:\n{selected.name}\n\n"
                    "Make sure you open the dataset folder\n"
                    "(double-click into it) before clicking OK.")
                initial = str(selected)
                continue

            self._try_add_dataset(selected)
            # Next dialog starts from same parent
            initial = str(selected.parent)

    def _remove_source(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx < len(self.sources):
            removed = self.sources.pop(idx)
            self._refresh_table()
            self.status_var.set(f"Removed: {removed.name}")

    def _move_up(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx > 0:
            self.sources[idx - 1], self.sources[idx] = \
                self.sources[idx], self.sources[idx - 1]
            self._refresh_table()
            # Re-select the moved item
            children = self.tree.get_children()
            self.tree.selection_set(children[idx - 1])

    def _move_down(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx < len(self.sources) - 1:
            self.sources[idx], self.sources[idx + 1] = \
                self.sources[idx + 1], self.sources[idx]
            self._refresh_table()
            # Re-select the moved item
            children = self.tree.get_children()
            self.tree.selection_set(children[idx + 1])

    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        total_eps = 0
        total_frames = 0
        for i, s in enumerate(self.sources):
            task_names = ", ".join(sorted(s.tasks.keys()))
            if len(task_names) > 40:
                task_names = task_names[:37] + "…"
            self.tree.insert("", tk.END, values=(
                i + 1,
                s.name,
                s.n_episodes,
                s.total_frames,
                s.fps,
                task_names,
                str(s.root),
            ))
            total_eps += s.n_episodes
            total_frames += s.total_frames

        n = len(self.sources)
        self.summary_var.set(
            f"{n} dataset{'s' if n != 1 else ''} · "
            f"{total_eps} episodes · "
            f"{total_frames} frames"
        )

    # ------------------------------------------------------------------
    # Run the merge
    # ------------------------------------------------------------------

    def _run_merge(self):
        dst = self.output_var.get().strip()
        if not self.sources:
            messagebox.showerror("Error", "Add at least one source dataset.")
            return
        if not dst:
            messagebox.showerror("Error", "Set an output dataset path.")
            return

        # Safety checks
        for s in self.sources:
            if _paths_overlap(str(s.root), dst):
                messagebox.showerror(
                    "⛔  Output path rejected",
                    f"Output path overlaps source '{s.name}'.\n\n"
                    "Choose a different output folder."
                )
                return

        if Path(dst).exists() and any(Path(dst).iterdir()):
            if not messagebox.askyesno(
                "Output exists",
                f"The folder already contains files:\n{dst}\n\n"
                "Files may be overwritten.  Continue?"
            ):
                return

        # Compatibility check
        ref = self.sources[0]
        for s in self.sources[1:]:
            if s.fps != ref.fps:
                messagebox.showerror(
                    "Incompatible datasets",
                    f"FPS mismatch: '{ref.name}' has {ref.fps} fps, "
                    f"'{s.name}' has {s.fps} fps.\n\n"
                    "All datasets must have the same FPS.")
                return
            if s.video_keys != ref.video_keys:
                messagebox.showerror(
                    "Incompatible datasets",
                    f"Video keys differ between '{ref.name}' and '{s.name}'.\n\n"
                    f"{ref.name}: {ref.video_keys}\n"
                    f"{s.name}: {s.video_keys}")
                return

        # Build summary
        total_eps = sum(s.n_episodes for s in self.sources)
        total_frames = sum(s.total_frames for s in self.sources)
        ds_names = ", ".join(s.name for s in self.sources)

        if not messagebox.askyesno(
            "Confirm merge",
            f"Merge {len(self.sources)} datasets into one?\n\n"
            f"Sources: {ds_names}\n"
            f"Total: {total_eps} episodes, {total_frames} frames\n\n"
            f"Output (WRITE):\n  {dst}"
        ):
            return

        # Disable button, run in thread
        self.merge_btn.config(state=tk.DISABLED, text="  Merging…  ")
        self.progress["value"] = 0

        def do_merge():
            try:
                merger = DatasetMerger(dst)
                for s in self.sources:
                    merger.add_source(s)
                merger.save(
                    progress_cb=lambda cur, tot, msg: self.root.after(
                        0, self._on_progress, cur, tot, msg
                    )
                )
                self.root.after(0, self._on_done, None)
            except Exception as exc:
                self.root.after(0, self._on_done, exc)

        threading.Thread(target=do_merge, daemon=True).start()

    def _on_progress(self, current: int, total: int, msg: str):
        pct = int(100 * current / max(total, 1))
        self.progress["value"] = pct
        self.status_var.set(msg)

    def _on_done(self, error: Exception | None):
        self.merge_btn.config(state=tk.NORMAL,
                              text="   ▶  RUN MERGE   ")
        self.progress["value"] = 100 if error is None else 0
        if error:
            messagebox.showerror("Merge failed", str(error))
            self.status_var.set(f"FAILED: {error}")
        else:
            dst = self.output_var.get()
            total_eps = sum(s.n_episodes for s in self.sources)
            messagebox.showinfo(
                "Merge complete ✓",
                f"Merged dataset with {total_eps} episodes saved to:\n{dst}")
            self.status_var.set("Merge complete ✓")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app = DatasetMergerApp()
    app.run()


if __name__ == "__main__":
    main()
