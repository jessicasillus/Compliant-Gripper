#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from pathlib import Path

SENSORS = ['A1','A2','A3','B1','B2','B3','C1','C2','C3']

# Weights — 0 = unloaded baseline anchor (REQUIRED for predict_weights.py)
WEIGHT_KEYS = {
    '0': ("0g",   0),
    '1': ("10g",  10),
    '2': ("20g",  20),
    '3': ("50g",  50),
    '4': ("100g", 100),
    '5': ("200g", 200),
    '6': ("500g", 500),
}
DEFAULT_WEIGHT = ("0g", 0)

BASELINE_SECONDS = 2.0
SMOOTH_WINDOW_S  = 0.25
MIN_STABLE_S     = 1.0
EDGE_TRIM_FRAC   = 0.20
EDGE_TRIM_S_MIN  = 0.30
THRESH_SIGMA     = 6.0
THRESH_ABS_MIN   = 8.0

def extract_weight(note_str):
    if note_str is None:
        return None
    s = str(note_str).strip().lower()
    if not s:
        return None
    if re.fullmatch(r'0\s*g?', s):
        return "0g", 0
    m = re.search(r'(\d+)\s*g', s)
    if m:
        v = int(m.group(1))
        return f"{v}g", v
    m = re.search(r'(\d+)', s)
    if m:
        v = int(m.group(1))
        return f"{v}g", v
    return None

def robust_sigma(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else np.nanstd(x)

def median_smooth(y, k):
    y = np.asarray(y, dtype=float)
    if k <= 1:
        return y
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = np.median(ypad[i:i+k])
    return out

def contiguous_blocks(active_idx):
    active_idx = np.asarray(active_idx, dtype=int)
    if active_idx.size == 0:
        return []
    cuts = np.where(np.diff(active_idx) != 1)[0] + 1
    return np.split(active_idx, cuts)

def dt_median(t):
    t = np.asarray(t, dtype=float)
    d = np.diff(t)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else 0.0

def trim_block_by_time(t, block_idx, trim_s, dt):
    if len(block_idx) < 3:
        return block_idx
    trim_n = int(np.ceil(trim_s / max(dt, 1e-9)))
    if 2 * trim_n >= len(block_idx):
        return block_idx
    return block_idx[trim_n:-trim_n]

def block_score(delta, block_idx):
    d = delta[block_idx]
    strength = np.nanmedian(np.abs(d))
    length = len(block_idx)
    return float(strength) * np.sqrt(max(length, 1))

def detect_stable_segment(t, y, baseline, dt):
    k = int(np.ceil(SMOOTH_WINDOW_S / max(dt, 1e-9)))
    ys = median_smooth(y, k)
    delta = ys - baseline
    sig = robust_sigma(delta)
    thr = max(THRESH_ABS_MIN, THRESH_SIGMA * (sig if np.isfinite(sig) else 0.0))
    active = np.where(np.abs(delta) >= thr)[0]
    blocks = contiguous_blocks(active)
    if not blocks:
        return None
    blocks_sorted = sorted(blocks, key=lambda b: block_score(delta, b), reverse=True)
    best = None
    for b in blocks_sorted:
        if (t[b[-1]] - t[b[0]]) >= MIN_STABLE_S:
            best = b
            break
    if best is None:
        return None
    trim_s = max(EDGE_TRIM_S_MIN, EDGE_TRIM_FRAC * (t[best[-1]] - t[best[0]]))
    best_trim = trim_block_by_time(t, best, trim_s, dt)
    if len(best_trim) < 3:
        best_trim = best
    return float(t[best_trim[0]]), float(t[best_trim[-1]])

def edit_notes(df):
    notes = df[df['note'].notna() & (df['note'].astype(str).str.strip() != "")][['timestamp','note']].copy()
    notes = notes.sort_values('timestamp').reset_index(drop=True)
    notes_list = []
    for _, r in notes.iterrows():
        w = extract_weight(r['note'])
        if w is not None:
            notes_list.append([float(r['timestamp']), w[0], int(w[1]), "Original"])

    t = df['timestamp'].to_numpy(dtype=float)
    y = df[SENSORS].astype(float).mean(axis=1).to_numpy()
    cursor_x     = [None]
    pending_mark = [None]
    fig, ax = plt.subplots(figsize=(12, 4))

    def draw():
        ax.clear()
        ax.plot(t, y, alpha=0.35, color='black')
        ax.set_title("NOTE EDITOR: move mouse | m=mark | 0..6=set weight (0=unloaded) | x=delete nearest | q=done")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean raw value")
        ax.grid(True)
        for tt, ws, _, _src in sorted(notes_list, key=lambda z: z[0]):
            col = "grey" if ws == "0g" else "steelblue"
            ax.axvline(tt, alpha=0.6, color=col)
            ax.text(tt, ax.get_ylim()[1], f"{ws}", fontsize=8, va="top", color=col)
        if pending_mark[0] is not None:
            ax.axvline(pending_mark[0], alpha=0.9, linestyle='--', color='orange')
            ax.text(pending_mark[0], ax.get_ylim()[0], "pending (press 0..6)", fontsize=9, va="bottom")
        fig.canvas.draw_idle()

    def on_move(event):
        if event.inaxes == ax and event.xdata is not None:
            cursor_x[0] = float(event.xdata)

    def delete_nearest(x):
        if not notes_list:
            return
        i = int(np.argmin([abs(n[0] - x) for n in notes_list]))
        notes_list.pop(i)

    def on_key(event):
        if event.key == 'm':
            if cursor_x[0] is not None:
                pending_mark[0] = cursor_x[0]
                draw()
        elif event.key in WEIGHT_KEYS:
            if pending_mark[0] is not None:
                ws, wv = WEIGHT_KEYS[event.key]
                notes_list.append([pending_mark[0], ws, wv, "Manual"])
                pending_mark[0] = None
                draw()
        elif event.key == 'x':
            if cursor_x[0] is not None:
                delete_nearest(cursor_x[0])
                draw()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show()
    out = pd.DataFrame(notes_list, columns=["timestamp","Weight","Weight_Val","Source"]).sort_values("timestamp")
    return out.reset_index(drop=True)

def build_weight_windows_from_note_table(df, note_table):
    nt = note_table.sort_values("timestamp").reset_index(drop=True)
    windows = []
    for i in range(len(nt)):
        t0 = float(nt.loc[i, "timestamp"])
        t1 = float(nt.loc[i+1, "timestamp"]) if i+1 < len(nt) else float(df["timestamp"].max())
        windows.append({"Weight": str(nt.loc[i,"Weight"]), "Weight_Val": int(nt.loc[i,"Weight_Val"]), "t0": t0, "t1": t1})
    return pd.DataFrame(windows)

def infer_weight_for_time(weight_windows, t_mid):
    hit = weight_windows[(weight_windows['t0'] <= t_mid) & (t_mid < weight_windows['t1'])]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return str(r["Weight"]), int(r["Weight_Val"])

def auto_clips(df, baselines, weight_windows):
    """
    Auto-detect stable spans for each sensor within each weight window.

    0g handling:
      - Always generates a 0g span from the recording's opening baseline
        window (first BASELINE_SECONDS), independently per sensor, so every
        sensor gets its own 0g anchor regardless of note placement.
      - If the user also marked explicit 0g note windows those are added too,
        giving the sensor-specific mean near that window.
      - Each sensor computes its own 0g mean from its own signal, so
        sensor-to-sensor baseline differences are captured correctly.
    """
    t_all = df['timestamp'].to_numpy(dtype=float)
    dt    = dt_median(t_all)
    clips = []

    # ── Always add a 0g span per sensor from the opening baseline window ──
    base_df = df[df['timestamp'] <= BASELINE_SECONDS]
    if base_df.empty:
        base_df = df.iloc[:min(len(df), 200)]
    if len(base_df) >= 10:
        ts_0g = float(base_df['timestamp'].iloc[0])
        te_0g = float(base_df['timestamp'].iloc[-1])
        for s in SENSORS:
            clips.append({"Weight": "0g", "Weight_Val": 0,
                          "Sensor": s, "Start": ts_0g, "End": te_0g,
                          "Source": "Auto-0g-baseline"})

    # ── Process each note window ──────────────────────────────────────────
    for _, w in weight_windows.iterrows():
        wdf = df[(df['timestamp'] >= w['t0']) & (df['timestamp'] < w['t1'])]
        if len(wdf) < 20:
            continue
        t = wdf['timestamp'].to_numpy(dtype=float)

        if int(w["Weight_Val"]) == 0:
            trim_s = max(0.5, 0.10 * (t[-1] - t[0]))
            ts = float(t[0]  + trim_s)
            te = float(t[-1] - trim_s)
            if te - ts < 0.5:
                ts, te = float(t[0]), float(t[-1])
            for s in SENSORS:
                clips.append({"Weight": "0g", "Weight_Val": 0,
                              "Sensor": s, "Start": ts, "End": te,
                              "Source": "Auto-0g"})
        else:
            for s in SENSORS:
                y   = wdf[s].to_numpy(dtype=float)
                seg = detect_stable_segment(t, y, float(baselines[s]), dt)
                if seg is None:
                    continue
                ts, te = seg
                clips.append({"Weight": w["Weight"], "Weight_Val": int(w["Weight_Val"]),
                              "Sensor": s, "Start": ts, "End": te, "Source": "Auto"})

    return pd.DataFrame(clips)

def edit_spans_for_sensor(df, sensor, baselines, weight_windows, clips_df):
    spans          = clips_df[clips_df["Sensor"] == sensor][["Start","End","Weight","Weight_Val","Source"]].to_dict("records")
    undo_stack     = []
    current_weight = [DEFAULT_WEIGHT]
    weight_explicit = [False]       # True once the user presses 0-6
    t = df["timestamp"].to_numpy(dtype=float)
    y = df[sensor].to_numpy(dtype=float)
    drag_active   = [False]
    drag_start    = [None]
    drag_end      = [None]
    cursor_x      = [None]
    preview_patch = [None]
    fig, ax = plt.subplots(figsize=(12, 4))

    def title_str():
        wlabel = current_weight[0][0]
        if weight_explicit[0]:
            wlabel += " *LOCKED*"
        mode = "DRAGGING - move mouse then press d to confirm" if drag_active[0] \
               else "a=start | d=confirm | x=delete/cancel | 0..6=weight({}) | u=undo | c=clear | q=next".format(wlabel)
        return f"{sensor}  |  {mode}"

    def span_colour(weight_val):
        return "lightgrey" if int(weight_val) == 0 else "steelblue"

    def redraw():
        ax.clear()
        preview_patch[0] = None
        ax.plot(t, y, color="black", alpha=0.25, label="raw")
        ax.axhline(float(baselines[sensor]), linestyle="--", alpha=0.5, label="baseline")
        ax.set_title(title_str(), fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Raw value")
        ax.grid(True)
        ax.legend(loc="best")
        for sp in spans:
            col = span_colour(sp["Weight_Val"])
            ax.axvspan(sp["Start"], sp["End"], alpha=0.25, facecolor=col)
            mid  = 0.5 * (sp["Start"] + sp["End"])
            ylim = ax.get_ylim()
            ax.text(mid, ylim[1], f'{sp["Weight"]}', fontsize=8, va="top", ha="center", color=col)
        if drag_active[0] and drag_start[0] is not None and drag_end[0] is not None:
            x0 = min(drag_start[0], drag_end[0])
            x1 = max(drag_start[0], drag_end[0])
            preview_patch[0] = ax.axvspan(x0, x1, alpha=0.35, facecolor="orange")
        fig.canvas.draw_idle()

    def update_preview(x):
        if not drag_active[0] or drag_start[0] is None:
            return
        drag_end[0] = x
        if preview_patch[0] is not None:
            try:
                preview_patch[0].remove()
            except Exception:
                pass
        x0 = min(drag_start[0], x)
        x1 = max(drag_start[0], x)
        preview_patch[0] = ax.axvspan(x0, x1, alpha=0.35, facecolor="orange")
        fig.canvas.draw_idle()

    def confirm_span():
        if drag_start[0] is None or drag_end[0] is None:
            drag_active[0] = False
            redraw()
            return
        x0 = float(min(drag_start[0], drag_end[0]))
        x1 = float(max(drag_start[0], drag_end[0]))
        drag_active[0] = False
        drag_start[0]  = None
        drag_end[0]    = None
        if (x1 - x0) < 0.05:
            redraw()
            return
        # If user explicitly pressed 0-6, always use that weight.
        # Otherwise try to infer from the note-based weight windows.
        if weight_explicit[0]:
            w_str, w_val = current_weight[0]
        else:
            tmid = 0.5 * (x0 + x1)
            w    = infer_weight_for_time(weight_windows, tmid)
            w_str, w_val = w if w is not None else current_weight[0]
        spans.append({"Start": x0, "End": x1, "Weight": w_str, "Weight_Val": w_val, "Source": "Manual"})
        undo_stack.append(("add",))
        redraw()

    def delete_nearest_span(x):
        if not spans:
            return
        containing = [i for i, sp in enumerate(spans) if sp["Start"] <= x <= sp["End"]]
        if containing:
            i_best = min(containing, key=lambda i: (spans[i]["End"] - spans[i]["Start"]))
        else:
            i_best = int(np.argmin([abs(0.5*(sp["Start"]+sp["End"]) - x) for sp in spans]))
        removed = spans.pop(i_best)
        undo_stack.append(("del", removed))
        redraw()

    def on_move(event):
        if event.inaxes != ax or event.xdata is None:
            return
        cursor_x[0] = float(event.xdata)
        if drag_active[0]:
            update_preview(float(event.xdata))

    def on_key(event):
        k = event.key
        if k == 'a':
            if cursor_x[0] is not None:
                drag_active[0] = True
                drag_start[0]  = cursor_x[0]
                drag_end[0]    = cursor_x[0]
                redraw()
        elif k == 'd':
            if drag_active[0]:
                confirm_span()
        elif k == 'x':
            if drag_active[0]:
                drag_active[0] = False
                drag_start[0]  = None
                drag_end[0]    = None
                redraw()
            elif cursor_x[0] is not None:
                delete_nearest_span(cursor_x[0])
        elif k in WEIGHT_KEYS:
            current_weight[0] = WEIGHT_KEYS[k]
            weight_explicit[0] = True
            redraw()
        elif k == 'u':
            if not undo_stack:
                return
            action = undo_stack.pop()
            if action[0] == "add" and spans:
                spans.pop()
            elif action[0] == "del":
                spans.append(action[1])
            redraw()
        elif k == 'c':
            spans.clear()
            undo_stack.clear()
            drag_active[0] = False
            drag_start[0]  = None
            drag_end[0]    = None
            redraw()
        elif k == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    redraw()
    plt.show()
    return pd.DataFrame(spans)

def compute_summary(df, baselines, clips_df):
    rows = []
    for _, c in clips_df.iterrows():
        seg = df[(df['timestamp'] >= c["Start"]) & (df['timestamp'] <= c["End"])]
        if seg.empty:
            continue
        y    = seg[c["Sensor"]].astype(float).to_numpy()
        mean = float(np.nanmean(y))
        std  = float(np.nanstd(y))
        base = float(baselines[c["Sensor"]])
        rows.append({"Weight": c["Weight"], "Weight_Val": int(c["Weight_Val"]),
                     "Sensor": c["Sensor"], "Start": float(c["Start"]), "End": float(c["End"]),
                     "Duration_s": float(c["End"] - c["Start"]), "Baseline": base,
                     "Mean": mean, "Std": std, "Delta(Baseline-Mean)": base - mean,
                     "Source": c["Source"]})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Sensor","Weight_Val","Start"]).reset_index(drop=True)
    return out

def plot_sensor_traces(df, baselines, summary):
    """Show interactive sensor trace figures. No auto-save — use the figure
    toolbar save button if you want to keep an image."""
    t_all         = df["timestamp"].to_numpy(dtype=float)
    weight_vals   = sorted(summary["Weight_Val"].unique()) if not summary.empty else []
    cmap          = plt.cm.get_cmap("tab10", max(len(weight_vals), 1))
    weight_colour = {wv: cmap(i) for i, wv in enumerate(weight_vals)}
    for sensor in SENSORS:
        sd    = summary[summary["Sensor"] == sensor]
        y_all = df[sensor].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t_all, y_all, color="black", alpha=0.3, linewidth=0.8, label="raw")
        ax.axhline(float(baselines[sensor]), color="grey", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"baseline ({baselines[sensor]:.1f})")
        legend_weights = {}
        for _, row in sd.iterrows():
            wv  = int(row["Weight_Val"])
            col = "lightgrey" if wv == 0 else weight_colour.get(wv, "steelblue")
            seg = df[(df["timestamp"] >= row["Start"]) & (df["timestamp"] <= row["End"])]
            if not seg.empty:
                ax.plot(seg["timestamp"], seg[sensor], color=col, linewidth=1.8, alpha=0.85)
            patch = ax.axvspan(row["Start"], row["End"], alpha=0.18, facecolor=col,
                               label=row["Weight"] if wv not in legend_weights else "")
            legend_weights[wv] = patch
            mid  = 0.5 * (row["Start"] + row["End"])
            ylim = ax.get_ylim()
            ax.text(mid, ylim[1], row["Weight"], fontsize=7, va="top", ha="center",
                    color=col, fontweight="bold")
        ax.set_title(f"Sensor {sensor} - accepted calibration spans", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Raw value")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
    plt.show()

def plot_summary_grid(summary):
    """Show interactive summary grid. No auto-save — use the figure
    toolbar save button if you want to keep an image."""
    active = [s for s in SENSORS if not summary[summary["Sensor"] == s].empty]
    if not active:
        return
    ncols     = 3
    nrows     = int(np.ceil(len(SENSORS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=False, sharey=False)
    axes_flat = axes.flatten()
    weight_vals   = sorted(summary["Weight_Val"].unique())
    cmap          = plt.cm.get_cmap("tab10", max(len(weight_vals), 1))
    weight_colour = {wv: cmap(i) for i, wv in enumerate(weight_vals)}
    for idx, sensor in enumerate(SENSORS):
        ax = axes_flat[idx]
        sd = summary[summary["Sensor"] == sensor].copy()
        if sd.empty:
            ax.set_title(f"{sensor} (no data)", fontsize=10, color="grey")
            ax.axis("off")
            continue
        agg = (sd.groupby("Weight_Val")["Delta(Baseline-Mean)"]
                 .agg(["mean", "std", "count"]).reset_index())
        agg.columns      = ["Weight_Val", "mean_delta", "std_delta", "n"]
        agg["std_delta"] = agg["std_delta"].fillna(0)
        for _, row in sd.iterrows():
            wv     = int(row["Weight_Val"])
            col    = weight_colour.get(wv, "steelblue")
            jitter = np.random.uniform(-0.5, 0.5)
            ax.scatter(wv + jitter, row["Delta(Baseline-Mean)"], color=col, alpha=0.5, s=30, zorder=3)
        ax.errorbar(agg["Weight_Val"], agg["mean_delta"], yerr=agg["std_delta"],
                    fmt='o-', color="black", linewidth=1.5, capsize=4, capthick=1.5,
                    elinewidth=1.2, markersize=6, zorder=4, label="mean +/- std")
        for _, row in agg.iterrows():
            ax.text(row["Weight_Val"], row["mean_delta"],
                    f'  n={int(row["n"])}', fontsize=7, va="center", color="dimgrey")
        ax.set_title(f"Sensor {sensor}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight (g)", fontsize=8)
        ax.set_ylabel("Delta (baseline - mean)", fontsize=8)
        ax.grid(True, alpha=0.35)
        ax.tick_params(labelsize=8)
    for idx in range(len(SENSORS), len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.suptitle("Calibration Summary - Weight vs Response", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.show()

def build_interp_table(summary):
    if summary.empty:
        return pd.DataFrame()
    grp = (summary.groupby(["Sensor", "Weight_Val"])
           .agg(Weight=("Weight","first"), Mean_Delta=("Delta(Baseline-Mean)","mean"),
                Std_Delta=("Delta(Baseline-Mean)","std"), N_Spans=("Delta(Baseline-Mean)","count"),
                Mean_Raw=("Mean","mean"), Baseline=("Baseline","first"))
           .reset_index()
           .sort_values(["Sensor", "Weight_Val"])
           .reset_index(drop=True))
    grp["Std_Delta"] = grp["Std_Delta"].fillna(0)
    return grp

def save_excel(summary, interp_table, note_table, final_clips, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        interp_table.to_excel(writer, sheet_name="Interpolation_Table", index=False)
        summary.to_excel(writer,      sheet_name="All_Spans_Detail",    index=False)
        final_clips.to_excel(writer,  sheet_name="Span_Boundaries",     index=False)
        note_table.to_excel(writer,   sheet_name="Notes_Used",          index=False)
    print(f"  Saved {path}")

def load_replacement_file():
    """Open a file dialog to pick a replacement Excel/CSV and return the loaded DataFrame."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select replacement sensor Excel/CSV",
        filetypes=[("Excel","*.xlsx"),("CSV","*.csv"),("All","*.*")]
    )
    root.destroy()
    if not file_path:
        return None, None
    df2 = pd.read_excel(file_path) if file_path.lower().endswith(".xlsx") else pd.read_csv(file_path)
    df2 = df2.copy()
    df2['timestamp'] = pd.to_numeric(df2['timestamp'], errors='coerce')
    for s in SENSORS:
        if s in df2.columns:
            df2[s] = pd.to_numeric(df2[s], errors='coerce')
    df2 = df2.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"  Loaded replacement file: {file_path}")
    return df2, file_path

def main():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select combined sensor Excel",
        filetypes=[("Excel","*.xlsx"),("CSV","*.csv"),("All","*.*")]
    )
    root.destroy()
    if not file_path:
        return

    # All outputs saved next to the input file
    out_dir = str(Path(file_path).parent)
    stem    = Path(file_path).stem
    print(f"Input:   {file_path}")
    print(f"Outputs: {out_dir}")

    df = pd.read_excel(file_path) if file_path.lower().endswith(".xlsx") else pd.read_csv(file_path)
    df = df.copy()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    for s in SENSORS:
        df[s] = pd.to_numeric(df[s], errors='coerce')
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    base_df = df[df["timestamp"] <= BASELINE_SECONDS]
    if base_df.empty:
        base_df = df.iloc[:min(len(df), 200)]
    baselines = base_df[SENSORS].median(numeric_only=True)

    print("\nStep 1/3: Edit notes.")
    print("  KEY 0 = unloaded (0g) -- mark at least one unloaded section per recording.")
    print("  This is the anchor point that makes predict_weights.py accurate.")
    note_table = edit_notes(df)
    if note_table.empty:
        raise RuntimeError("No notes found. Add notes with m then 0..6.")

    weight_windows = build_weight_windows_from_note_table(df, note_table)

    print("\nStep 2/3: Span editor.")
    print("  0g spans must be selected manually (auto-detect only works on loaded sections).")
    clips_auto = auto_clips(df, baselines, weight_windows)
    if clips_auto.empty:
        print("  Warning: auto-detection found no spans.")

    edited_all = []
    for i, s in enumerate(SENSORS):
        print(f"  Sensor {s} ({i+1}/{len(SENSORS)}) -- press q when done")
        edited = edit_spans_for_sensor(df, s, baselines, weight_windows, clips_auto)
        if not edited.empty:
            edited["Sensor"] = s
            edited_all.append(edited)
        if edited_all:
            partial_clips   = pd.concat(edited_all, ignore_index=True)
            partial_clips   = partial_clips[["Weight","Weight_Val","Sensor","Start","End","Source"]]
            partial_summary = compute_summary(df, baselines, partial_clips)
            partial_interp  = build_interp_table(partial_summary)
            save_excel(partial_summary, partial_interp, note_table, partial_clips,
                       path=str(Path(out_dir) / f"{stem}_calibration_partial.xlsx"))

    if not edited_all:
        print("No spans found after editing.")
        return

    print("\nStep 3/3: Computing summary and showing plots...")
    final_clips  = pd.concat(edited_all, ignore_index=True)
    final_clips  = final_clips[["Weight","Weight_Val","Sensor","Start","End","Source"]]
    final_clips  = final_clips.sort_values(["Sensor","Weight_Val","Start"]).reset_index(drop=True)
    summary      = compute_summary(df, baselines, final_clips)
    interp_table = build_interp_table(summary)

    # Warn if any sensor is missing a 0g span
    for sensor in SENSORS:
        sd = interp_table[interp_table["Sensor"] == sensor]
        if not sd.empty and 0 not in sd["Weight_Val"].values:
            print(f"  WARNING: {sensor} has no 0g span -- predict_weights.py will add a synthetic anchor.")

    plot_sensor_traces(df, baselines, summary)
    plot_summary_grid(summary)

    final_path = str(Path(out_dir) / f"{stem}_calibration_results.xlsx")
    save_excel(summary, interp_table, note_table, final_clips, path=final_path)

    # ── Sensor replacement loop ──────────────────────────────────────────
    print("\n" + "="*70)
    print("SENSOR REPLACEMENT")
    print("="*70)
    print("If any sensor looked bad, you can now replace its data from a")
    print("different Excel file and re-run calibration for just that sensor.")
    print("You will re-do the note editor on the replacement file so weight")
    print("windows match that file's timestamps.")
    print(f"  Sensors: {', '.join(SENSORS)}")
    print("  Type a sensor name (e.g. A3) to replace, or 'done' to finish.")

    # Cache: replacement filepath -> (df_repl, repl_note_table, repl_weight_windows, repl_baselines)
    replacement_cache = {}
    # Track ALL replaced sensors: sensor_name -> (df_repl, repl_baselines)
    replaced_sensors = {}

    while True:
        choice = input("\nReplace sensor (or 'done'): ").strip().upper()
        if choice in ('DONE', 'Q', ''):
            break
        if choice not in SENSORS:
            print(f"  Unknown sensor '{choice}'. Valid: {', '.join(SENSORS)}")
            continue

        sensor_to_replace = choice
        print(f"\n  Replacing {sensor_to_replace} — pick the replacement file...")
        df_repl, repl_path = load_replacement_file()
        if df_repl is None:
            print("  No file selected, skipping.")
            continue

        # Check if we already processed this replacement file's notes
        if repl_path in replacement_cache:
            print(f"  Reusing notes from previously loaded: {Path(repl_path).name}")
            df_repl, repl_note_table, repl_ww, repl_baselines = replacement_cache[repl_path]
        else:
            # Compute baselines for the replacement file
            repl_base_df = df_repl[df_repl["timestamp"] <= BASELINE_SECONDS]
            if repl_base_df.empty:
                repl_base_df = df_repl.iloc[:min(len(df_repl), 200)]
            repl_baselines = repl_base_df[SENSORS].median(numeric_only=True)

            # Run note editor on the replacement file so weight timestamps are correct
            print(f"\n  NOTE EDITOR for replacement file: {Path(repl_path).name}")
            print("  Mark the weight sections on THIS file's timeline.")
            repl_note_table = edit_notes(df_repl)
            if repl_note_table.empty:
                print("  No notes placed on replacement file, skipping.")
                continue
            repl_ww = build_weight_windows_from_note_table(df_repl, repl_note_table)
            replacement_cache[repl_path] = (df_repl, repl_note_table, repl_ww, repl_baselines)

        # Auto-detect clips on the replacement file for this sensor
        repl_clips = auto_clips(df_repl, repl_baselines, repl_ww)
        if not repl_clips.empty:
            repl_clips = repl_clips[repl_clips["Sensor"] == sensor_to_replace]

        # Let user edit spans on the replacement file's data and timeline
        print(f"  Opening span editor for {sensor_to_replace} on replacement data...")
        print("  Use 0-6 keys to set weight before drawing spans if auto-detect is wrong.")
        edited = edit_spans_for_sensor(
            df_repl, sensor_to_replace, repl_baselines, repl_ww, repl_clips
        )

        # Update final_clips: remove old entries for this sensor, add new ones
        final_clips = final_clips[final_clips["Sensor"] != sensor_to_replace]
        if not edited.empty:
            edited["Sensor"] = sensor_to_replace
            edited = edited[["Weight","Weight_Val","Sensor","Start","End","Source"]]
            final_clips = pd.concat([final_clips, edited], ignore_index=True)

        final_clips = final_clips.sort_values(["Sensor","Weight_Val","Start"]).reset_index(drop=True)

        # Remember this replacement so future iterations use the right data
        replaced_sensors[sensor_to_replace] = (df_repl, repl_baselines)

        # Recompute summary — use each sensor's own data source
        # (replacement file for any replaced sensor, original df for the rest)
        summary_parts = []
        for s in SENSORS:
            s_clips = final_clips[final_clips["Sensor"] == s]
            if s_clips.empty:
                continue
            if s in replaced_sensors:
                s_df, s_bl = replaced_sensors[s]
                s_summary = compute_summary(s_df, s_bl, s_clips)
            else:
                s_summary = compute_summary(df, baselines, s_clips)
            summary_parts.append(s_summary)

        if summary_parts:
            summary = pd.concat(summary_parts, ignore_index=True)
            summary = summary.sort_values(["Sensor","Weight_Val","Start"]).reset_index(drop=True)
        else:
            summary = pd.DataFrame()
        interp_table = build_interp_table(summary)

        # Show updated trace for the replaced sensor (from replacement file)
        print(f"\n  Showing updated trace for {sensor_to_replace} (from replacement file)...")
        sensor_summary = summary[summary["Sensor"] == sensor_to_replace]
        if not sensor_summary.empty:
            t_all = df_repl["timestamp"].to_numpy(dtype=float)
            y_all = df_repl[sensor_to_replace].to_numpy(dtype=float)
            weight_vals = sorted(summary["Weight_Val"].unique())
            cmap_local  = plt.cm.get_cmap("tab10", max(len(weight_vals), 1))
            wc = {wv: cmap_local(i) for i, wv in enumerate(weight_vals)}
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(t_all, y_all, color="black", alpha=0.3, linewidth=0.8, label="raw")
            ax.axhline(float(repl_baselines[sensor_to_replace]), color="grey", linestyle="--",
                       linewidth=1, alpha=0.7, label=f"baseline ({repl_baselines[sensor_to_replace]:.1f})")
            for _, row in sensor_summary.iterrows():
                wv  = int(row["Weight_Val"])
                col = "lightgrey" if wv == 0 else wc.get(wv, "steelblue")
                seg = df_repl[(df_repl["timestamp"] >= row["Start"]) & (df_repl["timestamp"] <= row["End"])]
                if not seg.empty:
                    ax.plot(seg["timestamp"], seg[sensor_to_replace], color=col, linewidth=1.8, alpha=0.85)
                ax.axvspan(row["Start"], row["End"], alpha=0.18, facecolor=col)
                mid = 0.5 * (row["Start"] + row["End"])
                ylim = ax.get_ylim()
                ax.text(mid, ylim[1], row["Weight"], fontsize=7, va="top", ha="center",
                        color=col, fontweight="bold")
            ax.set_title(f"Sensor {sensor_to_replace} (REPLACED from {Path(repl_path).name}) - calibration spans", fontsize=11)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Raw value")
            ax.grid(True, alpha=0.4)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            plt.show()

        # Show updated summary grid
        plot_summary_grid(summary)

        # Save updated results
        save_excel(summary, interp_table, note_table, final_clips, path=final_path)
        print(f"  Updated results saved to {final_path}")

    print(f"\nDone. Final results: {final_path}")

if __name__ == "__main__":
    main()