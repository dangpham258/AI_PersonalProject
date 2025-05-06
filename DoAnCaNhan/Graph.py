import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import re

class PuzzleVisualizer:
    def __init__(self, root):
        self.root = root

    def _abbreviate_algorithm_name(self, name, max_length=12):
        """Viết tắt tên thuật toán nếu quá dài."""
        if len(name) <= max_length:
            return name
        words = re.findall(r'[A-Za-z0-9]+', name)
        if len(words) > 1:
            abbr = ''.join(w[0].upper() for w in words)
            if len(abbr) <= max_length:
                return abbr
            return abbr[:max_length-1] + '…'
        return name[:max_length-1] + '…'

    def _add_data_labels(self, ax, bars, fmt="{:.2f}"):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    h * 1.02,
                    fmt.format(h),
                    ha='center', va='bottom', fontsize=9
                )

    def create_comparison_chart(self, data, algorithms, title="So sánh hiệu suất thuật toán"):
        valid = [alg for alg in algorithms if alg in data]
        if not valid:
            return None, "Không có dữ liệu hợp lệ để so sánh"

        display = []
        full_map = {}
        for alg in valid:
            abbr = self._abbreviate_algorithm_name(alg)
            display.append(abbr)
            if abbr != alg:
                full_map[abbr] = alg

        times = [data[alg][0] for alg in valid]
        steps = [data[alg][1] or 0 for alg in valid]
        spaces = [data[alg][2] for alg in valid]

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        titles = ["Thời gian (s)", "Số bước", "Không gian trạng thái"]
        values = [times, steps, spaces]
        colors = ["#4CAF50", "#2196F3", "#FF9800"]

        for ax, val, t, col in zip(axes, values, titles, colors):
            bars = ax.bar(display, val, color=col)
            ax.set_title(t)
            if ax != axes[-1]:
                ax.set_xticklabels([])
            else:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            self._add_data_labels(ax, bars, fmt="{:.4f}" if t == "Thời gian (s)" else "{}")

        plt.tight_layout(h_pad=2)

        idx_time = times.index(min(times)) if times else None
        best_time = valid[idx_time] if idx_time is not None else "N/A"
        nonzero_steps = [s for s in steps if s > 0]
        best_step = valid[steps.index(min(nonzero_steps))] if nonzero_steps else "N/A"
        idx_space = spaces.index(min(spaces)) if spaces else None
        best_space = valid[idx_space] if idx_space is not None else "N/A"

        def fm(x): return full_map.get(x, x)
        summary = (
            f"Tóm tắt hiệu suất tốt nhất:\n"
            f"• Nhanh nhất: {fm(best_time)} ({min(times):.4f}s)\n"
            f"• Ít bước nhất: {fm(best_step)} ({min(nonzero_steps) if nonzero_steps else 'N/A'} bước)\n"
            f"• Không gian nhỏ nhất: {fm(best_space)} ({min(spaces) if spaces else 'N/A'} nút)"
        )

        return fig, summary

    def display_chart(self, fig, summary, window_title="Kết quả so sánh"):
        if fig is None:
            return
        win = tk.Toplevel(self.root)
        win.title(window_title)
        win.geometry("1000x650")

        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas = FigureCanvasTkAgg(fig, master=left)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(frame, bd=2, relief=tk.RIDGE, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10,0))
        right.pack_propagate(False)

        lbl = tk.Label(right, text="KẾT QUẢ PHÂN TÍCH", font=("Arial", 12, "bold"), bg="#e0e0e0")
        lbl.pack(fill=tk.X)
        txt = tk.Label(right, text=summary, font=("Arial", 11), justify=tk.LEFT, wraplength=280)
        txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn = tk.Button(win, text="Đóng", command=win.destroy, font=("Arial", 11), width=10)
        btn.pack(pady=10)

    def create_custom_comparison(self, data, selected, title="So sánh tùy chọn"):
        valid = [alg for alg in selected if alg in data]
        if not valid:
            return None, "Không có thuật toán được chọn"

        display = []
        full_map = {}
        for alg in valid:
            abbr = self._abbreviate_algorithm_name(alg)
            display.append(abbr)
            if abbr != alg:
                full_map[abbr] = alg
        
        times = [data[alg][0] for alg in valid]
        steps = [data[alg][1] or 0 for alg in valid]
        spaces = [data[alg][2] for alg in valid]

        fig, ax = plt.subplots(figsize=(9, 6))
        idx = np.arange(len(display))
        w = 0.25

        b1 = ax.bar(idx - w, times, w, label='Thời gian', hatch='\\')
        b2 = ax.bar(idx, steps, w, label='Số bước')
        b3 = ax.bar(idx + w, spaces, w, label='Không gian')

        ax.set_xticks(idx)
        ax.set_xticklabels(display)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(title)
        ax.legend()

        self._add_data_labels(ax, b1, fmt="{:.4f}s")
        self._add_data_labels(ax, b2)
        self._add_data_labels(ax, b3)

        maxv = max(max(times), max(steps), max(spaces))
        minv = min(t for t in times+steps+spaces if t>0)
        if maxv / minv > 100:
            ax.set_yscale('log')

        idx_time = times.index(min(times))
        idx_step = steps.index(min(s for s in steps if s>0)) if any(steps) else None
        idx_space = spaces.index(min(spaces))

        def fm(x): return full_map.get(x, x)
        summary = (
            f"So sánh tùy chọn:\n"
            f"• Nhanh nhất: {fm(valid[idx_time])} ({min(times):.4f}s)\n"
            f"• Ít bước nhất: {fm(valid[idx_step])} ({min(s for s in steps if s>0)} bước)\n"
            f"• Không gian nhỏ nhất: {fm(valid[idx_space])} ({min(spaces)} nút)"
        )
        return fig, summary
