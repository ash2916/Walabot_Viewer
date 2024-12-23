import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import WalabotAPI as wb

# Predefined defaults for two profiles (Cartesian only)
PROFILE_DEFAULTS = {
    "PROF_SENSOR": {
        "xMin": -10, "xMax": 10, "xRes": 1.0,
        "yMin": -10, "yMax": 10, "yRes": 1.0,
        "zMin": 1,   "zMax": 20, "zRes": 1.0
    },
    "PROF_SHORT_RANGE_IMAGING": {
        "xMin": -5,  "xMax": 5,   "xRes": 0.5,
        "yMin": -5,  "yMax": 5,   "yRes": 0.5,
        "zMin": 1,   "zMax": 10,  "zRes": 0.5
    }
}

class WalabotArcApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Walabot - Target Viewer")
        self._walabot_initialized = False
        self._scanning = False
        self.plot_mode = "2D"
        self._create_widgets()
        self.after_id = None

    def _create_widgets(self):
        container = tk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(container, bd=2, relief=tk.GROOVE, padx=10, pady=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        self.plot_area = tk.Frame(container, bd=2, relief=tk.GROOVE)
        self.plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(controls, text="Profile:").pack(anchor=tk.W)
        self.profile_combo = ttk.Combobox(controls, state="readonly",
            values=["PROF_SENSOR","PROF_SHORT_RANGE_IMAGING"])
        self.profile_combo.current(0)
        self.profile_combo.pack(anchor=tk.W, fill=tk.X)
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_changed)

        tk.Label(controls, text="Filter:").pack(anchor=tk.W, pady=(10,0))
        self.filter_combo = ttk.Combobox(controls, state="readonly",
            values=["FILTER_TYPE_NONE","FILTER_TYPE_MTI","FILTER_TYPE_DERIVATIVE"])
        self.filter_dict = {
            "FILTER_TYPE_NONE": wb.FILTER_TYPE_NONE,
            "FILTER_TYPE_MTI": wb.FILTER_TYPE_MTI,
            "FILTER_TYPE_DERIVATIVE": wb.FILTER_TYPE_DERIVATIVE
        }
        self.filter_combo.current(1)
        self.filter_combo.pack(anchor=tk.W, fill=tk.X)

        arena_frame = tk.LabelFrame(controls, text="Cartesian Arena")
        arena_frame.pack(anchor=tk.W, fill=tk.X, pady=5)

        xx = tk.Frame(arena_frame)
        xx.pack(anchor=tk.W)
        tk.Label(xx, text="X min").pack(side=tk.LEFT)
        self.xMinVar = tk.DoubleVar()
        tk.Entry(xx, textvariable=self.xMinVar, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(xx, text="X max").pack(side=tk.LEFT)
        self.xMaxVar = tk.DoubleVar()
        tk.Entry(xx, textvariable=self.xMaxVar, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(xx, text="X res").pack(side=tk.LEFT)
        self.xResVar = tk.DoubleVar()
        tk.Entry(xx, textvariable=self.xResVar, width=5).pack(side=tk.LEFT, padx=5)

        yy = tk.Frame(arena_frame)
        yy.pack(anchor=tk.W)
        tk.Label(yy, text="Y min").pack(side=tk.LEFT)
        self.yMinVar = tk.DoubleVar()
        tk.Entry(yy, textvariable=self.yMinVar, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(yy, text="Y max").pack(side=tk.LEFT)
        self.yMaxVar = tk.DoubleVar()
        tk.Entry(yy, textvariable=self.yMaxVar, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(yy, text="Y res").pack(side=tk.LEFT)
        self.yResVar = tk.DoubleVar()
        tk.Entry(yy, textvariable=self.yResVar, width=5).pack(side=tk.LEFT, padx=5)

        zz = tk.Frame(arena_frame)
        zz.pack(anchor=tk.W)
        tk.Label(zz, text="Z min").pack(side=tk.LEFT)
        self.zMinVar = tk.DoubleVar()
        tk.Entry(zz, textvariable=self.zMinVar, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(zz, text="Z max").pack(side=tk.LEFT)
        self.zMaxVar = tk.DoubleVar()
        tk.Entry(zz, textvariable=self.zMaxVar, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(zz, text="Z res").pack(side=tk.LEFT)
        self.zResVar = tk.DoubleVar()
        tk.Entry(zz, textvariable=self.zResVar, width=5).pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(controls)
        btn_frame.pack(pady=(10,0), fill=tk.X)
        tk.Button(btn_frame, text="Start", command=self.on_start).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Stop", command=self.on_stop).pack(side=tk.LEFT, expand=True, fill=tk.X)

        mode_frame = tk.LabelFrame(controls, text="Plot Mode")
        mode_frame.pack(pady=(10,0), fill=tk.X)
        tk.Button(mode_frame, text="2D Arc", command=lambda: self.set_plot_mode("2D")).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(mode_frame, text="3D", command=lambda: self.set_plot_mode("3D")).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(mode_frame, text="RAW", command=lambda: self.set_plot_mode("RAW")).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.panel_2d = tk.Frame(self.plot_area)
        self.panel_3d = tk.Frame(self.plot_area)
        self.panel_raw = tk.Frame(self.plot_area)

        self.canvas_2d = tk.Canvas(self.panel_2d, bg="light gray")
        self.canvas_2d.pack(fill=tk.BOTH, expand=True)

        self.fig_3d = plt.Figure(figsize=(5,4), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.panel_3d)
        self.canvas_3d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_raw = plt.Figure(figsize=(5,4), dpi=100)
        self.ax_raw = self.fig_raw.add_subplot(111)
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.panel_raw)
        self.canvas_raw.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.panel_2d.pack(fill=tk.BOTH, expand=True)
        self.load_profile_defaults("PROF_SENSOR")

    def on_profile_changed(self, event):
        prof = self.profile_combo.get()
        self.load_profile_defaults(prof)

    def load_profile_defaults(self, prof):
        if prof not in PROFILE_DEFAULTS:
            return
        d = PROFILE_DEFAULTS[prof]
        self.xMinVar.set(d["xMin"])
        self.xMaxVar.set(d["xMax"])
        self.xResVar.set(d["xRes"])
        self.yMinVar.set(d["yMin"])
        self.yMaxVar.set(d["yMax"])
        self.yResVar.set(d["yRes"])
        self.zMinVar.set(d["zMin"])
        self.zMaxVar.set(d["zMax"])
        self.zResVar.set(d["zRes"])

    def set_plot_mode(self, mode):
        self.plot_mode = mode
        self.panel_2d.pack_forget()
        self.panel_3d.pack_forget()
        self.panel_raw.pack_forget()
        if mode == "2D":
            self.panel_2d.pack(fill=tk.BOTH, expand=True)
        elif mode == "3D":
            self.panel_3d.pack(fill=tk.BOTH, expand=True)
        else:
            self.panel_raw.pack(fill=tk.BOTH, expand=True)
        if not self._scanning:
            if mode == "2D":
                self.canvas_2d.delete("all")
            elif mode == "3D":
                self.fig_3d.clear()
                self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
                self.canvas_3d.draw()
            else:
                self.fig_raw.clear()
                self.ax_raw = self.fig_raw.add_subplot(111)
                self.canvas_raw.draw()

    def on_start(self):
        if self._scanning:
            return
        if not self._walabot_initialized:
            try:
                wb.Init()
                wb.SetSettingsFolder()
                wb.ConnectAny()
                self._walabot_initialized = True
            except wb.WalabotError as e:
                print("Cannot init/connect Walabot:", e)
                return
        prof = self.profile_combo.get()
        if prof == "PROF_SENSOR":
            wb.SetProfile(wb.PROF_SENSOR)
        else:
            wb.SetProfile(wb.PROF_SHORT_RANGE_IMAGING)

        ftxt = self.filter_combo.get()
        if ftxt in self.filter_dict:
            wb.SetDynamicImageFilter(self.filter_dict[ftxt])

        wb.SetArenaX(self.xMinVar.get(), self.xMaxVar.get(), self.xResVar.get())
        wb.SetArenaY(self.yMinVar.get(), self.yMaxVar.get(), self.yResVar.get())
        wb.SetArenaZ(self.zMinVar.get(), self.zMaxVar.get(), self.zResVar.get())

        wb.Start()
        wb.StartCalibration()
        stat, prog = wb.GetStatus()
        while stat == wb.STATUS_CALIBRATING and prog < 100:
            wb.Trigger()
            stat, prog = wb.GetStatus()
            print("Calibrating... ", prog, "%")

        self._scanning = True
        self.update_loop()
        print("Scanning started.")

    def on_stop(self):
        if not self._scanning:
            return
        self._scanning = False
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        try:
            wb.Stop()
        except wb.WalabotError as e:
            print("Error stopping:", e)
        print("Stopped scanning.")

    def update_loop(self):
        if not self._scanning:
            return
        try:
            wb.Trigger()
            if self.plot_mode == "2D":
                self.update_2d_arc()
            elif self.plot_mode == "3D":
                self.update_3d()
            else:
                self.update_raw()
        except wb.WalabotError as e:
            print("Walabot scanning error:", e)
        self.after_id = self.after(100, self.update_loop)

    # 2D arc
    def update_2d_arc(self):
        cv = self.canvas_2d
        w = cv.winfo_width()
        h = cv.winfo_height()
        cv.delete("all")
        cv.configure(bg="black")
        prof = self.profile_combo.get()
        if prof == "PROF_SHORT_RANGE_IMAGING":
            targets = wb.GetImagingTargets()
        else:
            targets = wb.GetSensorTargets()

        xMin = self.xMinVar.get()
        xMax = self.xMaxVar.get()
        zMin = self.zMinVar.get()
        zMax = self.zMaxVar.get()

        center_x = w / 2
        center_y = h - 10
        radius_px = min(w, h) * 0.9 / 2

        slice_count = 100
        x_range = xMax - xMin if (xMax - xMin) != 0 else 1
        for i in range(slice_count):
            frac1 = i / slice_count
            frac2 = (i + 1) / slice_count
            x_val1 = xMin + frac1 * x_range
            x_val2 = xMin + frac2 * x_range
            ang_deg1 = 180 * (1 - ((x_val1 - xMin) / x_range))
            ang_deg2 = 180 * (1 - ((x_val2 - xMin) / x_range))
            ang_rad1 = np.deg2rad(ang_deg1)
            ang_rad2 = np.deg2rad(ang_deg2)
            x1 = center_x + radius_px * np.cos(ang_rad1)
            y1 = center_y - radius_px * np.sin(ang_rad1)
            x2 = center_x + radius_px * np.cos(ang_rad2)
            y2 = center_y - radius_px * np.sin(ang_rad2)
            cv.create_polygon(
                center_x, center_y,
                x1, y1,
                x2, y2,
                fill="green", outline=""
            )

        line_count = 6
        for i in range(line_count + 1):
            frac = i / line_count
            ang_deg = 180 * (1 - frac)
            ang_rad = np.deg2rad(ang_deg)
            x2 = center_x + radius_px * np.cos(ang_rad)
            y2 = center_y - radius_px * np.sin(ang_rad)
            cv.create_line(center_x, center_y, x2, y2, fill="black", width=2)

        for i, t in enumerate(targets):
            x_val = t.xPosCm
            z_val = t.zPosCm
            if z_val < zMin:
                z_val = zMin
            if z_val > zMax:
                z_val = zMax

            x_range2 = (xMax - xMin) or 1
            frac_x = (x_val - xMin) / x_range2 if x_range2 != 0 else 0.5
            angle_deg = 180 * (1 - frac_x)
            angle_rad = np.deg2rad(angle_deg)

            z_range = (zMax - zMin) or 1
            frac_z = (z_val - zMin) / z_range if z_range != 0 else 0
            if frac_z < 0: frac_z = 0
            if frac_z > 1: frac_z = 1
            r_px = radius_px * frac_z

            tx = center_x + r_px * np.cos(angle_rad)
            ty = center_y - r_px * np.sin(angle_rad)

            size_px = 10
            cv.create_oval(
                tx - size_px, ty - size_px,
                tx + size_px, ty + size_px,
                fill="red", outline="black", width=1
            )
            cv.create_text(tx, ty, text=str(i + 1), fill="black", font=("Arial", 10, "bold"))

    # 3D
    def update_3d(self):
        self.fig_3d.clear()
        ax3d = self.fig_3d.add_subplot(111, projection='3d')
        ax3d.set_title("3D (x,y,z)")
        prof = self.profile_combo.get()
        if prof == "PROF_SHORT_RANGE_IMAGING":
            targets = wb.GetImagingTargets()
        else:
            targets = wb.GetSensorTargets()

        xMin, xMax = self.xMinVar.get(), self.xMaxVar.get()
        yMin, yMax = self.yMinVar.get(), self.yMaxVar.get()
        zMin, zMax = self.zMinVar.get(), self.zMaxVar.get()
        ax3d.set_xlim(xMin, xMax)
        ax3d.set_ylim(yMin, yMax)
        ax3d.set_zlim(zMin, zMax)

        xs = [t.xPosCm for t in targets]
        ys = [t.yPosCm for t in targets]
        zs = [t.zPosCm for t in targets]

        ax3d.scatter(xs, ys, zs, c='r', marker='o')
        self.canvas_3d.draw()

    # RAW
    def update_raw(self):
        self.fig_raw.clear()
        ax = self.fig_raw.add_subplot(111)
        ax.set_title("Raw Image Slice (2D Heatmap)")

        M, _, _, _, _ = wb.GetRawImageSlice()
        raw_data = np.array([val for row in M for val in row], dtype=np.float32)
        rows = len(M)
        cols = len(M[0]) if rows>0 else 0
        if rows*cols == raw_data.size and rows>0 and cols>0:
            arr2d = raw_data.reshape((rows, cols))
            ax.imshow(arr2d, cmap='jet', aspect='auto')
        else:
            ax.plot(raw_data, 'r-')
        self.canvas_raw.draw()

    def on_close(self):
        self.on_stop()
        if self._walabot_initialized:
            try:
                wb.Disconnect()
            except:
                pass
        self.destroy()

def main():
    app = WalabotArcApp()
    app.mainloop()

if __name__ == "__main__":
    main()
