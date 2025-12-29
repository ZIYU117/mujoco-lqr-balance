# balance_ui_fb_v7.py
# - UI only Forward/Stop/Reverse
# - Fix plot/logging so convergence figure ALWAYS appears (and always saved)
# - Generate the same 5-panel figure style you showed (Pitch/PitchRate/uL-uR/Omega/v)
# - Even if you stop with Ctrl+C, it will still save the plot.

import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import os
import matplotlib

# If no GUI display, use Agg so we can still save PNG.
HEADLESS = (os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "")
if HEADLESS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plot options
SHOW_PLOT_WINDOW = True  # Set False to never pop up a matplotlib window
PLOT_PITCH = True       # Disable Pitch plot
PLOT_PITCH_RATE = True  # Disable Pitch Rate plot

import inverted_pendulum_lqr_control as ip

XML_PATH = "crazydog_urdf/urdf/scene.xml"

S_IMU_QUAT = "imu_quat"
S_IMU_GYRO = "imu_gyro"
S_L_WVEL   = "L_wheel_vel"
S_R_WVEL   = "R_wheel_vel"

ACT_L_WHEEL = "L_wheel"
ACT_R_WHEEL = "R_wheel"
ACT_L_THIGH = "L_thigh"
ACT_L_CALF  = "L_calf"
ACT_R_THIGH = "R_thigh"
ACT_R_CALF  = "R_calf"


def quat_to_pitch(wxyz):
    # MuJoCo quat order: [w, x, y, z]
    w, x, y, z = [float(v) for v in wxyz]
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        return float(np.sign(sinp) * (np.pi / 2.0))
    return float(np.arcsin(sinp))


def infer_sign_from_joint_axes(model, left_act_id: int, right_act_id: int) -> float:
    """Guess if right wheel should be flipped by comparing hinge axes."""
    try:
        lj = int(model.actuator_trnid[left_act_id, 0])
        rj = int(model.actuator_trnid[right_act_id, 0])
        axisL = np.array(model.jnt_axis[lj], dtype=float)
        axisR = np.array(model.jnt_axis[rj], dtype=float)
        if float(axisL @ axisR) < 0.0:
            return -1.0
    except Exception:
        pass
    return 1.0


def print_config(act_sign_r: float, sense_sign_r: float, omega_sign: float):
    print(f"[CFG] ACT_SIGN_R={act_sign_r:+.0f}  SENSE_SIGN_R={sense_sign_r:+.0f}  OMEGA_SIGN={omega_sign:+.0f}")


# ----------------------------
# Tkinter UI (thread)
# ----------------------------

def start_ui(cmd, lock):
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Balance Teleop (Forward / Reverse only)")

    v_min, v_max = 0.0, -20.0  # (m/s)
    vmag_var = tk.DoubleVar(value=0.00)  # (m/s)
    en_var = tk.BooleanVar(value=True)
    dir_var = tk.IntVar(value=0)  # +1 forward, -1 reverse, 0 stop

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)

    info = ttk.Label(
        root,
        text=(
            "Hold: W(or ↑)=Forward, S(or ↓)=Reverse, Space=STOP.  Turn is disabled.\n"
            "Hotkeys: F flip ACT sign (right output), G flip SENSE sign (right vel), O flip omega sign, Esc quit.\n"
            "Tip: If it keeps spinning, try F then G."
        ),
        wraplength=520,
    )
    info.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 6))

    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    btn_frame.columnconfigure(0, weight=1)

    def set_dir(d):
        dir_var.set(int(d))

    def stop_all():
        dir_var.set(0)

    fwd = ttk.Button(btn_frame, text="▲  FORWARD  (hold)", width=22)
    fwd.grid(row=0, column=0, sticky="we", pady=(0, 8))
    stp = ttk.Button(btn_frame, text="■  STOP", command=stop_all, width=22)
    stp.grid(row=1, column=0, sticky="we")
    rev = ttk.Button(btn_frame, text="▼  REVERSE  (hold)", width=22)
    rev.grid(row=2, column=0, sticky="we", pady=(8, 0))

    fwd.bind("<ButtonPress-1>", lambda e: set_dir(+1))
    fwd.bind("<ButtonRelease-1>", lambda e: set_dir(0))
    rev.bind("<ButtonPress-1>", lambda e: set_dir(-1))
    rev.bind("<ButtonRelease-1>", lambda e: set_dir(0))

    ctrl_frame = ttk.Frame(root)
    ctrl_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
    ctrl_frame.columnconfigure(0, weight=1)

    ttk.Label(ctrl_frame, text="Speed magnitude (m/s)").grid(row=0, column=0, sticky="w")
    v_slider = ttk.Scale(ctrl_frame, from_=v_min, to=v_max, variable=vmag_var, length=280)
    v_slider.grid(row=1, column=0, sticky="we", pady=(2, 6))
    v_read = ttk.Label(ctrl_frame, text="+0.00")
    v_read.grid(row=2, column=0, sticky="e")

    en_chk = ttk.Checkbutton(ctrl_frame, text="Enable LQR", variable=en_var)
    en_chk.grid(row=3, column=0, sticky="w", pady=(8, 2))

    def update_cmd():
        vmag = float(vmag_var.get())
        d = int(dir_var.get())
        v_ref_mps = d * vmag  # UI command in m/s

        with lock:
            cmd["v_ref_mps"] = v_ref_mps
            cmd["enable"] = bool(en_var.get())
        disp = -v_ref_mps   # 你要顯示反值就用 -v_ref_mps；不反就用 v_ref_mps
        if abs(disp) < 1e-9:
            disp = 0.0
        v_read.config(text=f"{disp:+.2f}")

        # v_read.config(text=f"{(-v_ref_mps):+.2f}")
        root.after(30, update_cmd)

    def on_key_down(e):
        k = e.keysym.lower()
        if k in ("w", "up"):
            set_dir(+1)
        elif k in ("s", "down"):
            set_dir(-1)
        elif k == "space":
            stop_all()
        elif k == "f":
            with lock:
                cmd["flip_act"] = True
        elif k == "g":
            with lock:
                cmd["flip_sense"] = True
        elif k == "o":
            with lock:
                cmd["flip_omega"] = True
        elif k == "escape":
            with lock:
                cmd["quit"] = True
            root.destroy()

    def on_key_up(e):
        k = e.keysym.lower()
        if k in ("w", "up", "s", "down"):
            stop_all()

    root.bind("<KeyPress>", on_key_down)
    root.bind("<KeyRelease>", on_key_up)
    root.focus_force()

    def quit_all():
        with lock:
            cmd["quit"] = True
        root.destroy()

    ttk.Button(root, text="Quit (Esc)", command=quit_all).grid(row=2, column=1, sticky="e", padx=10, pady=(0, 12))

    update_cmd()
    root.mainloop()


# ----------------------------
# Plot helper (your screenshot style)
# ----------------------------


def save_and_show_plot(
    t_log,
    pitch_log,
    pitch_rate_log,
    uL_log,
    uR_log,
    omega_log,
    v_log,
    vref_ui_log,
    vref_f_log,
    enable_log,
    dir_log,
):
    """Save a convergence figure as PNG (and optionally show a window).

    Adds extra panels so you can see the *UI control state* on the plot:
    - v_ref_ui (what UI sends)
    - v_ref_f   (filtered command used by controller)
    - enable (0/1)
    - direction (REV/STOP/FWD)
    """
    if len(t_log) < 5:
        print("[PLOT] not enough data")
        return None

    # Build panels dynamically
    panels = []
    def draw_pitch(ax):
        ax.plot(t_log, pitch_log)
        ax.set_ylim(-0.10, 0.10)

    def draw_pitch_rate(ax):
        ax.plot(t_log, pitch_rate_log)

    if PLOT_PITCH:
        panels.append(("Pitch Angle vs Time", "Pitch (rad)", draw_pitch))
    if PLOT_PITCH_RATE:
        panels.append(("Pitch Rate vs Time", "Pitch rate (rad/s)", draw_pitch_rate))

    def draw_wheel(ax):
        ax.plot(t_log, uL_log, label="uL")
        ax.plot(t_log, uR_log, label="uR")
        ax.legend()

    def draw_omega(ax):
        ax.set_ylim(-0.10, 0.10)
        ax.plot(t_log, omega_log)

    def draw_v(ax):
        PLOT_SIGN = -1.0
        ax.plot(t_log, [PLOT_SIGN*x for x in v_log], label="v_meas")
        ax.plot(t_log, [PLOT_SIGN*x for x in vref_f_log], label="v_ref_f")
        ax.plot(t_log, [PLOT_SIGN*x for x in vref_ui_log], label="v_ref_ui")

        ax.legend()

    # def draw_enable(ax):
    #     ax.step(t_log, enable_log, where="post")
    #     ax.set_ylim(-0.05, 1.05)

    def draw_dir(ax):
        PLOT_SIGN = -1.0
        ax.step(t_log, [PLOT_SIGN*x for x in dir_log], where="post")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["REV", "STOP", "FWD"])
        ax.set_ylim(-1.2, 1.2)

    panels += [
        ("Wheel Control (Wheel Torque Command) vs Time", "ctrl", draw_wheel),
        ("Omega (yaw rate) vs Time", "omega", draw_omega),
        ("Forward speed v + UI command (m/s) vs Time", "m/s", draw_v),
        # ("UI Enable (1=on, 0=off)", "enable", draw_enable),
        ("UI Direction", "dir", draw_dir),
    ]

    fig, ax = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)), sharex=True)
    if len(panels) == 1:
        ax = [ax]

    for i, (title, ylabel, draw) in enumerate(panels):
        draw(ax[i])
        ax[i].set_title(title)
        ax[i].set_ylabel(ylabel)
        ax[i].grid(True)

    ax[-1].set_xlabel("Time (s)")

    plt.tight_layout()

    out_png = f"convergence_{time.strftime('%Y%m%d_%H%M%S')}.png"
    try:
        plt.savefig(out_png, dpi=160)
        print(f"[PLOT] saved -> {out_png}")
    except Exception as e:
        print(f"[PLOT] save failed: {e}")
        out_png = None

    # Only show a window if we have a GUI display.
    if SHOW_PLOT_WINDOW and (not HEADLESS):
        try:
            plt.show()
        except Exception as e:
            print(f"[PLOT] show failed: {e}")
    else:
        plt.close(fig)

    return out_png


def main():
    ACT_SIGN_R = 1.0
    SENSE_SIGN_R = 1.0
    OMEGA_SIGN = 1.0

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    L_wheel = model.actuator(ACT_L_WHEEL).id
    R_wheel = model.actuator(ACT_R_WHEEL).id
    L_thigh = model.actuator(ACT_L_THIGH).id
    L_calf  = model.actuator(ACT_L_CALF).id
    R_thigh = model.actuator(ACT_R_THIGH).id
    R_calf  = model.actuator(ACT_R_CALF).id

    auto = infer_sign_from_joint_axes(model, L_wheel, R_wheel)
    ACT_SIGN_R = float(auto)
    SENSE_SIGN_R = float(auto)

    print_config(ACT_SIGN_R, SENSE_SIGN_R, OMEGA_SIGN)

    KP, KD = 50.0, 5.0
    sim_dt = float(model.opt.timestep)
    control_dt = 0.01

    ip.delta_t = control_dt
    A, B = ip.get_model_matrix()
    K, _, _ = ip.dlqr(A, B, ip.Q, ip.R)
    np.set_printoptions(precision=6, suppress=True)
    print("\n=== System matrices (discrete) ===")
    print("control_dt =", control_dt)
    print("A=\n", A)
    print("B=\n", B)
    print("\n=== LQR config ===")
    print("Q=\n", ip.Q)
    print("R=\n", ip.R)
    print("K=\n", K)

    cmd = {
        "v_ref": 0.0,
        "enable": True,
        "flip_act": False,
        "flip_sense": False,
        "flip_omega": False,
        "quit": False,
    }
    lock = threading.Lock()
    threading.Thread(target=start_ui, args=(cmd, lock), daemon=True).start()


    # THIGH_BEND = -0.25
    # CALF_BEND  =  0.50
    THIGH_BEND =  -0.15
    CALF_BEND  =  0.15
    q_L_thigh0 = 1.27 + THIGH_BEND
    q_R_thigh0 = 1.27 + THIGH_BEND
    q_L_calf0  = -2.127 + CALF_BEND
    q_R_calf0  = -2.127 + CALF_BEND
    
    # ================= [新增] 初始化物理狀態 =================
    print("[INIT] Resetting robot state to remove randomness...")
    mujoco.mj_resetData(model, data)  # 1. 重置所有速度與位置到 XML 預設值

    # 2. 強制設定腿部關節位置 (透過 actuator 找到對應 joint)
    # 這樣機器人一出場就是 "蹲好" 的狀態，不會彈跳
    def set_joint_qpos(act_name, target_pos):
        act_id = model.actuator(act_name).id
        jnt_id = model.actuator_trnid[act_id, 0] # 找到該 actuator 控制的 joint
        qadr = model.jnt_qposadr[jnt_id]         # 找到 qpos 陣列中的索引
        data.qpos[qadr] = target_pos
        data.qvel[:] = 0.0  # 強制將所有速度歸零
        mujoco.mj_forward(model, data) # 確保 kinematics 更新
    set_joint_qpos(ACT_L_THIGH, q_L_thigh0)
    set_joint_qpos(ACT_R_THIGH, q_R_thigh0)
    set_joint_qpos(ACT_L_CALF, q_L_calf0)
    set_joint_qpos(ACT_R_CALF, q_R_calf0)

    # 3. 給 root free joint 一個固定的初始傾角 (pitch)
    # MuJoCo free joint 的 qpos 佈局是: [pos(3), quat(4)]，quat 順序為 [w, x, y, z]
    init_pitch = 0.01  # 初始傾角 (rad)
    free_jids = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    if free_jids.size == 0:
        raise RuntimeError("No free joint found; cannot set initial pitch.")
    root_jid = int(free_jids[0])
    qadr = int(model.jnt_qposadr[root_jid])
    quat_adr = qadr + 3
    data.qpos[quat_adr:quat_adr+4] = np.array([
        np.cos(init_pitch / 2.0),  # w
        0.0,                       # x
        np.sin(init_pitch / 2.0),  # y  (pitch about Y)
        0.0,                       # z
    ], dtype=float)
    
    # 執行一次前向運動學計算，確保位置更新
    mujoco.mj_forward(model, data)
    print("[INIT] Ready.")
    # ========================================================


    last_u_time = -1e9
    u_hold = np.zeros(2)  # [uL_cmd, uR_cmd] as in your original

    iv = 0.0
    iw = 0.0
    I_CLAMP = 10.0

    v_ref_f = 0.0
    alpha = 0.85

    # logs (IMPORTANT: all same length!)
    t_log, pitch_log, pitch_rate_log = [], [], []
    omega_log, v_log = [], []
    uL_log, uR_log = [], []  # log actual actuator ctrl values


    # UI state logs (same length as t_log)
    vref_ui_log, vref_f_log = [], []
    enable_log, dir_log = [], []
    v = omega = 0.0

    prev_enable = False

    out_png = None

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                with lock:
                    if cmd["quit"]:
                        break
                    if cmd.get("flip_act"):
                        ACT_SIGN_R *= -1.0
                        cmd["flip_act"] = False
                        print("[KEY] flip ACT_SIGN_R")
                        print_config(ACT_SIGN_R, SENSE_SIGN_R, OMEGA_SIGN)
                    if cmd.get("flip_sense"):
                        SENSE_SIGN_R *= -1.0
                        cmd["flip_sense"] = False
                        print("[KEY] flip SENSE_SIGN_R")
                        print_config(ACT_SIGN_R, SENSE_SIGN_R, OMEGA_SIGN)
                    if cmd.get("flip_omega"):
                        OMEGA_SIGN *= -1.0
                        cmd["flip_omega"] = False
                        print("[KEY] flip OMEGA_SIGN")
                        print_config(ACT_SIGN_R, SENSE_SIGN_R, OMEGA_SIGN)

                t0 = time.time()

                # --- leg PD lock ---
                qLth = float(data.sensor("L_thigh_pos").data[0]); dLth = float(data.sensor("L_thigh_vel").data[0])
                qLca = float(data.sensor("L_calf_pos").data[0]);  dLca = float(data.sensor("L_calf_vel").data[0])
                qRth = float(data.sensor("R_thigh_pos").data[0]); dRth = float(data.sensor("R_thigh_vel").data[0])
                qRca = float(data.sensor("R_calf_pos").data[0]);  dRca = float(data.sensor("R_calf_vel").data[0])

                data.ctrl[L_thigh] = KP*(q_L_thigh0 - qLth) - KD*dLth
                data.ctrl[L_calf]  = KP*(q_L_calf0  - qLca) - KD*dLca
                data.ctrl[R_thigh] = KP*(q_R_thigh0 - qRth) - KD*dRth
                data.ctrl[R_calf]  = KP*(q_R_calf0  - qRca) - KD*dRca

                # --- LQR update ---
                if data.time - last_u_time >= control_dt - 1e-12:
                    quat = data.sensor(S_IMU_QUAT).data
                    gyro = data.sensor(S_IMU_GYRO).data
                    pitch = quat_to_pitch(quat)
                    pitch_rate = float(gyro[1])

                    # wheel joint angular velocity (rad/s)
                    vL_w = float(data.sensor(S_L_WVEL).data[0])
                    vR_w_raw = float(data.sensor(S_R_WVEL).data[0])
                    vR_w = SENSE_SIGN_R * vR_w_raw

                    # Convert to body linear speed (m/s) and yaw rate (rad/s)
                    r = float(ip.r)  # wheel radius (m)
                    d = float(ip.d)  # wheel-to-wheel distance (m)
                    V_SIGN = -1.0   # 先用 -1 試試；如果變成倒著衝，再改回 +1
                    v = V_SIGN * 0.5 * r * (vL_w + vR_w)

                    omega = (OMEGA_SIGN * V_SIGN) * (r / d) * (vR_w - vL_w)

                    with lock:
                        v_ref_mps = float(cmd.get("v_ref_mps", 0.0))  # UI: body speed (m/s)
                        enable = bool(cmd.get("enable", True))

                    # # deadband for UI command (m/s)
                    # if abs(v_ref_mps) < 0.02:
                    #     v_ref_mps = 0.0

                    # Rising edge: when enabling LQR, reset integrators/filters to avoid kick
                    if enable and (not prev_enable):
                        iv = 0.0
                        iw = 0.0
                        v_ref_f = v
                        u_hold[:] = 0.0

                    prev_enable = enable

                    if enable:
                        # UI directly commands body linear speed (m/s)
                        v_ref = v_ref_mps
                        v_ref_f = alpha * v_ref_f + (1.0 - alpha) * v_ref

                        iv += (v - v_ref_f) * control_dt
                        iw += (omega - 0.0) * control_dt
                        iv = float(np.clip(iv, -I_CLAMP, I_CLAMP))
                        iw = float(np.clip(iw, -I_CLAMP, I_CLAMP))

                        enable_eff = (data.time > 0.2)  # small delay after reset
                        if enable_eff:
                            x = np.array([
                                [iv],           # x0: Integral of v_err
                                [v - v_ref_f],  # x1: v_err
                                [pitch],        # x2: pitch
                                [pitch_rate],   # x3: pitch rate
                                [iw],           # x4: Integral of omega
                                [omega]         # x5: omega
                            ], dtype=float)
                            u = (-K @ x).reshape(-1)
                            u = np.clip(u, -8.0, 8.0)
                            u_hold[:] = u
                        else:
                            u_hold[:] = 0.0
                    else:
                        # disabled: no wheel torque; keep filter aligned to current speed
                        u_hold[:] = 0.0
                        iv = iw = 0.0
                        v_ref_f = v
                    last_u_time = float(data.time)

                    # --- log at SAME rate as t_log ---
                    uL_cmd = float(u_hold[0])
                    uR_cmd = float(u_hold[1])
                    ctrlL = float(np.clip(uL_cmd, -8.0, 8.0))
                    ctrlR = float(np.clip(ACT_SIGN_R * uR_cmd, -8.0, 8.0))

                    t_log.append(float(data.time))
                    pitch_log.append(float(pitch))
                    pitch_rate_log.append(float(pitch_rate))
                    omega_log.append(float(omega))
                    v_log.append(float(v))
                    uL_log.append(ctrlL)
                    uR_log.append(ctrlR)


                    # --- UI state log (aligned with t_log) ---
                    vref_ui_log.append(float(v_ref_mps))
                    vref_f_log.append(float(v_ref_f))
                    enable_log.append(1.0 if enable else 0.0)
                    if v_ref_mps > 1e-6:
                        dir_log.append(1.0)
                    elif v_ref_mps < -1e-6:
                        dir_log.append(-1.0)
                    else:
                        dir_log.append(0.0)
                    if int(data.time * 5) != int((data.time - control_dt) * 5):
                        print(
                            f"[t={data.time:5.2f}] v_ref={v_ref_f:+.2f} v={v:+.2f} omega={omega:+.2f} "
                            f"ctrlL={ctrlL:+.2f} ctrlR={ctrlR:+.2f} ACT={ACT_SIGN_R:+.0f} SENSE={SENSE_SIGN_R:+.0f} OMG={OMEGA_SIGN:+.0f}"
                        )

                # --- write wheel torques (hold) ---
                uL_cmd = float(np.clip(u_hold[0], -8.0, 8.0))
                uR_cmd = float(np.clip(u_hold[1], -8.0, 8.0))
                data.ctrl[L_wheel] = uL_cmd
                data.ctrl[R_wheel] = ACT_SIGN_R * uR_cmd

                mujoco.mj_step(model, data)
                viewer.sync()

                dt = sim_dt - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) -> will still save plot.")

    finally:
        try:
            out_png = save_and_show_plot(t_log, pitch_log, pitch_rate_log, uL_log, uR_log, omega_log, v_log,
                                     vref_ui_log, vref_f_log, enable_log, dir_log)
        except Exception as e:
            print(f"[PLOT] failed: {e}")
        # 在 finally: 內、save_and_show_plot(...) 後面加
        import matplotlib.pyplot as plt
        phase_png = f"phase_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.figure(figsize=(6,4))
        plt.plot(pitch_log, pitch_rate_log)
        plt.title("Phase portrait: pitch vs pitch rate")
        plt.xlabel("pitch (rad)")
        plt.ylabel("pitch_rate (rad/s)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(phase_png, dpi=160)
        # ... 你的 plt.savefig(phase_png, dpi=160) 之後
        if SHOW_PLOT_WINDOW and (not HEADLESS):
            plt.show()
        plt.close()
        print(f"[PLOT] saved -> {phase_png}")
        plt.close()
        
    if out_png:
        # Helpful hint
        print("[TIP] If you don't see a window, open the PNG:")
        print("      ls -lt convergence_*.png | head")
        print("      xdg-open convergence_*.png")


if __name__ == "__main__":
    main()