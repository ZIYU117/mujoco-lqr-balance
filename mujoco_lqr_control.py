# mujoco_lqr_control.py
# 用 controller_lqr.py 讓雙輪平衡（直接跑這支）
import time
import numpy as np
import mujoco
import mujoco.viewer

from controller_lqr import LQRBalanceController, BalanceIO

XML_PATH = "crazydog_urdf/urdf/scene.xml"

# 你可以先用 mujoco_viewer.py 印出名稱後再改這裡
IO = BalanceIO(
    left_wheel_act="L_wheel",
    right_wheel_act="R_wheel",
    imu_quat_sensor="imu_quat",
    imu_gyro_sensor="imu_gyro",
    left_wheel_vel_sensor="L_wheel_vel",
    right_wheel_vel_sensor="R_wheel_vel",
    pitch_rate_axis=1,   # 0/1/2 → x/y/z
    pitch_sign=1.0,
    pitch_rate_sign=1.0,
    wheel_sign=1.0,
)

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # ✅ 建議：控制更新比模擬慢一點，先從 0.01s 開始（你也可以改成 0.005）
    control_dt = 0.01
    ctrl = LQRBalanceController(model, io=IO, control_dt=control_dt, u_limit=10.0)

    # 把 sim timestep 設好（通常 xml 已經有了）
    sim_dt = float(model.opt.timestep)

    last_u_time = -1e9
    u_hold = np.zeros(2)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 控制器每 control_dt 更新一次，其餘時間 hold 上一次 u
            if data.time - last_u_time >= control_dt - 1e-12:
                u_hold = ctrl.compute_u(data)
                last_u_time = float(data.time)

            ctrl.apply_u(data, u_hold)

            mujoco.mj_step(model, data)
            viewer.sync()

            # 盡量 real-time
            time_until_next_step = sim_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
