# controller_lqr.py
# 共用：雙輪平衡 LQR 控制器（把 LQR 接到 MuJoCo data.ctrl）
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import mujoco

import inverted_pendulum_lqr_control as ip


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def quat_to_euler_xyz(quat_wxyz: np.ndarray) -> Tuple[float, float, float]:
    """
    quat_wxyz: [w, x, y, z]
    回傳 (roll, pitch, yaw) ，採用常見的 XYZ (roll-pitch-yaw) Tait-Bryan 角
    """
    w, x, y, z = [float(v) for v in quat_wxyz]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = _clamp(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


@dataclass
class BalanceIO:
    # actuator names
    left_wheel_act: str = "L_wheel"
    right_wheel_act: str = "R_wheel"

    # sensors (建議用 mujoco_viewer.py 印出名稱再確認)
    imu_quat_sensor: str = "imu_quat"   # dim=4, [w,x,y,z]
    imu_gyro_sensor: str = "imu_gyro"   # dim=3, [wx,wy,wz]
    left_wheel_vel_sensor: str = "L_wheel_vel"  # dim=1
    right_wheel_vel_sensor: str = "R_wheel_vel" # dim=1

    # gyro axis used as pitch_rate: 0=x,1=y,2=z
    pitch_rate_axis: int = 1

    # sign flips (如果方向顛倒就改成 -1)
    pitch_sign: float = 1.0
    pitch_rate_sign: float = 1.0
    wheel_sign: float = 1.0


class LQRBalanceController:
    """
    雙輪平衡控制：
    x = [0, 0, pitch, pitch_rate, v, omega]^T
    u = [uL, uR]
    """
    def __init__(
        self,
        model: mujoco.MjModel,
        io: BalanceIO = BalanceIO(),
        control_dt: Optional[float] = None,
        u_limit: float = 10.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.io = io
        self.u_limit = float(u_limit)

        # 控制更新週期（K 的離散化 dt）
        self.control_dt = float(control_dt if control_dt is not None else model.opt.timestep)

        # 解析 actuator id
        self.L_act_id = model.actuator(io.left_wheel_act).id
        self.R_act_id = model.actuator(io.right_wheel_act).id

        # (可選) 覆蓋 Q/R
        if Q is None:
            Q = ip.Q
        if R is None:
            R = ip.R

        # 用你原本 inverted_pendulum_lqr_control 的模型，但把 dt 改成和控制週期一致
        ip.delta_t = self.control_dt
        A, B = ip.get_model_matrix()
        self.K, _, _ = ip.dlqr(A, B, Q, R)

        # hold last u（如果你採用 control_dt > sim_dt，會用到）
        self._u_last = np.zeros(2, dtype=float)

    def _read_pitch(self, data: mujoco.MjData) -> float:
        quat = data.sensor(self.io.imu_quat_sensor).data  # [w,x,y,z]
        _, pitch, _ = quat_to_euler_xyz(quat)
        return self.io.pitch_sign * float(pitch)

    def _read_pitch_rate(self, data: mujoco.MjData) -> float:
        gyro = data.sensor(self.io.imu_gyro_sensor).data  # [wx,wy,wz]
        pr = float(gyro[int(self.io.pitch_rate_axis)])
        return self.io.pitch_rate_sign * pr

    def _read_wheel_vel(self, data: mujoco.MjData) -> Tuple[float, float]:
        vL = float(data.sensor(self.io.left_wheel_vel_sensor).data[0])
        vR = float(data.sensor(self.io.right_wheel_vel_sensor).data[0])
        s = float(self.io.wheel_sign)
        return s * vL, s * vR

    def compute_u(self, data: mujoco.MjData) -> np.ndarray:
        pitch = self._read_pitch(data)
        pitch_rate = self._read_pitch_rate(data)
        vL, vR = self._read_wheel_vel(data)

        v = 0.5 * (vL + vR)
        omega = (vR - vL)

        x = np.array([[0.0], [0.0], [pitch], [pitch_rate], [v], [omega]], dtype=float)

        u = (-self.K @ x).reshape(-1)
        u = np.clip(u, -self.u_limit, self.u_limit)
        return u

    def apply_u(self, data: mujoco.MjData, u: np.ndarray) -> None:
        data.ctrl[self.L_act_id] = float(u[0])
        data.ctrl[self.R_act_id] = float(u[1])

    def step(self, data: mujoco.MjData) -> None:
        u = self.compute_u(data)
        self.apply_u(data, u)
        self._u_last[:] = u
