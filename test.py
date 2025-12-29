import mujoco
import mujoco.viewer
import time
import numpy as np

from inverted_pendulum_lqr_control import lqr_get_u

def pd_control(target_q, q, kp, target_dq, dq, kd):
    # """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

NUM_MOTOR = 6
# Load a sample model 
model = mujoco.MjModel.from_xml_path('/home/a/Downloads/mujoco_course-master/crazydog_urdf/urdf/scene.xml')
data = mujoco.MjData(model)
target_dof_pos = np.array([1.27, -2.127, 0,1.27, -2.127, 0])
target_dof_vel = np.array([0,0, 0.5,0,0,0.5])


simulation_dt = 0.005
kps = np.array([25.0, 25.0 ,0.0, 25.0, 25.0, 0.0])
kds = np.array([0.5, 0.5, 0.4, 0.5, 0.5, 0.4])

# Run a simple simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # 直接讀取機器人關節位置與速度
        q = data.qpos[:NUM_MOTOR]
        dq = data.qvel[:NUM_MOTOR]


        ##ur controller

        tau = pd_control(target_dof_pos, data.sensordata[:NUM_MOTOR], kps, target_dof_vel, data.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)
        # 限制馬達力矩，避免發瘋（依照你模型調整數值）
        tau = np.clip(tau, -5.0, 5.0)
        data.ctrl[:] = tau


        model.opt.timestep = simulation_dt
        mujoco.mj_step(model, data)
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


