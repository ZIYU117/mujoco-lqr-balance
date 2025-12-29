# mujoco_viewer.py
# 列出 body/joint/actuator/sensor 名稱與 index，方便你把名稱填到 controller 裡
import mujoco

XML_PATH = "crazydog_urdf/urdf/scene.xml"   # 你可以改成 test.xml / 你的 xml

def print_scene_information(model: mujoco.MjModel) -> None:
    print("\n<<------------- Body (Link) ------------->>")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            print(f"body_id={i:3d}  name={name}")

    print("\n<<------------- Joint ------------->>")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print(f"joint_id={i:3d} name={name}  type={model.jnt_type[i]}  qposadr={model.jnt_qposadr[i]} qveladr={model.jnt_dofadr[i]}")

    print("\n<<------------- Actuator ------------->>")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            print(f"act_id={i:3d}  name={name}  trntype={model.actuator_trntype[i]}  dyntype={model.actuator_dyntype[i]}")

    print("\n<<------------- Sensor ------------->>")
    adr = 0
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        dim = model.sensor_dim[i]
        if name:
            print(f"sensor_id={i:3d} name={name}  dim={dim}  adr={adr}")
        adr += dim

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    print_scene_information(model)
