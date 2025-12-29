# mujoco-lqr-balance
# LQR Self-Balancing Robot (MuJoCo)

使用 MuJoCo 進行的 LQR 自平衡機器人模擬，內容包含系統矩陣 A/B、
控制器參數 Q/R/K、收斂圖與相圖分析。

This repository contains a MuJoCo-based simulation of a self-balancing robot
controlled using discrete-time LQR (DLQR).

---

## 1. Environment / 執行環境

- OS: Ubuntu 22.04.6
- Python: 3.10.12
- MuJoCo
- Python packages:
  - numpy
  - scipy
  - matplotlib

---

## 2. Files / 檔案說明

- `balance_test_ok.py`  
  主程式，執行後會印出 A/B/Q/R/K，並產生收斂圖與相圖。

- `inverted_pendulum_lqr_control.py`  
  系統模型、狀態空間矩陣、LQR 設計。

- `controller_lqr.py`  
  LQR 控制器相關實作。

- `mujoco_lqr_control.py`, `mujoco_viewer.py`  
  MuJoCo 模擬與視覺化工具。

- `*.xml / assets / urdf`  
  MuJoCo 模型與相關資源檔。

---

## 3. How to Run / 執行方式

```bash
python3 balance_test_ok.py

---

## 4. Results / 結果
<img width="1600" height="2304" alt="convergence_20251229_204950" src="https://github.com/user-attachments/assets/e251595f-1043-4bc5-8992-f9deb14045ef" />
<img width="960" height="640" alt="phase_20251229_205008" src="https://github.com/user-attachments/assets/708af2f3-72bb-4236-90f1-1d56fe726352" />
