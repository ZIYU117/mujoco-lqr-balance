import math
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, eig

# ----------------------------------------
# Fast LQR interface for MuJoCo (lazy init)
# ----------------------------------------
_K_cache = None
_cache_dt = None

def lqr_get_u(x: np.ndarray) -> np.ndarray:
    """
    x: (6,1) column vector
    return: (2,) vector  [u_left, u_right]
    """
    global _K_cache, _cache_dt

    # ✅ 只有在第一次呼叫、或 dt 改變時才重算 K
    if (_K_cache is None) or (_cache_dt != delta_t):
        A, B = get_model_matrix()
        _K_cache, _, _ = dlqr(A, B, Q, R)
        _cache_dt = delta_t

    u = (-_K_cache @ x).reshape(-1)
    return u

# ---------------------------
# Model parameters
# ---------------------------
l_bar = 2.0
m = 0.28
M = 7.08
l = 0.37
r = 0.07
I = 0.5 * M * r * r
Jp = (1/3) * M * (l**2)
Qeq = M * Jp + (Jp + M*l*l) * (2*m + 2*I/r**2)

g = 9.8
d = 0.36
Jdelta = (1.0/12.0) * M * (d**2)

nx = 6
nu = 2
# Q = np.diag([20.0, 0.2, 80.0, 2.0, 10.0, 0.5])
# R = np.diag([0.12, 0.12])
# 以下balance使用
# Q = np.diag([20.0, 0.2, 200.0, 2.0, 10.0, 0.5])
# R = np.diag([2.0, 2.0])
# 以下v3使用
# Q = np.diag([400.0, 5.0, 200.0, 10.0, 10.0, 0.5])
# R = np.diag([0.2935, 0.2935])
# 以下v5使用
# Q = np.diag([0.5, 0.5, 2000.0, 300.0, 0.5, 10.0])
# R = np.diag([0.50, 0.50])
# 以下v5使用
# Q = np.diag([0.5, 0.5, 1000.0, 300.0, 0.5, 7.78909])#7.2381=200
# R = np.diag([0.50, 0.50])
# 強力修正版 (來自你的 board 檔案)
# Q = np.diag([0.5, 0.5, 2000.0, 300.0, 0.5, 10.0])
# R = np.diag([5.0, 5.0])
# 嘗試這個較為柔和的參數=ok
Q = np.diag([0.5, 0.5, 2000.0, 175.40, 0.5, 10.0])
R = np.diag([0.01, 0.01])
# 註：如果震盪過大，可以稍微加大 R (例如 [10, 10]) 來抑制輸出，或減小 Q 裡的 2000。
# Q = np.diag([50.0, 5.0,   400.0, 25.0,   200.0, 20.0])
# R = np.diag([0.30, 0.30])

delta_t = 0.01
sim_time = 5.0
show_animation = True


# ----------------------------------------
# Solve DARE
# ----------------------------------------
def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    P = Q
    for _ in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ \
            inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if np.max(np.abs(Pn - P)) < eps:
            break
        P = Pn
    return Pn


# ----------------------------------------
# LQR
# ----------------------------------------
def dlqr(A, B, Q, R):
    P = solve_DARE(A, B, Q, R)
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    eigVals, _ = eig(A - B @ K)
    return K, P, eigVals


def lqr_control(x):
    A, B = get_model_matrix()
    K, _, _ = dlqr(A, B, Q, R)
    return -K @ x


# ----------------------------------------
# System Matrices
# ----------------------------------------
def get_model_matrix():
    A23 = -(M**2 * l**2 * g) / Qeq
    A43 = (M * l * g * (M + 2*m + 2*I/r**2)) / Qeq

    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, A23, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, A43, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    A = np.eye(nx) + delta_t * A

    B21 = (Jp + M*l*l + M*l*r) / (Qeq * r)
    B41 = -((M*l)/r + M + 2*m + 2*I/r**2) / Qeq
    B42 = B41
    Bd = 1.0 / (r * (m*d + (I*d)/(r*r) + (2*Jdelta)/d))

    B61 = Bd
    B62 = -Bd

    B = np.array([
        [0, 0],
        [B21, B21],
        [0, 0],
        [B41, B42],
        [0, 0],
        [B61, B62]
    ])

    B = B * delta_t
    return A, B


# ----------------------------------------
# Simulation
# ----------------------------------------
def simulation(x, u):
    A, B = get_model_matrix()
    return A @ x + B @ u


# ----------------------------------------
# Plot cart
# ----------------------------------------
def plot_cart(xt, theta):
    cart_w = 1.0
    cart_h = 0.5
    radius = 0.1

    cx = np.array([-cart_w/2, cart_w/2, cart_w/2, -cart_w/2, -cart_w/2]) + xt
    cy = np.array([0, 0, cart_h, cart_h, 0]) + radius*2

    bx = np.array([0, l_bar*math.sin(-theta)]) + xt
    by = np.array([cart_h, l_bar*math.cos(-theta) + cart_h]) + radius*2

    angles = np.linspace(0, 2*math.pi, 60)
    ox = radius * np.cos(angles)
    oy = radius * np.sin(angles)

    rwx = ox + cart_w/4 + xt
    rwy = oy + radius
    lwx = ox - cart_w/4 + xt
    lwy = oy + radius

    plt.plot(cx, cy, "-b")
    plt.plot(bx, by, "-k")
    plt.plot(rwx, rwy, "-k")
    plt.plot(lwx, lwy, "-k")
    plt.title(f"x={xt:.2f}, θ={math.degrees(theta):.2f}°")
    plt.axis("equal")


# ----------------------------------------
# Main
# ----------------------------------------
def main():
    x = np.array([[0.0], [0.0], [0.5], [0.0], [0.0], [0.0]])
    t = 0.0

    while t < sim_time:
        u = lqr_control(x)
        x = simulation(x, u)
        t += delta_t

        if show_animation:
            plt.clf()
            plot_cart(float(x[0]), float(x[2]))
            plt.pause(0.001)

    print("Finish")
    print(f"x = {float(x[0]):.6f} m, theta = {math.degrees(float(x[2])):.6f} deg")


if __name__ == "__main__":
    main()