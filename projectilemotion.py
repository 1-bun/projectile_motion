import numpy as np

import matplotlib.pyplot as plt

# Constants (Alter time_step to note decreased accuracy for larger steps)
G = 9.81  # gravity (m/s^2)
v_initial = 20  # m/s
launch_angle = 45  # degrees
time_step = 0.001 # seconds
t_max = 8  # seconds

# Convert angle to radians
angle_rad = np.radians(launch_angle)
initial_vx = v_initial * np.cos(angle_rad)
initial_vy = v_initial * np.sin(angle_rad)

# Euler Method
def euler_method(vx, vy, x, y, dt, max_time):
    times = [0]
    positions_x = [x]
    positions_y = [y]
    
    while y >= 0 and times[-1] < max_time:
        vy -= G * dt
        y += vy * dt
        x += vx * dt
        
        times.append(times[-1] + dt)
        positions_x.append(x)
        positions_y.append(y)
    
    return np.array(times), np.array(positions_x), np.array(positions_y)

# RK4 Method
def rk4_method(vx, vy, x, y, dt, max_time):
    times = [0]
    positions_x = [x]
    positions_y = [y]
    
    while y >= 0 and times[-1] < max_time:
        # RK4 for vertical velocity and position
        k1_vy = -G
        k1_y = vy
        
        k2_vy = -G
        k2_y = vy + 0.5 * k1_vy * dt
        
        k3_vy = -G
        k3_y = vy + 0.5 * k2_vy * dt
        
        k4_vy = -G
        k4_y = vy + k3_vy * dt
        
        vy += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6 * dt
        y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6 * dt
        x += vx * dt
        
        times.append(times[-1] + dt)
        positions_x.append(x)
        positions_y.append(y)
    
    return np.array(times), np.array(positions_x), np.array(positions_y)

# Run simulations
t_euler, x_euler, y_euler = euler_method(initial_vx, initial_vy, 0, 0, time_step, t_max)
t_rk4, x_rk4, y_rk4 = rk4_method(initial_vx, initial_vy, 0, 0, time_step, t_max)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_euler, y_euler, 'b-', label='Euler Method', linewidth=2)
plt.plot(x_rk4, y_rk4, 'r--', label='RK4 Method', linewidth=2)
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Vertical Height (m)', fontsize=12)
plt.title('Projectile Motion: Euler vs RK4', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()