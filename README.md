import pygame
import numpy as np
from numba import njit
RES = 800
M = 4.8
STEP_SIZE = 0.2
MAX_STEPS = 110
DISK_RS = (2.2, 9.0)
@njit
def calculate_physics(time, cam_p):
x = np.linspace(-1.0, 1.0, RES)
y = np.linspace(-1.0, 1.0, RES)
final_img = np.zeros((RES, RES, 3))
for i in range(RES):
for j in range(RES):
uv = np.array([x[i], -y[j], 1.3])
mag = np.sqrt(uv[0]**2 + uv[1]**2 + uv[2]**2)
ray_dir = uv / mag
pos = cam_p.copy()
acc_color = np.array([0.0, 0.0, 0.0])
transparency = 1.0
hit_bh = False
for _ in range(MAX_STEPS):
r_mag = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
if r_mag < 1.7:
hit_bh = True
break
gravity = (pos * STEP_SIZE * M) / (r_mag**3 + 0.0001)
ray_dir = ray_dir - gravity
ray_dir = ray_dir / np.sqrt(ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2)
pos = pos + ray_dir * STEP_SIZE
if abs(pos[1]) < 0.28 and r_mag > DISK_RS[0] and r_mag < DISK_RS[1]:
angle = np.arctan2(pos[2], pos[0])
doppler = np.exp(0.7 * np.cos(angle + time))
redshift = (r_mag - 1.7) / 2.0
redshift = max(0.1, min(1.0, redshift))
spin = (np.sin(angle * 7.0 + time * 2.5) * 0.5 + 0.5) * doppler
intensity = (4.0 / r_mag) * spin
acc_color[0] += (0.4 + (1.0 - redshift) * 0.6) * intensity * transparency
acc_color[1] += (0.7 * redshift) * intensity * transparency
acc_color[2] += (1.0 * redshift) * intensity * transparency
transparency *= np.exp(-0.1 * intensity)
if not hit_bh:
s_noise = np.sin(ray_dir[0]*45.0) * np.sin(ray_dir[1]*45.0) * np.sin(ray_dir[2]*45.0)
if s_noise > 0.985:
acc_color += transparency * 0.8
else:
acc_color *= 0.0
final_img[i, j, 0] = min(acc_color[0] * 255, 255)
final_img[i, j, 1] = min(acc_color[1] * 255, 255)
final_img[i, j, 2] = min(acc_color[2] * 255, 255)
return final_img.astype(np.uint8)
def run_sim():
pygame.init()
screen = pygame.display.set_mode((RES, RES))
clock = pygame.time.Clock()
time_val = 0.0
cam_p = np.array([0.0, 1.3, -16.0])
while True:
for event in pygame.event.get():
if event.type == pygame.QUIT: return
keys = pygame.key.get_pressed()
speed = 0.5 if not keys[pygame.K_LSHIFT] else 1.5
if keys[pygame.K_w]: cam_p[2] += speed
if keys[pygame.K_s]: cam_p[2] -= speed
if keys[pygame.K_a]: cam_p[0] -= speed
if keys[pygame.K_d]: cam_p[0] += speed
if keys[pygame.K_q]: cam_p[1] += speed
if keys[pygame.K_e]: cam_p[1] -= speed
frame = calculate_physics(time_val, cam_p)
surf = pygame.surfarray.make_surface(frame)
screen.blit(surf, (0, 0))
pygame.display.flip()
time_val += 0.12
clock.tick(60)
if __name__ == "__main__":
run_sim()
