import numpy as np
from sgp4.api import Satrec, WGS72 # WGS72 is a gravity constant model used by SGP4
from sgp4.api import jday # julian date
import datetime
from skyfield.api import load, EarthSatellite
from skyfield.framelib import itrs
from skyfield.timelib import julian_date
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# -- READ --#
# https://towardsdatascience.com/use-python-to-create-two-body-orbits-a68aed78099c/
# https://pypi.org/project/sgp4/

# consts
TLE_FILE_PATH = 'tle.txt'

# functions
def model_2BP(state, t):
    mu = 3.986004418E+05
    x = state[0]
    y = state[1]
    z = state[2]
    x_dot = state[3]
    y_dot = state[4]
    z_dot = state[5]
    x_ddot = -mu * x / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    y_ddot = -mu * y / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    z_ddot = -mu * z / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    dstate_dt = [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
    return dstate_dt


# number of flying satellites (manmade)
satellites = []
with open(TLE_FILE_PATH, 'r') as f:
    lines = f.readlines()

# process every 3 lines
for i in range(0, len(lines), 3):
    try:
        name = lines[i].strip()
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()

        satellite = Satrec.twoline2rv(line1, line2, WGS72)

        satellites.append({'name': name, 'model': satellite})

    except IndexError:
        print(f"skipping incomplete set starting at line {i+1}")
        continue
    except Exception as e:
        print(f"parsing error for line {i+1}: {e}")

print(f"Successfully loaded {len(satellites)} debris objects.")

state = []

# TODO: add real-time values
jd, fr = jday(2025, 2, 11, 13, 57, 0)

for i in range(len(satellites) - 1):
    satellite_model = satellites[i]["model"]
    e, r, v = satellite_model.sgp4(jd, fr)
    
for i in range(3):
    state.append(r[i])

for i in range(3):
    state.append(v[i])    

print(state)

# Time Array
t = np.linspace(0, 6*3600, 200)  # Simulates for a time period of 6
                                 # hours [s]

# Solving ODE
sol = odeint(model_2BP, state, t)
X_Sat = sol[:, 0]  # X-coord [km] of satellite over time interval 
Y_Sat = sol[:, 1]  # Y-coord [km] of satellite over time interval
Z_Sat = sol[:, 2]  # Z-coord [km] of satellite over time interval

# Setting up Spherical Earth to Plot
N = 50
phi = np.linspace(0, 2 * np.pi, N)
theta = np.linspace(0, np.pi, N)
theta, phi = np.meshgrid(theta, phi)

r_Earth = 6378.14  # Average radius of Earth [km]
X_Earth = r_Earth * np.cos(phi) * np.sin(theta)
Y_Earth = r_Earth * np.sin(phi) * np.sin(theta)
Z_Earth = r_Earth * np.cos(theta)

# Plotting Earth and Orbit
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)
ax.plot3D(X_Sat, Y_Sat, Z_Sat, 'black')
ax.view_init(30, 145)  # Changing viewing angle (adjust as needed)

plt.title('Two-Body Orbit')
ax.set_xlabel('X [km]')
ax.set_ylabel('Y [km]')
ax.set_zlabel('Z [km]')

xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(),      
                   ax.get_zlim3d()]).T
XYZlim = np.asarray([min(xyzlim[0]), max(xyzlim[1])])
ax.set_xlim3d(XYZlim)
ax.set_ylim3d(XYZlim)
ax.set_zlim3d(XYZlim * 3/4)
plt.show()
