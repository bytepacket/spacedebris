import numpy as np
import pyvista as pv
from pyvista import examples
import datetime as dt
import math as m
import matplotlib.cm as cm

# Constants
mu = 398600.4418  # Standard gravitational parameter (km^3/s^2)
r = 6378.1  # Earth radius (km)
D = 24 * 0.997269  # Sidereal day in hours

def calculate_orbit(tle1, tle2):
    """Calculate orbit points from TLE data."""
    if tle1[0] != "1":
        return None, None, None
    
    try:
        year_str = tle1[18:20]
        if int(year_str) > int(dt.date.today().year % 100):
            year_prefix = "19"
        else:
            year_prefix = "20"
        
        orb = {"t": dt.datetime.strptime(
            year_prefix + year_str + " " + tle1[20:23] + " " +
            str(int(24 * float(tle1[23:33]) // 1)) + " " +
            str(int(((24 * float(tle1[23:33]) % 1) * 60) // 1)) + " " +
            str(int((((24 * float(tle1[23:33]) % 1) * 60) % 1) // 1)),
            "%Y %j %H %M %S"
        )}
        
        orb.update({
            "name": tle2[2:7],
            "e": float("." + tle2[26:34]),
            "a": (mu / ((2 * m.pi * float(tle2[52:63]) / (D * 3600)) ** 2)) ** (1. / 3),
            "i": float(tle2[9:17]) * m.pi / 180,
            "RAAN": float(tle2[17:26]) * m.pi / 180,
            "omega": float(tle2[34:43]) * m.pi / 180
        })
        
        orb.update({"b": orb["a"] * m.sqrt(1 - orb["e"] ** 2),
                    "c": orb["a"] * orb["e"]})
        
        R = np.matmul(np.array([[m.cos(orb["RAAN"]), -m.sin(orb["RAAN"]), 0],
                                 [m.sin(orb["RAAN"]), m.cos(orb["RAAN"]), 0],
                                 [0, 0, 1]]),
                      (np.array([[1, 0, 0],
                                 [0, m.cos(orb["i"]), -m.sin(orb["i"])],
                                 [0, m.sin(orb["i"]), m.cos(orb["i"])]])))
        R = np.matmul(R, np.array([[m.cos(orb["omega"]), -m.sin(orb["omega"]), 0],
                                 [m.sin(orb["omega"]), m.cos(orb["omega"]), 0],
                                 [0, 0, 1]]))
        
        # Calculate orbit points
        num_points = 200  # Smooth orbit curves
        x = []
        y = []
        z = []
        for i in np.linspace(0, 2 * m.pi, num_points):
            P = np.matmul(R, np.array([[orb["a"] * m.cos(i)],
                                     [orb["b"] * m.sin(i)],
                                     [0]])) - np.matmul(R, np.array([[orb["c"]],
                                                                     [0],
                                                                     [0]]))
            x.append(P[0][0])
            y.append(P[1][0])
            z.append(P[2][0])
        
        return np.array(x), np.array(y), np.array(z)
    except Exception as e:
        print(f"Error calculating orbit: {e}")
        return None, None, None

def plot_satellites_pyvista(tle_data):
    """Plot satellite orbits around PyVista Earth."""
    # Create PyVista plotter with space background
    pl = pv.Plotter(lighting="none")
    
    # Add space cubemap background
    print("Loading space environment...")
    try:
        cubemap = examples.download_cubemap_space_16k()
        pl.add_actor(cubemap.to_skybox())
        pl.set_environment_texture(cubemap, True)
    except:
        print("Could not load space cubemap, using default background")
    
    # Add lighting
    light = pv.Light()
    light.set_direction_angle(30, -20)
    pl.add_light(light)
    
    # Create Earth with texture
    print("Loading Earth...")
    earth = examples.planets.load_earth(radius=r)
    earth_texture = examples.load_globe_texture()
    pl.add_mesh(earth, texture=earth_texture, smooth_shading=True)
    
    # Calculate and plot orbits
    print("Calculating satellite orbits...")
    orbit_count = 0
    # Generate colors for orbits
    color_map = cm.get_cmap('tab20')
    colors = [color_map(i / max(len(tle_data), 1))[:3] for i in range(len(tle_data))]
    
    orbit_actors = []
    for idx, (sat_name, tle) in enumerate(tle_data.items()):
        tle1, tle2 = tle
        x, y, z = calculate_orbit(tle1, tle2)
        
        if x is not None:
            # Create polyline for orbit
            points = np.column_stack([x, y, z])
            # Create a single polyline connecting all points
            orbit_line = pv.PolyData()
            orbit_line.points = points
            # Create line cell - single polyline connecting all points
            n_points = len(points)
            orbit_line.lines = np.concatenate([[n_points], np.arange(n_points)])
            
            # Add orbit to plot
            actor = pl.add_mesh(orbit_line, color=colors[idx % len(colors)], 
                               line_width=2, opacity=0.8)
            orbit_actors.append((sat_name, actor))
            orbit_count += 1
    
    print(f"Plotted {orbit_count} satellite orbits")
    
    # Set up the plot
    pl.add_text("Satellite Orbits around Earth", font_size=18, position='upper_left')
    pl.set_background('black')
    
    # Add legend if not too many satellites
    if len(orbit_actors) <= 20:
        legend_labels = [(name, colors[i % len(colors)]) 
                        for i, (name, _) in enumerate(orbit_actors)]
        pl.add_legend(labels=legend_labels, face='rectangle', size=(0.2, 0.4))
    
    # Set camera position
    pl.camera_position = 'iso'
    pl.camera.zoom(0.7)
    
    return pl

# Define the TLE data as a dictionary
tle_data = {
    "LILACSAT-2": (
        '1 40908U 15049K   25040.75362640  .00014926  00000-0  44552-3 0  9999',
        '2 40908  97.5165  58.4844 0008714 220.7406 139.3186 15.34689321519828'
    ),
    "IO-86": (
        '1 40931U 15052B   25038.81889873  .00001817  00000-0  15548-3 0  9993',
        '2 40931   6.0001 108.1159 0012838 187.7041 172.2889 14.78568669506492'
    ),
    "Horyu-4": (
        '1 41340U 16012D   25040.50964053  .00024423  00000-0  83068-3 0  9998',
        '2 41340  30.9968 215.0775 0005942 258.1882 101.8095 15.29227906494341'
    ),
    "Lapan A3": (
        '1 41603U 16040E   25040.82089363  .00005724  00000-0  20899-3 0  9994',
        '2 41603  97.1391  50.2222 0010651 346.8368  13.2593 15.28446454479488'
    ),
    "CAS-2T": (
        '1 41847U 16066G   25040.81371518  .00003589  00000-0  52043-3 0  9990',
        '2 41847  98.4339 116.3024 0348372 180.5013 179.5879 14.42893172433742'
    ),
    "CAS-4B": (
        '1 42759U 17034B   25040.80858593  .00043921  00000-0  84809-3 0  9990',
        '2 42759  43.0123 306.9566 0016345 180.6949 179.3890 15.46942498423955'
    ),
    "CAS-4A": (
        '1 42761U 17034D   25040.83991301  .00043338  00000-0  82365-3 0  9998',
        '2 42761  43.0161 304.7417 0017725 187.1675 172.8933 15.47379287423971'
    ),
    "TechnoSat": (
        '1 42829U 17042E   25040.39008121  .00004774  00000-0  38520-3 0  9997',
        '2 42829  97.3894 214.9223 0012152 342.6431  17.4377 15.00254609412697'
    ),
    "AO-91": (
        '1 43017U 17073E   25040.58219161  .00013379  00000-0  72834-3 0  9997',
        '2 43017  97.5498 274.4797 0181459 214.7102 144.2175 15.01471107391187'
    ),
    "S-Net D": (
        '1 43186U 18014G   25040.29591181  .00007102  00000-0  44445-3 0  9997',
        '2 43186  97.5738 304.5247 0007708 281.1198  78.9164 15.09684555384211'
    ),
    "S-Net B": (
        '1 43187U 18014H   25040.29990714  .00006884  00000-0  43107-3 0  9993',
        '2 43187  97.5744 304.4942 0008350 275.0534  84.9742 15.09670862384215'
    ),
    "S-Net A": (
        '1 43188U 18014J   25040.69435521  .00008315  00000-0  51936-3 0  9998',
        '2 43188  97.5751 305.0192 0008334 274.3627  85.6650 15.09697017384278'
    ),
    "S-Net C": (
        '1 43189U 18014K   25040.63344299  .00007613  00000-0  47604-3 0  9991',
        '2 43189  97.5769 304.9524 0009830 265.3769  94.6337 15.09675364382815'
    ),
    "Ten-Koh": (
        '1 43677U 18084G   25040.72854077  .00005254  00000-0  43457-3 0  9991',
        '2 43677  98.0827 208.2460 0011283 307.3159  52.7033 14.99330140342270'
    )
}

if __name__ == "__main__":
    print("Creating satellite orbits around realistic Earth...")
    print("=" * 60)
    
    plotter = plot_satellites_pyvista(tle_data)
    
    print("\nDisplaying interactive 3D visualization...")
    
    # Show the plot
    plotter.show()
