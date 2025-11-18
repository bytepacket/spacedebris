import numpy as np
import pyvista as pv
from pyvista import examples
import datetime as dt
import math as m
import matplotlib.cm as cm
from tle_data import tle_data
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

def get_satellite_statistics(tle_data):
    """Get statistics about satellites for UI display."""
    stats = {
        'total': 0,
        'low': 0,      # 0-30° or low altitude
        'medium': 0,   # 30-60°
        'high': 0,     # 60-90°
        'polar': 0,    # 90-120°
        'retrograde': 0,  # 120-180°
        'inclinations': []
    }
    
    for sat_name, tle in tle_data.items():
        tle1, tle2 = tle
        if tle1[0] != "1":
            continue
        
        try:
            inclination = float(tle2[9:17])  # Inclination in degrees
            stats['inclinations'].append(inclination)
            stats['total'] += 1
            
            if 0 <= inclination < 30:
                stats['low'] += 1
            elif 30 <= inclination < 60:
                stats['medium'] += 1
            elif 60 <= inclination < 90:
                stats['high'] += 1
            elif 90 <= inclination < 120:
                stats['polar'] += 1
            else:  # 120-180
                stats['retrograde'] += 1
        except:
            continue
    
    # Calculate percentages
    if stats['total'] > 0:
        stats['low_pct'] = (stats['low'] / stats['total']) * 100
        stats['medium_pct'] = (stats['medium'] / stats['total']) * 100
        stats['high_pct'] = (stats['high'] / stats['total']) * 100
        stats['polar_pct'] = (stats['polar'] / stats['total']) * 100
        stats['retrograde_pct'] = (stats['retrograde'] / stats['total']) * 100
    else:
        stats['low_pct'] = stats['medium_pct'] = stats['high_pct'] = stats['polar_pct'] = stats['retrograde_pct'] = 0
    
    return stats

def get_inclination_color(inclination):
    """Get color based on inclination angle."""
    if 0 <= inclination < 30:
        return (1.0, 0.0, 0.0, 1.0)  # Red - Equatorial
    elif 30 <= inclination < 60:
        return (1.0, 0.65, 0.0, 1.0)  # Orange - Low
    elif 60 <= inclination < 90:
        return (1.0, 1.0, 0.0, 1.0)  # Yellow - Medium
    elif 90 <= inclination < 120:
        return (0.0, 1.0, 0.0, 1.0)  # Green - High
    else:  # 120-180
        return (0.0, 0.5, 1.0, 1.0)  # Blue - Retrograde

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

if __name__ == "__main__":
    print("Creating satellite orbits around realistic Earth...")
    print("=" * 60)
    
    plotter = plot_satellites_pyvista(tle_data)
    
    print("\nDisplaying interactive 3D visualization...")
    
    # Show the plot
    plotter.show()
