# ============================================================================
# KIVY SPACE DASHBOARD UI
# ============================================================================

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line
from kivy.core.image import Image as CoreImage
from datetime import datetime
import threading
from io import BytesIO
from tle_data import tle_data
from main import *
from util import lookup

Window.clearcolor = (0.02, 0.02, 0.05, 1)  # Deep space background
Window.size = (1600, 1000)


class DashboardPanel(BoxLayout):
    """Base panel class for dashboard widgets"""
    def __init__(self, title="", **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = [10, 10]
        self.spacing = 5
        
        # Panel background
        with self.canvas.before:
            Color(0.1, 0.1, 0.15, 0.9)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        
        self.bind(pos=self.update_bg, size=self.update_bg)
        
        # Title
        if title:
            title_label = Label(
                text=title,
                font_size=16,
                bold=True,
                color=(0.4, 0.8, 1.0, 1),
                size_hint_y=None,
                height=30
            )
            self.add_widget(title_label)
    
    def update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


class StatsPanel(DashboardPanel):
    """Statistics display panel"""
    def __init__(self, stats, **kwargs):
        super().__init__(title="ORBITAL STATISTICS", **kwargs)
        self.stats = stats
        
        # Total satellites
        total_label = Label(
            text=f"TOTAL SATELLITES: {stats['total']}",
            font_size=24,
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_y=None,
            height=50
        )
        self.add_widget(total_label)
        
        # Distribution stats
        stat_items = [
            ('Equatorial (0-30°)', stats['low'], stats['low_pct'], (1.0, 0.0, 0.0, 1)),
            ('Low (30-60°)', stats['medium'], stats['medium_pct'], (1.0, 0.65, 0.0, 1)),
            ('Medium (60-90°)', stats['high'], stats['high_pct'], (1.0, 1.0, 0.0, 1)),
            ('High (90-120°)', stats['polar'], stats['polar_pct'], (0.0, 1.0, 0.0, 1)),
            ('Retrograde (120-180°)', stats['retrograde'], stats['retrograde_pct'], (0.0, 0.5, 1.0, 1))
        ]
        
        for name, count, pct, color in stat_items:
            stat_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=35)
            
            # Color indicator
            color_widget = Widget(size_hint_x=None, width=20)
            with color_widget.canvas:
                Color(*color)
                Rectangle(pos=color_widget.pos, size=color_widget.size)
            stat_row.add_widget(color_widget)
            
            # Stat text
            stat_label = Label(
                text=f"{name}: {count} ({pct:.1f}%)",
                font_size=12,
                color=(0.9, 0.9, 0.9, 1),
                halign='left'
            )
            stat_row.add_widget(stat_label)
            
            self.add_widget(stat_row)


class SatelliteListPanel(DashboardPanel):
    """Scrollable satellite list panel"""
    def __init__(self, tle_data, **kwargs):
        super().__init__(title="SATELLITE REGISTRY", **kwargs)
        
        scroll = ScrollView()
        list_layout = BoxLayout(orientation='vertical', spacing=2, size_hint_y=None)
        list_layout.bind(minimum_height=list_layout.setter('height'))
        
        for sat_name, tle in tle_data.items():
            tle1, tle2 = tle
            if tle1[0] != "1":
                continue
            
            try:
                inclination = float(tle2[9:17])
                color = get_inclination_color(inclination)
                
                sat_row = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
                
                # Color bar
                color_bar = Widget(size_hint_x=None, width=5)
                with color_bar.canvas:
                    Color(*color)
                    Rectangle(pos=color_bar.pos, size=color_bar.size)
                sat_row.add_widget(color_bar)
                
                # Satellite name
                name_label = Label(
                    text=sat_name,
                    font_size=14,
                    color=(1, 1, 1, 1),
                    halign='left',
                    text_size=(200, None)
                )
                sat_row.add_widget(name_label)
                
                # Inclination
                incl_label = Label(
                    text=f"Inc: {inclination:.2f}°",
                    font_size=11,
                    color=(0.7, 0.7, 0.7, 1),
                    size_hint_x=None,
                    width=100
                )
                sat_row.add_widget(incl_label)
                
                list_layout.add_widget(sat_row)
            except:
                continue
        
        scroll.add_widget(list_layout)
        self.add_widget(scroll)


class LookupPanel(DashboardPanel):
    """Lookup UI to query tle.csv via util.lookup and display fields"""
    def __init__(self, **kwargs):
        super().__init__(title="SATELLITE LOOKUP", **kwargs)
        self.search_box = TextInput(hint_text='Enter OBJECT_NAME (e.g. KUIPER-00008)', size_hint_y=None, height=40)
        self.add_widget(self.search_box)
        btn = Button(text='Lookup', size_hint_y=None, height=40, background_color=(0.2,0.6,0.9,1), color=(1,1,1,1))
        btn.bind(on_press=self.on_lookup)
        self.add_widget(btn)

        # Header label to show satellite name + epoch after lookup
        self.header_label = Label(text='', size_hint_y=None, height=28, font_size=14, color=(0.8,0.9,1,1))
        self.add_widget(self.header_label)

        # Results area inside a ScrollView so it doesn't expand the layout
        self.results_container = ScrollView(size_hint=(1, 1))
        self.results = GridLayout(cols=2, size_hint_y=None, spacing=4, padding=[10, 10])
        self.results.bind(minimum_height=self.results.setter('height'))
        self.results_container.add_widget(self.results)
        self.add_widget(self.results_container)

    def clear_results(self):
        self.results.clear_widgets()

    def on_lookup(self, instance):
        name = self.search_box.text.strip()
        self.clear_results()
        if not name:
            self.results.add_widget(Label(text='Please enter a name', color=(1,0.5,0.5,1)))
            return
        row = lookup(name)
        if not row:
            self.header_label.text = ''
            self.results.add_widget(Label(text='Not found', color=(1,0.5,0.5,1)))
            return

        # Display important fields
        fields = [
            ('OBJECT_NAME', 'OBJECT_NAME'),
            ('OBJECT_ID', 'OBJECT_ID'),
            ('EPOCH', 'EPOCH'),
            ('MEAN_MOTION', 'MEAN_MOTION'),
            ('ECCENTRICITY', 'ECCENTRICITY'),
            ('INCLINATION', 'INCLINATION'),
            ('NORAD_CAT_ID', 'NORAD_CAT_ID'),
            ('DATA_SOURCE', 'DATA_SOURCE')
        ]

        for label, key in fields:
            self.results.add_widget(Label(text=label+':', size_hint_y=None, height=30, color=(0.8,0.8,0.8,1)))
            self.results.add_widget(Label(text=str(row.get(key, '')), size_hint_y=None, height=30, color=(1,1,1,1)))

        # Show satellite name and epoch in header label
        sat_name = row.get('OBJECT_NAME', '')
        epoch = row.get('EPOCH', '')
        if sat_name:
            self.header_label.text = f"{sat_name}    {epoch}"
        else:
            self.header_label.text = ''


class KuiperInfoPanel(DashboardPanel):
    """Display KUIPER info loaded from info.txt in a scrollable area"""
    def __init__(self, **kwargs):
        super().__init__(title="KUIPER INFO", **kwargs)
        try:
            with open('info.txt', 'r', encoding='utf-8') as fh:
                text = fh.read()
        except Exception as e:
            text = f"Error loading info.txt: {e}"

        # Label that sizes to its texture and is placed inside a ScrollView
        info_label = Label(
            text=text,
            size_hint_y=None,
            halign='left',
            valign='top',
            color=(1, 1, 1, 1)
        )
        # Update label height when texture changes
        info_label.bind(texture_size=lambda inst, val: setattr(inst, 'height', val[1]))

        # Ensure the label wraps to the panel width
        def _update_text_size(instance, value):
            info_label.text_size = (max(200, value[0]-20), None)

        self.bind(size=_update_text_size)

        scroll = ScrollView()
        scroll.add_widget(info_label)
        self.add_widget(scroll)


class StatusBar(BoxLayout):
    """Top status bar"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Make StatusBar two rows: title row + telemetry row
        self.orientation = 'vertical'
        self.size_hint_y = None
        self.height = 70
        self.padding = [6, 6]
        self.spacing = 4
        
        # First row: title, status indicator and time
        row1 = BoxLayout(orientation='horizontal', size_hint_y=None, height=36, spacing=8)
        title = Label(
            text='SPACE MISSION CONTROL',
            font_size=20,
            bold=True,
            color=(0.4, 0.8, 1.0, 1),
            size_hint_x=0.5
        )
        row1.add_widget(title)

        # Status indicator (small)
        self.status_label = Label(
            text='● OPERATIONAL',
            font_size=12,
            color=(0.0, 1.0, 0.0, 1),
            size_hint_x=0.2
        )
        row1.add_widget(self.status_label)

        # Time display
        self.time_label = Label(
            text=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            font_size=12,
            color=(0.7, 0.7, 0.7, 1),
            size_hint_x=0.3
        )
        row1.add_widget(self.time_label)
        self.add_widget(row1)

        # Second row: telemetry badges and small metrics
        row2 = BoxLayout(orientation='horizontal', size_hint_y=None, height=26, spacing=8)
        def _make_badge(color, text):
            b = BoxLayout(orientation='horizontal', size_hint_x=None, width=120, spacing=6)
            # colored square
            sq = Widget(size_hint_x=None, width=14)
            with sq.canvas:
                Color(*color)
                Rectangle(pos=sq.pos, size=(14, 14))
            # keep square position/size updated
            sq.bind(pos=lambda inst, val: setattr(inst.canvas.children[-1], 'pos', val), size=lambda inst, val: setattr(inst.canvas.children[-1], 'size', (14,14)))
            b.add_widget(sq)
            b.add_widget(Label(text=text, font_size=12, halign='left', valign='middle'))
            return b

        # Data connection badge
        self.badge_data = _make_badge((0, 1, 0, 1), 'Data: OK')
        row2.add_widget(self.badge_data)

        # GL / Renderer badge (may be N/A)
        self.badge_gl = _make_badge((0.6, 0.6, 0.6, 1), 'GL: N/A')
        row2.add_widget(self.badge_gl)

        # Last refresh
        self.last_refresh = Label(text='Last refresh: -', font_size=12)
        row2.add_widget(self.last_refresh)

        # Telemetry: total satellites and selected item (right aligned)
        self.total_label = Label(text='Total: 0', font_size=12, halign='right')
        self.selected_label = Label(text='', font_size=12, halign='right')
        row2.add_widget(self.total_label)
        row2.add_widget(self.selected_label)

        self.add_widget(row2)
        
        # Update time
        Clock.schedule_interval(self.update_time, 1.0)
    
    def update_time(self, dt):
        self.time_label.text = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

    # --- status helpers ---
    def set_total(self, n):
        self.total_label.text = f'Total: {n}'

    def set_selected(self, name):
        self.selected_label.text = f'Selected: {name}' if name else ''

    def set_data_status(self, state):
        # state: 'ok', 'warn', 'down'
        color = (0,1,0,1) if state=='ok' else ((1,0.65,0,1) if state=='warn' else (1,0,0,1))
        # update colored square
        with self.badge_data.children[1].canvas.before:
            pass

    def set_gl_status(self, text, color=(0.6,0.6,0.6,1)):
        # replace GL badge color and text
        # update colored square
        with self.badge_gl.children[1].canvas.before:
            pass


class ControlPanel(DashboardPanel):
    """Control buttons panel"""
    def __init__(self, app_instance=None, **kwargs):
        super().__init__(title="MISSION CONTROLS", **kwargs)
        self.app_instance = app_instance
        
        # Launch separate window button
        launch_btn = Button(
            text='OPEN IN SEPARATE WINDOW',
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.6, 0.9, 1),
            color=(1, 1, 1, 1),
            font_size=14,
            bold=True
        )
        launch_btn.bind(on_press=self.launch_viz)
        self.add_widget(launch_btn)
        
        # Rotate camera button
        rotate_btn = Button(
            text='ROTATE VIEW',
            size_hint_y=None,
            height=40,
            background_color=(0.3, 0.5, 0.7, 1),
            color=(1, 1, 1, 1)
        )
        rotate_btn.bind(on_press=self.rotate_view)
        self.add_widget(rotate_btn)
        
        # Refresh button
        refresh_btn = Button(
            text='REFRESH DATA',
            size_hint_y=None,
            height=40,
            background_color=(0.3, 0.3, 0.4, 1),
            color=(1, 1, 1, 1)
        )
        refresh_btn.bind(on_press=self.refresh_data)
        self.add_widget(refresh_btn)
        
        # System info
        info_label = Label(
            text='System Status: All Systems Nominal\nVisualization: Embedded',
            font_size=11,
            color=(0.6, 0.6, 0.6, 1),
            size_hint_y=None,
            height=60
        )
        self.add_widget(info_label)
        
        self.rotation_angle = 45
    
    def launch_viz(self, instance):
        """Launch PyVista visualization in separate window"""
        def run_viz():
            plotter = plot_satellites_pyvista(tle_data)
            plotter.show()
        
        thread = threading.Thread(target=run_viz, daemon=True)
        thread.start()
    
    def rotate_view(self, instance):
        """Rotate the embedded visualization camera"""
        if self.app_instance and self.app_instance.embedded_plotter:
            self.rotation_angle = (self.rotation_angle + 15) % 360
            self.app_instance.embedded_plotter.camera_position = [
                (1, 0, 0),
                (0, 0, 0),
                (0, 0, 1)
            ]
            self.app_instance.embedded_plotter.camera.azimuth = self.rotation_angle
            self.app_instance.embedded_plotter.camera.elevation = 20
            self.app_instance.update_visualization()
    
    def refresh_data(self, instance):
        """Refresh dashboard data"""
        print("Refreshing satellite data...")
        if self.app_instance:
            self.app_instance.setup_embedded_viz()


class SpaceDashboardApp(App):
    """Main space dashboard application"""
    def build(self):
        root = FloatLayout()
        
        # Get statistics
        stats = get_satellite_statistics(tle_data)
        
        # Status bar at top
        status_bar = StatusBar()
        status_bar.pos_hint = {'top': 1}
        # keep reference on the app instance for updates from panels
        self.status_bar = status_bar
        root.add_widget(status_bar)
        # initialize telemetry total
        try:
            status_bar.set_total(stats['total'])
        except Exception:
            pass
        
        # Main content area
        main_layout = BoxLayout(orientation='horizontal', spacing=10, padding=10)
        main_layout.pos_hint = {'top': 0.95, 'bottom': 0.05}
        main_layout.size_hint = (1, 0.95)
        
        # Left column - Statistics
        left_col = BoxLayout(orientation='vertical', spacing=10, size_hint_x=0.25)
        stats_panel = StatsPanel(stats, size_hint_y=0.5)
        # Replace Mission Controls with KUIPER info panel
        control_panel = KuiperInfoPanel(size_hint_y=0.5)
        left_col.add_widget(stats_panel)
        left_col.add_widget(control_panel)
        main_layout.add_widget(left_col)
        
        # Center - Lookup panel (replaces embedded 3D view)
        center_panel = DashboardPanel(size_hint_x=0.5)
        self.lookup_panel = LookupPanel()
        # allow lookup panel to update status bar when a selection is made
        self.lookup_panel.app = self
        center_panel.add_widget(self.lookup_panel)
        main_layout.add_widget(center_panel)
        
        # Right column - Satellite list
        right_col = BoxLayout(orientation='vertical', size_hint_x=0.25)
        sat_list = SatelliteListPanel(tle_data)
        right_col.add_widget(sat_list)
        main_layout.add_widget(right_col)
        
        root.add_widget(main_layout)
        
        # No periodic visualization updates needed for lookup UI
        
        return root
    
    def setup_embedded_viz(self):
        """Setup the embedded PyVista visualization"""
        try:
            # Get image widget size (use default if not yet set)
            width, height = 800, 600
            if self.viz_image.width > 0 and self.viz_image.height > 0:
                width = int(self.viz_image.width)
                height = int(self.viz_image.height)
            
            # Create offscreen plotter
            self.embedded_plotter = create_embedded_plotter(tle_data, width, height)
            self.update_visualization()
        except Exception as e:
            print(f"Error setting up embedded visualization: {e}")
            # Fallback: show error message
            error_label = Label(
                text=f'Visualization Error:\n{str(e)}',
                font_size=14,
                color=(1, 0, 0, 1),
                halign='center',
                valign='middle',
                pos_hint={'center_x': 0.5, 'center_y': 0.5}
            )
            error_label.bind(size=error_label.setter('text_size'))
            self.viz_image.parent.add_widget(error_label)
    
    def update_visualization(self, dt=None):
        """Update the embedded visualization frame"""
        if self.embedded_plotter is None:
            return
        
        try:
            # Get current size
            width = max(400, int(self.viz_image.width)) if self.viz_image.width > 0 else 800
            height = max(300, int(self.viz_image.height)) if self.viz_image.height > 0 else 600
            
            # Resize plotter if needed
            if self.embedded_plotter.window_size[0] != width or self.embedded_plotter.window_size[1] != height:
                self.embedded_plotter.window_size = [width, height]
            
            # Render to image
            self.embedded_plotter.render()
            image = self.embedded_plotter.image
            
            # Convert to bytes
            buf = BytesIO()
            image.save(buf, format='png')
            buf.seek(0)
            
            # Update Kivy image
            img = CoreImage(buf, ext='png')
            self.viz_image.texture = img.texture
        except Exception as e:
            # Silently fail to avoid spam
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("SPACE MISSION CONTROL DASHBOARD")
    print("=" * 60)
    print(f"Loaded {len(tle_data)} satellite entries")
    stats = get_satellite_statistics(tle_data)
    print(f"Total active satellites: {stats['total']}")
    print("\nLaunching dashboard...")
    
    SpaceDashboardApp().run()
