
#64D7FA
#B4E044
#FFDB26
#FF9F46

from math import sin, exp
import threading
import dearpygui.dearpygui as dpg
import time
import collections

# --- Simulation Parameters ---
MAX_POINTS = 1000  # The number of data points to display on the plot at once
AMPLITUDE = 0.5     # Initial amplitude
FREQUENCY = 16.0    # Oscillation frequency (omega)
DAMPING = 1.0       # Damping factor (lambda)
TIME_STEP = 0.005   # Simulation time step

# --- Data Structures ---
# Use a deque as a circular/rolling buffer for our data
# Initialize with zeros
x_data = collections.deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
y_data = collections.deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
# Global time variable for the simulation
current_time = 0.0

dpg.create_context()

def update_series_data():
    global current_time

    while True:
        update_frame_counter()
        amplitude = dpg.get_value("amplitude_slider")
        frequency = dpg.get_value("frequency_slider")
        damping = dpg.get_value("damping_slider")
        new_y_value = amplitude * exp(-damping * current_time) * sin(frequency * current_time)
        x_data.append(current_time)
        y_data.append(new_y_value)
        dpg.set_value('series_tag', [list(x_data), list(y_data)])
        dpg.set_axis_limits("x_axis", x_data[0], x_data[-1])
        current_time += TIME_STEP
        time.sleep(0.01)

def reset_simulation():
    global current_time
    current_time = 0.0
    x_data.clear()
    y_data.clear()
    x_data.extend([0.0] * MAX_POINTS)
    y_data.extend([0.0] * MAX_POINTS)
    print("Simulation Reset")


# --- GUI Layout ---
VIEWPORT_WIDTH = 2300
VIEWPORT_HEIGHT = 1300

with dpg.window(label="Plot Panel",
                tag="win_left",
                pos=[0, 0],
                width=VIEWPORT_WIDTH * 0.6,
                height=VIEWPORT_HEIGHT,
                no_move=True,
                no_resize=True,
                no_close=True,
                no_collapse=True,
                no_title_bar=True):

    with dpg.theme(tag="plot_theme"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, value=(255,216,38) , category=dpg.mvThemeCat_Plots)
            # dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            # dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 2, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots)

    with dpg.plot(label="Mass-Spring-Damper System", height=-1, width=-1):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis")
        dpg.add_plot_axis(dpg.mvYAxis, label="Displacement", tag="y_axis")
        dpg.set_axis_limits("y_axis", -1.0, 1.0)
        dpg.add_line_series(list(x_data), list(y_data), label="Displacement", parent="y_axis", tag="series_tag")
        dpg.bind_item_theme("series_tag", "plot_theme")

with dpg.window(label="Control Panel",
                tag="win_right",
                pos=[VIEWPORT_WIDTH * 0.6, 0],
                width=VIEWPORT_WIDTH * 0.4,
                height=VIEWPORT_HEIGHT,
                no_move=True,
                no_resize=True,
                no_close=True,
                no_collapse=True,
                no_title_bar=True):

    dpg.add_text("Simulation Controls")
    dpg.add_separator()
    dpg.add_spacer(height=10)
    
    dpg.add_slider_float(
        label="Amplitude",
        tag="amplitude_slider",
        default_value=AMPLITUDE,
        min_value=0.1,
        max_value=1.0,
        width=300
    )
    dpg.add_slider_float(
        label="Frequency (ω)",
        tag="frequency_slider",
        default_value=FREQUENCY,
        min_value=1.0,
        max_value=100.0,
        width=300
    )
    dpg.add_slider_float(
        label="Damping (λ)",
        tag="damping_slider",
        default_value=DAMPING,
        min_value=0.0,
        max_value=5.0,
        width=300
    )
    dpg.add_spacer(height=20)
    dpg.add_button(label="Reset Simulation", callback=reset_simulation, width=300, height=40)
    dpg.add_spacer(height=20)
    dpg.add_separator()
    dpg.add_text(f"Frame: {dpg.get_frame_count()}", tag="frame_text")


def update_frame_counter():
    dpg.set_value("frame_text", f"Frame: {dpg.get_frame_count()}")

with dpg.font_registry():
    try:
        mono = dpg.add_font("/usr/share/fonts/OTF/GeistMonoNerdFont-Medium.otf", 20)
        dpg.bind_font(mono)
    except Exception as e:
        print(f"Could not load font: {e}. Using default font.")

dpg.create_viewport(title='Mass-Spring-Damper Simulation', width=VIEWPORT_WIDTH, height=VIEWPORT_HEIGHT)
dpg.set_viewport_vsync(True)
dpg.setup_dearpygui()
update_thread = threading.Thread(target=update_series_data, args=(), daemon=True)
update_thread.start()
dpg.show_viewport()
dpg.start_dearpygui() # Old way
dpg.destroy_context()
