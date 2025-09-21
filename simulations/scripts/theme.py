
#64D7FA 100, 215, 250 blue
#B4E044 180, 224, 68  green
#FFDB26 225, 216, 38  yellow
#FF9F46 255, 159, 70  orange

def load_font(dpg):
    with dpg.font_registry():
        try:
            mono = dpg.add_font("/usr/share/fonts/OTF/GeistMonoNerdFont-Medium.otf", 20)
            sans = dpg.add_font("/usr/share/fonts/Geist/otf/Geist-Medium.otf", 20)
            dpg.bind_font(mono)
        except Exception as e:
            print(f"Could not load font: {e}. Using default font.")

def create_line_series_theme(dpg):
    with dpg.theme() as line_series_theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, value=(255,216,38) , category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots)
    return line_series_theme

def create_global_theme(dpg):
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20, 20, 20))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 30, 30))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 40))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (60, 60, 60))
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, (30, 30, 30), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (20, 20, 20), category=dpg.mvThemeCat_Plots)
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
    return global_theme

def scatter_green(dpg):
    with dpg.theme() as green:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, value=(180, 224, 68) , category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3, category=dpg.mvThemeCat_Plots)
    return green

def scatter_orange(dpg):
    with dpg.theme() as orange:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, value=(255, 159, 70) , category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 3, category=dpg.mvThemeCat_Plots)
    return orange
