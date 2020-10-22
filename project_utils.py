from pathlib import Path

cwd = Path().cwd()
project_path_ = Path('/'.join(cwd.parts[:cwd.parts.index('MyProject') + 1]))

TIME_SERIES_PLOT_KWARGS = {
    "x_axis_format": "%y-%b-%d %H",
    "x_label": "Date and Time [Year-Month-Day Hour]"
}
