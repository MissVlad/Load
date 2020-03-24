import numpy as np
import pandas as pd
from pandas import DataFrame
from nilmtk import DataSet, MeterGroup, Building
from prepare_datasets import load_datasets, load_ampds2_weather
from typing import Tuple
import copy
from matplotlib import pyplot as plt
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_legend_in_ax
from Ploting.fast_plot_Func import series, time_series
import datetime
from dateutil import tz
from project_path_Var import project_path_
import os


def fraction_of_energy_consumption_for_one_dataset(meter_group: MeterGroup):
    fraction = meter_group.submeters().fraction_per_meter().dropna()


def __fraction_of_energy_consumption_for_all_datasets(datasets: Tuple[DataSet, ...]):
    for this_dataset in datasets:
        for this_building in this_dataset.buildings.values():
            elec = this_building.elec  # type: MeterGroup
            fraction_of_energy_consumption_for_one_dataset(elec)


def plot_active_power_for_meter_group(meter_group: MeterGroup, sample_period=60, **kwargs):
    ax_ = None
    for this_meter in meter_group.meters:
        if isinstance(this_meter, MeterGroup):
            raise Exception("this_meter是MeterGroup")
        else:
            if this_meter.appliances.__len__() > 1:
                raise Exception("this_meter.appliances的个数超过1个")
            else:
                label = 'WHE' if this_meter.is_site_meter() else this_meter.appliances[0].metadata['original_name']
                this_meter_df = next(this_meter.load(ac_type='active', sample_period=sample_period))  # type: DataFrame
                x = this_meter_df.index.to_pydatetime()
                y = this_meter_df.iloc[:, 0].values
                ax_ = time_series(x=x, y=y, ax=ax_,
                                  y_label='Active power (W)', label=label, ncol=3,
                                  y_lim=(-5, None),
                                  x_axis_format=kwargs.get('x_axis_format'), tz=kwargs.get('tz'))
    # 调整线形和legend位置
    ax_ = reassign_linestyles_recursively_in_ax(ax_)
    return ax_


def energies_paper_one_day_visulisation_for_ampds2_dataset():
    def one_day_visulisation(this_dataset, start, end):
        this_dataset.set_window(start=start, end=end)
        elec = this_dataset.buildings[1].elec  # type: MeterGroup
        _ax = plot_active_power_for_meter_group(elec, x_axis_format="%H", tz=tz.gettz('Canada/Pacific'))
        adjust_legend_in_ax(_ax, protocol='Outside center left', ncol=3)
        _ax.set_xlabel('Time (hour of a day, time zone: Pacific Time Zone)')
        return _ax

    # ampds2_dataset
    # ax = one_day_visulisation(ampds2_dataset, "2013-02-06", "2013-02-07")
    # plt.savefig(os.path.join(project_path_,
    #                          'Report/ampds2_dataset_2013-02-06.svg'))
    # ax = one_day_visulisation(ampds2_dataset, "2013-08-07", "2013-08-08")
    # plt.savefig(os.path.join(project_path_,
    #                          'Report/ampds2_dataset_2013-08-07.svg'))

    # temperature correlation
    site_df = next(ampds2_dataset.buildings[1].elec.mains().load(ac_type='active', sample_period=60))
    site_df.join(pd.DataFrame)
    ampds2_weather_df = load_ampds2_weather()
    merge = pd.merge(site_df, ampds2_weather_df, right_index=True, left_index=True)
    tt=1


if __name__ == '__main__':
    ampds2_dataset, refit_dataset, uk_dale_dataset = load_datasets()
    energies_paper_one_day_visulisation_for_ampds2_dataset()
