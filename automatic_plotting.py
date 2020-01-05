import os
from automatic_plot_helper import load_settings


def automatic_plot(sim_name):
    settings = load_settings(sim_name)
    if settings['energy_model']:
        os.system("python plot__anything_combined {} avg_energy".format(sim_name))
    else:
        os.system("python plot__anything_combined {} fitness".format(sim_name))

if __name__=='__main__':
    sim_name = 'sim-20200103-170627-ser_-f_40_-s_-b_10_-ie_2_-a_0_200_500_1000_1999'
    automatic_plot(sim_name)

