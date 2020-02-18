import os
from automatic_plot_helper import load_settings
from automatic_plot_helper import load_isings
import plot_anything_combined
import plot_anythingXY_scatter
from os import path, listdir
import glob

def main(sim_name, load_isings_list=True):
    settings = load_settings(sim_name)
    if load_isings_list:
        isings_list = load_isings(sim_name)
    plot_anything_auto(sim_name, ['Beta', 'avg_velocity', 'food'], settings, isings_list = isings_list, autoLoad=False)
    plot_var_tuples = [('Beta', 'avg_velocity'), ('avg_energy', 'avg_velocity'), ('avg_energy', 'food')]
    plot_scatter_auto(sim_name, settings, plot_var_tuples, isings_list, autoLoad=False)

def plot_scatter_auto(sim_name, settings, plot_var_tuples, isings_list, autoLoad = True):
    for plot_var_x, plot_var_y in plot_var_tuples:
        plot_anythingXY_scatter.main(sim_name, settings, isings_list, plot_var_x, plot_var_y, s = 0.8, alpha = 0.05, autoLoad = autoLoad)

def plot_anything_auto(sim_name, plot_vars, settings, isings_list = None, autoLoad = True):
    '''
    :param plot_vars: List of string of which each represents an attribute of the isings class
    :param isings_list: List of all isings generations in case it has been loaded previously
    '''

    if settings['energy_model']:
        #os.system("python plot__anything_combined {} avg_energy".format(sim_name))
        plot_anything_combined.main([sim_name], 'avg_energy', isings_lists=[isings_list], autoLoad=autoLoad)
    else:
        #os.system("python plot__anything_combined {} fitness".format(sim_name))
        plot_anything_combined.main([sim_name], 'fitness', isings_lists=[isings_list], autoLoad=autoLoad)

    for plot_var in plot_vars:
        plot_anything_combined.main([sim_name], plot_var, isings_lists=[isings_list], autoLoad=autoLoad)



if __name__ == '__main__':
    # sim_name = 'sim-20200120-004759-p_50_-t_2000_-g_8000_-a_7999_-ie_2_-ef_-sf_-zs_1_-n_ANN_energy_is_fitness_Beta_foodshare'
    # main(sim_name)

    # sim_names = ['sim-20200206-192604-p_50_-t_2000_-g_2000_-a_1999_-ie_2_-ef_-b_1_-zs_1_-fr_0_-n_ANN']

    folder = 'exp_2'
    d = path.join('save', folder)
    sim_names = [path.join(folder, o) for o in listdir(d)
                 if path.isdir(path.join(d, o))]

    # sim_names = glob.glob('save/sim-20200211*')
    # sim_names = [os.path.basename(d) for d in sim_names]


    for s in sim_names:
        main(s)
