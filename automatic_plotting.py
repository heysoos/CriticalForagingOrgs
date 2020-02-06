import os
from automatic_plot_helper import load_settings
from automatic_plot_helper import load_isings
import plot_anything_combined
import plot_anythingXY_scatter

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

    sim_names = ['sim-20200128-151458-p_50_-t_2000_-g_8000_-a_7999_-b_0.1_-ie_2_-ef_-fr_0.1_-zs_0_'
                 '-n_ANN_new_physics_FIXED',
                 'sim-20200128-151506-p_50_-t_2000_-g_8000_-a_7999_-b_1_-ie_2_-ef_-fr_0.1_-zs_0_'
                 '-n_ANN_new_physics_FIXED',
                 'sim-20200128-151513-p_50_-t_2000_-g_8000_-a_7999_-b_10_-ie_2_-ef_-fr_0.1_-zs_0_'
                 '-n_ANN_new_physics_FIXED',
                 'sim-20200128-151522-p_50_-t_2000_-g_8000_-a_7999_-b_0.1_-ie_2_-ef_-fr_0.1_-zs_1_'
                 '-n_ANN_new_physics_FIXED',
                 'sim-20200128-151529-p_50_-t_2000_-g_8000_-a_7999_-b_1_-ie_2_-ef_-fr_0.1_-zs_1_'
                 '-n_ANN_new_physics_FIXED',
                 'sim-20200128-151534-p_50_-t_2000_-g_8000_-a_7999_-b_10_-ie_2_-ef_-fr_0.1_-zs_1_'
                 '-n_ANN_new_physics_FIXED',
                 ]

    for s in sim_names:
        main(s)
