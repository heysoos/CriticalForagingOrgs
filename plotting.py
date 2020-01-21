import matplotlib as mpl
import numpy as np
mpl.use('Agg') #For server use
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines
import matplotlib.animation as animation
import time

from math import sin
from math import cos
from math import radians
import os


#plt.rcParams["figure.figsize"] = [8,8]

#--- FUNCTIONS ----------------------------------------------------------------+



def animate_plot_Func(isings_all_timesteps, foods_all_timesteps, settings, ax, fig, rep, t, save_folder):
    ''' Uses FuncAnimation - works and currently implemented'''
    my_path = os.path.abspath(__file__)
    #mpl.rcParams["savefig.directory"] = my_path + 'tmp/'
    if settings['server_mode']:
        plt.rcParams['animation.ffmpeg_path'] = '/data-uwks159/home/jprosi/ffmpeg-4.2.1-linux-64/ffmpeg'
        #'/usr/local/bin/ffmpeg'
    else:
        pass
        # plt.rcParams['animation.ffmpeg_path'] = "D:/Program Files/ffmpeg-20191217-bd83191-win64-static/bin/ffmpeg.exe"

    if settings['LoadIsings']:
        path = '/save/{}/animation_gen{}/'.format(settings['loadfile'], rep)
    else:
        path = '/{}animation_gen{}/'.format(save_folder, rep)
    savename = 'ani-{}-{}ts-gen{}.mp4'.format(time.strftime("%Y%m%d-%H%M%S"), t, rep)

    savepath = savename
    cur_wdir = os.getcwd()
    path = cur_wdir.replace('\\','/') + path
    if not os.path.exists(path):
        os.makedirs(path)

    os.chdir(path)
    design_figure(settings, fig, ax)
    initial_plot(isings_all_timesteps[0], foods_all_timesteps[0], settings, ax)
    #plt.savefig('firstframe.png', dpi =100, bbox_inches = 'tight')
    ani = animation.FuncAnimation(fig, __update_plot,
                                  fargs=[isings_all_timesteps, foods_all_timesteps, settings, ax, fig],
                                  interval=1, frames=len(isings_all_timesteps))

    if True:
        #ffmpeg does not work on server, therefore default writer used
        Writer = animation.FFMpegFileWriter
        writer = Writer(fps=settings['animation_fps'], metadata=dict(artist='Sina Khajehabdollahi, Jan Prosi'),
                        bitrate=1800)
        writer.frame_format = 'png'
        ani.save(savepath, writer=writer)
    elif False:
        #Using defaul writer instead of imagemagick
        ani.save(savepath, dpi=100, writer='imagemagick', fps=settings['animation_fps']) #TODO: dpi=100 writer='imagemagick',
    elif False:
        writer = animation.ImageMagickFileWriter(fps=settings['animation_fps'], metadata=dict(artist='Sina Abdollahi, Jan Prosi'), bitrate=1800)
        ani.save('location.gif', writer=writer, dpi = 100)#
    print('\nAnimation successfully saved at {}'.format(savepath))
    os.chdir(cur_wdir)


def animate_plot(all_artists, settings, ax, fig):
    '''
    Uses ArtistAnimation - currently not implemented as ryceptions occur when saving
    '''
    if settings['server_mode']:
        plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    else:
        plt.rcParams['animation.ffmpeg_path'] = "D:/Program Files/ffmpeg-20191217-bd83191-win64-static/bin/ffmpeg.exe"
    design_figure(settings, fig, ax)
    #initial_plot(isings_all_timesteps[0], foods_all_timesteps[0], settings, ax)
    #Writer = animation.FFMpegWriter
    savepath ='save/{}/animation-{}.mp4'.format(settings['loadfile'], time.strftime("%Y%m%d-%H%M%S"))
    Writer = animation.FFMpegFileWriter
    writer = Writer(fps=settings['animation_fps'], metadata=dict(artist='Sina Abdollahi, Jan Prosi'), bitrate=1800)
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.ArtistAnimation(fig, all_artists)
    #ani.save(savepath, writer=writer, dpi = 100)
    mpl.verbose.set_level("helpful") #<-- This error occured in FuncAnimate when savepath did not exist yet
    ani.save(savepath, writer=writer)
    print('Animation successfully saved at {}'.format(savepath))


def __update_plot(t, isings_all_timesteps, foods_all_timesteps, settings, ax, fig):
    #[a.remove for a in reversed(ax.artists)]

    tt = np.max([1, t - 10])  # number of past frames to draw
    # isings = isings_all_timesteps[t]
    isings = [isings_all_timesteps[i] for i in range(tt, t)]
    foods = foods_all_timesteps[t]
    # foods = [foods_all_timesteps[i] for i in range(tt, t)]
    ax.cla()
    design_figure(settings, fig, ax)
    initial_plot(isings, foods, settings, ax)

    return ax.artists

def design_figure(settings, fig, ax):
    # fig, ax = plt.subplots()
    fig.set_size_inches(9.6, 5.4)

    # plt.xlim([settings['x_min'] + settings['x_min'] * 0.25,
    #           settings['x_max'] + settings['x_max'] * 0.25])
    # plt.ylim([settings['y_min'] + settings['y_min'] * 0.25,
    #           settings['y_max'] + settings['y_max'] * 0.25])
    pad = 1

    # plt.xlim([settings['x_min'] - pad,
    #           settings['x_max'] + pad])
    # plt.ylim([settings['y_min'] - pad,
    #           settings['y_max'] + pad])

    plt.xlim([settings['x_min'],
              settings['x_max']])
    plt.ylim([settings['y_min'],
              settings['y_max']])



    # MISC PLOT SETTINGS
    ax.set_aspect('equal')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    #plt.figtext(0.025, 0.90, r'T_STEP: ' + str(time))



    #ax.plot()
    #plt.pause(1e-5)
    #plt.draw()
    #plt.cla()
    #plt.clf()
    #frame.close()

def plot_frame(settings, folder, fig, ax, isings, foods, time, rep):
    # fig, ax = plt.subplots()
    fig.set_size_inches(9.6, 5.4)

    # plt.xlim([settings['x_min'] + settings['x_min'] * 0.25,
    #           settings['x_max'] + settings['x_max'] * 0.25])
    # plt.ylim([settings['y_min'] + settings['y_min'] * 0.25,
    #           settings['y_max'] + settings['y_max'] * 0.25])
    pad = 0.5

    plt.xlim([settings['x_min'] - pad,
              settings['x_max'] + pad])
    plt.ylim([settings['y_min'] - pad,
              settings['y_max'] + pad])


    # PLOT ORGANISMS

    #plotting.initial_plot(isings, foods, settings, ax)




    #line, = ax.plot(0,0)

    # MISC PLOT SETTINGS
    ax.set_aspect('equal')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    plt.figtext(0.025, 0.90, r'T_STEP: ' + str(time))

    # if settings['plotLive'] == True:
    #     plt.show()
    if settings['save_data'] == True:
        filename = folder + 'figs/iter-' + str(rep) + 'time-' + str(time).zfill(4) + '.png'
        plt.savefig(filename, dpi=300)
        #plt.close()
    #ax.plot()
    plt.pause(1e-5)
    #plt.draw()
    plt.cla()
    #plt.clf()
    #frame.close()


def initial_plot(isings, foods, settings, ax):
    if len(np.shape(isings)) == 3:
        alpha = np.logspace(-1, 0, len(isings))
        for i, It in enumerate(isings):
            for I in It:
                __plot_organism_init(settings, I[0], I[1], I[2], I[3], ax, alpha[i])
    else:
        for I in isings:
            __plot_organism_init(settings, I[0], I[1], I[2], I[3], ax)

    # PLOT FOOD PARTICLES
    for food in foods:
        __plot_food_init(settings, food[0], food[1], ax)

def __plot_organism_init(settings, x1, y1, theta, energy, ax, alpha=1):

    if energy < 0.5:
        energy = 0.5

    circle = Circle([x1,y1], settings['org_radius'], edgecolor='g', facecolor='lightgreen', zorder=8, alpha=alpha)
    ax.add_artist(circle)

    edge = Circle([x1,y1], settings['org_radius'], facecolor='None', edgecolor='darkgreen', zorder=8, alpha=alpha)
    ax.add_artist(edge)

    tail_len = settings['org_radius']*(np.log(energy + 1)) * 1.25
    
    x2 = cos(radians(theta)) * tail_len + x1
    y2 = sin(radians(theta)) * tail_len + y1


    ax.add_line(lines.Line2D([x1,x2],[y1,y2], color='darkgreen', linewidth=1, zorder=10, alpha=alpha))

    pass


def __plot_food_init(settings, x1, y1, ax):

    circle = Circle([x1,y1], settings['food_radius']/2, edgecolor = 'darkslateblue', facecolor = 'mediumslateblue', zorder=5)
    ax.add_artist(circle)
    
    pass

#--------------Functions used for ArtistAnimation--------------------------
"""
def create_artists_append(isings, foods, settings):
    '''Creates artists and apends the to artist list'''
    artists_this_gen = []
    for I in isings:
        artists_this_gen = __create_artists_organisms(artists_this_gen,settings, I.xpos, I.ypos, I.r)
    for food in foods:
        artists_this_gen = __create_food_artist(artists_this_gen, settings, food.xpos, food.ypos)
    return artists_this_gen

def __create_artists_organisms(artist_list, settings, x1, y1, theta):
    #Circles
    artist_list.append(Circle([x1,y1], settings['org_radius'], edgecolor = 'g', facecolor = 'lightgreen', zorder=8))
    #Edges
    artist_list.append(Circle([x1,y1], settings['org_radius'], facecolor='None', edgecolor = 'darkgreen', zorder=8))
    tail_len = settings['org_radius'] * 1.25

    x2 = cos(radians(theta)) * tail_len + x1
    y2 = sin(radians(theta)) * tail_len + y1

    # Does this work??
    #artist_list.append(lines.Line2D([x1, x2], [y1, y2], color='darkgreen', linewidth=1, zorder=10))
    return artist_list

def __create_food_artist(artist_list, settings, x1, y1):
    artist_list.append(Circle([x1, y1], settings['food_radius'] / 2, edgecolor='darkslateblue', facecolor='mediumslateblue',
                    zorder=5))
    return artist_list

"""
#--- END ----------------------------------------------------------------------+
