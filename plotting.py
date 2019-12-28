import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines
import matplotlib.animation as animation
import time

from math import sin
from math import cos
from math import radians

plt.rcParams['animation.ffmpeg_path'] = "D:\\Program Files\\ffmpeg-20191217-bd83191-win64-static\\bin\\ffmpeg.exe"
#plt.rcParams["figure.figsize"] = [8,8]

#--- FUNCTIONS ----------------------------------------------------------------+



def animate_plot_Func(isings_all_timesteps, foods_all_timesteps, settings, ax, fig):
    ''' Uses FuncAnimation - works and currently implemented'''
    savepath = 'save\\{}\\animation-{}.mp4'.format(settings['loadfile'], time.strftime("%Y%m%d-%H%M%S"))
    design_figure(settings, fig, ax)
    initial_plot(isings_all_timesteps[0], foods_all_timesteps[0], settings, ax)
    #plt.savefig('firstframe.png', dpi =100, bbox_inches = 'tight')
    ani = animation.FuncAnimation(fig, __update_plot, fargs=[isings_all_timesteps, foods_all_timesteps, settings, ax, fig], interval=1, frames=len(isings_all_timesteps))
    #Writer = animation.FFMpegWriter
    Writer = animation.FFMpegFileWriter
    writer = Writer(fps=settings['animation_fps'], metadata=dict(artist='Sina Abdollahi, Jan Prosi'), bitrate=1800)
    ani.save(savepath, writer=writer)
    print('Animation successfully saved at {}'.format(savepath))
    #plt.show()

def animate_plot(all_artists, settings, ax, fig):
    '''
    Uses ArtistAnimation - currently not implemented as ryceptions occur when saving
    '''
    design_figure(settings, fig, ax)
    #initial_plot(isings_all_timesteps[0], foods_all_timesteps[0], settings, ax)
    #Writer = animation.FFMpegWriter
    savepath ='save\\{}\\animation-{}.mp4'.format(settings['loadfile'], time.strftime("%Y%m%d-%H%M%S"))
    Writer = animation.FFMpegFileWriter
    writer = Writer(fps=settings['animation_fps'], metadata=dict(artist='Sina Abdollahi, Jan Prosi'), bitrate=1800)
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani = animation.ArtistAnimation(fig, all_artists)
    #ani.save(savepath, writer=writer, dpi = 100)
    mpl.verbose.set_level("helpful")
    ani.save(savepath, writer=writer)
    print('Animation successfully saved at {}'.format(savepath))


def __update_plot(t, isings_all_timesteps, foods_all_timesteps, settings, ax, fig):
    #[a.remove for a in reversed(ax.artists)]

    isings = isings_all_timesteps[t]
    foods = foods_all_timesteps[t]
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
    pad = 0.5

    plt.xlim([settings['x_min'] - pad,
              settings['x_max'] + pad])
    plt.ylim([settings['y_min'] - pad,
              settings['y_max'] + pad])



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
    for I in isings:
        __plot_organism_init(settings, I[0], I[1], I[2], ax)

    # PLOT FOOD PARTICLES
    for food in foods:
        __plot_food_init(settings, food[0], food[1], ax)

def __plot_organism_init(settings, x1, y1, theta, ax):

    circle = Circle([x1,y1], settings['org_radius'], edgecolor = 'g', facecolor = 'lightgreen', zorder=8)
    ax.add_artist(circle)

    edge = Circle([x1,y1], settings['org_radius'], facecolor='None', edgecolor = 'darkgreen', zorder=8)
    ax.add_artist(edge)

    tail_len = settings['org_radius']*1.25
    
    x2 = cos(radians(theta)) * tail_len + x1
    y2 = sin(radians(theta)) * tail_len + y1


    ax.add_line(lines.Line2D([x1,x2],[y1,y2], color='darkgreen', linewidth=1, zorder=10))

    pass


def __plot_food_init(settings, x1, y1, ax):

    circle = Circle([x1,y1], settings['food_radius']/2, edgecolor = 'darkslateblue', facecolor = 'mediumslateblue', zorder=5)
    ax.add_artist(circle)
    
    pass

#--------------Functions used for ArtistAnimation--------------------------

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


#--- END ----------------------------------------------------------------------+
