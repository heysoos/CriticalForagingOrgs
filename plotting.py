from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

from math import sin
from math import cos
from math import radians

#--- FUNCTIONS ----------------------------------------------------------------+

def plot_organism(settings, x1, y1, theta, ax):

    circle = Circle([x1,y1], settings['org_radius'], edgecolor = 'g', facecolor = 'lightgreen', zorder=8)
    ax.add_artist(circle)

    edge = Circle([x1,y1], settings['org_radius'], facecolor='None', edgecolor = 'darkgreen', zorder=8)
    ax.add_artist(edge)

    tail_len = settings['org_radius']*1.25
    
    x2 = cos(radians(theta)) * tail_len + x1
    y2 = sin(radians(theta)) * tail_len + y1

    ax.add_line(lines.Line2D([x1,x2],[y1,y2], color='darkgreen', linewidth=1, zorder=10))

    pass


def plot_food(settings, x1, y1, ax):

    circle = Circle([x1,y1], settings['food_radius']/2, edgecolor = 'darkslateblue', facecolor = 'mediumslateblue', zorder=5)
    ax.add_artist(circle)
    
    pass

#--- END ----------------------------------------------------------------------+
