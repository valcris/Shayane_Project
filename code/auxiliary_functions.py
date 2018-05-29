'''
This file contains some auxiliary functions used for plotting maps,
histograms, defining color scale bounds and also calculating some
useful quantities.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.mlab as mlab
from fatiando.utils import gaussian2d
import cPickle as pickle
from fatiando.gridder import regular
from fatiando.mesher import Prism
from fatiando.gravmag import transform
from fatiando.constants import PERM_FREE_SPACE, T2NT

def multiplotmap (ax, x, y, data, shape, area, color_scheme,
                  prism_projection, projection_style, model,
                  unit = None, ranges = None,
                  figure_label=None, label_color='k', label_size=10,
                  label_position = (0.02,0.93),
                  label_x = True, label_y = True,
                  observations=False, point_style=None, point_size=None):
    '''
    Simple function for ploting maps.

    input
    ax:  single axis object of matplotlib.pyplot
    x: numpy array - x coordinates of the data to be plotted.
    y: numpy array - y coordinates of the data to be plotted.
    data: None or numpy array - data to be plotted. If None,
          it does not creates a contour map.
    shape: tuple - number of points along x and y directions.
    area: list - contains the boundaries along x and y directions.
    color_scheme: string - color scheme to plot the data
                  (http://matplotlib.org/api/colors_api.html#matplotlib.colors.Colormap).
    prism_projection: True or False - If True, plot the projection of the
                      synthetic bodies. If not True, it does not plot.

    projection_style: string - defines the style of the lines representing
                      the projection of the synthetic bodies.
    model: list - horizontal coordinates of the synthetic
           bodies.
    unit: None or string - title of the label.
    ranges: None or tuple - If None, it defines bounds of the color scale automatically.
            If not, it uses the values defined in ranges[0] and ranges[1] as the
            lower and upper bounds, respectively.
    figure_label: string - used for multiple figures, e.g., 2a, 2b, 2c.
    label_color: string - defines the color of the label.
    label_size: int - defines the size of the label.
    label_position: tuple - defines the position of the label.
    label_x: boolean - defines if the x label is shown.
    label_y: boolean - defines if the y label is shown.
    observations: boolean - If True, it plots the points.
    point_style: None or string - defines the style of the
                 points representing the observations.
    point_size: None or int - defines the size of the
                points representing the observations.

    '''

    N = shape[0]*shape[1]

    if (data is not None):
        assert (x.size == N) and (y.size == N) and (data.size == N), \
                'x, y and data must have the same size difined by shape'
    if (data is None):
        assert (x.size == N) and (y.size == N), \
                'x and y must have the same size difined by shape'

    ax.axis('scaled')

    if (data is not None) and (ranges is None):
        cs = ax.contourf(np.reshape(y, shape), np.reshape(x, shape), np.reshape(data, shape),
                         20, cmap=plt.get_cmap(color_scheme))
        #cbar = plt.colorbar(pad=0.01, aspect=40, shrink=1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(cs, ax=ax, cax=cax)
        if unit is not None:
            cbar.set_label(unit)
    if (data is not None) and (ranges is not None):
        cs=ax.contourf(np.reshape(y, shape), np.reshape(x, shape), np.reshape(data, shape),
                       20, cmap=plt.get_cmap(color_scheme),
                       vmin=ranges[0], vmax=ranges[1])
        #cbar = plt.colorbar(pad=0.01, aspect=40, shrink=1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(cs, ax=ax, cax=cax)
        if unit is not None:
            cbar.set_label(unit)

    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs = [x1, x1, x2, x2, x1]
            ys = [y1, y2, y2, y1, y1]
            ax.plot(xs, ys, projection_style, linewidth=1.0)

    if figure_label is not None:
        ax.annotate(s=figure_label, xy=label_position,
                    xycoords = 'axes fraction', color=label_color,
                    fontsize=label_size)

    if observations is True:
        ax.plot(y,x, point_style, markersize=point_size)

    ax.set_xlim(area[2], area[3])
    ax.set_ylim(area[0], area[1])

    if label_x is True:
        ax.set_xlabel('y (km)')

    if label_y is True:
        ax.set_ylabel('x (km)')

    #transform the axes from m to km
    ax.set_xticklabels(['%g' % (0.001 * l) for l in ax.get_xticks()])
    ax.set_yticklabels(['%g' % (0.001 * l) for l in ax.get_yticks()])

    plt.tight_layout()

def plotmap (x, y, data, shape, area, color_scheme,
             prism_projection, projection_style, model, unit = True,
             ranges = None,
             figure_label=None, label_color='k', label_size=10,
             label_position = (0.02,0.93),
             label_x = True, label_y = True,figure_name=None,
             observations=False, point_style=None, point_size=None):
    '''
    Simple function for ploting maps.

    input
    x: numpy array - x coordinates of the data to be plotted.
    y: numpy array - y coordinates of the data to be plotted.
    data: None or numpy array - data to be plotted. If None,
          it does not creates a contour map.
    shape: tuple - number of points along x and y directions.
    area: list - contains the boundaries along x and y directions.
    color_scheme: string - color scheme to plot the data
                  (http://matplotlib.org/api/colors_api.html#matplotlib.colors.Colormap).
    prism_projection: True or False - If True, plot the projection of the
                      synthetic bodies. If not True, it does not plot.

    projection_style: string - defines the style of the lines representing
                      the projection of the synthetic bodies.
    model: list - horizontal coordinates of the synthetic
           bodies.
    unit: None or string - title of the label.
    ranges: None or tuple - If None, it defines bounds of the color scale automatically.
            If not, it uses the values defined in ranges[0] and ranges[1] as the
            lower and upper bounds, respectively.
    figure_label: string - used for multiple figures, e.g., 2a, 2b, 2c.
    label_color: string - defines the color of the label.
    label_size: int - defines the size of the label.
    label_position: tuple - defines the position of the label.
    label_x: boolean - defines if the x label is shown.
    label_y: boolean - defines if the y label is shown.
    figure_name: string - name of the file (including the extension)
                 showing the map.
    observations: boolean - If True, it plots the points.
    point_style: None or string - defines the style of the
                 points representing the observations.
    point_size: None or int - defines the size of the
                points representing the observations.

    output
    A matplotlib figure.
    '''

    N = shape[0]*shape[1]

    if (data is not None):
        assert (x.size == N) and (y.size == N) and (data.size == N), \
                'x, y and data must have the same size difined by shape'
    if (data is None):
        assert (x.size == N) and (y.size == N), \
                'x and y must have the same size difined by shape'

    plt.close('all')

    fig = plt.figure(figsize=(3.33333, 2.66667))
    #fig = plt.figure(figsize=(4., 3.))

    ax = plt.gca()

    ax.axis('scaled')
    #ax.axis('auto')

    if (data is not None) and (ranges is None):
        plt.contourf(np.reshape(y, shape), np.reshape(x, shape), np.reshape(data, shape),
                     20, cmap=plt.get_cmap(color_scheme))
        #cbar = plt.colorbar(pad=0.01, aspect=40, shrink=1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(cax=cax)
        cbar.set_label(unit)
    if (data is not None) and (ranges is not None):
        plt.contourf(np.reshape(y, shape), np.reshape(x, shape), np.reshape(data, shape),
                     20, cmap=plt.get_cmap(color_scheme),
                     vmin=ranges[0], vmax=ranges[1])
        #cbar = plt.colorbar(pad=0.01, aspect=40, shrink=1.0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(cax=cax)
        cbar.set_label(unit)

    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs = [x1, x1, x2, x2, x1]
            ys = [y1, y2, y2, y1, y1]
            ax.plot(xs, ys, projection_style, linewidth=1.0)

    if figure_label is not None:
        ax.annotate(s=figure_label, xy=label_position,
                    xycoords = 'axes fraction', color=label_color,
                    fontsize=label_size)

    if observations is True:
        ax.plot(y,x, point_style, markersize=point_size)

    ax.set_xlim(area[2], area[3])
    ax.set_ylim(area[0], area[1])

    if label_x is True:
        ax.set_xlabel('y (km)')

    if label_y is True:
        ax.set_ylabel('x (km)')

    #transform the axes from m to km
    ax.set_xticklabels(['%g' % (0.001 * l) for l in ax.get_xticks()])
    ax.set_yticklabels(['%g' % (0.001 * l) for l in ax.get_yticks()])

    plt.tight_layout()

    if figure_name is not None:
        plt.savefig(figure_name, dpi=600)

    plt.show()

def multiplothist (ax, data, file_name = None, color = 'grey',
                   text_position = [0.5, 0.93, 0.86, 0.05], text_fontsize = 8,
                   unit = None, figure_label = None, label_color='k',
                   label_size=10, label_position = (0.02,0.93),
                   label_x = True, label_y = True):
    '''
    Plot a normalized histogram. the normalization consists in
    removing the sample mean from each data and dividing the
    result by the sample standard deviation.

    input
    data: numpy array - it contains the data.
    file_name: None or string - If None (default), it does not creates
               a file. If not None, saves the figure in a file whose name
               is defined by the string.
    color: string - defines the facecolor of the histogram.
    text_position: list - it controls the position where the
                   sample mean and standard deviation will be plotted.
    text_fontsize: float - it defines the size of the fontsize of
                   the sample mean and standard deviation.
    unit: None or string - defines the unit of the estimated mu and
          sigma.
    figure_label: string - used for multiple figures, e.g., 2a, 2b, 2c.
    label_color: string - defines the color of the label.
    label_size: int - defines the size of the label.
    label_position: tuple - defines the position of the label.
    label_x: boolean - defines if the x label is shown.
    label_y: boolean - defines if the y label is shown.

    output
    A matplotlib figure.
    '''

    data_mean = np.mean(data)
    data_std = np.std(data)

    ax.axis('auto')

    n, bins, patches = ax.hist(data, 50, normed=1,
                               histtype='stepfilled', facecolor=color, alpha=0.75)

    bincenters = 0.5*(bins[1:]+bins[:-1])

    y = mlab.normpdf( bincenters, data_mean, data_std)
    l = ax.plot(bincenters, y, 'k--', linewidth=1)

    xlim = np.max(np.abs([np.min(data), np.max(data)]))
    ax.set_xlim(-xlim, xlim)
    ymax = 1.2*n.max()
    ax.set_ylim(0, ymax)
    ax.annotate(s=r'$\mu$' % (data_mean),
                xy=(text_position[0],text_position[1]),
                xycoords = 'axes fraction', color='k',
                fontsize=text_fontsize+4)
    ax.annotate(s=r'$\sigma$' % (data_mean),
                xy=(text_position[0],text_position[2]),
                xycoords = 'axes fraction', color='k',
                fontsize=text_fontsize+4)

    if unit is not None:
        ax.annotate(s=r' = %.1e %s' % (data_mean, unit),
                    xy=(text_position[0] + text_position[3],text_position[1]),
                    xycoords = 'axes fraction', color='k',
                    fontsize=text_fontsize)
        ax.annotate(s=r' = %.1e %s' % (data_std, unit),
                    xy=(text_position[0] + text_position[3],text_position[2]),
                    xycoords = 'axes fraction', color='k',
                    fontsize=text_fontsize)

        if label_x is True:
            ax.set_xlabel('Residuals '+unit)

    if unit is None:
        ax.annotate(s=r' = %.1e' % (data_mean),
                    xy=(text_position[0] + text_position[3],text_position[1]),
                    xycoords = 'axes fraction', color='k',
                    fontsize=text_fontsize)
        ax.annotate(s=r' = %.1e' % (data_std),
                    xy=(text_position[0] + text_position[3],text_position[2]),
                    xycoords = 'axes fraction', color='k',
                    fontsize=text_fontsize)

        if label_x is True:
            ax.set_xlabel('Residuals')

    if figure_label is not None:
        ax.annotate(s=figure_label, xy=label_position,
                    xycoords = 'axes fraction', color=label_color,
                    fontsize=label_size)

    if label_y is True:
        ax.set_ylabel('Probability')

    plt.tight_layout()

def plothist (data, file_name = None, color = 'grey',
              text_position = [0.3, 0.95, 0.88], text_fontsize = 8,
              unit = None,
              figure_label = None, label_color='k'):
    '''
    Plot a normalized histogram. the normalization consists in
    removing the sample mean from each data and dividing the
    result by the sample standard deviation.

    input
    data: numpy array - it contains the data.
    file_name: None or string - If None (default), it does not creates
               a file. If not None, saves the figure in a file whose name
               is defined by the string.
    color: string - defines the facecolor of the histogram.
    text_position: list - it controls the position where the
                   sample mean and standard deviation will be plotted.
    text_fontsize: float - it defines the size of the fontsize of
                   the sample mean and standard deviation.
    unit: None or string - defines the unit of the estimated mu and
          sigma.
    figure_label: string - used for multiple figures, e.g., 2a, 2b, 2c.
    label_color: string - defines the color of the label.

    output
    A matplotlib figure.
    '''

    data_mean = np.mean(data)
    data_std = np.std(data)

    plt.close('all')
    plt.figure(figsize=(3.33333,2.66667))

    plt.axis('auto')

    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(data, 50, normed=1,
                               histtype='stepfilled', facecolor=color, alpha=0.75)

    bincenters = 0.5*(bins[1:]+bins[:-1])

    y = mlab.normpdf( bincenters, mu, sigma)
    l = ax.plot(bincenters, y, 'k--', linewidth=1)

    #ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    ax.set_xlim(40, 160)
    ymax = 1.2*n.max()
    ax.set_ylim(0, ymax)
    plt.text(text_position[0]*xlim, text_position[1]*ymax,
             r'$\mu$', fontsize=text_fontsize+4)
    plt.text(text_position[0]*xlim, text_position[2]*ymax,
             r'$\sigma$', fontsize=text_fontsize+4)
    if unit is not None:
        plt.text(text_position[0]*xlim + 0.4, text_position[1]*ymax,
                 r' = %.3e %s' % (data_mean, unit), fontsize=text_fontsize)
        plt.text(text_position[0]*xlim + 0.4, text_position[2]*ymax,
                r' = %.3e %s' % (data_std, unit), fontsize=text_fontsize)
        ax.set_xlabel(unit)
    if unit is None:
        plt.text(text_position[0]*xlim + 0.4, text_position[1]*ymax,
                 r' = %.3e' % data_mean, fontsize=text_fontsize)
        plt.text(text_position[0]*xlim + 0.4, text_position[2]*ymax,
                r' = %.3e' % data_std, fontsize=text_fontsize)
        ax.set_xlabel('Residuals')

#    ax.grid(True)

    if figure_label is not None:
        plt.annotate(s=figure_label, xy=(0.02,0.93),
                     xycoords = 'axes fraction', color=label_color)

    if file_name is not None:
        plt.savefig(file_name, dpi=600)

    ax.set_ylabel('Probability')

    plt.tight_layout()

    plt.show()

def scale_bounds (data, div=False):
    '''
    Defines the bounds of a color scale.

    input
    data: numpy array - data to be plotted with the specified color scale.
    div: boolean - If True, it defines the limits for a divergent
         color scale. If False (default), it defines the limits based on
         the data limits.

    output
    ranges: tuple - contains the lower and upper limits of the color scale.
    '''

    if div is True:
        ranges = np.max(np.abs([np.min(data), np.max(data)]))
        return (-ranges, ranges)
    if div is False:
        return (np.min(data), np.max(data))

def gaussian2D(x, y, sigma_x, sigma_y, x0=0., y0=0., angle=0.0, amp=1.0, shift=0.):
    '''
    It calculates a two-dimensional Gaussian function.

    input
    x: numpy array - x coordinates where the 2D Gaussian will be calculated.
    y: numpy array - y coordinates where the 2D Gaussian will be calculated.
    sigma_x: float - standard deviation along the x direction.
    sigma_y: float - standard deviation along the y direction.
    x0: float - x coordinate of the center of the Gaussian function.
    y0: float - y coordinate of the center of the Gaussian function.
    angle: float - clockwise angle bettwen the x-axis of the coordinate system
           and the x-axis of the Gaussian function.
    amp: float - amplitude of the Gaussian function.
    shift: float - constant level added to the Gaussian function.

    output
    f: numpy array - Gaussian function evaluated at the coordinates x and y.
    fx: numpy array - First derivative of the Gaussian function with respect
        to the variable x, evaluated at the coordinates x and y.
    fy: numpy array - First derivative of the Gaussian function with respect
        to the variable y, evaluated at the coordinates x and y.
    '''
    tempx = 1./sigma_x**2.
    tempy = 1./sigma_y**2.
    theta = np.deg2rad(angle)
    cos = np.cos(theta)
    sin = np.sin(theta)
    sin2 = np.sin(2.*theta)
    a = 0.5*(tempx*cos**2. + tempy*sin**2.)
    b = 0.25*sin2*(tempy - tempx)
    c = 0.5*(tempx*sin**2. + tempy*cos**2.)

    f = shift + amp*np.exp(-(a*(x - x0)**2. - 2.*b*(x - x0)*(y - y0) + c*(y - y0)**2.))
    fx = f*2.*(-a*(x - x0) + b*(y - y0))
    fy = f*2.*(-c*(y - y0) + b*(x - x0))

    return f, fx, fy

def observation_surface(x, y):
    '''
    Evaluates the function defining the observation
    surface at the points with Cartesian coordinates
    x and y.

    input
    x: numpy array - Cartesian coordinates of the points along the x-axis.
    y: numpy array - Cartesian coordinates of the points along the y-axis.

    output
    z: numpy array - observation surface.
    '''

    assert (x.size == y.size), 'x and y must have the same size'

    surface = -550. - 700.*gaussian2d(x, y, 20000., 10000., 12500., 22500., angle=45.)

    return surface

def eqlayer_surface(x, y):
    '''
    Evaluates a function f(x,y) defining the surface
    containing the equivalent sources with Cartesian
    coordinates x and y.

    input
    x: numpy array - Cartesian coordinates of the equivalent sources
       along the x-axis.
    y: numpy array - Cartesian coordinates of the equivalent sources
       along the y-axis.

    output
    f: numpy array -  surface defined by function f(x,y).
    fx: numpy array - spatial derivative of the function f(x,y)
        with respect to the coordinate x.
    fy: numpy array - spatial derivative of the function f(x,y)
        with respect to the coordinate y.
    '''

    assert (x.size == y.size), 'x and y must have the same size'

    f1, fx1, fy1 = gaussian2D(x, y, 8000, 7000, 10000., 15000., angle=-30., amp=200., shift=-250.)
    f2, fx2, fy2 = gaussian2D(x, y, 8000, 5000, 25000., 25000., angle=-30., amp=200., shift=-250.)
    f = f1 + f2
    fx = fx1 + fx2
    fy = fy1 + fy2

    return f, fx, fy

def set_grid(path):
    '''
    Defines a regular 2D grid of Cartesian coordinates
    x and y from a pickle file containing located at path.

    input
    path: string - contains the path for the pickle file
          containing the pickle file with a previously
          defined grid.

    output
    xp: numpy arrays - Cartesian coordinates of the
        points located on the grid.
    yp: numpy arrays - Cartesian coordinates of the
        points located on the grid.
    shape: tuple - number of points along the x and y
           directions.
    '''

    with open('../data/regular_grid.pickle') as f:
        grid = pickle.load(f)

    #coordinates x and y of the data
    xp, yp = regular(grid['area'], grid['shape'])

    return xp, yp, grid['shape'], grid['area']

def p_mag_by_Fourier (x, y, data, shape, inc, dec, sinc, sdec, zc, z0):
    assert zc > z0, 'zc must be greater than z0'
    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = transform._pad_data(data, shape)
    kx, ky = transform._fftfreqs(x, y, shape, padded.shape)
    kz = np.sqrt(kx**2 + ky**2)
    fx, fy, fz = utils.ang2vec(1, inc, dec)
    mx, my, mz = utils.ang2vec(1, sinc, sdec)
    theta_f = fz + 1j*((fx*kx + fy*ky)/kz)
    theta_h = hz + 1j*((hx*kx + hy*ky)/kz)

    data_ft = np.fft.fft2(padded)

    p_ft = (np.exp(kz*(zc - z0))*data_ft)/(theta_f*theta_h*kz)

    p_pad = (2*np.pi/(PERM_FREE_SPACE*T2NT))*np.real(np.fft.ifft2(p_ft))
    # Remove padding from derivative
    p = p_pad[padx: padx + nx, pady: pady + ny]
    return p.ravel()
