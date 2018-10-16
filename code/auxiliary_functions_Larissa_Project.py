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
from fatiando.vis import mpl
from fatiando.gridder import regular
from fatiando.mesher import Prism
from fatiando.gravmag import transform
from fatiando.constants import PERM_FREE_SPACE, T2NT
from fatiando.constants import G, SI2MGAL,SI2EOTVOS


from scipy.stats import norm


# 
# Function to compute the sensitivity matrix of the Total Field anomaly caused by magnetized DIPOLE 
# x axis is north, z axis is down
#
# Input parameters:
# Observation point located at (xp,yp,zp). 
#
# Dipole centered at (xq,yq,zq)  
# Magnetization of dipole is defined by the direction cossines mx_s, my_s, mz_s
#
# The Geomagnetic Field is defined by the direction cossines: MX_Geomag, MY_Geomag, MZ_Geomag
#
# ATENTION:  THE dipole HAS: UNIT VOLUME AND UNIT  MAGNETIC MOMENT 
#
# The Units:
# of distance irrelevant but must be consistent. 
#
# Output parameters:
# tf = Total field anomaly in units of nT.
#
# Adapted from function Dipole of Blakely (1995)


def dipole_unit_moment(xp, yp, zp, xc, yc, zc, mx_s, my_s, mz_s, MX_Geomag, MY_Geomag, MZ_Geomag):
    #: Proportionality constant used in the magnetic method in henry/m (SI)
    CM = 10. ** (-7)
    #: Conversion factor from tesla to nanotesla
    T2NT = 10. ** (9)
    
    rx = xp - xc
    ry = yp - yc
    rz = zp - zc
    r2 = rx*rx + ry*ry + rz*rz
    r  = np.sqrt(r2)
    if np.any(r==0): print "Bad argument detected"
    r5 = r**5
    dot = rx*mx_s + ry*my_s + rz*mz_s
    volume = 1.0
    moment = 1.0
    # moment = 4.*np.pi*(a**3)*m/3.
    
    bx = T2NT*CM*moment*(3.*dot*rx-r2*mx_s)/r5
    by = T2NT*CM*moment*(3.*dot*ry-r2*my_s)/r5
    bz = T2NT*CM*moment*(3.*dot*rz-r2*mz_s)/r5
    
    tf = bx*MX_Geomag + by*MY_Geomag + bz*MZ_Geomag
    return tf


# Function DIPOLE computes the Total Field anomaly caused by a uniformly magnetized # sphere, x axis
# is north, z axis is down
#
# Input parameters:
# Observation point located at (xp,yp,zp). 
#
# Sphere centered at (xq,yq,zq) 
# Sphere with radius a. 
# Magnetization of sphere defined by:
# intensity m, inclination incs, and declination decs. 
#
# Magnetization direction of the Geomagnetic Field defined by:
# inclination inc, and declination dec. 
#
# 
# Units:
# of distance irrelevant but must be consistent. All angles
# in degrees. Intensity of magnetization in A/m. Requires
# function dircos_blakely.

# Output parameters:
# The three components of magnetic induction (bx,by,bz) in units of nT.
# Dipole Blakely 


def TF_sphere(xp, yp, zp, xc, yc, zc, a, incs, decs, m, inc, dec):
    #: Proportionality constant used in the magnetic method in henry/m (SI)
    CM = 10. ** (-7)
    #: Conversion factor from tesla to nanotesla
    T2NT = 10. ** (9)
    # The three direction cosines of the sphere (source)
    mx, my, mz = dircos_blakely(incs,decs,0.)
    rx = xp - xc
    ry = yp - yc
    rz = zp - zc
    r2 = rx**2 + ry**2 + rz**2
    r = np.sqrt(r2)
    if np.any(r==0): print "Bad argument detected"
    r5 = r**5
    dot = rx*mx + ry*my + rz*mz
    moment = 4.*np.pi*(a**3)*m/3.
    bx = T2NT*CM*moment*(3.*dot*rx-r2*mx)/r5
    by = T2NT*CM*moment*(3.*dot*ry-r2*my)/r5
    bz = T2NT*CM*moment*(3.*dot*rz-r2*mz)/r5
    
    MX_Geomag, MY_Geomag, MZ_Geomag = dircos_blakely(inc,dec,0.)
    tf = bx*MX_Geomag + by*MY_Geomag + bz*MZ_Geomag
    return tf


# DIRCOS computes direction cosines from inclination and declination,
#
# Input parameters:
# incl: inclination in degrees positive below horizontal,
# decl: declination in degrees positive east of true north,
# azim: azimuth of x axis in degrees positive east of north,
#
# Output parameters:
# mx,my,mz: the three direction cosines


def dircos_blakely(incl,decl,azim):
    d2rad = 0.017453293
    xincl=incl*d2rad
    xdecl=decl*d2rad
    xazim=azim*d2rad
    mx=np.cos(xincl)*np.cos(xdecl-xazim)
    my=np.cos(xincl)*np.sin(xdecl-xazim)
    mz=np.sin(xincl)
    return mx, my, mz


# gz component of a point of mass located at xk, yk, zk computed at the observation point xi, yi, zi
def AZ(xi,yi,zi,xk,yk,zk):
    raio = 1.0
    volume = 1.0 
    rz = zi-zk
    rx = xi-xk
    ry = yi-yk
    r  = np.sqrt(rx**2+ry**2+rz**2)
    r3=r**3
    gz = -volume*G*rz/r3
    Az = gz*SI2MGAL
    return Az


# FTG - second derivatives of a point of mass located at xk, yk, zk computed at the observation point xi, yi, zi

def A_DERIVES(xi,yi,zi,xk,yk,zk):
    volume = 1.0 
    rx = xi-xk
    ry = yi-yk
    rz = zi-zk
    r  = np.sqrt(rx**2+ry**2+rz**2)
    r3=r**3
    r5=r**5
    
    gxx = volume*G*( (-1./r3)  + 3.*(rx*rx/r5) )
    gxy = volume*G*( (3.*rx*ry)/r5 )
    gxz = volume*G*( (3.*rx*rz)/r5 )
    gyy = volume*G*( (-1./r3)  + 3.*(ry*ry/r5) )
    gyz = volume*G*( (3.*ry*rz)/r5 )
    gzz = volume*G*( (-1./r3)  + 3.*(rz*rz/r5) )
    
    AXX = gxx*SI2EOTVOS
    AXY = gxy*SI2EOTVOS
    AXZ = gxz*SI2EOTVOS
    AYY = gyy*SI2EOTVOS
    AYZ = gyz*SI2EOTVOS
    AZZ = gzz*SI2EOTVOS
    
    return AXX, AXY, AXZ, AYY, AYZ, AZZ

# First-order polynomial surface

def Polynomial_1th(xp,yp,coef):
    assert (xp.size == yp.size), 'xp and yp must have the same size '
    assert (len(coef) == 3), 'First-order polynomial must have 3 coefficients '
    
    Poly_1th = coef[0] + np.dot(xp, coef[1]) + np.dot(yp, coef[2]) 
    
    return Poly_1th

# Second-order polynomial surface
def Polynomial_2th(xp,yp,coef):
    assert (xp.size == yp.size), 'xp and yp must have the same size '
    assert (len(coef) == 6), 'Secon-order polynomial must have 6 coefficients '
    
    Poly_2th = coef[0] + np.dot(xp, coef[1]) + np.dot(yp, coef[2]) + np.dot(xp*xp, coef[3]) \
                       + np.dot(xp*yp, coef[4]) + np.dot(yp*yp, coef[5])
   
    return Poly_2th



def Plot_Onemap(x, y, data, shape, 
                prism_projection, projection_style, line_width,model, 
                figure_title, label_x, label_y, label_size,
                observations, point_style, point_size, unit):
    
    levels = mpl.contourf(y, x, data, shape, 20, interp= True)
    cbar = plt.colorbar()
    mpl.contour(y, x, data, shape, levels, clabel=False, interp=True)
    

    if observations is True:
        plt.plot(y,x, point_style, markersize = point_size)
    
    if unit is not None:
            cbar.set_label(unit, fontsize = label_size)

    ax = plt.gca() 
    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax.set_ylabel(label_y, fontsize = label_size)

    if figure_title is not None:
          ax.set_title(figure_title, fontsize = label_size)
    
    
    mpl.m2km()


def Plot_Twomaps(x, y, data1, data2, shape, 
                prism_projection, projection_style, line_width,model, 
                figure_title1, figure_title2, label_x, label_y, label_size,
                observations, point_style, point_size, unit):
    
    ax1=plt.subplot(1,2,1)
    ax1.text(0.01, 1.05, figure_title1, 
        horizontalalignment='left',
        verticalalignment='top',
        fontsize = label_size,
        transform = ax1.transAxes)
    
    levels = mpl.contourf(y, x, data1, shape, 20, interp= True)
    cbar = plt.colorbar()
    mpl.contour(y, x, data1, shape, levels, clabel=False, interp=True)
    

    if observations is True:
        plt.plot(y,x, point_style, markersize = point_size)
    
    if unit is not None:
            cbar.set_label(unit, fontsize = label_size)

    ax1 = plt.gca() 
    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax1.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax1.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax1.set_ylabel(label_y, fontsize = label_size)

      
    
    mpl.m2km()

    ax2=plt.subplot(1,2,2)
    ax2.text(0.01, 1.05, figure_title2, 
        horizontalalignment='left',
        verticalalignment='top',
        fontsize = label_size,
        transform = ax2.transAxes)
    
    levels = mpl.contourf(y, x, data2, shape, 20, interp= True)
    cbar = plt.colorbar()
    mpl.contour(y, x, data2, shape, levels, clabel=False, interp=True)
    

    if observations is True:
        plt.plot(y,x, point_style, markersize = point_size)
    
    if unit is not None:
            cbar.set_label(unit, fontsize = label_size)

    ax2 = plt.gca() 
    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax2.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax2.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax2.set_ylabel(label_y, fontsize = label_size)

      
    
    mpl.m2km()

######    

def Plot_Threemaps(x, y, data1, data2, data3, shape, 
                prism_projection, projection_style, line_width,model, 
                figure_title1, figure_title2, figure_title3, label_x, label_y, label_size,
                observations, point_style, point_size, unit):
    
    ax1=plt.subplot(1,3,1)
    ax1.text(0.01, 1.05, figure_title1, 
        horizontalalignment='left',
        verticalalignment='top',
        fontsize = label_size,
        transform = ax1.transAxes)
    
    levels = mpl.contourf(y, x, data1, shape, 20, interp= True)
    cbar = plt.colorbar()
    mpl.contour(y, x, data1, shape, levels, clabel=False, interp=True)
    

    if observations is True:
        plt.plot(y,x, point_style, markersize = point_size)
    
    if unit is not None:
            cbar.set_label(unit, fontsize = label_size)

    ax1 = plt.gca() 
    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax1.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax1.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax1.set_ylabel(label_y, fontsize = label_size)

      
    
    mpl.m2km()

    ax2=plt.subplot(1,3,2)
    ax2.text(0.01, 1.05, figure_title2, 
        horizontalalignment='left',
        verticalalignment='top',
        fontsize = label_size,
        transform = ax2.transAxes)
    
    levels = mpl.contourf(y, x, data2, shape, 20, interp= True)
    cbar = plt.colorbar()
    mpl.contour(y, x, data2, shape, levels, clabel=False, interp=True)
    

    if observations is True:
        plt.plot(y,x, point_style, markersize = point_size)
    
    if unit is not None:
            cbar.set_label(unit, fontsize = label_size)

    ax2 = plt.gca() 
    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax2.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax2.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax2.set_ylabel(label_y, fontsize = label_size)

      
    
    mpl.m2km()

    
    ax3=plt.subplot(1,3,3)
    ax3.text(0.01, 1.05, figure_title3, 
        horizontalalignment='left',
        verticalalignment='top',
        fontsize = label_size,
        transform = ax3.transAxes)
    
    levels = mpl.contourf(y, x, data3, shape, 20, interp= True)
    cbar = plt.colorbar()
    mpl.contour(y, x, data3, shape, levels, clabel=False, interp=True)
    

    if observations is True:
        plt.plot(y,x, point_style, markersize = point_size)
    
    if unit is not None:
            cbar.set_label(unit, fontsize = label_size)

    ax3 = plt.gca() 
    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax3.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax3.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax3.set_ylabel(label_y, fontsize = label_size)

      
    
    mpl.m2km()




######
    
def Plot_Onemap_Histog(x, y, data, shape, 
                prism_projection, projection_style, line_width,model, 
                figure_title1, figure_title2, label_x, label_y, label_size,
                observations, point_style, point_size, unit):
    
    ax1=plt.subplot(1,2,1)
    ax1.text(0.01, 1.10, figure_title1, 
        horizontalalignment='left',
        verticalalignment='top',
        fontsize = label_size,
        transform = ax1.transAxes)
    
    levels = mpl.contourf(y, x, data, shape, 20, interp= True)
    cbar = plt.colorbar()
    mpl.contour(y, x, data, shape, levels, clabel=False, interp=True)
    

    if observations is True:
        plt.plot(y,x, point_style, markersize = point_size)
    
    if unit is not None:
            cbar.set_label(unit, fontsize = label_size)

    ax1 = plt.gca() 
    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax1.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax1.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax1.set_ylabel(label_y, fontsize = label_size)

      
    
    mpl.m2km()

    ax2=plt.subplot(1,2,2)
    ax2.text(0.01, 1.10, figure_title2, 
        horizontalalignment='left',
        verticalalignment='top',
        fontsize = label_size,
        transform = ax2.transAxes)
    
  
    (mu, sigma) = norm.fit(data)
    # the histogram of the difference between the true and the estimated RTP
    n_h, bins, patches = plt.hist(data, 120, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y_histogram = mlab.normpdf(bins, mu, sigma)
    l_histogram = plt.plot(bins, y_histogram, 'r--', linewidth=2)
    if label_x is not None:
        ax2.set_xlabel('Residual', fontsize = label_size)

    if label_y is not None:
        ax2.set_ylabel('Probability', fontsize = label_size)
    
    if figure_title2 is not None:
        ax2.set_title(r'$ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma), fontsize = label_size)
    
    plt.grid(True) 
    
    mpl.m2km()

######    
        
    


def Plot_FTG(x, y, data_xx, data_xy, data_xz, data_yy, data_yz, data_zz, shape, 
                prism_projection, projection_style, line_width, model,
                label_x, label_y, label_size, unit):

    #   $G_{xx}$
    ax1=plt.subplot(3,3,1)
    ax1.text(0, 1.10,'(a) $G_{xx}$',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax1.transAxes)


    levels = mpl.contourf(y, x, data_xx, shape, 12, interp=True, cmap='Greys')
    cbar = plt.colorbar()
    mpl.contour(y, x, data_xx, shape, levels, clabel=False, interp=True)
    cbar.set_label(unit)

    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax1.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    if label_x is not None:
        ax1.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax1.set_ylabel(label_y, fontsize = label_size)

    
    mpl.m2km()

    #   $G_{xy}$

    ax2=plt.subplot(3,3,2)
    ax2.text(0, 1.10,'(b) $G_{xy}$',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax2.transAxes)


    levels = mpl.contourf(y, x, data_xy, shape,12, interp=True, cmap='Greys')
    cbar = plt.colorbar()
    mpl.contour(y, x, data_xy, shape, levels, clabel=False, interp=True)
    cbar.set_label(unit)

    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax2.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    
    mpl.m2km()


    #   $G_{xz}$

    ax3=plt.subplot(3,3,3)
    ax3.text(0, 1.10,'(c) $G_{xz}$',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax3.transAxes)


    levels = mpl.contourf(y, x, data_xz, shape, 12, interp=True, cmap='Greys')
    cbar = plt.colorbar()
    mpl.contour(y, x, data_xz, shape, levels, clabel=False, interp=True)
    cbar.set_label(unit)

    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax3.plot(xs_project , ys_project, projection_style, linewidth = line_width)
    mpl.m2km()


    #   $G_{yy}$
    ax4=plt.subplot(3,3,5)
    ax4.text(0, 1.10,'(d) $G_{yy}$',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax4.transAxes)


    levels = mpl.contourf(y, x, data_yy, shape, 12, interp=True, cmap='Greys')
    cbar = plt.colorbar()
    mpl.contour(y, x, data_yy, shape,levels, clabel=False, interp=True)
    cbar.set_label(unit)

    if prism_projection is True:
        for i, sq in enumerate(model):
            y1, y2, x1, x2 = sq
            xs_project  = [x1, x1, x2, x2, x1]
            ys_project  = [y1, y2, y2, y1, y1]
            ax4.plot(xs_project , ys_project, projection_style, linewidth = line_width)
            
    if label_x is not None:
        ax4.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax4.set_ylabel(label_y, fontsize = label_size)

    
    mpl.m2km()
    
    
    #   $G_{yz}$
    ax5=plt.subplot(3,3,6)
    ax5.text(0, 1.10,'(e) $G_{yz}$',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax5.transAxes)


    levels = mpl.contourf(y, x, data_yz, shape, 12, interp=True, cmap='Greys')
    cbar = plt.colorbar()
    mpl.contour(y, x, data_yz, shape, levels, clabel=False, interp=True)
    cbar.set_label(unit)

    for i, sq in enumerate(model):
        y1, y2, x1, x2 = sq
        xs_project = [x1, x1, x2, x2, x1]
        ys_project = [y1, y2, y2, y1, y1]
        ax5.plot(xs_project, ys_project, projection_style,linewidth = 1.0 )

    #plt.xlabel('Easting coordinate y(km)')
    #plt.ylabel('Northing coordinate x(km)')
    mpl.m2km()


    #   $G_{zz}$
    ax6=plt.subplot(3,3,9)
    ax6.text(0, 1.10,'(f) $G_{zz}$',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax6.transAxes)


    levels = mpl.contourf(y, x, data_zz, shape, 12, interp=True, cmap='Greys')
    cbar = plt.colorbar()
    mpl.contour(y, x, data_zz, shape, levels, clabel=False, interp=True)
    cbar.set_label(unit)

    for i, sq in enumerate(model):
        y1, y2, x1, x2 = sq
        xs_project = [x1, x1, x2, x2, x1]
        ys_project = [y1, y2, y2, y1, y1]
        projection_style = '-r'
        ax6.plot(xs_project, ys_project, projection_style,linewidth = 1.0 )

    if label_x is not None:
        ax6.set_xlabel(label_x, fontsize = label_size)

    if label_y is not None:
        ax6.set_ylabel(label_y, fontsize = label_size)

    
    mpl.m2km()


#######   
def Plot_FTG_Histog(xp, yp, res_gxx, res_gxy, res_gxz,res_gyy, res_gyz, res_gzz,
                label_size, unit):


    #   $G_{xx}$
    ax1=plt.subplot(3,3,1)
    ax1.text(0.05, 0.90,'(a) $G_{xx}$ Residual',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax1.transAxes)


    (mu_xx, sigma_xx) = norm.fit(res_gxx)
    # the histogram of the difference between the true and the estimated RTP
    n_h, bins, patches = plt.hist(res_gxx, 120, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y_histogram = mlab.normpdf(bins, mu_xx, sigma_xx)
    l_histogram = plt.plot(bins, y_histogram, 'r--', linewidth=2)
    plt.xlabel('Residual $G_{xx}$ (Eotvos)')
    plt.ylabel('Probability')
    plt.title(r'$ \mu_{xx}=%.3f,\ \sigma_{xx}=%.3f$' %(mu_xx, sigma_xx))
    plt.grid(True)

    #   $G_{xy}$

    ax2=plt.subplot(3,3,2)
    ax2.text(0.05, 0.90,'(b) $G_{xy}$ Residual',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax2.transAxes)


    (mu_xy, sigma_xy) = norm.fit(res_gxy)
    # the histogram of the difference between the true and the estimated RTP
    n_h, bins, patches = plt.hist(res_gxy, 120, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y_histogram = mlab.normpdf(bins, mu_xy, sigma_xy)
    l_histogram = plt.plot(bins, y_histogram, 'r--', linewidth=2)
    #plt.xlabel('Residual $G_{xy}$ (Eotvos)')
    plt.ylabel('Probability')
    plt.title(r'$ \mu_{xy}=%.3f,\ \sigma_{xy}=%.3f$' %(mu_xy, sigma_xy))
    plt.grid(True)

    #   $G_{xz}$

    ax3=plt.subplot(3,3,3)
    ax3.text(0.05, 0.90,'(c) $G_{xz}$ Residual',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax3.transAxes)


    (mu_xz, sigma_xz) = norm.fit(res_gxz)
    # the histogram of the difference between the true and the estimated RTP
    n_h, bins, patches = plt.hist(res_gxz, 120, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y_histogram = mlab.normpdf(bins, mu_xz, sigma_xz)
    l_histogram = plt.plot(bins, y_histogram, 'r--', linewidth=2)
    #plt.xlabel('Residual $G_xz$ (Eotvos)')
    plt.ylabel('Probability')
    plt.title(r'$ \mu_{xz}=%.3f,\ \sigma_{xz}=%.3f$' %(mu_xz, sigma_xz))
    plt.grid(True)

    #   $G_{yy}$
    ax4=plt.subplot(3,3,5)
    ax4.text(0.05, 0.90,'(d) $G_{yy}$ Residual',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax4.transAxes)


    (mu_yy, sigma_yy) = norm.fit(res_gyy)
    # the histogram of the difference between the true and the estimated RTP
    n_h, bins, patches = plt.hist(res_gyy, 120, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y_histogram = mlab.normpdf(bins, mu_yy, sigma_yy)
    l_histogram = plt.plot(bins, y_histogram, 'r--', linewidth=2)
    plt.xlabel('Residual $G_{yy}$ (Eotvos)')
    plt.ylabel('Probability')
    plt.title(r'$ \mu_{yy}=%.3f,\ \sigma_{yy}=%.3f$' %(mu_yy, sigma_yy))
    plt.grid(True)


    #   $G_{yz}$
    ax5=plt.subplot(3,3,6)
    ax5.text(0.05, 0.90,'(e) $G_{yz}$ Residual',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax5.transAxes)


    (mu_yz, sigma_yz) = norm.fit(res_gyz)
    # the histogram of the difference between the true and the estimated RTP
    n_h, bins, patches = plt.hist(res_gyz, 120, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y_histogram = mlab.normpdf(bins, mu_yz, sigma_yz)
    l_histogram = plt.plot(bins, y_histogram, 'r--', linewidth=2)
    #plt.xlabel('Residual   $G_xy$ (Eotvos))')
    plt.ylabel('Probability')
    plt.title(r'$ \mu_{yz}=%.3f,\ \sigma_{yz}=%.3f$' %(mu_yz, sigma_yz))
    plt.grid(True)


    #   $G_{zz}$
    ax6=plt.subplot(3,3,9)
    ax6.text(0.05, 0.90,'(f) $G_{zz}$ Residual',
         horizontalalignment='left',
         verticalalignment='top',
         transform = ax6.transAxes)


    (mu_zz, sigma_zz) = norm.fit(res_gzz)
    # the histogram of the difference between the true and the estimated RTP
    n_h, bins, patches = plt.hist(res_gzz, 120, normed=1, facecolor='blue', alpha=0.75)
    # add a 'best fit' line
    y_histogram = mlab.normpdf(bins, mu_zz, sigma_zz)
    l_histogram = plt.plot(bins, y_histogram, 'r--', linewidth=2)
    plt.xlabel('Residual  $G_(zz}$ (Eotvos)')
    plt.ylabel('Probability')
    plt.title(r'$ \mu_{zz}=%.3f,\ \sigma_{zz}=%.3f$' %(mu_zz, sigma_zz))
    plt.grid(True)
    
    return mu_xx,sigma_xx,mu_xy,sigma_xy,mu_xz,sigma_xz,mu_yy,sigma_yy,mu_yz,sigma_yz,mu_zz,sigma_zz

    #######   


######
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

print 'Executed auxiliary_functions_Larissa_Project.py'
