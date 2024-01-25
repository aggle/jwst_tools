"""
This script computes the offsets needed for TA
"""
import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt

from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time

import pysiaf
from pysiaf import Siaf



###############################
########## USER INPUT #########
###############################

# Star positions - make sure to enter all values.
# The coordinates should be given *at the time of observation*
# By default, we expect the coordinates to be given in RA, Dec system,
# in units of degrees, and in the ICRS frame.
# Users who are comfortable with astropy.coordiantes.SkyCoord may set them
# as they like.

# The "slew_to" variable stores the position of the final target of the observations
slew_to = SkyCoord( # B component
    272.81221024, 69.24905315,
    unit='deg',
    frame='icrs',
    )
# The "slew_from" variable stores the position of the star that is used for TA.
slew_from = SkyCoord( # A component
    272.81369648, 69.25014279,
    unit='deg',
    frame='icrs',
    )
# Telescope V3PA
# enter the PA angle of the *telescope* V3 axis, at the time of the observation
v3 = 320.074

# Choose a coronagraph by uncommenting one of these choices
coron_id = [
    # '1065',
    # '1140',
    '1550',
    # 'LYOT',
]

# Plotting - set to False if you don't want to show plots
show_plots = True

###############################
####### END USER INPUT ########
###############################
# Script takes over from here #
###############################

coron_id = coron_id[0]
star_positions = {
    # the TA star
    'TA': slew_from,
    # The star you will eventually slew to
    'Target': slew_to,
    'v3': v3
}

# Offsets
sep = star_positions['TA'].separation(star_positions['Target']).to(units.arcsec)
pa = star_positions['TA'].position_angle(star_positions['Target']).to(units.deg)
print("Separation and PA: ", f"{sep.mas:0.2f} mas, {pa.degree:0.2f} deg")


# Siaf
miri = Siaf("MIRI")
# now that we have the MIRI object, let's get the 1550 coronagraph apertures used in 1618.
# There are two relevant apertures: MIRIM_MASK1550, which is the entire subarray, and MIRIM_CORON1550, which is just the portion that gets illuminated
mask = miri[f'MIRIM_MASK{coron_id}']
coro = miri[f'MIRIM_CORON{coron_id}']
# we also want the upper right (UR) and central upper right (CUR) target acquisition apertures
ta_ur = miri[f'MIRIM_TA{coron_id}_UR']
ta_cur = miri[f'MIRIM_TA{coron_id}_CUR']

def sky_to_idl(ta_pos, targ_pos, aper, pa):
    """
    Convert RA and Dec positions of a TA star and its target with an offset
    into a detector position (measured from the reference point, in arcsec)
    for a given PA of the V3 axis w.r.t. North.
    Assume the TA star is centered on the aperture (idl = (0, 0))

    Parameters
    ----------
    ta_pos : SkyCoord object of the TA star position
    targ_pos : SkyCoord object of the target star position
    aper : SIAF object for the aperture being used (e.g. MIRIM_CORON1550)
    pa : the PA in degrees of the V3 axis of the telescope (measured eastward of North) at the observation epoch

    Output
    ------
    idl_coords : dict {'ta': tuple, 'targ': tuple}
      a dictionary of IDL x, y coordinates for the TA star and the science target
      the TA star coordinates should be very close to 0
    """
    v2, v3 = aper.reference_point('tel')
    # compute the attitude matrix when we're pointing directly at the TA target
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=ta_pos.ra.deg, 
                                                    dec=ta_pos.dec.deg, 
                                                    pa=pa)
    aper.set_attitude_matrix(attmat)
    idl_coords = {}
    # ta star - should be close to 0
    idl_coords['ta'] = np.array(aper.sky_to_idl(ta_pos.ra.deg, ta_pos.dec.deg))
    # eps Mus
    idl_coords['targ'] = np.array(aper.sky_to_idl(targ_pos.ra.deg, targ_pos.dec.deg))
    return idl_coords



# Plotting helpers\
import matplotlib.ticker as ticker

# let's combine all the SIAF objects in a dict for convenience
all_apers = {}
all_apers['UR'] = ta_ur
all_apers['CUR'] = ta_cur
all_apers['coro'] = coro
all_apers['mask'] = mask


def plot_apers(ax, attmat, format_dict):
    """
    Helper function to plot the apertures for a given part of the TA sequence
    ax : axis to plot on
    attmat : attitude matrix
    format_dict : aperture plot formatting parameters
    """
    for k, aper in all_apers.items():
        aper.set_attitude_matrix(attmat)
        formatting = format_dict.copy()
        if k == 'mask':
            formatting['mark_ref'] = True
        if k == 'coro':
            # skip the illuminated region aperture, it's too crowded
            pass
        else:
            aper.plot(ax=ax, label=False, frame='sky', fill=False, **formatting)


def make_plots() -> None:
    """
    Generate the plots

    Parameters
    ----------
    None

    Output
    ------
    Displays 4 plots

    """
    colors = mpl.cm.plasma(np.linspace(0.2, 0.9, 4))

    # plot 1 : POV of the detector
    fig, ax = plt.subplots(1, 1, )
    ax.set_title("""Positions *before* offset slew""")
    frame = 'idl' # options are: tel (telescope), det (detector), sci (aperture)
    mask.plot(ax=ax, label=False, frame=frame, c='C0')
    coro.plot(ax=ax, label=False, frame=frame, c='C1', mark_ref=True)

    ax.scatter(0, 0, c='k', label='TA', marker='x', s=100)
    ax.scatter(*idl_coords['targ'], label="Target", marker="*", c='k')

    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, ls='--', c='grey', alpha=0.5)

    # Select your TA quadrant. 
    # Options are: UR (upper right, Q1), UL (upper left, Q2), LL (lower left, Q3) or LR (lower right, Q4).
    # A preceding "C" (CUL, CUL, CLL, CLR) indicates the inner TA region.
    ta_apers = {"UR": ta_ur,
                "CUR": ta_cur,
               }
    # use these apertures to compute the relative positions for every TA option.
    # really though it should be the same relative offset everywhere on the detector
    ta_sequence = {}
    for aper_id in ['UR', 'CUR']:
        aper = all_apers[aper_id]
        ta_sequence[aper_id] = sky_to_idl(star_positions['TA'], 
                                          star_positions['Target'], 
                                          aper, 
                                          star_positions['v3'])

    # Plot 2: The TA sequence on the detector
    nrows = 1
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*ncols, 4*nrows), sharex=True, sharey=True)
    fig.suptitle(f"TA sequence, as seen by the detector")

    # plot the SIAF apertures on every plot
    for ax in axes:
        mask.plot(ax=ax, label=False, frame='det', fill=False, c='C0')
        ax.plot([], [], c='C0', label='Readout')
        coro.plot(ax=ax, label=False, frame='det', mark_ref=True, fill=False, c='C1')
        ax.plot([], [], c='C1', label='Illuminated')
        for aper in ta_apers.values():
            aper.plot(ax=ax, label=False, frame='det', mark_ref=True, fill=False, c='C2')
        ax.plot([], [], c='C2', label='TA regions')

    # plot the positions of the stars at each step in the TA sequence

    # Outer TA
    ax = axes[0]
    ta_aper_id = "UR"
    ax.set_title("Step 1\n" + f"{ta_aper_id} TA region")

    # use the TA aperture object to convert coordinates
    ta_aper = ta_apers[ta_aper_id]
    ta_pos = ta_aper.idl_to_det(*ta_sequence[ta_aper_id]['ta'])
    targ_pos = ta_aper.idl_to_det(*ta_sequence[ta_aper_id]['targ'])

    ax.scatter(*ta_pos, 
               c='k', label='TA star', marker='x', s=100)
    ax.scatter(*targ_pos,
               c='k', label='Target', marker='*', s=100)

    # put the legend on this plot
    ax.legend(loc='best', ncol=1, fontsize='small', markerscale=0.7)

    # Inner TA
    ax = axes[1]
    ta_aper_id = 'CUR'
    ax.set_title("Step 2\n" + f"{ta_aper_id} TA region")
    # use the TA aperture object to convert coordinates
    ta_aper = ta_apers[ta_aper_id]
    ta_aper.plot(ax=ax, label=False, frame='det', mark_ref=True, fill=False, c='C2')

    ta_pos = ta_aper.idl_to_det(*ta_sequence[ta_aper_id]['ta'])
    targ_pos = ta_aper.idl_to_det(*ta_sequence[ta_aper_id]['targ'])

    ax.scatter(*ta_pos, 
               c='k', label='TA star', marker='x', s=100)
    ax.scatter(*targ_pos, 
               c='k', label='Target', marker='*', s=100)

    # TA star centered
    ax = axes[2]
    ax.set_title("Step 3\n" + "TA star centered")
    # plot the final TA before the offset is applied
    ax.scatter(*coro.idl_to_det(*idl_coords['ta']), c='k', label='TA star', marker='x', s=100)
    ax.scatter(*coro.idl_to_det(*idl_coords['targ']), c='k', label='Target', marker='*', s=100)

    # Offset applied
    ax = axes[3]
    ax.set_title("Step 4\n" + "Offset applied")
    # apply the offset to the position
    ta_pos  = coro.idl_to_det(*np.array(ta_sequence['UR']['ta']) + offset)
    targ_pos = coro.idl_to_det(*np.array(ta_sequence['UR']['targ']) + offset)
    ax.scatter(*ta_pos, 
               c='k', label='TA star', marker='x', s=100)
    ax.scatter(*targ_pos, 
               c='k', label='Target', marker='*', s=100)


    for ax in axes:
        # plot customizations
        ax.label_outer()
        ax.set_aspect('equal')
        ax.grid(True, ls='--', c='grey', alpha=0.5)

    fig.tight_layout()  



    # Plot 3: The TA sequence in RA and Dec, split into separate plots
    nrows = 4
    ncols = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             #figsize=(6*ncols, 10*nrows),
                             sharex=True, sharey=True)

    for ax in axes.ravel():
        ta_pos = (star_positions['TA'].ra.deg, star_positions['TA'].dec.deg)
        targ_pos = (star_positions['Target'].ra.deg, star_positions['Target'].dec.deg)    
        ax.scatter(*ta_pos,
                   c='k', label='TA', marker='x', s=100)
        ax.scatter(*targ_pos,
                   c='k', label='Target', marker='*', s=100)    



    # We start TA in the outer TA region
    ax = axes[0]
    ax.set_title(f"Step 1: UR TA region")

    aper = all_apers['UR']

    # the telescope is now pointing the *outer* TA region at the TA star
    v2, v3 = aper.reference_point('tel')
    # compute the attitude matrix when we're pointing directly at the TA target
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=star_positions['TA'].ra.deg, 
                                                    dec=star_positions['TA'].dec.deg, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[0], alpha=1, ls='-')
    plot_apers(ax, attmat, formatting)
    #plot_apers(ax, all_apers, attmat, {'c': colors[0]})



    # Continue to step 2 of TA, in the inner TA region
    ax = axes[1]

    ax.set_title(f"Step 2: CUR TA region")

    aper = all_apers['CUR']

    # the telescope is now pointing the *inner* TA region at the TA star
    v2, v3 = aper.reference_point('tel')
    # compute the attitude matrix when we're pointing the inner TA region at the TA target
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=star_positions['TA'].ra.deg, 
                                                    dec=star_positions['TA'].dec.deg, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[1], alpha=1, ls='-')
    plot_apers(ax, attmat, formatting)



    # plot the final TA before the offset is applied
    ax = axes[2]
    ax.set_title("Step 3: Centered")

    # the telescope is now pointing the center of the coronagraph at the TA star
    v2, v3 = coro.reference_point('tel')
    # compute the attitude matrix when we're pointing directly at the TA target
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=star_positions['TA'].ra.deg, 
                                                    dec=star_positions['TA'].dec.deg, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[2], alpha=1, ls='-')
    plot_apers(ax, attmat, formatting)



    ax = axes[3]
    ax.set_title("Step 4: Offset applied")
    # the telescope now slews to the Target
    v2, v3 = coro.reference_point('tel')
    # compute the RA and Dec of the pointing using the offset values you found earlier
    # # note that you must CHANGE THE SIGN OF THE SLEW with respect to the previous plot

    ra, dec = coro.idl_to_sky(*(-offset))

    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=ra, 
                                                    dec=dec, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[3], alpha=1, ls='-')
    plot_apers(ax, attmat, formatting)



    for ax in axes:
        # plot customizations
        ax.set_ylabel("Dec [deg]")
        ax.set_xlabel("RA [deg]")
        ax.label_outer()
        ax.set_aspect("equal") 
        ax.grid(True, ls='--', c='grey', alpha=0.5)    
        # fix x-axis labels
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

    fig.tight_layout()

    # Plot 4: The TA sequence in RA and Dec on a single plot
    nrows = 1
    ncols = 1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5))
    fig.suptitle(f"TA sequence, in RA/Dec")
    colors = mpl.cm.plasma(np.linspace(0.2, 0.9, 4))

    ta_pos = (star_positions['TA'].ra.deg, star_positions['TA'].dec.deg)
    targ_pos = (star_positions['Target'].ra.deg, star_positions['Target'].dec.deg)    
    ax.scatter(*ta_pos,
               c='k', label='TA', marker='x', s=100)
    ax.scatter(*targ_pos,
               c='k', label='Target', marker='*', s=100)    

    # We start TA in the outer TA region
    aper = all_apers["UR"]
    # the telescope is now pointing the *outer* TA region at the TA star
    v2, v3 = aper.reference_point('tel')
    # compute the attitude matrix when we're pointing directly at the TA target
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=star_positions['TA'].ra.deg, 
                                                    dec=star_positions['TA'].dec.deg, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[0], alpha=1, ls='dotted')
    plot_apers(ax, attmat, formatting)
    ax.plot([], [], 
            **formatting,
            label='Step 1: Outer TA step')


    # Continue to step 2 of TA, in the inner TA region
    aper = all_apers["CUR"]
    # the telescope is now pointing the *outer* TA region at the TA star
    v2, v3 = aper.reference_point('tel')
    # compute the attitude matrix when we're pointing directly at the TA target
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=star_positions['TA'].ra.deg, 
                                                    dec=star_positions['TA'].dec.deg, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[1], alpha=1, ls='dashdot')
    plot_apers(ax, attmat, formatting)
    ax.plot([], [], 
            **formatting,
            label='Step 2: Inner TA step')    

    # plot the final TA before the offset is applied

    # the telescope is now pointing the center of the coronagraph at the TA star
    v2, v3 = coro.reference_point('tel')
    # compute the attitude matrix when we're pointing directly at the TA target
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=star_positions['TA'].ra.deg, 
                                                    dec=star_positions['TA'].dec.deg, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[2], alpha=1, ls='dashed')
    plot_apers(ax, attmat, formatting)
    ax.plot([], [], 
            **formatting, 
            label='Step 3: Before offset')    


    # ax = axes[3]
    # the telescope now places the TA star at the commanded offset
    v2, v3 = coro.reference_point('tel')
    # note that you must CHANGE THE SIGN OF THE OFFSET to get the position of the reference point
    ra, dec = coro.idl_to_sky(*(-offset))
    attmat = pysiaf.utils.rotations.attitude_matrix(v2, v3, 
                                                    ra=ra, 
                                                    dec=dec, 
                                                    pa=star_positions['v3'])
    formatting = dict(c=colors[3], alpha=1, ls='solid')    
    plot_apers(ax, attmat, formatting)
    ax.plot([], [],
            **formatting,
            label='Step 4: After offset')    


    # plot formatting
    ax.set_ylabel("Dec [deg]")
    ax.set_xlabel("RA [deg]")
    ax.set_aspect("equal") 
    ax.grid(True, ls='--', c='grey', alpha=0.5)    
    # fix x-axis labels
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    fig.tight_layout()
    ax.legend()

    plt.show()


if __name__ == "__main__":
    idl_coords = sky_to_idl(star_positions['TA'], 
                            star_positions['Target'],
                            coro,
                            star_positions['v3'])


    print("After TA but before slewing, the position of the TA star should be close to (0, 0):")
    print(f"\t", ', '.join(f"{i:+0.3e}" for i in idl_coords['ta']), "arcsec")
    print("... and the position of the Target star is:")
    print(f"\t", ', '.join(f"{i:+0.3e}" for i in idl_coords['targ']), "arcsec")
    print()



    offset = -1*np.array(idl_coords['targ'])

    print("When the TA star is centered, the Target star is at:")
    print(f"\tdX: {idl_coords['targ'][0]:+2.3f} arcsec")
    print(f"\tdY: {idl_coords['targ'][1]:+2.3f} arcsec")
    print()

    print("Therefore, the commanded offsets that will move the coronagraph from the TA star to the Target are:")
    print(f"\tdX: {offset[0]:+2.3f} arcsec")
    print(f"\tdY: {offset[1]:+2.3f} arcsec")

    if show_plots == True:
        make_plots()
