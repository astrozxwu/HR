# %%
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from dustmaps.sfd import SFDQuery 
from dustmaps.bayestar import BayestarQuery
from scipy.interpolate import CubicSpline
bayestar = BayestarQuery(max_samples=2)
sfd = SFDQuery()

# %%
def filter(df):
    df = df[df['E_BP_RP_'] < 1.6]
    bprp = df['BP-RP']
    e = df['E_BP_RP_']
    data = np.array([bprp, e]).T
    slices = np.linspace(0.8, 3.0, 11)
    xys = []
    for i in range(len(slices)-1):
        tmp = data[[t < slices[i+1] and t > slices[i] for t in data.T[0]]].T
        xys.append([np.median(tmp[0]), np.median(tmp[1])])
    xys = np.array(xys).T
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(xys[0], xys[1])
    df = df[~df['E_BP_RP_'].mask]
    flt = [np.abs(i['E_BP_RP_']-cs(i['BP-RP'])) < 0.02 for i in df]
    df = df[flt]
    return df


def h2d(string):
    ra, dec = string.split()
    sgn = 1
    if '-' in dec:
        sgn = -1
    dec = dec.replace('-', '')
    a, b, c = [float(i) for i in ra.split(':')]
    d, e, f = [float(i) for i in dec.split(':')]

    alpha = (a + b/60 + c/3600)*15
    delta = sgn*(d + e/60 + f/3600)
    return alpha, delta


def corrPlot_HR_diagram(ra_dec,ra=None,dec=None,figtype='HR'):
    if ra_dec:
        ra, dec = h2d(ra_dec)
    src = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    kwd = ['+_r', 'RA_ICRS', 'DE_ICRS', 'Source', 'Dist', 'b_Dist', 'B_Dist', 'BP-RP', 'Gmag',
           'e_Gmag', 'e_BPmag', 'Plx', 'e_Plx','RPlx', 'e_RPmag', 'RUWE', 'epsi', 'sepsi',
            'E(BP/RP)', 'AG','RFG','RFBP','RFRP','Nper','NgAL','chi2AL']
    v = Vizier(columns=kwd)
    v.ROW_LIMIT = -1
    result = v.query_region(src, radius="15m", catalog=["I/355/gaiadr3"])
    tab = result[0]
#   filtering
    tab = tab[np.logical_and(tab['RPlx'] > 10,tab['Nper'] > 8)]
    #tab = tab[tab['RUWE'] < 1.4]
    tab = tab[tab['NgAL'] > 5]
#    tab = tab[tab['chi2AL']/(tab['NgAL']-5) < 1.44*np.array([max([1,i]) for i in np.exp(-0.4*(tab['Gmag']-19.5))])]
#    tab = tab[np.logical_and(tab['RFG'] > 50,tab['RFBP'] > 50,tab['RFRP'] > 20)]
    tab = tab[1.0+0.015*tab['BP-RP']**2 < tab['E_BP_RP_']]
    tab = tab[tab['BP-RP']< 1.3+0.06*tab['BP-RP']**2]

    tab = tab[np.logical_or(tab['epsi'] < 1, tab['sepsi'] < 2)]
    ascii.write(tab, 'temp.csv', format='csv', overwrite=True)

    from astroquery.xmatch import XMatch
    tab = XMatch.query(cat1=open('./temp.csv'), cat2='vizier:I/352/gedr3dis',
                       max_distance=1*u.arcsec, colRA1='RA_ICRS', colDec1='DE_ICRS')
    tab = tab[tab['Source_1'] == tab['Source_2']]

# filtering here
    #tab = tab[tab['B_rgeo']-tab['b_rgeo']<1/3*tab['rgeo']]
    #tab = filter(tab)
    plt.scatter(tab['BP-RP'], tab['E_BP_RP_'], s=1)
    plt.show()

    ra = tab['RA_ICRS_1']
    dec = tab['DE_ICRS_1']
    dist = tab['rgeo']
    coords = SkyCoord(ra*u.deg, dec*u.deg, distance=dist*u.pc, frame='icrs')
    ebv = bayestar(coords, mode='median')
    if np.isnan(ebv[0]) and np.isnan(ebv[-1]):
        print('3d failed , turn to sfd')
        ebv = sfd(coords)
    ebprp = 1.329*ebv
    R_G = 2.516
    BP_RP = tab['BP-RP'] - ebprp
    Gmag = tab['Gmag'] + 5*np.log10(10/tab['rgeo'])-ebv*R_G
    m_G = tab['Gmag']
    m_BP_RP = tab['BP-RP']

# plotter
    import matplotlib.pyplot as plt
    from matplotlib import colors
    fig, ax = plt.subplots(figsize=(6, 6))
#    h = ax.hist2d(BP_RP, Gmag, bins=300, cmin=3, norm=colors.PowerNorm(0.5), zorder=0.5)
    if figtype=='HR':
        ax.scatter(BP_RP, Gmag, s=1)
        ax.invert_yaxis()
    #    cb = fig.colorbar(h[3], ax=ax, pad=0.02)
        ax.set_xlabel(r'$G_{BP} - G_{RP}$')
        ax.set_ylabel(r'$M_G$')
    #    cb.set_label(r"$\mathrm{Stellar~density}$")
    elif figtype == 'CMB':
        ax.scatter(m_BP_RP, m_G, s=1)
        ax.invert_yaxis()
    #    cb = fig.colorbar(h[3], ax=ax, pad=0.02)
        ax.set_xlabel(r'$m_{G_{BP}} - m_{G_{RP}}$')
        ax.set_ylabel(r'$m_G$')
    plt.show()





# fill the rest with scatter (set rasterized=True if saving as vector graphics)

corrPlot_HR_diagram("22:09:49.38   58:56:43.12",figtype='HR')

# %%
