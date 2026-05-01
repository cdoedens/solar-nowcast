import pvlib
import numpy as np
import pandas as pd

def csi(ghi, time, lat, lon):
    '''
    Calculate the clear sky index (CSI). PVLib functions handle 1D data (e.g. dataframes) fine but do not work on higher dimension.

    Inputs
    ghi: observations of global horizontal irradiance
    time, lat, lon: for the respective ghi observations

    Output
    CSI: ratio between observed irradiance and irradiance during clear sky conditions (i.e. no cloud)
    '''

    # find the position of the sun, based off the time and coordinates
    solpos = pvlib.solarposition.get_solarposition(
        time,
        lat,
        lon,
    )

    # airmass that the solar irradiance passes through
    airmass_relative = pvlib.atmosphere.get_relative_airmass(
        solpos['apparent_zenith'].values
    )
    airmass_absolute = pvlib.atmosphere.get_absolute_airmass(
        airmass_relative,
    )
    
    linke_turbidity = np.maximum(2 + 0.1 * airmass_absolute, 2.5)
    
    doy = pd.to_datetime(time).dayofyear
    
    dni_extra = pvlib.irradiance.get_extra_radiation(doy)
    
    clear_sky_irradiance = pvlib.clearsky.ineichen(
        apparent_zenith=solpos['apparent_zenith'].values,
        airmass_absolute=airmass_absolute,
        linke_turbidity=linke_turbidity,
        dni_extra=dni_extra
    )

    clear_sky_ghi = clear_sky_irradiance['ghi']
    return ghi / clear_sky_ghi