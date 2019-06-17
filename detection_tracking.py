__author__ = 'Christoph Morgan'
__copyright__ = '2019'
__license__ = 'MIT'
__version__ = 0.4
__maintainer__ = 'Christoph Morgan'
__email__ = 'christoph.morgan@gmail.com'
__status__ = 'prototype'

# IMPORT REQUIREMENTS
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from scipy.ndimage import mean
from scipy.ndimage import minimum_filter
from metpy.calc import vorticity
import datetime as dt
import math
import string
import random
from natsort import natsorted

#YAML IMPORT
import yaml
with open('main.yaml') as main:
    init = open(yaml.full_load(main)['main'])
    config_main, config_region, config_intensity, config_rules, config_filenames, config_var = yaml.full_load_all(
        init)


def deg_rounder(number, degs):
    '''
    Takes a number and rounds it to the nearest values defined in degs.
    i.e. degs = 0.25, number = 3.15 --> rounds to --> 3.25

    Parameters
    ----------
    number : float
        The number that should be rounded.
    degs : float
        The nearest value to which the number should be rounded to

    Returns
    -------
    float
        The given number, rounded to the nearest deg.
    '''
    x = 1 / degs
    return round(number * x) / x


def _convert_coords(region, coord, mode):
    '''
    Take a tuple of local/global coordinates and returns the global/regional coordinates for it.

    Parameters
    ----------
    region : __main__.Region
        The region the local coordinates are located in.
    coord : tuple
        Coordinate tuple in the form of (latitude, longitude)
    mode : string
        Will accept either 'to_global' or 'to_regional' as input.
    '''
    if mode == 'to_global':
        glo_lats = region.reg_lats[coord[0]]
        glo_lons = region.reg_lons[coord[1]]

        return (glo_lats, glo_lons)

    elif mode == 'to_regional':
        reg_lats = np.where(coord[0] == region.reg_lats)[0][0]
        reg_lons = np.where(coord[1] == region.reg_lons)[0][0]

        return (reg_lats, reg_lons)


class RegionClass():
    def __init__(self, name, path, degs, bbox, ts, config_intensity):
        '''
        Required parameters for creating a Region class object. Parameters are loaded from
        individual modules located in the region subdirectory.

        Parameters
        ----------
        name : string
            The name of the class object.
        path : string
            Absolute path to the directory where the netCDF files for the selected mode are stored
        degs : float
            Spatial resolution of the datasets in the unit of earth degrees. Applies to both
            analysis and ensemble datasets.
        bbox : list
            List of floats that determines the bounding box extend of the class object.
            Format should be [West, South, East, North]
        ts : integer
            Length of the time step for the temporal resolution for the selected mode datasets.
            Unit should be seconds.

        Returns
        -------
        class object
            Returns a RegionClass class object with the respective class attributes.
        '''

        self.name = name
        self.path = path
        self.degs = degs
        self.bbox = [deg_rounder(coord, self.degs) for coord in bbox]
        self.ts = ts

        self.glo_lats = np.linspace(90, -90, int(180 / self.degs) + 1)
        self.glo_lons = np.linspace(-180, 180 - self.degs,
                                    int(360 / self.degs))
        self.reg_lats = np.arange(self.bbox[3], self.bbox[1] - self.degs,
                                  -self.degs)
        self.reg_lons = np.arange(self.bbox[0], self.bbox[2] + self.degs,
                                  self.degs)

        def intensity_conversion(value, unit):
            if unit == 'ms':
                pass
            elif unit == 'kts':
                value = value * 0.514444
            elif unit == 'kmh':
                value = value * 0.277778
            elif unit == 'mph':
                value = value * 0.447040
            else:
                raise ValueError(
                    'The provided unit does not match any of the units in our catalogue. \
                \nPlease use either "ms", "kts", "kmh" or "mph"')

            return value

        self.scale_names = config_intensity['INTENSITY_NAMES']
        self.scale_values = [
            intensity_conversion(value, config_intensity['INTENSITY_UNIT'])
            for value in config_intensity['INTENSITY_VALUES']
        ]

        self.rules = Rules(self, config_rules)

        def __repr__(self):
            return
            f'RegionClass:'
            f'nName: {self.name}'
            f'nData Path: {self.path}'
            f'nSpatial Resolution: {self.degs}\u00b0'
            f'nTemporal Resolution: {self.ts} seconds'
            f'\n\n#Regional Boundaries'
            f'\nWest: \t{self.bbox[0]}\u00b0  E'
            f'\nEast: \t{self.bbox[2]}\u00b0 E'
            f'\nSouth: \t{self.bbox[1]}\u00b0 N'
            f'\nNorth: \t{self.bbox[3]}\u00b0 N'
            f'\n\n#Intensity Scala'
            f'\nIntensity Names: {self.scale_names}'
            f'\nIntensity Lower Bounds: {self.scale_values}'

    def createTracking(self):
        self.Tracking = Tracking(self, self.path, config_main['MODE'],
                                 config_main['START'], config_main['STOP'],
                                 config_main['DAYS'])


class Rules:
    def __init__(self, region, config_rules):
        self.region = region

        def _reset_default_rules():

            degs = self.region.degs
            ts_hours = self.region.ts / 3600

            self.pressure_min = 1015 * 100
            self.pressure_neighbourhood = 7.5 / degs
            self.vmax_radius = 2 / degs
            self.vmax_thresh = 16.5
            self.vort_radius = 2 / degs
            self.vort_thresh = 0.00001
            self.core_inner = 1 / degs
            self.core_outer = 3.5 / degs
            self.core_rule = None
            self.duration = 24 / ts_hours
            self.cyclosis = self.duration * 2
            self.update = (ts_hours / 1.5) / degs
            self.exclude_extratropical = True

        def _custom_rules(config_rules):

            self.pressure_min = config_rules['pressure_min']
            self.pressure_neighbourhood = config_rules[
                'pressure_neighbourhood']
            self.vmax_radius = config_rules['vmax_radius']
            self.vmax_thresh = config_rules['vmax_thresh']
            self.vort_radius = config_rules['vort_radius']
            self.vort_thresh = config_rules['vort_thresh']
            self.core_inner = config_rules['core_inner']
            self.core_outer = config_rules['core_outer']
            self.core_rule = config_rules['core_rule']
            self.duration_analysis = config_rules['duration_analysis']
            self.duration_ensemble = config_rules['duration_ensemble']
            self.cyclosis_analysis = config_rules['cyclosis_analysis']
            self.cyclosis_ensemble = config_rules['cyclosis_ensemble']
            self.update_analysis = config_rules['update_analysis']
            self.update_ensemble = config_rules['update_ensemble']

            self.exclude_extratropical = config_rules['exclude_extratropical']

        if config_rules['RULES_DEFAULT'] == True:
            _reset_default_rules()
        else:
            _custom_rules(config_rules)


class Tracking:
    def __init__(self, region, path, mode, start, stop, days):
        '''
        Creates an Analysis class for using the algorithm with Analysis
        type data inputs. Accepts a start and stop datetime to set the temporal
        boundaries. These string datetimes are parsed to Pandas datetime and then to
        UNIX epochtime.

        Parameters
        ----------
        region : class object
            Automatically parsed from the self in Region.Analysis function.
        start : string (YYYY-MM-DD hh:mm:ss)
            The datetime corresponding to the start of the analysis period
        stop : string (YYYY-MM-DD hh:mm:ss)
            The datetime corresponding to the end of the analysis period

        Raises
        ------
        Exception
            Raises an exception if the start and stop datetimes are not fully divisible
            with the analysis timestep. Future iterations may automatically round this.
        '''

        self._region = region
        self.path = path
        self.mode = mode
        self.start = dt.datetime.strptime(start, '%Y-%m-%d %H:%M')
        self.days = days
        self.ensembles = np.arange(0, 50, 1)
        if mode == 'analysis':
            self.stop = dt.datetime.strptime(
                stop, '%Y-%m-%d %H:%M') + dt.timedelta(seconds=self._region.ts)
        else:
            self.stop = self.start + dt.timedelta(days=self.days)

        #Parse datetime to epochtime
        self._start_epochtime \
                = int(self.start.replace(tzinfo=dt.timezone.utc).timestamp())
        self._stop_epochtime \
                = int(self.stop.replace(tzinfo=dt.timezone.utc).timestamp())

        #Parse datetime to string (for use in ensemble file finding)
        self._start_stringtime = self.start.strftime("%Y%m%dT%H")

        #Check for Exceptions:
        if  self._start_epochtime % self._region.ts != 0 or \
            self._stop_epochtime % self._region.ts != 0:
            raise Exception(
                'Start and Stop datetime must be fully divisible with the timestep'
            )

        #create list of times
        self.times = np.arange(self._start_epochtime, self._stop_epochtime,
                               self._region.ts)

        #AUTOMATIC PARAMETER INITILIASATION AND TC detection
        self._initialise_parameters()

    def __repr__(self):
        return f'Region: {self._region} \
        \nStart: {self.start} \
        \nEnd: {self.stop} \
        \nLength: {self.times.shape[0]} \tTimeSteps: {self._region.ts / 3600}h '

    def _initialise_parameters(self):
        '''
        A series of functions that first creates Python slices for 'cutting' away the relevant
        data from the netCDF arrays. In a second step, these data packages are assigned to Parameter
        class objects and linked to the Tracking class.
        Returns
        -------
        class objects
            A series of Paramter class objects with the data from netCDF arrays
        '''
        if self.mode == 'analysis':
            self._dataset = {
                'PRMSL': Dataset(self.path + config_filenames['PSL'], 'r'),
                'VGRD_10M': Dataset(self.path + config_filenames['V10M'], 'r'),
                'UGRD_10M': Dataset(self.path + config_filenames['U10M'], 'r'),
                'TMP': Dataset(self.path + config_filenames['TMP'], 'r'),
                'U850': Dataset(self.path + config_filenames['U850'], 'r'),
                'V850': Dataset(self.path + config_filenames['V850'], 'r')
            }
        if self.mode == 'ensemble':
            self._dataset = {
                'PRMSL':
                Dataset(
                    self.path + self._start_stringtime +
                    config_filenames['PSL'], 'r'),
                'VGRD_10M':
                Dataset(
                    self.path + self._start_stringtime +
                    config_filenames['V10M'], 'r'),
                'UGRD_10M':
                Dataset(
                    self.path + self._start_stringtime +
                    config_filenames['U10M'], 'r'),
                'TMP':
                Dataset(
                    self.path + self._start_stringtime +
                    config_filenames['TMP'], 'r'),
                'U850':
                Dataset(
                    self.path + self._start_stringtime +
                    config_filenames['U850'], 'r'),
                'V850':
                Dataset(
                    self.path + self._start_stringtime +
                    config_filenames['V850'], 'r')
            }

        def boundary_slice():
            '''
            Creates two slices from the defined bounding box for the region. Slices are for
            the extents in longitudinal and lateral direction.

            Returns
            -------
            slice
                A slice in lateral (iy) and in longitudinal (ix) dimensions for array
                slicing
            '''
            lats = self._dataset['PRMSL']['latitude'][:]
            lons = self._dataset['PRMSL']['longitude'][:]
            bbox = self._region.bbox  #lon0, lat0, lon1, lat1

            lon0 = np.where(bbox[0] == lons)[0][0]
            lon1 = np.where(bbox[2] == lons)[0][0]
            lat0 = np.where(bbox[1] == lats)[0][0]
            lat1 = np.where(bbox[3] == lats)[0][0]

            ix = slice(lon0, lon1, 1)
            iy = slice(lat1, lat0, 1)

            return ix, iy

        def timeslice():
            '''
            Creates a slice from

            Returns
            -------
            slice
                A slice in temporal dimension (it) for array slicing
            '''

            glo_times = self._dataset['PRMSL']['time'][:]
            start_idx = np.where(glo_times == self._start_epochtime)[0][0]
            stop_idx = np.where(glo_times == self._stop_epochtime)[0][0]

            return slice(start_idx, stop_idx, 1)

        ix, iy = boundary_slice()
        it = timeslice()
        il = slice(0, 5, 1)
        ie = slice(0, 50, 1)

        self.ie, self.ix, self.iy, self.it, self.il = ie, ix, iy, it, il

        #HACK defining il should not require user to edit code and/or have knowledge of level size.

        #abbr
        data = self._dataset
        region = self._region

        if self.mode == 'analysis':
            slices = [it, iy, ix]
            slices_level = [it, iy, ix, il]
        elif self.mode == 'ensemble':
            slices = [ie, it, iy, ix]
            slices_level = [ie, it, iy, ix, il]

        self.pressure = Parameter(data, region, 'PRMSL', config_var['PSL'],
                                  slices)
        self.ugrd = Parameter(data, region, 'UGRD_10M', config_var['U10M'],
                              slices)
        self.vgrd = Parameter(data, region, 'VGRD_10M', config_var['V10M'],
                              slices)
        self.vmax = Parameter(data, region, 'VGRD_10M', config_var['V10M'],
                              slices)
        self.vmax.values = np.sqrt(self.ugrd.values**2 + self.vgrd.values**2)
        self.u850 = Parameter(data, region, 'U850', config_var['U850'],
                              slices_level)
        self.v850 = Parameter(data, region, 'V850', config_var['V850'],
                              slices_level)
        self.tmp = Parameter(data, region, 'TMP', config_var['TMP'],
                             slices_level)

    def detection_tracking_algorithm(self, mode=config_main['MODE']):
        '''
        This is the main tracking algorithm segment of the code. Within this function are
        multiple sub-functions that piece together the tracking algorithm.

        Parameters
        ----------
        mode : string, optional
            Parsed automatically from YAML config file, by default config_main['MODE']
            Should not be changed.

        '''
        self.mode = mode

        def detection_algorithm(self, ie, tstamp):
            def cyclone(self, ie, tstamp):
                '''
                A cyclone is a synoptic low-level pressure area. This function looks at the
                regional pressure dataset and locates the local pressure minima through use of a SciPy
                minimum filter. Filter size is defined in regional rules.

                Parameters
                ----------
                tstamp : int
                    The current activated timestamp in the for-loop of the requested time range

                Returns
                -------
                list (of tuples)
                    Returns a list of global coordinates (latitude, longitude) that represent all of the local
                    pressure minima in the region for that timestamp
                '''
                neighborhood_size = self._region.rules.pressure_neighbourhood

                #Run a minimum filter on a 2D array
                filtered_pressure_min = minimum_filter(
                    self.pressure.values[ie, tstamp],
                    neighborhood_size,
                    mode='nearest')
                #Create bool array of filter output onto data
                minima = (
                    self.pressure.values[ie, tstamp] == filtered_pressure_min)
                #Return y,x arrays where bool == True
                y, x = np.where(minima)
                #convert lats, lons to global and store in list of tuples
                pressure_minima = [
                    _convert_coords(self._region, coord, mode='to_global')
                    for coord in list(zip(y, x))
                ]

                #HACK: typhoons keep on appearing on the boundaries
                def remove_boundary_values(minima_list):
                    '''
                    Removes all pressure minimas that are located along the boundaries of the bounding box. 
                    This is to prevent

                    Parameters
                    ----------
                    minima_list : list of tuples
                        The list of global coordinates (latitude, longitude) that represent all of the local
                        pressure minima in the region for that timestamp

                    Returns
                    -------
                    list of tuples
                        returns the input list of pressure minima, minus the excluded minima along the bounding
                        box edges.
                    '''
                    for coord in minima_list:
                        if coord[0] == self._region.bbox[2] or coord[
                                0] == self._region.bbox[3]:
                            minima_list.remove(coord)
                        elif coord[1] == self._region.bbox[0] or coord[
                                1] == self._region.bbox[1]:
                            minima_list.remove(coord)
                    return minima_list

                return remove_boundary_values(pressure_minima)

            def tropical_cyclone(self, cyc_coords, ie, tstamp):
                '''
                For a single pressure minimas for the current timestamp, the function makes a check for several
                criteria that exhibit characteristics of a tropical_cyclone :
                - pressure criterium
                - vmax criterium
                - temperature anomaly criterium
                - vorticity criterium
                The criteria check values are provided in the the Region.Rules class.

                Parameters
                ----------
                tstamp : int
                    The current activated timestamp in the for-loop of the requested time range
                pressure_minima : list of tuples
                    The list of global coordinates (latitude, longitude) that represent all of the local
                        pressure minima in the region for that timestamp

                Returns
                -------
                list of tuples
                    Returns a list of tropical cyclone candidates that match all of the aforemnetioned criteria
                '''
                #link rules class for easier typing
                rules = self._region.rules

                def pressure_criteria(ie, tstamp, coord):
                    '''
                    Verifies if the pressure value at the cyclone center is sufficiently low to be considered a tropical cylone.
                    Function:
                    1.  Converts global coords from input (pressure_minima) to regional coords (for use with sliced dataset)
                    2.  Collects pressure value at specific spatial and temporal position
                    3.  Requires pressure to be lower than pressure_min rule to pass on coord
                    '''
                    #1
                    reg_idx = _convert_coords(self._region,
                                              coord,
                                              mode='to_regional')
                    #2
                    pressure = self.pressure.values[ie, tstamp, reg_idx[0],
                                                    reg_idx[1]]
                    #3
                    if pressure < rules.pressure_min:  #unit hPa to Pa
                        return coord

                def vmax_criteria(ie, tstamp, coord):
                    '''
                    Verifies if the maximum wind speed in a given area around the cyclone position is sufficiently intense
                    to be qualified as a TC.
                    Function:
                    1.  Creates two slices in lat (yslice) and lon (xslice) directions around the center of the cyclone.
                        Size of slice is determined by vmax_radius rule.
                    2.  Uses these slices to create a NxN array of vmax values surrounding the cyclone
                    3.  Checks to determine if any of the values in this area are higher than the minimum vmax requirements
                        from rules.vmax_thresh
                     '''
                    #1
                    yslice, xslice, _ = self.vmax.box_slice(
                        coord, rules.vmax_radius)
                    #2
                    vmax_area = self.vmax.values[ie, tstamp, yslice, xslice]
                    #3
                    if np.any(vmax_area >= rules.vmax_thresh):
                        return coord

                def tmp_criteria(ie, tstamp, coord):
                    '''
                    Verifies if the temperature anomaly of the cyclone core sufficiently strong to qualify as a TC.
                    Method is to subtract the mean temperature of a snmaller inner box directly surrounding the cyclone
                    from a larger outer box.

                    1.  Creates two slices in lat (yslice) and lon (xslice) directions around the center of the cyclone.
                        Size of slice is determined by core_outer rule.
                    2.  Uses these slices to create a NxN array of temperature values surrounding the cyclone. This is
                        done for atmospheric level of 700 hPa, 500 hPa and 300hPa
                    3.  Creates a 2d slice for the inner box, dependant on the rules core_outer (r_out) and core_inner (r_in)
                    4.  Creates a twos same-size (as in #1) arrays of ones and zeros. The outer shape array does not calculate
                        the values in the inner shape, and vice-versa.
                    5.  Ndimage.mean function can take a shape, denoted as an array of ones and zeroes (see #4) and calculate
                        the mean for all cells where there is a one. Using the inner and outer shapes created in #4, we calculate
                        the temperature anomaly for the three pressure levels by subtracting the outer temperature mean from the 
                        inner temperature mean.
                    6.  If the sum of all three temperature anoamlies is greater than 0, then the criteria has been passed.

                    '''
                    #1
                    r_out = rules.core_outer
                    r_in = rules.core_inner
                    yslice, xslice, _ = self.tmp.box_slice(coord, r_out)
                    #2
                    if self.mode == 'analysis':
                        tmp_700 = self.tmp.values[ie, tstamp, yslice, xslice,
                                                  3]  #ens =2, ans = 3, lan=2
                        tmp_500 = self.tmp.values[ie, tstamp, yslice, xslice,
                                                  2]  #ens = 1, ans = 2, lan=1
                        tmp_300 = self.tmp.values[ie, tstamp, yslice, xslice,
                                                  1]  # ens =0, ans = 1, lan=0
                    if self.mode == 'ensemble':
                        tmp_700 = self.tmp.values[ie, tstamp, yslice, xslice,
                                                  2]  #ens =2, ans = 3, lan=2
                        tmp_500 = self.tmp.values[ie, tstamp, yslice, xslice,
                                                  1]  #ens = 1, ans = 2, lan=1
                        tmp_300 = self.tmp.values[ie, tstamp, yslice, xslice,
                                                  0]  # ens =0, ans = 1, lan=0

                    #3
                    x = (r_out - r_in) / 2
                    inner_side = slice(int(x), int(r_out - x))
                    inner_slice = (inner_side, inner_side)
                    #4
                    outer_shape = np.ones_like(tmp_700)
                    outer_shape[inner_slice] = 0
                    inner_shape = np.zeros_like(tmp_700)
                    inner_shape[inner_slice] = 1
                    #5
                    #calculate anomaly from inner and outer core with ndimage.filer.mean
                    anomaly700 = mean(tmp_700, inner_shape) - mean(
                        tmp_700, outer_shape)
                    anomaly500 = mean(tmp_500, inner_shape) - mean(
                        tmp_500, outer_shape)
                    anomaly300 = mean(tmp_300, inner_shape) - mean(
                        tmp_300, outer_shape)
                    #6
                    if (anomaly700 + anomaly500 + anomaly300) > 0:
                        return coord

                def vort_criteria(ie, tstamp, coord):
                    '''
                    Function defines a box of specified radius around the center of the cyclone.
                    1.  Creates two slices in lat (yslice) and lon (xslice) directions around the center of the cyclone.
                        Size of slice is determined by vort_radius rule. 
                    2.  Calculate dx and dy for search area via Haversine formula. create a mean dx and dy for forcing in the vorticity creation
                    3.  Slice u and v component at 850hPa level
                    4.  Force dx, dy, u and v to calculate vorcitiy fields around candidate. Detemine if a value in field
                        greater than designated threshold.
                    '''

                    def calc_dx_dy(longitude, latitude):
                        '''
                        This definition calculates the distance between grid points that are in
                        a latitude/longitude format. Necessary for vorticity calculations as dx changes
                        as a function of latitude.

                        Equation and code from:
                        http://andrew.hedges.name/experiments/haversine/

                        dy should be close to 55600 m
                        dx at pole should be 0 m
                        dx at equator should be close to 55600 m

                        Accepts, 1D arrays for latitude and longitude

                        Returns: dx, dy; 2D arrays of distances between grid points
                        in the x and y direction in meters
                        '''

                        dlat = np.abs(latitude[1] - latitude[0]) * np.pi / 180
                        dy = 2 * (np.arctan2(
                            np.sqrt((np.sin(dlat / 2))**2),
                            np.sqrt(1 - (np.sin(dlat / 2))**2))) * 6371000
                        dy = np.ones(
                            (latitude.shape[0], longitude.shape[0])) * dy

                        dx = np.empty((latitude.shape))
                        dlon = np.abs(longitude[1] -
                                      longitude[0]) * np.pi / 180
                        for i in range(latitude.shape[0]):
                            a = (np.cos(latitude[i] * np.pi / 180) *
                                 np.cos(latitude[i] * np.pi / 180) *
                                 np.sin(dlon / 2))**2
                            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                            dx[i] = c * 6371000
                        dx = np.repeat(dx[:, np.newaxis],
                                       longitude.shape,
                                       axis=1)
                        return dx, dy

                    #1
                    yslice, xslice, _ = self.vmax.box_slice(
                        coord, npad=rules.vort_radius)
                    lats = self._region.reg_lats[yslice]
                    lons = self._region.reg_lons[xslice]
                    #2
                    dx, dy = calc_dx_dy(lons, lats)
                    dx = dx.mean()
                    dy = dy.mean()
                    #3
                    if self.mode == 'analysis':
                        u = self.u850.values[ie, tstamp, yslice, xslice,
                                             4]  # ens =1, ans =4, lan=0
                        v = self.v850.values[ie, tstamp, yslice, xslice,
                                             4]  # ens =1, ans =4, lan=0
                    if self.mode == 'ensemble':
                        u = self.u850.values[ie, tstamp, yslice, xslice,
                                             0]  # ens =1, ans =4, lan=0
                        v = self.v850.values[ie, tstamp, yslice, xslice,
                                             0]  # ens =1, ans =4, lan=0
                    #4
                    vort = vorticity(u, v, dx, dy).magnitude

                    if np.any(vort > rules.vort_thresh):
                        return coord

                def check_criteria(ie, tstamp, cyc_coords):
                    '''
                    Function applies all of the aforementioned criteria to the list of cyclones of
                    cyclones for the given timestamp.
                    1.  Creates four seperate lists for each of the TC criteria. These lists are filled
                    with a list of all coordinate tuples that passed the respective criteria tests for that
                    metric.
                    2.  The four lists are intersected to find the unqiue set of coordinate tuples that satisfy
                    all of the criteria.
                    3. Removes all None instances from the list.
                    '''
                    pressure_candidates, vmax_candidates, tmp_candidates, vort_candidates = [], [], [], []
                    #1
                    for coord in cyc_coords:
                        pressure_candidates.append(
                            pressure_criteria(ie, tstamp, coord))
                        vmax_candidates.append(vmax_criteria(
                            ie, tstamp, coord))
                        tmp_candidates.append(tmp_criteria(ie, tstamp, coord))
                        vort_candidates.append(vort_criteria(
                            ie, tstamp, coord))

                    #2
                    tc_candidates = set(pressure_candidates).intersection(
                        vmax_candidates, vort_candidates)  #tmp_candidates
                    #3
                    tc_candidates = list(filter(None, tc_candidates))

                    return tc_candidates

                return check_criteria(ie, tstamp, cyc_coords)

            #Run main functions to determine list of tropical cyclone candidate coordinates
            cyc_coords = cyclone(self, ie, tstamp)
            tc_coords = tropical_cyclone(self, cyc_coords, ie, tstamp)

            return tc_coords

        def tracking_algorithm(self, tc_coords, ie, tstamp, key):
            '''
            '''

            def prepare_history(self, coord, ie, tstamp):
                position = coord
                #max_windspeed
                yslice, xslice, _ = self.vmax.box_slice(
                    coord, self._region.rules.vmax_radius)
                vmax = self.vmax.values[ie, tstamp, yslice, xslice].max()

                #intensity class
                def classify(vmax):
                    for ii in range(len(self._region.scale_values)):
                        if vmax < self._region.scale_values[ii]:
                            return self._region.scale_names[ii]
                    return self._region.scale_names[len(
                        self._region.scale_names) - 1]  # last element

                intensity_class = classify(vmax)

                #min pressure
                lat = np.where(coord[0] == self._region.reg_lats)[0][0]
                lon = np.where(coord[1] == self._region.reg_lons)[0][0]
                pmin = self.pressure.values[ie, tstamp, lat, lon]

                history = {
                    'class': intensity_class,
                    'pos': position,
                    'vmax': vmax,
                    'pmin': pmin
                }

                return history

            def def_search_area(self, coord):
                '''
                Creates a 2-dimensional search grid surrounding the coord variable. Used to check if a new coord tuple is
                located within the searchg grid.

                REQUIRES: coordinate pair (coord)
                RETURNS: array of coord pairs in radius around coord

                '''
                _, _, coords = self.pressure.box_slice(
                    coord, npad=self._region.rules.update)
                yy, xx = np.meshgrid(coords[0], coords[1])
                coord_array = np.stack((np.ravel(yy), np.ravel(xx)), axis=-1)

                return list(map(tuple, coord_array))

            def create_candidate(self, coord, tstamp, key):
                '''
                Creates a new member of the candidate class with a sequential number based on the total number of candidates detected
                so far in the current ensemble. Additonal metadata is additionally sourced and saved to the class memebr.
                '''
                name = ''.join(random.choices(string.ascii_uppercase, k=4))
                region = self._region
                position = coord
                active = True
                detection = tstamp
                detection_pos = coord
                history = {}
                history[tstamp] = prepare_history(self, coord, ie, tstamp)
                last_update = tstamp
                search_area = def_search_area(self, coord)
                color = np.random.rand(3, )

                #ensemble key
                self.candidates[key][name] = Candidate(region, position,
                                                       active, history,
                                                       search_area, detection,
                                                       detection_pos,
                                                       last_update, color)

            def update_candidate(self, entry, coord, tstamp, key):
                '''
                Updates the relevant class attribtutes for a given entry.
                '''
                candidate = self.candidates[key][entry]
                candidate.position = coord
                candidate.history[tstamp] = prepare_history(
                    self, coord, ie, tstamp)
                candidate.search_area = def_search_area(self, coord)
                candidate.last_update = tstamp

            def iterate_candidate(self, coord, tstamp, key):
                '''
                This function iterates through the list of candidates in self.candidates.
                If the the coordinate pair (from pressure_list) is found in the current search area of any of the candidates,
                the coord_pair is append to the history list. If not found, a new candidate is created.
                '''
                for entry in self.candidates[key]:
                    if self.candidates[key][entry].active == True:
                        if coord in self.candidates[key][entry].search_area:
                            update_candidate(self, entry, coord, tstamp, key)
                        return
                create_candidate(self, coord, tstamp, key)

            #run main functions
            for coord in tc_coords:
                iterate_candidate(self, coord, tstamp, key)

        def duration_criterion(self, key):
            '''
            Removes candidates that do not meet the required duration criteria. Duration criteria are defined
            in the region class and are a function of the timestep interval of the dataset.
            REQUIRES: None
            RETURNS: Candidates dictionary
            '''
            remove_list = []
            for entry in self.candidates[key]:
                if len(self.candidates[key]
                       [entry].history) < self._region.rules.duration:
                    remove_list.append(entry)
            for entry in remove_list:
                self.candidates[key].pop(entry)

            #rename dictionary keys to be numeric:
            count = 0
            for entry in list(self.candidates[key]):
                self.candidates[key][count] = self.candidates[key].pop(entry)
                count = +1

        def cyclolysis(self, tstamp, key):
            '''
            '''
            for entry in self.candidates[key]:
                if (tstamp - self.candidates[key][entry].last_update
                    ) > self._region.rules.cyclosis:
                    self.candidates[key][entry].active = False

        def single_pass(self, ie):
            '''
            [summary]

            Parameters
            ----------
            ie : [type]
                [description]
            tstamp : [type]
                [description]
            '''
            if self.mode.lower() == 'analysis':
                key = 'Analysis'
            elif self.mode.lower() == 'ensemble':
                key = 'Ensemble ' + str(ie)

            self.candidates[key] = {}
            self.candidates[key]['XXX'] = Candidate(self._region, (-1, 1),
                                                    False, {'pos': (-1, -1)},
                                                    [(-1, -1)], 0, (-1, -1), 0,
                                                    'b')
            for tstamp in range(len(self.times)):
                tc_coords = detection_algorithm(self, ie, tstamp)
                tracking_algorithm(self, tc_coords, ie, tstamp, key)
                cyclolysis(self, tstamp, key)
            duration_criterion(self, key)

        def find_tc_in_analysis_timestep():
            '''
            '''
            key = 'Analysis'
            self.candidates[key] = {}
            tc_coords = detection_algorithm(self, ie=0, tstamp=0)
            tracking_algorithm(self, tc_coords, ie=0, tstamp=0, key='Analysis')

        #define dictionaries
        self.candidates = {}

        if self.mode.lower() == 'analysis':
            ie = None
            single_pass(self, ie)

        elif self.mode.lower() == 'ensemble':
            for ie in self.ensembles:
                single_pass(self, ie)


#            find_tc_in_analysis_timestep()


class Parameter:
    def __init__(self, dataset, region, key, var, slices):
        '''
        Creates a class called Paramter which serves as a storage unit for paramater values and functions. Usually
        created from a larger dataset and thus requires slicing parameters for efficiency. Assumes that dictionary
        key name and netCDF variable name are identical. If dataset has levels, il (ilevels) must be specified as
        slice.
        REQUIRES: dataset(library), abbreviation(string), region (class obj), it(slice), ix (slice), iy (slice),
                  il (slice)*
        RETURN: values (n-dimensional array), region(obj reference)
        '''

        self.values = dataset[key][var][slices]
        self.region = region

    def box_slice(self, coord, npad):
        '''
        Creates a bounding box around a coordinate tuple ie.(40.5,160) and returns a slice in longitudinal and
        lateral direction as well as a list of all coordinate tuples within the box. The size of the bounding box
        can be defined by the npad keyboard. Bounding boxes can currently only be square and npad is halt the side of the box.
        Box will always have an odd number side length to preverse symmetry around box center. Function checks to ensure that
        box does not extend beyond region domain.  
        REQUIRES: coord (tuple), npad (int)
        RETURNS: yslice (slice), xslice (slice), coord ()
        '''
        # convert global coords to a regional index
        regional_idx = _convert_coords(self.region, coord, mode='to_regional')

        y0 = regional_idx[0] - npad
        y1 = regional_idx[0] + npad + 1
        x0 = regional_idx[1] - npad
        x1 = regional_idx[1] + npad + 1

        # ensures that bounding box does not extend out beyond domain borders
        if self.region.Tracking.mode.lower() == 'analysis':
            shape_y = 1
            shape_x = 2
        elif self.region.Tracking.mode.lower() == 'ensemble':
            shape_y = 2
            shape_x = 3

        if y0 < 0:
            y0 = 0
        if y1 > self.values.shape[shape_y]:
            y1 = self.values.shape[shape_y]
        if x0 < 0:
            x0 = 0
        if x1 > self.values.shape[shape_x]:
            x1 = self.values.shape[shape_x]

        yslice = slice(int(y0), int(y1))
        xslice = slice(int(x0), int(x1))
        # list of all coordinates within the bounding box
        box_coords = [
            self.region.reg_lats[yslice], self.region.reg_lons[xslice]
        ]

        return yslice, xslice, box_coords

    def get(self, relation, parthresh=None):
        '''
        Takes a parameter variable and searchs for all squares that match operator
        ie. min, max, smaller, greater, etc
        REQUIRES: relation operator (string): ['min', 'max', '>', '<', '>=', '<=', '=']
        RETURNS: index of location(s) where operator value is true
        '''

        import operator
        ops = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '=': operator.eq
        }

        if relation == 'min':
            coords = np.where(self.values == self.values.min())
        elif relation == 'max':
            coords = np.where(self.values == self.values.max())
        else:
            coords = np.where(ops[relation](self.values, parthresh))
        return coords


class Candidate:
    def __init__(self, region, position, active, history, search_area,
                 detection, detection_pos, last_update, color):

        self.region = region
        self.position = position
        self.active = active
        self.history = history
        self.search_area = search_area
        self.detection = detection
        self.detection_pos = detection_pos
        self.last_update = last_update
        self.color = color

    def __repr__(self):
        return "\nposition: {}, \nlength: {}, \ndetection: {}, \ndetection position {}, \nlast update: {}\n".format(
            self.position, len(self.history), self.detection,
            self.detection_pos, self.last_update)


###############################################################################################################
################################################### MAIN ######################################################
###############################################################################################################

Region = {}
Region[config_region['NAME']] = RegionClass(config_region['NAME'], config_region['PATH'], config_region['DEGS'], \
     config_region['BBOX'], config_region['TS'], config_intensity)
Region[config_region['NAME']].createTracking()
Region[config_region['NAME']].Tracking.detection_tracking_algorithm()
candidates = Region[config_region['NAME']].Tracking.candidates

print(list(candidates.keys()))
