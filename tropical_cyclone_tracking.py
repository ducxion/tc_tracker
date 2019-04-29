__author__ = 'Christoph Morgan'
__copyright__ = '2019'
__license__ = 'MIT'
__version__ = 0.4
__maintainer__ = 'Christoph Morgan'
__email__ = 'christoph.morgan@gmail.com'
__status__ = 'prototype'

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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib import colors
from matplotlib.animation import FuncAnimation

init_dataset = {
    'ensemble': Dataset(r'C:\user\typhoon tracking\data\gfs\ensemble\20170909T00_msl.nc', 'r'),
    'analysis': Dataset(r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\PRMSL.nc', 'r')
}

analysis = {
    'PRMSL_10M':
        Dataset(r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\PRMSL.nc', 'r'),
    'UGRD_10M':
        Dataset(r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\UGRD_10M.nc', 'r'),
    'VGRD_10M':
        Dataset(r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\VGRD_10M.nc', 'r'),
    'TMP':
        Dataset(r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\TMP.nc', 'r'),
    'UGRD':
        Dataset(r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\UGRD.nc', 'r'),
    'VGRD':
        Dataset(r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\VGRD.nc', 'r')
}

### GLOBAL FUNCTIONS ###


def _to_global(region, region_coord):
  ''' This will take a coordinate from a regional slice (ie. se_asia) in the
    form of (lat,lon) and return the corresponding pair of global coordinates.'''
  #FIXME Automatic rounding to be divisible by globe.degs
  glo_lats = region.lats[region_coord[0]]
  glo_lons = region.lons[region_coord[1]]

  return (glo_lats, glo_lons)


def _to_regional(region, global_coord):
  ''' This will take a global coordinate entry and return the local
    coordinate (indices) that correspond to the local region.'''
  #FIXME Automatic rounding to be divisible by globe.degs
  reg_lats = np.where(global_coord[0] == region.lats)[0][0]
  reg_lons = np.where(global_coord[1] == region.lons)[0][0]

  return (reg_lats, reg_lons)


class Globe:

  def __init__(self, ensemblepath=None, analysispath=None):
    #FIXME: Ensemble and analysispath should be automatically set to folder
    #where file is: +ensemble/ an
    #TODO: Allow for different spatial analysis and ensemeble resolutions?
    # *This would require a massive rewriting of large portions of the
    # code
    self.ensemblepath = ensemblepath
    self.analysispath = analysispath

    def metadata(path):
      '''Performs a metadata analysis on the spatial and temperal
            resolution of the initial dataset.Creates a new lats and lons array
            according to the spatial step.'''
      metadata = Dataset(path, 'r')

      #create latitude array, assumption: square grid
      degs = abs(metadata['latitude'][0] - metadata['latitude'][1])
      lats = np.linspace(90, -90, int(180 / degs) + 1)
      lons = np.linspace(-180, 180 - degs, int(360 / degs))

      #create timestep
      time = metadata['time'][:]
      ts_seconds = int(abs(metadata['time'][0] - metadata['time'][1]))
      ts_hours = ts_seconds / 3600

      return lats, lons, degs, time, ts_seconds

    self.lats, self.lons, self.degs, self.time_analysis, self.ts_analysis = metadata(
        self.analysispath)

    if self.ensemblepath != None:
      _, _, _, self.time_ensemble, self.ts_ensemble = metadata
      (self.ensemblepath)
      #different spatial resolutions not supported

  def region(self, name, top_left, bot_right):
    return Region(self, name, top_left, bot_right)

  def _validate(self):
    '''Would be used to validate datasets. For example, ensure that all
        global datasets have the same size, or that all the time stamps of the
        datasets line up. 
        '''
    pass


class Region:

  def __init__(self, earth, name, top_left, bottom_right):
    '''Enter the coordinates of two corners of the bounding box. First the
        top left corner and then bottom right corner. Coordinates should be
        given as tuples: (latitude,longitude). Currently only square shapes
        allowed. Bounding numbers are rounded to the nearest integer. 
        A regional lats and lons array is created from the bounding box. '''
    self.bounds = [
        round(top_left[1]),  #lon0
        round(bottom_right[1]),  #lon1
        round(top_left[0]),  #lat1
        round(bottom_right[0])  #lat0
    ]
    self._globe = earth
    self._name = name
    lats = np.arange(self.bounds[3], self.bounds[2] + self._globe.degs, self._globe.degs)
    lons = np.arange(self.bounds[0], self.bounds[1] + self._globe.degs, self._globe.degs)
    self.lats = np.flip(lats)  #convention is to count down
    self.lons = lons
    self.besttracks = {}

    def define_default_rules():
      ''' These are the default values for the algorithm parameters. They
            are dependant on spatial (degs) and temporal (ts_...) resolution
            and should update accordingly, however no guarentee is made for
            the accurate results at resolutions other than intended ones
            (0.5 & 1 degrees), (3,6,12 hour steps) '''
      degs = self._globe.degs
      rules = {
          'pressure_neighbourhood': 7.5 / degs,
          'vmax_radius': 2 / degs,
          'vmax_thresh': 16.5,
          'vort_radius': 2 / degs,
          'vort_thresh': 0.00001,
          'core_inner': 1 / degs,
          'core_outer': 3.5 / degs,
          'core_rule': None,
          'duration_analysis': 24 / (self._globe.ts_analysis / 3600),
          'duration_ensemble': 24 / (self._globe.ts_ensemble / 3600),
          'cyclosis_analysis': 24 / (self._globe.ts_analysis / 3600) * 2,
          'cyclosis_ensemble': 24 / (self._globe.ts_ensemble / 3600) * 2,
          'update_analysis': 2 / degs,
          'update_ensemble': 4 / degs,
      }
      return rules

    self.rules = define_default_rules()

  def __repr__(self):
    return 'Class Region: \
        \nName: {} \
        \nBounding: \tTopLeft: {} \tBottomRight: {}\
        \n\nRegional Rules: \
        \nPressure Neighbourhood: {}\
        \nVmax Radius: {} \
        \nVmax Threshold: {}\
        \nVorticity Radius: {}\
        \nVorticity Threshold: {}\
        \nTemperature Anomaly Inner Core Radius: {}\
        \nTemperature Anomaly Outer Core Radius: {}\
        \nTemperature Anomaly Rule: {}\
        \nMinimum # of TS for Analysis Cyclogenesis: {}\
        \nMinimum # of TS for Ensemble Cyclognesis: {}\
        \nMinimum # of TS for Analysis Cyclosis: {}\
        \nMinimum # of TS for Ensemble Cyclosis: {}\
        \nUpdate Radius, Analysis: {}\
        \nUpdate Boxsize, Ensemble: {}\
        '                                         .format(\
    self._name, \
    (self.bounds[0],self.bounds[1]), (self.bounds[2],self.bounds[3]), \
    self.rules['pressure_neighbourhood'],\
    self.rules['vmax_radius'],\
    self.rules['vmax_thresh'],\
    self.rules['vort_radius'],\
    self.rules['vort_thresh'],\
    self.rules['core_inner'],\
    self.rules['core_outer'],\
    self.rules['core_rule'],\
    self.rules['duration_analysis'],\
    self.rules['duration_ensemble'],\
    self.rules['cyclosis_analysis'],\
    self.rules['cyclosis_ensemble'].\
    self.rules['update_analysis'],\
    self.rules['update_ensemble'])

  def _regional_datasets():
    '''if regional datasets are to be used, this is were they would be
        initialised '''
    pass

  def regional_rules(**kwargs):
    ''' Function that allows user to change the algorithm rules. Current
        possible parameters are:
        pressure_neighbourhood, vmax_radius, vmax_thresh, vort_radius,
        vort_thresh, core_boxsize, core_rule, duration_analysis,
        duration_ensmble, update_analysis, update_ensemble. 
        Syntax according to standard **kwargs syntax,
        i.e. regional_rules(vmax_thresh=19) 
        '''
    for entry in kwargs:
      self.rules[entry] = kwargs[entry]

  def define_intensity_scale(level_name, upper_threshold, unit='ms'):
    ''' level_name should be a list of string. e.g ['Tropical Depression',
        'Tropical Storm', 'Typhoon', 'Super Typhoon'].Upper_threshold should be
        a list of int or floats that is the same length as the level_name: e.g.[17,33,55,70]
        This represenets the upper intensity threshold for the TC to fall into
        that category.'''

    def unit_conversion(level_threshold, unit):
      if unit == 'ms':
        pass
      elif unit == 'kt':
        level_threshold = level_threshold * 0.514444
      elif unit == 'kmh':
        level_threshold = level_threshold * 0.277778
      elif unit == 'mph':
        level_threshold = level_threshhold * 0.447040
      return level_threshold

    def create_level_range(level_threshhold):
      bounds = [5, 8, 13]
      classifications = ["c1", "c2", "c3", "c4"]

    def classify(speed):
      for ii in range(len(bounds)):
        if speed < bounds[ii]:
          return classifications[ii]
      return classifications[len(classifications) - 1]  # last element

    for jj in np.arange(0, 20, 0.3):
      print("speed", jj, "is in class", classify(jj))

  def _region_slice(self):
    ''' Create slicing arrays based on the defined boundaries in
        region.__init__. Used for parameter initialising'''
    y = self._globe.lats
    x = self._globe.lons
    iy = (y >= self.bounds[3]) & (y <= self.bounds[2])
    ix = (x >= self.bounds[0]) & (x <= self.bounds[1])
    x, y = x[ix], y[iy]
    self.lons = x
    self.lats = y

  def Ensemble(self, timestamp, path, forecastdays=10):
    self.Ensemble = Ensemble(self, timestamp, path, forecastdays)
    #FIXME: timestamp should automatically round to next 6 hour mark.

  def Analysis(self, start, stop):
    self.Analysis = Analysis(self, start, stop)


class Analysis:

  def __init__(self, region, start, stop):
    ''' Enter the start and stop datetimes here (YYYY-MM-DD hh:mm:ss)'
        Dates must be fully divisble by the temporal resolution (ie. every 3,6
        hours). Init functions convert this string to a pandas datetime. UNIX
        epochtime conversion is from the datetime format to an integer number
        of seconds since 01.01.1970.

        INPUT: designated region, start (string), stop (string)
        RETURN: start(datetime), stop (datetime), start_epochtime (int),
        stop_epochtime (int), times (1D-array)
        '''
    self._region = region
    self._globe = region._globe
    self.start = dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    self.stop = dt.datetime.strptime(
        stop, '%Y-%m-%d %H:%M:%S') + dt.timedelta(seconds=self._globe.ts_analysis)
    #Extra timestep added so that stop timestep is included in data values
    self._start_epochtime \
            = int(self.start.replace(tzinfo=dt.timezone.utc).timestamp())
    self._stop_epochtime \
            = int(self.stop.replace(tzinfo=dt.timezone.utc).timestamp())

    self.times = self._globe.time_analysis[self._timeslice()]
    self._dataset = analysis

    #AUTOMATIC PARAMETER INITILIASATION AND TC detection
    self._initialise_parameters()

  def __repr__(self):
    return 'Region: {} \
        \nStart: {} \
        \nEnd: {} \
        \nLength: {} \tTimeSteps: {}h\
        '.format(self._region._name, self.start, self.stop, self.times.shape[0],
                 self._globe.ts_analysis / 3600)

  def _timeslice(self):
    '''Finds the index location of the start and stop times in the global timestamp list. 
        Exception is raised if the start and stop times are not fully divisible with the length of a timestep.
        INPUT: (self)  
        RETURNS: index of start (int), index of stop (int), slice 
        '''
    if self._start_epochtime % self._globe.ts_analysis != 0 or \
       self._stop_epochtime % self._globe.ts_analysis != 0:
      raise Exception('Start and Stop datetime must be fully divisible with the Analysis timestep')
    start_idx = np.where(self._globe.time_analysis == self._start_epochtime)[0][0]
    stop_idx = np.where(self._globe.time_analysis == self._stop_epochtime)[0][0]
    return slice(start_idx, stop_idx)

  def _initialise_parameters(self):
    '''This initialises all the defined parameters (defined in the function). Arrays in the form (timeslice,lat,lon) 
        are created from the global dataset. ts_slice,iy and ix are slices used when creating the dataset.
        Self is used to refer the parameter back to it's "parent class'''

    def boundary_slice():
      ''' Creates two slice from the defined bounding box for the region. Slice are for the extents
            in longitudinal and lateral direction. 
                INPUT: (self)
                RETURNS: slice of longitude (ix), slice of latitude (iy)
            '''

      lats = self._globe.lats
      lons = self._globe.lons
      bounds = self._region.bounds  #lon0,lon1,lat1,lat0
      #Find index of bounding box coordinates in global list. i.e latitude 40 = global_lats[100]
      lon0 = np.where(bounds[0] == lons)[0][0]
      lon1 = np.where(bounds[1] == lons)[0][0]
      lat1 = np.where(bounds[2] == lats)[0][0]
      lat0 = np.where(bounds[3] == lats)[0][0]

      ix = slice(lon0, lon1, 1)
      iy = slice(lat1, lat0, 1)
      return ix, iy

    ix, iy = boundary_slice()
    it = self._timeslice()
    #HACK defining il should not require user to edit code and/or have knowledge of level size.
    il = slice(0, 5, 1)
    region = self._region

    self.lats = self._region.lats
    self.lons = self._region.lons
    self.pressure = Parameter(self._dataset, 'PRMSL_10M', region, it, iy, ix)
    self.ugrd = Parameter(self._dataset, 'UGRD_10M', region, it, iy, ix)
    self.vgrd = Parameter(self._dataset, 'VGRD_10M', region, it, iy, ix)
    self.vmax = Parameter(self._dataset, 'VGRD_10M', region, it, iy, ix)
    self.vmax.values = np.sqrt(self.ugrd.values**2 + self.vgrd.values**2)
    self.u850 = Parameter(self._dataset, 'UGRD', region, it, iy, ix, il)
    self.v850 = Parameter(self._dataset, 'VGRD', region, it, iy, ix, il)
    self.tmp = Parameter(self._dataset, 'TMP', region, it, iy, ix, il)

  def pressure_candidates(self, mode='analysis'):
    '''New attempt using a pressure min'''
    self.mode = mode

    def _filler_candidate(self):
      '''Creates a temporary filler candidate. Is deleted after usage'''
      name = 'XXXX0'
      region = self._region
      position = (-9999, -9999)
      active = False
      history = [position]
      search_area = [(-9999, -9999)]
      detection = 0
      last_update = detection
      color = rand_color = np.random.rand(3,)
      self.candidates[name] = TyphoonCandidate(name, region, position, \
           active, history, search_area, detection, last_update, color)

    def _clean_up(self):
      ''' Removes candidates that do not meet the required duration criteria. Duration criteria are defined
            in the region class and are a function of the timestep interval of the dataset. 
            REQUIRES: None
            RETURNS: Candidates dictionary
            '''
      if self.mode == 'analysis':
        remove_list = []
        for entry in self.candidates:
          if len(self.candidates[entry].history) < self._region.rules['duration_analysis']:
            remove_list.append(entry)
        for entry in remove_list:
          self.candidates.pop(entry)

      elif self.mode == 'ensemble':
        remove_list = []
        for entry in self.candidates:
          if len(self.candidates[entry].history) < self._region.rules['duration_ensemble']:
            remove_list.append(entry)
        for entry in remove_list:
          self.candidates.pop(entry)

      else:
        print('Mode must be either "analysis" or "ensemble"!')

    def deactivate_old_tc(self, tstamp):
      '''
            '''
      if mode == 'analysis':
        for entry in self.candidates:
          if (tstamp -
              self.candidates[entry].last_update) > self._region.rules['cyclosis_analysis']:
            self.candidates[entry].active = False
      elif mode == 'ensemble':
        for entry in self.candidates:
          if (tstamp -
              self.candidates[entry].last_update) > self._region.rules['cyclosis_ensemble']:
            self.candidates[entry].active = False

    def single_timestep(self, tstamp):

      def cyclone(self, tstamp):
        '''A cyclone is a synoptic low-level pressure area. This function looks at the regional pressure dataset and locates the local pressure minima through use of a scipy minimum filter. Filter size is defined in regional rules, default is 7.5 degrees / spatial resolution. Only valid for a 2D array (time element is provided in a for loop of the function) Returns a list of global coords for pressure minima.
                REQUIRES: current timestamp (int)
                RETURN: pressure_minima (list) 
                '''
        neighborhood_size = self._region.rules['pressure_neighbourhood']

        #Run a minimum filter on a 2D array
        filtered_pressure_min = minimum_filter(
            self.pressure.values[tstamp], neighborhood_size, mode='nearest')
        #Create bool array of filter output onto data
        minima = (self.pressure.values[tstamp] == filtered_pressure_min)
        #Return y,x arrays where bool == True
        y, x = np.where(minima)
        #convert lats, lons to global and store in list of tuples
        pressure_minima = [_to_global(self._region, coord) for coord in list(zip(y, x))]

        #HACK: typhoons keep on appearing on the boundaries
        def remove_boundary_values(pressure_minima):
          for coord in pressure_minima:
            if coord[0] == self._region.bounds[2] or coord[0] == self._region.bounds[3]:
              pressure_minima.remove(coord)
            elif coord[1] == self._region.bounds[0] or coord[1] == self._region.bounds[1]:
              pressure_minima.remove(coord)
          return pressure_minima

        pressure_minima = remove_boundary_values(pressure_minima)
        return pressure_minima

      def tropical_cyclone(self, tstamp, pressure_minima):
        ''' 
                From the list of pressure minimas for the current timestamp, the function makes a check for several criteria defined in Region class if the pressure minima exhibits characteristics of a tropical_cyclone. 
                REQUIRES: pressure_minima (list of tuples)
                OUTPUT: list of tropical cyclone candidates (tc_candidates)
                '''
        #link rules dict for easier typing
        rules = self._region.rules

        def pressure_criteria(tstamp, coord):
          ''' Determines if cyclone has a sufficiently low pressure '''

          reg_idx = _to_regional(self._region, coord)
          pressure = self.pressure.values[tstamp, reg_idx[0], reg_idx[1]]
          if pressure < (1015 * 100):
            return coord

        def vmax_criteria(tstamp, coord):
          ''' Determines if cyclone has sufficient vmax in a search area around cyclone center '''
          yslice, xslice, _ = self.vmax.box_slice(coord, rules['vmax_radius'])
          vmax_area = self.vmax.values[tstamp, yslice, xslice]
          if np.any(vmax_area >= rules['vmax_thresh']):
            return coord

        def tmp_criteria(tstamp, coord):
          ''' Determines if cylone has the correct temperature anomaly structure 
                        ie. Temperature anomaly at 850hPa > 0 '''
          r_out = rules['core_outer']
          r_in = rules['core_inner']
          yslice, xslice, _ = self.tmp.box_slice(coord, r_out)
          #create levels !!! check that index is correct
          if self.mode == 'analysis':
            tmp_700 = self.tmp.values[tstamp, yslice, xslice, 3]  #ens =2, ans = 3, lan=2
            tmp_500 = self.tmp.values[tstamp, yslice, xslice, 2]  #ens = 1, ans = 2, lan=1
            tmp_300 = self.tmp.values[tstamp, yslice, xslice, 1]  # ens =0, ans = 1, lan=0
          if self.mode == 'ensemble':
            tmp_700 = self.tmp.values[tstamp, yslice, xslice, 2]  #ens =2, ans = 3, lan=2
            tmp_500 = self.tmp.values[tstamp, yslice, xslice, 1]  #ens = 1, ans = 2, lan=1
            tmp_300 = self.tmp.values[tstamp, yslice, xslice, 0]  # ens =0, ans = 1, lan=0

          #create inner slice
          inner_slice = slice(int(r_out - r_in), int(r_out + r_in))
          #create shape of array for calculating mean
          outer_shape = np.ones_like(tmp_700)
          outer_shape[inner_slice] = 0
          inner_shape = np.zeros_like(tmp_700)
          inner_shape[inner_slice] = 1
          #calculate anomaly from inner and outer core with ndimage.filer.mean
          anomaly700 = mean(tmp_700, inner_shape) - mean(tmp_700, outer_shape)
          anomaly500 = mean(tmp_500, inner_shape) - mean(tmp_500, outer_shape)
          anomaly300 = mean(tmp_300, inner_shape) - mean(tmp_300, outer_shape)

          if (anomaly700 + anomaly500 + anomaly300) > 0:
            return coord

        def vort_criteria(tstamp, coord):
          '''Function defines a box of specified radius around the center of the cyclone. In this box, an alogirithm from MetPy package calculates the vertical vorticity component from u and v wind vectors. As longitudal distance between cells dx is a function of latitude an additional algorithm calculates dx and dy for the latitudes in the search box and the mean dx, dy caluclated. 
                    '''

          def calc_dx_dy(longitude, latitude):
            ''' 
            This definition calculates the distance between grid points that are in
            a latitude/longitude format. Necessary for vorticity calculations as dx changes as a function of latitude.
            
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
                np.sqrt((np.sin(dlat / 2))**2), np.sqrt(1 - (np.sin(dlat / 2))**2))) * 6371000
            dy = np.ones((latitude.shape[0], longitude.shape[0])) * dy

            dx = np.empty((latitude.shape))
            dlon = np.abs(longitude[1] - longitude[0]) * np.pi / 180
            for i in range(latitude.shape[0]):
              a = (np.cos(latitude[i] * np.pi / 180) * np.cos(latitude[i] * np.pi / 180) *
                   np.sin(dlon / 2))**2
              c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
              dx[i] = c * 6371000
            dx = np.repeat(dx[:, np.newaxis], longitude.shape, axis=1)
            return dx, dy

          yslice, xslice, _ = self.vmax.box_slice(coord, npad=rules['vort_radius'])
          lats = self.lats[yslice]
          lons = self.lons[xslice]
          dx, dy = calc_dx_dy(lons, lats)
          dx = dx.mean()
          dy = dy.mean()
          if self.mode == 'analysis':
            u = self.u850.values[tstamp, yslice, xslice, 4]  # ens =1, ans =4, lan=0
            v = self.v850.values[tstamp, yslice, xslice, 4]  # ens =1, ans =4, lan=0
          if self.mode == 'ensemble':
            u = self.u850.values[tstamp, yslice, xslice, 0]  # ens =1, ans =4, lan=0
            v = self.v850.values[tstamp, yslice, xslice, 0]  # ens =1, ans =4, lan=0

          vort = vorticity(u, v, dx, dy).magnitude

          if np.any(vort > rules['vort_thresh']):
            return coord

        pressure_candidates, vmax_candidates, tmp_candidates, vort_candidates = [], [], [], []

        for coord in pressure_minima:
          pressure_candidates.append(pressure_criteria(tstamp, coord))
          vmax_candidates.append(vmax_criteria(tstamp, coord))
          tmp_candidates.append(tmp_criteria(tstamp, coord))
          vort_candidates.append(vort_criteria(tstamp, coord))

        # find duplicates in all lists
        tc_candidates = set(pressure_candidates).intersection(vmax_candidates, tmp_candidates,
                                                              vort_candidates)
        tc_candidates = filter(None, tc_candidates)
        return tc_candidates

      def _search_area(self, coord):
        '''Creates an array of coordinate pairs around a specified center. Used to create an array that can be counterreferenced to see if a new tc_candidate is within the movement range of an existing one. 
                REQUIRES: coordinate pair (coord)
                RETURNS: array of coord pairs in radius around coord
                '''
        if self.mode == 'analysis':
          _, _, coords = self.pressure.box_slice(coord, npad=self._region.rules['update_analysis'])
        elif self.mode == 'ensemble':
          _, _, coords = self.pressure.box_slice(coord, npad=self._region.rules['update_ensemble'])
        yy, xx = np.meshgrid(coords[0], coords[1])
        coord_array = np.stack((np.ravel(yy), np.ravel(xx)), axis=-1)
        return list(map(tuple, coord_array))

      def new_candidate(self, coord, tstamp):
        '''Function creates a new candidate with a random alphanumeric name
                REQUIRES: coordinate pair, time index
                RETURNS: dictionary entry of a TyphoonCandidate class object'''

        name = ''.join(random.choices(string.ascii_uppercase, k=4))
        region = self._region
        position = coord
        active = True
        detection = tstamp
        history = {detection: coord}
        last_update = detection
        search_area = _search_area(self, coord)
        color = np.random.rand(3,)

        self.candidates[name] = TyphoonCandidate(name, region, position, active, history,
                                                 search_area, detection, last_update, color)

      def update_candidate(self, entry, coord, tstamp):
        ''' Function updates relevant attributes of the Typhoon candidate. 
                '''
        self.candidates[entry].postition = coord
        self.candidates[entry].history[tstamp] = coord
        self.candidates[entry].search_area = _search_area(self, coord)
        self.candidates[entry].last_update = tstamp

      def iterate_candidates(self, coord, tstamp):
        '''This function iterates through the list of candidates in self.candidates. 
                If the the coordinate pair (from pressure_list) is found in the current search area of any of the candidates, the coord_pair is append to the history list. If not found, a new candidate is created.
                '''
        for entry in self.candidates:
          if self.candidates[entry].active == True:
            if coord in self.candidates[entry].search_area:
              update_candidate(self, entry, coord, tstamp)
              return
        new_candidate(self, coord, tstamp)

      pressure_minima = cyclone(self, tstamp)
      tc_candidates = tropical_cyclone(self, tstamp, pressure_minima)
      for coordpair in tc_candidates:
        iterate_candidates(self, coordpair, tstamp)

    self.candidates = {}
    _filler_candidate(self)

    for tstamp in range(len(self.times)):
      single_timestep(self, tstamp)
      deactivate_old_tc(self, tstamp)
    self.candidates.pop('XXXX0')
    _clean_up(self)


class Ensemble:

  def __init__(self, region, start, path=None, forecastdays=10):
    self._region = region
    self.start = dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    self._start_epochtime = int(self.start.replace(tzinfo=dt.timezone.utc).timestamp())
    self.forecastdays = forecastdays
    self.path = path

  def find_datasets(self):
    date = '\\' + dt.datetime.strftime(self.start, '%Y%m%dT%H') + '_'

    surface = self.path + '\\surface' + date
    wind = self.path + '\\wind' + date
    temp = self.path + '\\temp' + date

    ensemble = {
        'msl': Dataset(surface + 'msl.nc', 'r'),
        '10u': Dataset(surface + '10u.nc', 'r'),
        '10v': Dataset(surface + '10v.nc', 'r'),
        'TMP': Dataset(temp + 'TMP.nc', 'r'),
        'u': Dataset(wind + '10u.nc', 'r'),
        'v': Dataset(wind + '10v.nc', 'r')
    }

    self._dataset = ensemble
    self.ensembles = ensemble['msl']['ensemble'][:]
    self.times = ensemble['msl']['time'][:]

  def hijack_analysis(self):

    def hijack_parameters(self, ensemble):
      '''
            #FIXME: Currently, Analysis must already have run in order to initiate the Parameters. At the moment, this function only overwrite the values attribute, instead of creating a new Parameter object. 
            In the future ensembel should work automatically, and not require Analysis. '''

      def creating_boundary_slice():
        ''' THis function is necessary in case the ensemble dataset is not a global one. 
                Functions looks at the ensemble dataset, represented a netcdf file in init_dataset, and returns a slice.
                '''

        init_lats = self._dataset['msl']['latitude'][:]
        init_lons = self._dataset['msl']['longitude'][:]
        bounds = self._region.bounds  #lon0,lon1,lat1,lat0
        lon0 = np.where(bounds[0] == init_lons)[0][0]
        lon1 = np.where(bounds[1] == init_lons)[0][0]
        lat1 = np.where(bounds[2] == init_lats)[0][0]
        lat0 = np.where(bounds[3] == init_lats)[0][0]

        lon_slice = slice(lon0, lon1, 1)
        lat_slice = slice(lat1, lat0, 1)
        return init_lats, init_lons, lon_slice, lat_slice

      lats, lons, ix, iy = creating_boundary_slice()

      self.lats = lats
      self.lons = lons

      tslice = slice(0, self.forecastdays * 4 + 1, 1)

      #hijack times
      self._region.Analysis.times = self.times

      #hijack parameters
      self._region.Analysis.pressure.values = self._dataset['msl']['msl'][ensemble, tslice, iy, ix]
      self._region.Analysis.ugrd.values = self._dataset['10u']['10u'][ensemble, tslice, iy, ix]
      self._region.Analysis.vgrd.values = self._dataset['10v']['10v'][ensemble, tslice, iy, ix]
      self._region.Analysis.vmax.values = self._dataset['10v']['10v'][ensemble, tslice, iy, ix]
      self._region.Analysis.vmax.values = np.sqrt(self._region.Analysis.vgrd.values**2 + \
                                                  self._region.Analysis.ugrd.values**2)

      self._region.Analysis.tmp.values = self._dataset['TMP']['TMP'][ensemble, tslice, iy, ix][:]
      self._region.Analysis.u850.values = self._dataset['u']['10u'][ensemble, tslice, iy, ix][:]
      self._region.Analysis.v850.values = self._dataset['v']['10v'][ensemble, tslice, iy, ix][:]

    def hijack_cyclonefinder(self, ensemble):
      self._region.Analysis.pressure_candidates(mode='ensemble')
      self.EnsembleCandidates['Ensemble ' + str(ensemble)] = self._region.Analysis.candidates

    self.EnsembleCandidates = {}
    for ensemble in self.ensembles:

      hijack_parameters(self, ensemble)
      hijack_cyclonefinder(self, ensemble)

  def ensemble_plotting(self,
                        ensemble='all',
                        timestamps='all',
                        ensemble_dict='default',
                        besttrack=None,
                        annotate=False):
    '''
        '''
    #initialising boundary
    fig = plt.figure(figsize=(30, 16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, color='black', alpha=0.2, linestyle='--')
    ax.set_extent(self._region.bounds)

    #define dictionary
    if ensemble_dict != 'default':
      dictionary = ensemble_dict
    else:
      dictionary = self.EnsembleCandidates

    #define timestamps
    if timestamps != 'all':
      print(
          'Please enter a timestamp index for your starting point: (timestamp inteval = {})'.format(
              self._region._globe.ts_ensemble / 3600))
      start = int(input())
      print(
          'Please enter a timestamp index for your stopping point: (timestamp inteval = {})'.format(
              self._region._globe.ts_ensemble / 3600))
      end = int(input())
    else:
      start = 0
      end = len(self.times)

    #define members:
    if ensemble != 'all':
      print(
          'Please select which ensemble members you would like to view (Multiple entries seperated by comma & no spaces!'
      )
      input_string = input()
      members = input_string.split(',')
      members = ['Ensemble ' + str(n) for n in members]
    else:
      members = []
      for member in self.ensembles:
        members.append('Ensemble ' + str(member))

    if dictionary == 'default':
      for ensemble in members:
        for cyclone in dictionary[ensemble]:
          x_list = []
          y_list = []
          for tidx in dictionary[ensemble][cyclone].history:
            if tidx >= start and tidx <= end:
              coord = dictionary[ensemble][cyclone].history[tidx]
              ax.plot(coord[1], coord[0], marker='.', color=dictionary[ensemble][cyclone].color)
              if annotate == True:
                ax.annotate(
                    tidx,
                    xy=(coord[1], coord[0]),
                    xycoords='data',
                    xytext=(coord[1] + 0.5, coord[0] + 0.5))
              x_list.append(coord[1])
              y_list.append(coord[0])
          ax.plot(x_list, y_list, linewidth=1, color=dictionary[ensemble][cyclone].color)

    else:
      for cyclone in members:
        x_list = []
        y_list = []
        for tidx in dictionary[cyclone].history:
          if tidx >= start and tidx <= end:
            coord = dictionary[cyclone].history[tidx]
            ax.plot(coord[1], coord[0], marker='.', color=dictionary[cyclone].color)
            if annotate == True:
              ax.annotate(
                  tidx,
                  xy=(coord[1], coord[0]),
                  xycoords='data',
                  xytext=(coord[1] + 0.5, coord[0] + 0.5))
            x_list.append(coord[1])
            y_list.append(coord[0])
        ax.plot(x_list, y_list, linewidth=1, color=dictionary[cyclone].color)

    if besttrack != None:
      x_list = []
      y_list = []
      besttrack = self._region.besttracks[besttrack]
      for tidx in besttrack.history:
        if tidx >= start and tidx <= end:
          coord = besttrack.history[tidx]
          ax.plot(coord[1], coord[0], marker='.', color=besttrack.color)
          if annotate == True:
            ax.annotate(
                tidx,
                xy=(coord[1], coord[0]),
                xycoords='data',
                xytext=(coord[1] + 0.5, coord[0] + 0.5))
          x_list.append(coord[1])
          y_list.append(coord[0])
      ax.plot(x_list, y_list, linewidth=3, color=besttrack.color)

    plt.draw()
    plt.show()


class Parameter:

  def __init__(self, dataset, abbr, region, it, iy, ix, il=None, plot=None):
    ''' 
        Creates a class called Paramter which serves as a storage unit for paramater values and functions. Usually      created from a larger dataset and thus requires slicing parameters for efficiency. Assumes that dictionary      key name and netCDF variable name are identical. If dataset has levels, il (ilevels) must be specified as       slice.  
        REQUIRES: dataset(library), abbreviation(string), region (class obj), it(slice), ix (slice), iy (slice), 
                  il (slice)*
        RETURN: values (n-dimensional array), region(obj reference)        
        '''
    if il == None:
      self.values = dataset[abbr][abbr][it, iy, ix]
    else:
      self.values = dataset[abbr][abbr][it, iy, ix, il]

    self.region = region
    self.plot = plot

  def box_slice(self, coord, npad):
    '''
        Creates a bounding box around a coordinate tuple ie.(40.5,160) and returns a slice in longitudinal and lateral direction as well as a list of all coordinate tuples within the box. The size of the bounding box can be defined by the npad keyboard. Bounding boxes can currently only be square and npad is halt the side of the box. Box will always have an odd number side length to preverse symmetry around box center. Function checks to ensure that box does not extend beyond region domain.  
        REQUIRES: coord (tuple), npad (int)
        RETURNS: yslice (slice), xslice (slice), coord ()
        '''
    # convert global coords to a regional index
    regional_idx = _to_regional(self.region, coord)

    y0 = regional_idx[0] - npad
    y1 = regional_idx[0] + npad + 1
    x0 = regional_idx[1] - npad
    x1 = regional_idx[1] + npad + 1

    # ensures that bounding box does not extend out beyond domain borders
    if y0 < 0:
      y0 = 0
    if y1 > self.values.shape[1]:
      y1 = self.values.shape[1]
    if x0 < 0:
      x0 = 0
    if x1 > self.values.shape[2]:
      x1 = self.values.shape[2]

    yslice = slice(int(y0), int(y1))
    xslice = slice(int(x0), int(x1))
    # list of all coordinates within the bounding box
    box_coords = [self.region.lats[yslice], self.region.lons[xslice]]

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


class TyphoonCandidate:

  def __init__(self, name, region, position, active, history, search_area, detection, last_update,
               color):
    self.name = name
    self.region = region
    self.position = position
    self.active = active
    self.history = history
    self.search_area = search_area
    self.detection = detection
    self.last_update = last_update
    self.color = color

  def plot(self):
    #initialising
    fig, ax = plt.subplots(figsize=(16, 8))
    pc = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, color='black', alpha=0.2, linestyle='--')
    ax.set_extent(self.region.bounds)

    x_list = []
    y_list = []
    for coord in self.history.values():
      ax.plot(coord[1], coord[0], marker='.', color=self.color)
      x_list.append(coord[1])
      y_list.append(coord[0])
    ax.plot(x_list, y_list, linewidth=1)
    ax.annotate(
        self.name,
        xy=(coord[1], coord[0]),
        xycoords='data',
        xytext=(coord[1] + 1, coord[0] + 1),
        arrowprops=dict(arrowstyle="->"))

  def __repr__(self):
    return "TyphoonCandidate: \nname: {}, \nposition: {}, \nlength: {}, \ndetection: {}, \nlast update: {} \n".format(
        self.name, self.position, len(self.history), self.detection, self.last_update)

  #    def check_conditions:
  #        pass
  #        def temp_anomaly_condition
  #            pass
  #        def vorticity_condition
  #            pass
  #        def vmax_condition
  #            pass


class Plotting:

  def __init__(self, name, cmap, bounds, ncolors=256):
    self.name = name
    self.cmap = cmap
    self.bounds = bounds
    self.ncolors = ncolors
    self.norm = colors.BoundaryNorm(boundaries=self.bounds, ncolors=self.ncolors)


###############################################################################################################
################################################### MAIN ######################################################
###############################################################################################################
earth = Globe(
    ensemblepath=r'C:\user\typhoon tracking\data\gfs\ensemble\20170908T12_msl.nc',
    analysispath=r'C:\user\typhoon tracking\data\gfs\analysis-levels\gfsanl\PRMSL.nc')
se_asia = earth.region('South-East Asia', (40, 90), (0, 160))
se_asia.Analysis('2017-10-16 00:00:00', '2017-10-24 00:00:00')
se_asia.Analysis.pressure_candidates()

###################################################ENSEMBLE TRACKS##############################################
se_asia.Ensemble(
    '2017-10-16 00:00:00', path='C:\\user\\typhoon tracking\\data\\ecmwf\\ensemble\\lan')
se_asia.Ensemble.find_datasets()
se_asia.Ensemble.hijack_analysis()


#############################
### BEST TRACK CREATION #####
#############################
def best_track(name, filepath, start, mode='interpolate', mode2='analysis'):

  def create_history(filepath, start, mode='interpolate', mode2='analysis'):
    '''Reads a JTWC besttrack .dat file and creates a history of the location and time
        that can be plotted by the tracking algorithm. 
        REQUIRES:   filepath to a .dat JTWC file (string)
                    start-date reference for creating the date index 
        RETURNS:    dicionarty of time_idx : (y,x)
        '''
    df = pd.read_csv(filepath, header=None)
    df = df[[2, 6, 7, 8, 9]]
    df.columns = ['Date', 'lats', 'lons', 'vmax', 'mslp']
    df.head()

    #lats,lons & time conversion
    def coord_change(coord):
      coord = float(coord[:-1]) / 10
      #coord = round(coord * 2.0) / 2.0
      return coord

    def datetime_change(date):
      date = dt.datetime.strptime(str(date), '%Y%m%d%H')
      return date

    #conversion to time_index
    def datetime_to_idx(datetime, start):
      step = 6 * 3600
      start = dt.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
      #temp
      #datetime = dt.datetime.strptime(datetime, '%Y-%m-%d %H:%M:%S')
      start_epochtime = int(start.replace(tzinfo=dt.timezone.utc).timestamp())
      #loop
      epoch_datetime = int(datetime.replace(tzinfo=dt.timezone.utc).timestamp())
      idx = (epoch_datetime - start_epochtime) / step
      return int(idx)

    def history_dict():
      history = {}
      for idx in range(len(df)):
        history[df.Date_IDX[idx]] = (df.lats[idx], df.lons[idx])
      return history

    df['lats'] = df['lats'].map(lambda a: coord_change(a))
    df['lons'] = df['lons'].map(lambda a: coord_change(a))

    df = df.drop_duplicates('Date')
    df['Date'] = df['Date'].map(lambda a: datetime_change(a))
    df['Date_IDX'] = df['Date'].map(lambda a: datetime_to_idx(a, start))

    #create history dictionary
    df = df[df.Date_IDX >= 0]
    df = df.reset_index()  #reset index numbers after removing duplicate rows
    history = history_dict()

    ######## OPTIONAL ########
    def interpolate_history(history):
      ''' Does a simple interpolation of the data in the history dictionary. Used for a visual comparision of 
            best track 
            '''
      history_odd = {}
      for key in history:
        if key == max(history):
          return history_odd
        y = history[key][0] + (history[key + 2][0] - history[key][0])
        x = history[key][1] + (history[key + 2][1] - history[key][1])
        odd_key = key + 1
        history_odd[odd_key] = (y, x)

    if mode == 'interpolate':
      history_odd = interpolate_history(history)
      history.update(history_odd)
      #create an ordered dictionary
      from collections import OrderedDict
      history = OrderedDict(sorted(history.items(), key=lambda history: history[0]))
      return history
    else:
      return history

  def create_besttrack(name, history):
    ''' Uses a history file to create a candidate dictionary entry for the best-track TC. 
            REQEQUIRES: name (string), history (dict)
            OUTPUT: candidate ( class obj)
            '''
    name = name
    region = se_asia
    position = max(history)
    active = False
    history = history
    search_area = None
    detection = 0
    last_update = (len(history) - 1) * 2
    color = 'k'

    return TyphoonCandidate(name, region, position, active, history, search_area, detection,
                            last_update, color)

  history = create_history(filepath, start, mode, mode2)
  se_asia.besttracks[name] = create_besttrack(name, history)
  #if mode2 == 'analysis':
  #    se_asia.Analysis.candidates[name] = create_besttrack(name, history)

  #if mode2 == 'ensemble':
  #    se_asia.Ensemble.EnsembleCandidates[name] = {}
  #    se_asia.Ensemble.EnsembleCandidates[name][name] = create_besttrack(name, history)


def remove_odd_tstamps(name):
  '''Removes odd tstamps in the list for accurate statistical comparision between deterministic and observed track
    '''
  history = se_asia.Analysis.candidates[name].history
  odd_history = []
  for key in history:
    if key % 2 != 0:
      odd_history.append(key)
  for key in odd_history:
    history.pop(key)
  se_asia.Analysis.candidates[name].history = history


def extract_TC_ensemble(common_start):

  def create_common_dict(common_start):
    main_dict = se_asia.Ensemble.EnsembleCandidates
    common_dict = {}
    for ensemble in main_dict:
      for cyclone in main_dict[ensemble]:
        if main_dict[ensemble][cyclone].detection == 0:
          if main_dict[ensemble][cyclone].history[0] == common_start:
            common_dict[ensemble] = main_dict[ensemble][cyclone]
    return common_dict

  def plot_common_dict(common_dict):

    #initialising
    fig = plt.figure(figsize=(30, 16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, color='black', alpha=0.2, linestyle='--')
    ax.set_extent(self.bounds)

    #ensemble
    for cyclone in common_dict:
      x_list = []
      y_list = []
      for coord in common_dict[cyclone].history.values():
        ax.plot(coord[1], coord[0], marker='.', color=common_dict[cyclone].color)
        x_list.append(coord[1])
        y_list.append(coord[0])
      ax.plot(x_list, y_list, linewidth=1, color=common_dict[cyclone].color)
      #ax.annotate(entry,xy=(coord[1],coord[0]), xycoords='data', xytext=(coord[1]+1,coord[0]+1), arrowprops=dict(arrowstyle="->"))
    plt.draw()
    plt.show()

  common_dict = create_common_dict(common_start)
  #plot_common_dict(common_dict)
  return common_dict


def add_missing_cyclones(dictionary, number_list):

  def create_cyclone_list(number_list):
    ens_list = []
    cyclone_list = []

    for number in number_list:
      ens_list.append('Ensemble ' + str(number))

    for entry in ens_list:
      length = 0
      longest_cyclone = ''
      for cyclone in se_asia.Ensemble.EnsembleCandidates[entry]:
        cyc_length = len(se_asia.Ensemble.EnsembleCandidates[entry][cyclone].history)
        if cyc_length > length:
          length = cyc_length
          longest_cyclone = cyclone
      cyclone_list.append(longest_cyclone)
    return cyclone_list

  def add_cyclones(cyclone_list, number_list, dictionary):
    idx = range(len(number_list))
    for idx in idx:
      ensemble = 'Ensemble ' + str(number_list[idx])
      cyclone = cyclone_list[idx]
      new_member = se_asia.Ensemble.EnsembleCandidates[ensemble][cyclone]
      dictionary[ensemble] = new_member
    return dictionary

  cyclone_list = create_cyclone_list(number_list)
  dictionary = add_cyclones(cyclone_list, number_list, dictionary)

  return dictionary


##############################
###### ERROR STATISTICS ######
##############################


def error_statistics(besttrack, track_dictionary, simtrack=None, tidx=None, mode='track'):
  '''Calculates the error according to one of two modes. 
    
        if mode set to 'track', error will be calculated for all pairs the observed and simulated track        
            REQUIRES: besttrack history (besttrack), deterministic history (simtrack)
            RETURN: error statistics for entire track
        
        if mode set to 'tidx', the error will be calculated for the error along a single time stamp, 
        commonly used for ensemble statistics        
            REQUIRES: best track history (bestrack), track dictionary (track_dictioanry, time stamp (tidx)
            RETURNS: error statistics for entire time stamp
    '''
  besttrack = se_asia.besttracks[besttrack]
  if mode == 'track':
    simtrack = track_dictionary[simtrack]

  def error_distance(obs, sim):
    '''Calculates the error distance from the observed pair of coordinates and the simulated pair.
            REQUIRES: observed coordinate pair (obs) {tuple}, simulation coordinate pair (sim) {tuple}
            RETURN: error {float}
        '''
    error = np.sqrt((obs[0] - sim[0])**2 + (obs[1] - sim[1])**2)
    return error

  def track_error(besttrack, simtrack):
    '''
        '''
    error_list = []
    for tidx in simtrack:
      error = error_distance(besttrack[tidx], simtrack[tidx])
      error_list.append(error)
    return error_list

  def tidx_error(bestrack, track_dictionary, tidx):
    '''
        '''
    error_list = []
    obs = besttrack.history[tidx]

    for cyclone in natsorted(track_dictionary):  #HACK: sorting a dictionary
      try:
        sim = track_dictionary[cyclone].history[tidx]
        error = error_distance(obs, sim)
        error_list.append(error)
      except KeyError:
        #print(cyclone,' does not have an entry for timestamp ',tidx)
        error_list.append(np.nan)
        continue

    return error_list

  def dataframe_statistics(error_list):
    '''
        '''
    df_error = pd.Series(error_list)
    return df_error.describe(), df_error

  if mode == 'track':
    error_list = track_error(besttrack, simtrack)
    error_statistics = dataframe_statistics(error_list)
    return error_statistics, error_list
  elif mode == 'ensemble':
    error_list = tidx_error(besttrack, track_dictionary, tidx)
    error_summary, error_series = dataframe_statistics(error_list)
    return error_summary, error_series


def ensemble_plotting(ensemble='all',
                      timestamps='all',
                      ensemble_dict='default',
                      besttrack=None,
                      annotate=False):
  '''
    '''
  self = se_asia.Ensemble
  #initialising boundary
  fig = plt.figure(figsize=(30, 16))
  ax = plt.axes(projection=ccrs.PlateCarree())
  ax.coastlines()
  ax.gridlines(draw_labels=True, color='black', alpha=0.2, linestyle='--')
  ax.set_extent(self._region.bounds)

  #define dictionary
  if ensemble_dict != 'default':
    dictionary = ensemble_dict
  else:
    dictionary = self.EnsembleCandidates

  #define timestamps
  if timestamps != 'all':
    print('Please enter a timestamp index for your starting point: (timestamp inteval = {})'.format(
        self._region._globe.ts_ensemble / 3600))
    start = int(input())
    print('Please enter a timestamp index for your stopping point: (timestamp inteval = {})'.format(
        self._region._globe.ts_ensemble / 3600))
    end = int(input())
  else:
    start = 0
    end = len(self.times)

  #define members:
  if ensemble != 'all':
    print(
        'Please select which ensemble members you would like to view (Multiple entries seperated by comma & no spaces!'
    )
    input_string = input()
    members = input_string.split(',')
    members = ['Ensemble ' + str(n) for n in members]
  else:
    members = []
    for member in self.ensembles:
      members.append('Ensemble ' + str(member))

  if ensemble_dict == 'default':
    for ensemble in members:
      for cyclone in dictionary[ensemble]:
        x_list = []
        y_list = []
        for tidx in dictionary[ensemble][cyclone].history:
          if tidx >= start and tidx <= end:
            coord = dictionary[ensemble][cyclone].history[tidx]
            ax.plot(coord[1], coord[0], marker='.', color=dictionary[ensemble][cyclone].color)
            if annotate == True:
              ax.annotate(
                  tidx,
                  xy=(coord[1], coord[0]),
                  xycoords='data',
                  xytext=(coord[1] + 0.5, coord[0] + 0.5))
            x_list.append(coord[1])
            y_list.append(coord[0])
        ax.plot(x_list, y_list, linewidth=1, color=dictionary[ensemble][cyclone].color)

  else:
    for cyclone in members:
      x_list = []
      y_list = []
      for tidx in dictionary[cyclone].history:
        if tidx >= start and tidx <= end:
          coord = dictionary[cyclone].history[tidx]
          ax.plot(coord[1], coord[0], marker='.', color=dictionary[cyclone].color)
          if annotate == True:
            ax.annotate(
                tidx,
                xy=(coord[1], coord[0]),
                xycoords='data',
                xytext=(coord[1] + 0.5, coord[0] + 0.5))
          x_list.append(coord[1])
          y_list.append(coord[0])
      ax.plot(x_list, y_list, linewidth=1, color=dictionary[cyclone].color)

  if besttrack != None:
    x_list = []
    y_list = []
    besttrack = self._region.besttracks[besttrack]
    for tidx in besttrack.history:
      if tidx >= start and tidx <= end:
        coord = besttrack.history[tidx]
        ax.plot(coord[1], coord[0], marker='.', color=besttrack.color)
        if annotate == True:
          ax.annotate(
              tidx,
              xy=(coord[1], coord[0]),
              xycoords='data',
              xytext=(coord[1] + 0.5, coord[0] + 0.5))
        x_list.append(coord[1])
        y_list.append(coord[0])
    ax.plot(x_list, y_list, linewidth=3, color=besttrack.color)

  x_list = []
  y_list = []
  counter = 0
  for coord in mean_track:
    ax.plot(coord[1], coord[0], marker='.', color='r')
    x_list.append(coord[1])
    y_list.append(coord[0])
    #coord = (round(coord[1]), round(coord[0]))
    #ax.annotate(counter, xy=(coord[1],coord[0]), xycoords='data', xytext=(coord[1]+0.5,coord[0]+0.5))
    counter += 1
  ax.plot(x_list, y_list, linewidth=3, color='maroon')

  plt.draw()
  plt.show()
