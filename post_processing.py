from collections import defaultdict
import random
import copy
import string
import numpy as np
#plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation

from detection_tracking import candidates


def isolate_systems(master_dict, finesse=5, degs=0.5):
    '''
    Analyses the cloud of ensemble tracks and attributes them to number of independant systems.
    Assumes that an ensemble will not have more than 10 individual tracks associated to it.

    Returns
    -------
    dict
        A dictionary with one or more entries. The dictionary keys are random alphabetical sequences
        (ie. SYNL) and contain a list of all Ensemble tracks associated with the system
    '''

    #create list of all ensembles and TCs
    master_list = []
    for ens in master_dict:
        for tc in master_dict[ens]:
            master_list.append(ens + '-' + str(tc))

    #initiase other lists
    unassigned_tracks = copy.deepcopy(master_list)
    assigned_tracks = []
    cyclones = defaultdict(dict)

    def create_cyclone():  #cyclones, unassigned_tracks):
        '''
        Randomly picks a track from the unassigned tracks list and assigns this track to a new
        system in the cyclones dictionary. Removes the entry from the unassigned_list and adds it
        to the assigned_list.
        '''
        #create random system name and select random entry from unassigned tracks
        system_name = ''.join(random.choices(string.ascii_uppercase, k=4))
        rand_tc = random.choice(unassigned_tracks)

        #create system and append/ remove random track from lists
        cyclones[system_name] = []
        unassigned_tracks.remove(rand_tc)
        assigned_tracks.append(rand_tc)
        cyclones[system_name].append(rand_tc)

    def search_lats_lons(coord, radius, degs):
        '''
        Function that creates two sequences of lats and lons in a radius around a specified
        coordinate. Used for proximity comparisions.

        Parameters
        ----------
        coord : tuple
            A tuple of global coordinates in the form (lat, lon)
        radius : int
            The radius in degrees for which the lats and lons sequence should be generated
        degs : float
            The resolution of the spatial domain. Radius is divided by degs to obtain the radius
            in number of grid cells.

        Returns
        -------
        nested list
            Returns a list with two numpy arrays: lats sequence and lons sequence
        '''
        lats = np.arange(coord[0] - radius, coord[0] + radius, degs)
        lons = np.arange(coord[1] - radius, coord[1] + radius, degs)
        return [lats, lons]

    def detect_adjacent(system, track, master_dict, finesse):
        '''
        Retraces the positions of a specified track and checks at every time step for proximity
        to other tracks. If adjacent tracks are found, they are added to the cyclone system
        dictionary of the track.These sequence is iterated through all (growing) members of the
        cyclone system until exhausted.

        Parameters
        ----------
        system : string
            The dictionary key name of the cyclone system
        track : string
            The string name of the selected track to check adjacency for. In the following 
            format: 'Ensemble n-k'
        master_dict : dictionary
            The result of detection_tracking.py which is a dictionary of all tracks and
            Ensembles as well as relevant metadata
        '''

        #Assumption: Ensemble will not consist of more than 10 tracks (highly unlikely)
        ensemble_num = track[:-2]
        track_num = int(track[-1])

        for tstamp in master_dict[ensemble_num][track_num].history:
            coord = master_dict[ensemble_num][track_num].history[tstamp]['pos']
            latlon = search_lats_lons(coord, finesse, degs)

            for member in unassigned_tracks:
                ens = member[:-2]
                tc = int(member[-1])
                if (tstamp in master_dict[ens][tc].history and master_dict[ens]
                    [tc].history[tstamp]['pos'][0] in latlon[0]
                        and master_dict[ens][tc].history[tstamp]['pos'][1] in
                        latlon[1]):
                    entry = ens + '-' + str(tc)
                    cyclones[system].append(entry)
                    assigned_tracks.append(entry)
                    unassigned_tracks.remove(entry)

    def iterate_cyclones():
        create_cyclone()
        for system in list(cyclones.keys()):
            for track in cyclones[system]:
                detect_adjacent(system, track, candidates, finesse)

    while len(unassigned_tracks) != 0:
        iterate_cyclones()

    return cyclones


#Plotting
def ensemble_plotting(master_dict,
                      tc_list,
                      ensemble='all',
                      timestamps='all',
                      annotate=False):
    '''
    '''
    region = Region['se_asia']
    #initialising boundary
    fig = plt.figure(figsize=(30, 16))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, color='black', alpha=0.2, linestyle='--')
    bbox = [region.bbox[0], region.bbox[2], region.bbox[1], region.bbox[3]]
    ax.set_extent(bbox)

    #define dictionary

    #define timestamps
    if timestamps != 'all':
        print(
            'Please enter a timestamp index for your starting point: (timestamp inteval = {})'
            .format(region.ts / 3600))
        start = int(input())
        print(
            'Please enter a timestamp index for your stopping point: (timestamp inteval = {})'
            .format(region.ts / 3600))
        end = int(input())
    else:
        start = 0
        end = len(region.Tracking.times)

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
        for member in region.Tracking.ensembles:
            members.append('Ensemble ' + str(member))

    for track in tc_list:
        ensemble = track[:-2]
        tc = int(track[-1])

        x_list = []
        y_list = []
        for tidx in dictionary[ensemble][tc].history:
            if tidx >= start and tidx <= end:
                coord = dictionary[ensemble][tc].history[tidx]['track']
                ax.plot(coord[1],
                        coord[0],
                        marker='.',
                        color=dictionary[ensemble][tc].color)
                if annotate == True:
                    ax.annotate(tidx,
                                xy=(coord[1], coord[0]),
                                xycoords='data',
                                xytext=(coord[1] + 0.5, coord[0] + 0.5))
                x_list.append(coord[1])
                y_list.append(coord[0])
        ax.plot(x_list,
                y_list,
                linewidth=1,
                color=dictionary[ensemble][tc].color)

    plt.draw()
    plt.show()


cyclones = isolate_systems(candidates)

print(cyclones)