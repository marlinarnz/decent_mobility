import os
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime
import osmnx as ox
import networkx as nx
import gtfs_kit as gk
import uuid
from tqdm import tqdm
from shapely import geometry, Point, distance


from quetzal.model import stepmodel
from quetzal.engine import engine
from quetzal.engine.pathfinder import PublicPathFinder
from quetzal.io.gtfs_reader import importer
from quetzal.analysis import analysis
from quetzal.engine.msa_utils import default_bpr
from syspy.transitfeed import feed_links


def biking_speed(wq, bike_type):
    '''
    map way_quality variable from each link type to speed (depending on type of bicycle)
    Data from: doi.org/10.1016/j.ssci.2015.07.027
    '''
    #wq = link.wayquality
    if bike_type == 'normal':
        if wq == 'designated':
            return 16.7
        elif wq == 'with_foot':
            return 13.3
        elif wq in ['on_street','primary','main_road','small_road']:
            return 16.4
        elif wq == 'path':
            return 13.7
        else:
            return 9.9    
    elif bike_type == 'pedelec':
        if wq == 'designated':
            return 18.4
        elif wq == 'with_foot':
            return 13.9
        elif wq in ['on_street','primary','main_road','small_road']:
            return 18.8
        elif wq == 'path':
            return 14.5
        else:
            return 9.4
    elif bike_type == 's-pedelec':
        if wq == 'designated':
            return 23.6
        elif wq == 'with_foot':
            return 17.6
        elif wq in ['on_street','primary','main_road','small_road']:
            return 25.6
        elif wq == 'path':
            return 16.4
        else:
            return 14.2


def driving_speed(vmax,rt ):
    '''
    Determine "average free-flow" driving speed depending on speed limit and road type
    '''

    #Motorway Data from https://www.umweltbundesamt.de/publikationen/klimaschutz-durch-tempolimit
    if rt in ['motorway','trunk']:
        if vmax > 130:
            return 124.7
        elif vmax > 120:
            return 118.3
        elif vmax > 100:
            return 115.6
        elif vmax > 80:
            return 103.3
        elif vmax > 60:
            return 87.4
        else: 
            return vmax*1.2
    else:
        #estimated
        return vmax*0.8


class RegionalModel(stepmodel.StepModel):
    """
    A transport network model for a region of choice, including all relevant
    transport network types and corresponding pathfinding.
    """

    def __init__(self, polygon, buffer_size, timeseg = 5, *args, **kwargs):
        """
        Initializes a new RegionalModel for a study region with buffer size.

        Parameters:
        ----------
        polygon : shapely.geometry.Polygon
            A geometric object representing the study region's boundaries.
        
        buffer_size : float
            The size of the buffer to be applied around the study region
            polygon. The unit of the buffer size depends on the coordinate
            system used by the model and the polygon object. You can specify
            the model's ccordinate system with the `epsg` attribute and the
            coordinate unit (meter or degrees) with `coordinates_unit`.
                       
        daily_timeseg: int or pd.DataFrame
            if int: bulids DataFrame self.timeseg_dict to contain the given number of equitemporal time slots
            if DataFrame: sets self.timeseg_dict as the given DataFrame. The given DataFrame must be in the format of self.timeseg_dict
        
        *args : tuple
            Additional positional arguments to be passed to the quetzal StepModel.
        
        **kwargs : dict
            Additional keyword arguments to be passed to the quetzal StepModel.
        
        Builds: 
        ----------
        self.timeseg_dict: pd.DataFrame
            DataFrame defining the time intervalls in which the analysis of the moddel will be performed. Time intervalls will be the same for every day.
            Each row corresponds to a time intervall. The columns 'starttime' and 'endtime' give the respective bounds of the time interval. 
            The index of each row defines the interval name.
        Notes:
        -----
        This method calls the constructor of the superclass to initialize the inherited 
        attributes before setting up the study region polygon and buffer size specific 
        to this instance.
        """
        super().__init__(*args, **kwargs)

        self.study_region_polygon = polygon
        self.buffer_size = buffer_size
        assert (self.coordinates_unit=='degree' and self.epsg==4326 and self.buffer_size<10) \
               or (self.coordinates_unit=='meter' and self.epsg!=4326), \
               'The coordinate reference system (attribute `epsg`) should be either geographic '\
               +'(coordinates in lat and lon, as in EPSG:4326) with the attribute `coordinates_unit="degree"`, '\
               +'or cartesian with coordinate units in meter (such as EPSG:25832 for most of Germany). '\
               +'The buffer size must match the corresponding coorinate unit.'

        # Total area to observe: inital polygon + buffer region
        # Must be saved in EPSG:4326
        # Switch to EPSG:4326 after setting the buffer area
        if self.epsg!=4326:
            geo = gpd.GeoSeries([self.study_region_polygon],
                                crs='EPSG:'+str(self.epsg))
            self.study_region_polygon = geo.to_crs('EPSG:4326')[0]
            self.total_area = geo.buffer(self.buffer_size).to_crs('EPSG:4326')[0]
            self.epsg = 4326
            self.coordinates_unit = 'degree'
            print('Converted the coordinate reference system to EPSG:4326.')
        else:
            self.total_area = self.study_region_polygon.buffer(self.buffer_size)

        # Initialize level-of-service tables
        cols = ['time_range']
        self.auto_los = pd.DataFrame(columns=cols)
        self.cycle_los = pd.DataFrame(columns=cols)
        self.walk_los = pd.DataFrame(columns=cols)
        self.pt_los = pd.DataFrame(columns=cols)
        self.pr_los = pd.DataFrame(columns=cols)
        self.pc_los = pd.DataFrame(columns=cols)

        # And geo-tables
        self.pt_links = gpd.GeoDataFrame()
        self.pt_nodes = gpd.GeoDataFrame()
        self.pt_footpaths = gpd.GeoDataFrame()
        self.zones = gpd.GeoDataFrame(columns=['geometry'], crs=4326)
        self.zone_to_road = gpd.GeoDataFrame()
        self.zone_to_transit = gpd.GeoDataFrame()

        if isinstance(timeseg, int):
            # Initialize parent classes
            # Build timesg_df: Distribution to five equally sized time segments per day
            timeseg_intervals = [a*((24*3600)//timeseg) for a in range(0,timeseg+1,1)]
            seg_labels = [a for a in range(1,timeseg+1,1)]
            timeseg_dict = pd.DataFrame(index  = seg_labels, data = {'interval':list(zip(timeseg_intervals[:-1],timeseg_intervals[1:]))})
            timeseg_dict['starttime'] = timeseg_dict.interval.map(lambda x: x[0])
            timeseg_dict['endtime'] = timeseg_dict.interval.map(lambda x: x[1])
            self.timeseg_dict = timeseg_dict.drop('interval',axis = 1)
        elif isinstance(timeseg, pd.DataFrame):
            assert set(timeseg.columns) == set(['starttime', 'endtime']), 'Wrong column names in daily_timeseg DataFrame'
            self.timeseg_dict = timeseg

    def generate_street_networks(self, path_to_graphml,
                                 car=True, bike=True, walk=True,
                                 cycle_speed=18, walking_speed=4,
                                 car_fallback_speed = {"motorway": 140, "trunk": 100, 
                                        "primary": 100, "secondary": 100, "tertiary": 80, "unclassified": 60,
                                        "motorway_link": 80, "trunk_link": 60, 
                                        "primary_link": 60, "secondary_link": 50, "tertiary_link": 30,
                                        "residential": 30, "rest_area":30, 'living_street': 10},
                                 bikeable=['designated', 'road_markings', 'small_road', 'path',
                                           'main_road', 'with_foot', 'primary'],
                                 walkable=['designated', 'on_street'],
                                 node_consolidation_tolerance = 15,
                                 simplify_strict = True,
                                 truncate_graph_strongly = True):
        """
        Generates street network for different modes of transport by 
        extracting road, cycling, and walking networks from a GraphML file.
        It uses the OSMNX package for efficient data extraction and then
        filters the networks for full coverage, but minimum size
        Loads Data from online sourrce using osmnx and saves graphml if given filepath is empty.

        Parameters:
        ----------
        path_to_graphml : str
            The file path to the GraphML file that contains the network data.
            This file can be downloaded from OpenStreetMap.
        car             : bool
            Generate the network for motorised individual travel from OSM data.
        bike             : bool
            Generate the network for cycling from OSM data.
        walk             : bool
            Generate the network for short-distance walking from OSM data.
        cycle_speed     : int, float or dict
            A fixed speed at all road types or a dict with specific values
            for each key in 'designated', 'road_markings', 'small_road',
            'path', 'main_road', 'with_foot'
        walking_speed   : int or float
            A fixed speed for walking
        car_fallback_speed: dict
            A dictionary translating OSM highway types into default speed limits
        cycleable:      : list of string
            A list of cycling path qualities to consider as cycling paths.
            Options: designated, main_road, with_foot, road_markings,
            small_road, primary, unuseable, prohibited, unknown
        walkable:      : list of string
            A list of walking path qualities to consider as walking paths.
            Options: designated, on_street, main_road, unuseable, prohibited,
            unknown
        truncate_graph_strongly: bool
            Whether to truncate each street network to the largest subgraph
            based on strong (default) or weak connection (would be a one-way
            connection)

        Attributes:
        ----------
        self.auto_links : geopandas.GeoDataFrame
            A GeoDataFrame containing the links (edges) for the car network
        self.auto_nodes : geopandas.GeoDataFrame
            A GeoDataFrame containing the nodes (vertices) for the car network

        self.cycle_links : geopandas.GeoDataFrame
            A GeoDataFrame containing the links (edges) for the cycle network
        self.cycle_nodes : geopandas.GeoDataFrame
            A GeoDataFrame containing the nodes (vertices) for the cycle network

        self.walk_links : geopandas.GeoDataFrame
            A GeoDataFrame containing the links (edges) for the walk network
        self.walk_nodes : geopandas.GeoDataFrame
            A GeoDataFrame containing the nodes (vertices) for the walk network

        """
        #attributes to save in graph
        ox.settings.useful_tags_way = ["oneway", "lanes",
                               "ref", "name", 
                               "highway", "maxspeed",
                               "service", "access",
                               "area", "junction",
                               "cycleway", "cycleway:both", "cycleway:left","cycleway:right",
                               "bicycle_road", "cyclestreet", "bicycle",
                               "surface","tracktype", 'foot', 
                               'sidewalk',"sidewalk:left","sidewalk:right","sidewalk:both",
                               'footway']
        #Load OSM network as osmnx-graph
        try:
            #load graph from file
            print('try loading from file')
            R = ox.load_graphml(path_to_graphml)
            print('graph loaded from file')
        except:
            #load graph from online source
            print('load graph from OpenStreetMap: Full detail for study region, only driving networks for buffer area')
            R_region = ox.graph_from_polygon(
                self.study_region_polygon, network_type="all",
                simplify=False, retain_all=False, truncate_by_edge=True)

            R_drive = ox.graph_from_polygon(
                self.total_area, network_type='drive',
                simplify=False, retain_all=False, truncate_by_edge=True)
            R = nx.compose_all([R_drive, R_region])
            ox.save_graphml(R, path_to_graphml)
            print('graph loaded from online source and saved')

        #get nodes and links from graph as Geopandas Dataframes
        nodes_osm, links_osm = ox.graph_to_gdfs(R)
        

        
        def fix_maxpeed(link):
            """ determine Value for maxspeed attribute for all links"""
            speed = link.maxspeed
            try:
                return int(speed)
            except:
                if speed in ['walk','DE:living_street','living_street']:
                    return 10
                elif link.junction == 'roundabout':
                    return 20
                elif link.highway in car_fallback_speed.keys():
                    return car_fallback_speed[link.highway]
                else:
                    return 50
        
        def fix_lanecount(link):
            """ Determine number of lanes for each link"""
            try:
                #lanenumber given as attribute
                return int(link.lanes)
            except:
                if (link.highway in ['motorway','trunk','primary','secondary','teriary']
                    and link.oneway != True):
                    #main roads without oneway traffic: one lane for each direction
                    return 2
                elif link.highway in ['motorway','trunk']:
                    #highway and trunk roads with oneway traffic: both directions mapped seperately, two lanes in each direction
                    return 2
                else:
                    #remaining: oneway on main roads, links, smaller/residential streets: only one lane
                    return 1
        
        def calc_road_capacity(link, min_capacity = 300, capacity_per_lane = {"motorway": 2000, "trunk": 2000, 
                                                                "primary": 1500, "secondary": 1000, "tertiary": 600, "unclassified": 600,
                                                                "motorway_link": 1500, "trunk_link": 1500, 
                                                                "primary_link": 1500, "secondary_link": 1000, "tertiary_link": 600,
                                                                "residential": 600, "rest_area":300, 'living_street': 300}):
            '''Determines the capacity for each road segment based on number of lanes per direction and road type
                Capacity estimations according to Jafari et al. 2022 (doi.org/10.1016/j.simpat.2021.102398)
            '''

            if link.oneway == True:
                num_lanes = int(link.lanes)
            else:# if link can be driven in both direction, only half the number of total lanes per direction
                num_lanes = int(0.5*link.lanes)
            if link.highway in capacity_per_lane.keys():
                return max([min_capacity, capacity_per_lane[link.highway]*num_lanes])
            else:
                return min_capacity
        
        print('fix maximum speed from OSM data')
        links_osm.maxspeed = links_osm.apply(fix_maxpeed, axis = 1)
            
        print('fix lane count from OSM data')
        links_osm.lanes = links_osm.apply(fix_lanecount, axis = 1)

        print('calculate road capacity')
        links_osm['capacity'] = links_osm.apply(calc_road_capacity, axis = 1)

        ####################
        # road network for cars
        ####################
        if car:
            print('build road network')
            print('simplifing car network')
            #simplify osmnx graph 
            AutoGraph = ox.simplify_graph(R,edge_attrs_differ=['maxspeed','lanes','highway'])
            AutoGraph = ox.truncate.largest_component(AutoGraph, strongly = truncate_graph_strongly)
            #get nodes and links from graph
            ant, alt = ox.graph_to_gdfs(AutoGraph)
            
            #filter only main & residential roads for car network
            links_auto_temp = alt[alt.highway.map(lambda x: x in car_fallback_speed.keys())].copy()
            nodes_auto_temp = ant[ant.index.map(lambda x: x in  alt.index)].copy()
           
            #AutoGraph_temp = ox.graph_from_gdfs(gdf_nodes = nodes_auto_temp, gdf_edges = links_auto_temp)
            #AutoGraph_temp = ox.truncate.largest_component(AutoGraph_temp, strongly = truncate_graph_strongly)
            #get links and nodes from graph
            #nodes_auto, links_auto = ox.graph_to_gdfs(AutoGraph_temp)
           
            print('fix maximum speed from OSM data')
            links_auto_temp.maxspeed = links_auto_temp.apply(fix_maxpeed, axis = 1)
                
            print('fix lane count from OSM data')
            links_auto_temp.lanes = links_auto_temp.apply(fix_lanecount, axis = 1)
            
            print('calculate road capacity')
            links_auto_temp['capacity'] = links_auto_temp.apply(calc_road_capacity, axis = 1)

            #drop unnecessary columns
            links_auto_temp = links_auto_temp.dropna(axis=1, how= 'any')
            links_auto_temp = links_auto_temp.drop(columns = ['osmid', 'reversed'])

            #Calculate time on link
            ########################
            ########################
            links_auto_temp['speed']= links_auto_temp.apply(lambda x: driving_speed(x.maxspeed, x.highway), axis = 1)
            links_auto_temp['length'] = links_auto_temp['length']
            links_auto_temp['time'] = links_auto_temp.apply(lambda x: x['length']/(x['speed'])*3.6, axis = 1) 
            

            agt = ox.graph_from_gdfs(gdf_nodes = nodes_auto_temp, gdf_edges = links_auto_temp)
            if simplify_strict:
                agt = ox.simplification.simplify_graph(agt,edge_attr_aggs = {'length': sum, 'time': sum,'capacity':min})
                agt = ox.truncate.largest_component(agt, strongly = truncate_graph_strongly)
                agt = ox.project_graph(agt, to_crs = 3035)
                agt = ox.simplification.consolidate_intersections(agt, tolerance = node_consolidation_tolerance)
                agt = ox.project_graph(agt, to_crs = 4326)
            else:
                agt = ox.truncate.largest_component(agt, strongly = truncate_graph_strongly)

            nodes_auto, links_auto = ox.graph_to_gdfs(agt)

            #set unique index for links and nodes:
            links_auto = links_auto.reset_index(drop= False)
            links_auto.u = links_auto.u.map(lambda x: 'node-car_'+f'{x:012d}')
            links_auto.v = links_auto.v.map(lambda x: 'node-car_'+f'{x:012d}')
            links_auto.index = links_auto.index.map(lambda x: 'link-car_'+f'{x:012d}')
            nodes_auto.index = nodes_auto.index.map(lambda x: 'node-car_'+f'{x:012d}')
            nodes_auto.index = nodes_auto.index.rename('index')
            links_auto = links_auto.rename(columns={'u':'a','v':'b'})
   

            #drop unnecessary columns
            links_auto = links_auto.dropna(axis=1, how= 'any')
            links_auto = links_auto.drop(columns = ['key','u_original','v_original'],
                                         errors = 'ignore')
            nodes_auto = nodes_auto[['geometry']]

            self.auto_links = links_auto.copy()
            self.auto_nodes = nodes_auto.copy()
            self.auto_links = self.auto_links.loc[~self.auto_links.duplicated(['a', 'b'])]
       
        ####################
        # cycling network
        ####################
        if bike:
            print('build cycling network')
            links_bike_temp = links_osm.copy()
            nodes_bike_temp = nodes_osm.copy()

            #replace NaN values for relevant attributes:
            for attr in ["bicycle","cycleway",  "bicycle_road",
                "cycleway:right", "cycleway:both","cycleway:left",
                "tracktype" ,"surface",'cyclestreet']:
                try:
                    links_bike_temp[attr] = links_bike_temp[attr].fillna('unknown')
                except:
                    links_bike_temp[attr] = 'unknown'

            def classify_cycle_quality(link):
                '''Determine cycleway-class based on OSM-attributes
                    
                    possible categories:
                    --------------------
                    > designated(6):       cyclestreet or way designated for bikes, seperated from streets, also tracks/path with paved surface
                    > with_foot(5):        way seperated from car traffic but shared with pedestrians
                    > road_markings(4):    lane for cyclist marked on road, no physical seperation from car traffic
                    > small_road(3):       residential or access roads
                    > primary(5):          Bundesstraße (OSM: highway= primary) without bicycle infrastructure and max speed >50 km/h
                    > main_road(1):        Remaining roads without bicycle infrastructure
                    > prohibited(-1):       cycling not allowed either implied by road type or explicitly marked
                    > unknown(0):          otherwise
                '''

                if link.bicycle in ['designated']:
                    return 'designated'
                elif link.bicycle in ['no', 'use_sidepath']:
                    return 'prohibited'
                elif link.bicycle_road in ['yes'] or link.cyclestreet in ['yes']:
                    return 'designated'
                elif (link.highway in ['pedestrian', 'bridleway','footway']) and (link.bicycle =='yes'):
                    return 'with_foot'
                elif  link.highway in ['motorway','trunk','motorway_link','trunk_link','bridleway','pedestrian', 'steps','footway']:
                    return 'prohibited'
                elif link.highway in ['cycleway']:
                    return 'designated'
                elif link.cycleway in ['track']:
                    return 'designated'
                elif link.cycleway in ['lane', 'shared_lane', 'share_busway']:
                    return 'road_markings'
                elif link["cycleway:both"] in ['track']:
                    return 'designated'
                elif link["cycleway:both"] in ['lane', 'shared_lane', 'share_busway']:
                    return 'road_markings'
                elif link.reversed == False and link["cycleway:right"] in ['track']:
                    return 'designated'
                elif link.reversed == False and link["cycleway:right"] in ['lane', 'shared_lane', 'share_busway']:
                    return 'road_markings'
                elif link.reversed == True and link["cycleway:left"] in ['track']:
                    return 'designated'
                elif link.reversed == True and link["cycleway:left"] in ['lane', 'shared_lane', 'share_busway']:
                    return 'road_markings'
                elif link.highway in ['residential','living_street','unclassified']:
                    return 'small_road'
                elif link.highway in ["primary", "primary_link" ]:
                    if link.maxspeed >50:
                        return 'primary'
                    else:
                        return 'main_road'
                elif link.highway in ["primary", "secondary", "tertiary", "primary_link", "secondary_link", "tertiary_link"]:
                    return 'main_road'
                elif link.highway in ['path', 'track']:
                    if link.bicycle =='yes' and link.tracktype in ['grade1','grade2', '1','2']:
                        return 'designated' 
                    elif link.surface in ['asphalt','concrete','paved']:
                        return 'designated'
                    elif link.tracktype in ['grade1','grade2', '1','2']:
                        return 'path'
                    elif link.tracktype in ['grade5','grade4', 'grade3']:
                        return 'unuseable' 
                else:
                    return('not classified')

            print('classify quality of cycling paths')
            links_bike_temp['way_quality'] = links_bike_temp.apply(classify_cycle_quality, axis = 1)

            #select only relevant road types for cycling network
            links_bike_temp = links_bike_temp[links_bike_temp.way_quality.isin(bikeable)]
            links_bike_temp = links_bike_temp[['lanes', 'highway','reversed', 'oneway','maxspeed', 'way_quality', 'length','geometry']]
        
            #simplify cycling network
            print('simplifing cycling network')
            CycleGraph = ox.graph_from_gdfs(gdf_nodes = nodes_bike_temp, gdf_edges = links_bike_temp)
            CycleGraph = ox.simplify_graph(CycleGraph, edge_attrs_differ=['maxspeed','lanes','highway','way_quality'])
            CycleGraph = ox.truncate.largest_component(CycleGraph, strongly = truncate_graph_strongly)
            #get links and nodes from graph
            nodes_bike_t, links_bike_t = ox.graph_to_gdfs(CycleGraph)

            print('Calculate time on link')
            if isinstance(cycle_speed, dict):
                links_bike_t['speed'] = links_bike_t['way_quality'].map(cycle_speed)
            else:
                links_bike_t['speed'] = cycle_speed
            
            links_bike_t['length'] = links_bike_t['length']
            links_bike_t['time'] = links_bike_t['length']/(links_bike_t['speed'].fillna(18))*3.6

            wq_id = {'designated':7,
                    'with_foot':6,        
                    'road_markings':5,    
                    'small_road':4,
                    'path':3 ,      
                    'primary':2,         
                    'main_road':1,        
                    'prohibited':-2,
                    'unuseable':-1,      
                    'unknown':0}
            wq_df = pd.DataFrame(index = wq_id.keys(), data = wq_id.values(), columns = ['wq_idx'])

            links_bike_t['way_quality_id'] = links_bike_t.way_quality.map(lambda x:wq_df.loc[x, 'wq_idx'])

            agt = ox.graph_from_gdfs(gdf_nodes = nodes_bike_t, gdf_edges = links_bike_t.drop(columns = ['lanes', 'highway','reversed', 'oneway','maxspeed'], errors= 'ignore'))
            if simplify_strict:
                agt = ox.simplification.simplify_graph(agt,edge_attr_aggs = {'length': sum, 'time': sum,'way_quality_id':min})
                agt = ox.truncate.largest_component(agt, strongly = truncate_graph_strongly)
                agt = ox.project_graph(agt, to_crs = 3035)
                agt = ox.simplification.consolidate_intersections(agt, tolerance = node_consolidation_tolerance)
                agt = ox.project_graph(agt, to_crs = 4326)
            else:
                agt = ox.truncate.largest_component(agt, strongly = truncate_graph_strongly)

            nodes_bike, links_bike = ox.graph_to_gdfs(agt)
            
            wq_df['wq'] = wq_df.index
            wq_df = wq_df.set_index('wq_idx')
            links_bike['way_quality'] = links_bike.way_quality_id.map(lambda x: wq_df.loc[x, 'wq'])
            
            #give unique indices to links and nodes
            nodes_bike.index = nodes_bike.index.map(lambda x: f'node-bike_{x:012d}')
            nodes_bike.index = nodes_bike.index.rename('index')
            links_bike = links_bike.reset_index()
            links_bike.u = links_bike.u.map(lambda x: f'node-bike_{x:012d}')
            links_bike.v = links_bike.v.map(lambda x: f'node-bike_{x:012d}')
            links_bike.index = links_bike.index.map(lambda x: f'link-bike_{x:012d}')
            links_bike.index = links_bike.index.rename('index')
            links_bike = links_bike.rename(columns={'u':'a','v':'b'})
            
            self.cycle_links = links_bike[['a','b','way_quality','length','speed','time','geometry']].copy()
            self.cycle_nodes = nodes_bike[['geometry']].copy()
            self.cycle_links = self.cycle_links.loc[~self.cycle_links.duplicated(['a', 'b'])]
        
        ####################
        # walking network
        ####################
        if walk:
            links_walk_temp = links_osm.copy()
            nodes_walk_temp = nodes_osm.copy()

            #replace NaN values for relevant attributes:
            for attr in ["foot","sidewalk",
               "sidewalk:right", "sidewalk:both","sidewalk:left",
               "tracktype" ,"surface"]:
                try:
                    nodes_walk_temp[attr] = nodes_walk_temp[attr].fillna('unknown')
                except:
                    nodes_walk_temp[attr] = 'unknown'

            def get_walking_paths(link):
                '''
                Determine the class of way for pedestrians:

                possible categories:
                --------------------
                > designated:   Designated way for pedestrians or sidewalk along street 
                > on_street:    residential streets or other streets with a speed limit of 50 km/h without sidewalk
                > main_road:    Main roads without sidewalk
                > unuseable:    unpaved way with bad surface    
                > prohibited:   walking prohibited either implied by highway type or explicitly marked
                > unknown:      otherwise
                '''
                try:
                    if link.foot in ['yes','designated']:
                        return 'designated'
                    if link.foot in ['no', 'use_sidepath']:
                        return 'prohibited'
                except KeyError:
                    pass
                if link.highway in ['motorway','trunk','motorway_link','trunk_link']:
                    return 'prohibited'
                if link.highway in ['pedestrian','footway','crossing','living_street','steps']:
                    return 'designated'
                if link.highway in ['path', 'track']:
                    try:
                        if link.tracktype in ['grade5','grade4']:
                            return 'unuseable' 
                    except: 
                        return 'designated'
                if link.highway in ['primary','secondary','tertiary','bridleway','cycleway']:
                    if link.foot == 'yes':
                        return 'on_street'
                    if link.maxspeed<=60:
                        # roads in cities have usually sidewalks
                        return 'on_street'
                    if link.length < 150:
                        return 'on_street'
                    else:
                        return 'main_road'
                if link.highway in ['residential', 'unclassified']:
                    return 'on_street'
                try:
                    if link.junction == 'roundabout':
                        return 'prohibited'
                except KeyError:
                    pass
                try:
                    if link['sidewalk'] in ['both','right','left','yes','shared']:
                        return 'designated'
                    if link['sidewalk'] in ['separate']:
                        return 'prohibited'
                    if link['sidewalk:both'] in ['yes','shared']:
                        return 'designated'
                    if link['sidewalk:both'] in ['separate','no']:
                        return 'prohibited'    
                    if link['sidewalk:left'] in ['yes','shared']:
                        return 'designated'
                    if link['sidewalk:right'] in ['yes','shared']:
                        return 'designated'
                    if link['sidewalk:left'] in ['separate','no'] and link['sidewalk:right'] in ['separate','no']:
                        return 'prohibited'
                except KeyError:
                    pass
                return 'unknown'

            print('classify walking paths')
            links_walk_temp['footway_class'] = links_walk_temp.apply(get_walking_paths, axis = 1)
            links_walk_temp = links_walk_temp[links_walk_temp.footway_class.isin(walkable)]
            
            print('simplifing walking network')
            WalkGraph = ox.graph_from_gdfs(gdf_nodes=nodes_walk_temp,gdf_edges=links_walk_temp)
            WalkGraph = ox.simplify_graph(WalkGraph,edge_attrs_differ=['maxspeed','lanes','highway','footway_class'])
            WalkGraph = ox.truncate.largest_component(WalkGraph, strongly = truncate_graph_strongly)
            nodes_walk_t, links_walk_t = ox.graph_to_gdfs(WalkGraph)

            print('Calculate time on link')   
            links_walk_t['speed'] = walking_speed
            links_walk_t['length'] = links_walk_t['length']
            links_walk_t['time'] = links_walk_t['length']/(links_walk_t['speed'])*3.6
            
            
            wq_id = {'designated':7,    
                    'on_street':4,   
                    'prohibited':-2,
                    'unuseable':-1,      
                    'unknown':0}
            wq_df = pd.DataFrame(index = wq_id.keys(), data = wq_id.values(), columns = ['wq_idx'])

            
            links_walk_t['way_quality_id'] = links_walk_t.footway_class.map(lambda x:wq_df.loc[x, 'wq_idx'])

            self.ld = links_walk_t
            self.nd = nodes_walk_t
            agt = ox.graph_from_gdfs(gdf_nodes = nodes_walk_t, gdf_edges = links_walk_t[['time','geometry','way_quality_id','length']])
            if simplify_strict:
                agt = ox.simplification.simplify_graph(agt,edge_attr_aggs = {'length': sum, 'time': sum,'way_quality_id':min})
                agt = ox.truncate.largest_component(agt, strongly = truncate_graph_strongly)
                agt = ox.project_graph(agt, to_crs = 3035)
                agt = ox.simplification.consolidate_intersections(agt, tolerance = node_consolidation_tolerance)
                agt = ox.project_graph(agt, to_crs = 4326)
            else:
                agt = ox.truncate.largest_component(agt, strongly = truncate_graph_strongly)

            nodes_walk, links_walk = ox.graph_to_gdfs(agt)

            wq_df['wq'] = wq_df.index
            wq_df = wq_df.set_index('wq_idx')
            links_walk['footway_class'] = links_walk.way_quality_id.map(lambda x: wq_df.loc[x, 'wq'])

            #set unique indices to nodes and links
            nodes_walk.index = nodes_walk.index.map(lambda x: f'node-walk_{x:012d}')
            nodes_walk.index = nodes_walk.index.rename('index')
            links_walk = links_walk.reset_index()
            links_walk.u = links_walk.u.map(lambda x: f'node-walk_{x:012d}')
            links_walk.v = links_walk.v.map(lambda x: f'node-walk_{x:012d}')
            links_walk.index = links_walk.index.map(lambda x: f'link-walk_{x:012d}')
            links_walk.index = links_walk.index.rename('index')
            links_walk = links_walk.rename(columns={'u':'a','v':'b'})
            
            self.walk_links = links_walk[['a','b','footway_class','length','time','geometry']].copy()
            self.walk_nodes = nodes_walk[['geometry']].copy()
            self.walk_links = self.walk_links.loc[~self.walk_links.duplicated(['a', 'b'])]


    def get_pr_parking(self, path_to_parking_df, path_to_stations_df, max_distance = 300, min_area = 1000): 
        """Build connections between transit and road network.
        Map larger parking spots close to public transit stations. If no file
        is given, use OSM data for parking spots and stations, respectively
        (OSM tag: public_dtransit = station).

        Requires:
        ----------
        self.pt_nodes
        
        Parameters:
        ----------
        path_to_parking_df: str
            Filepath to GeoDataframe with all possible parkingspots and their OSM attributes. 
            If no file is found Data is downloaded using osmnx and saved as GeoJSON at given path.
            GeoDataFrame columns required (and corresponding filter criteria):
            - element_type (=='way')
            - park_ride (!='no')
            - hiking (!='yes')
            - access (not private or similar, can be NaN)
            - maxstay (unrestricted, e.g. 'no', can be NaN)
            - maxstay:conditional (unrestricted, e.g. 'no', can be NaN)
            - parking (not on roadside and similar, can be NaN)
            - geometry in CRS EPSG:4326
        path_to_stations_df: str
            Filepath to GeoDataframe with all possible transit stations and a `geometry` column in CRS EPSG:4326. 
            If no file is found, data is downloaded using osmnx (OSM tag: public_dtransit = station)
            and saved as GeoJSON at given path.
        max_distance: int
            Maximum distance in m between station and parking spot in order to count parkingspot as park and ride
        min_area: int
            Minimum area (in sqm) of parkingspot to be considerd 
        
        Attributes:
        ----------
        self.pr_parking
        self.road_to_park
        self.road_to_transit
        """
        try:
            parking = gpd.read_file(path_to_parking_df)
            stations = gpd.read_file(path_to_stations_df)
            print(f'loaded {len(stations)} stations and {len(parking)} parking spots from files')
        except:
            #Load Parkinspots from OSM
            parking = ox.features_from_polygon(self.total_area,{'amenity':'parking'})
            parking.to_file(path_to_parking_df, driver= 'GeoJSON')
            #load stations from OSM, OSM-tag public_transit = station ist not perfect but works well enough
            stations = ox.features_from_polygon(self.total_area,{'public_transport':'station'})
            stations.to_file(path_to_stations_df, driver= 'GeoJSON')
            print(f'loaded {len(stations)} stations and {len(parking)} parking spots from OpenStreetMap')
        parking = parking.reset_index()
        #Filter parkingspots where attributes do not fit for park and ride
        parking = parking[parking.element_type == 'way']
        parking = parking[parking.park_ride != 'no']
        parking = parking[parking.hiking != 'yes']
        #drop parkingspots without public access
        parking = parking[parking.access.map(
            lambda x: x not in ['customers','permissive','private','permit','visitors','employees','destination'])]
        #drop parkingspots with any form of maximum stay
        parking[['maxstay','maxstay:conditional']] = parking[['maxstay','maxstay:conditional']].fillna('unknown')
        parking = parking[parking.maxstay.map(lambda x: x in['unknown','no','unlimited'])]
        parking = parking[parking['maxstay:conditional'].map(lambda x: x in['unknown','no','unlimited'])]
        parking = parking[parking['parking'].map(lambda x: x not in['lane','layby','shoulder','street_side'])]

        #calculate area of parking spots as capacity indicator
        p3035 = parking.to_crs(crs= 3035)
        p3035['area'] = p3035.area
        #drop very small parking spots
        p3035 = p3035[p3035['area']>min_area]

        #drop unused columns
        p3035=p3035.dropna(axis = 1, how='all')
        #simplify parkingspot as point
        p3035=p3035.set_geometry('geometry')
        p3035['geometry'] = p3035.centroid

        #give unique index to stations from OSM data
        stations3035 = stations.to_crs(crs = 3035)
        stations3035 = stations3035.reset_index()
        stations3035.geometry = stations3035.geometry.centroid
        stations3035['station_id'] = stations3035.index.map(lambda x: f'station-osm_{x:04d}')
        stations3035= stations3035.set_index('station_id',drop= False)

        # assign close parkingspots within given radius to each transit station
        # and Combine all parkingspots in given radius around station 
        name = []
        dist = []
        area = []
        loc = []
        for s in tqdm(range(len(stations3035))):
            station = stations3035.iloc[s]
            #determine distances between sation and all parkingspots
            p3035['dist'] = p3035.distance(station.geometry)
            #Only take into account parkingspots that are not further than 300m from a station
            relP = p3035[p3035['dist']<max_distance]
            #relP=relP.dropna(axis = 1, how='all')
            if len(relP)==0:#no parking spot ist close to station
                name.append(None)
                dist.append(None)
                area.append(None)
                loc.append(None)
            elif len(relP)==1:
                name.append(station.station_id)
                dist.append(relP['dist'].values[0])
                area.append(relP['area'].values[0])
                loc.append(relP['geometry'].values[0])
            else:
                #multiple parkingspots wihtin radius: combine all spots
                center = relP.dissolve().centroid
                name.append(station.station_id)
                dist.append(np.mean(relP['dist']))
                area.append(sum(relP['area']))#summ area of all parkingspots
                loc.append(center.values[0])#position of combined parking spot is now at the center
        dct = {'geometry':loc,'name':name,'area':area,'dist':dist}
        # build dataframe with relevant parkingspot/station combinations
        # combine parkingstpot- and osm-station data
        df = pd.DataFrame.from_dict(dct)
        df = df.set_index('name')
        df = df.dropna(axis = 0, how = 'any')
        df = gpd.GeoDataFrame(df)

        #get corresponding stations(OSM data)
        stations3035 = stations3035.loc[df.index]
        stations3035 = stations3035.dropna(axis = 1, how='any')
        stations3035 = stations3035.rename(columns={'geometry':'loc_station'})
        
        #Map which stops from the GTFS feed corrspond to the stations with Parkingspots from OSM data
        df = df.rename(columns={'geometry':'loc_parking'})
        df = gpd.GeoDataFrame(df)

        df = df.set_geometry('loc_parking')
        df = df.set_crs(crs = 3035)
        df = df.to_crs(crs = 4326)
        stations3035 = stations3035.set_geometry('loc_station')
        pr_spots = stations3035.to_crs(crs = 4326)
        pr_spots = pr_spots.merge(df,left_index=True,right_index=True)
        nodes_parking = pr_spots[['loc_parking','area']]
        nodes_parking = nodes_parking.rename(columns={'loc_parking':'geometry'})

        stations = pr_spots[['loc_station','area']]
        stations = stations.rename(columns={'loc_station':'geometry'})

        nodes_parking = nodes_parking.reset_index(drop=True)
        nodes_parking.index = nodes_parking.index.map(lambda x: f'node-parking_{x:04d}')

        #Create list with all staion names where parking is available at the station
        osm2gtfsStation = engine.ntlegs_from_centroids_and_nodes(stations,self.pt_nodes,n_neighbors=1)
        osm2gtfsStation= gpd.GeoDataFrame(osm2gtfsStation)
        stationWithParkingList = np.unique(osm2gtfsStation.a[osm2gtfsStation.a.map(lambda x: 'node-pt'in x)].values)
        # Find and build access/egress links (footpaths) between parking and nearest station/road-node to be able to connect the networks
        park2station = engine.ntlegs_from_centroids_and_nodes(nodes_parking,self.pt_nodes.loc[stationWithParkingList],n_neighbors=1)
        park2station=gpd.GeoDataFrame(park2station)
        #road2park = engine.ntlegs_from_centroids_and_nodes(nodes_parking,self.auto_nodes,n_neighbors=1)
        #road2park=gpd.GeoDataFrame(road2park)
        p=park2station[park2station.a.map(lambda x: 'node-parking'in x)]
        p = p[['a','b','geometry']]
        p= p.rename(columns={'a': 'parking_id','b':'station_id'})
        p = p.set_index('parking_id')
        p['location_parking']= p.geometry.map(lambda x: Point(x.coords[0]))
        p = p.drop(columns=['geometry'])
        p = p.rename(columns={'location_parking':'geometry'})
        pp = p.merge(nodes_parking.area,left_index=True,right_index=True)
        pp = gpd.GeoDataFrame(pp, geometry='geometry',crs = 4326)
        pp.index =pp.index.rename('node-parking_id')
        self.pr_parking = pp.drop_duplicates()
    

    def fix_pt_network_integrity(self):
        '''
        Tests PT network attributes relevant for pathfinding and fixes errors,
        if circular lines occur (by renaming nodes), if nodes without links
        exist (dropping them), if links without nodes exist (dropping the entire
        trip), and if sequences are corrupted (reindex sequence numbers). This
        process takes several minutes in a mid-sized region (e.g. city with
        surrounding countryside).
        '''
        self.links = self.pt_links
        self.nodes = self.pt_nodes
        
        try:
            self.integrity_test_nodeset_consistency()
        except:
            if 'orphan_nodes' in self.__dict__.keys() and len(self.orphan_nodes) > 0:
                print('Found {} orphan nodes. Dropping them...'.format(len(self.orphan_nodes)))
                self.nodes.drop(self.orphan_nodes, inplace=True)
            if 'missing_nodes' in self.__dict__.keys() and len(self.missing_nodes) > 0:
                print('Found {} missing nodes. Fixing...'.format(len(self.missing_nodes)))
                affected_trips = list(self.links.loc[(self.links['a'].isin(self.missing_nodes))
                                                        | (self.links['b'].isin(self.missing_nodes)), 'trip_id'])
                self.links = self.links.loc[~self.links['trip_id'].isin(affected_trips)]
                print('Dropped {} entire trips.'.format(len(affected_trips)))
        
        def repair_sequence(trip):
            b_stops = tuple(trip['b'])[:-1]
            a_stops = tuple(trip['a'])[1:]
            if a_stops != b_stops:
                if len(trip) > 1:
                    for i in range(len(b_stops)):
                        if a_stops[i] != b_stops[i]:
                            ind = trip.iloc[i+1:].index
                            trip.loc[ind, 'trip_id'] = \
                                trip.loc[ind, 'trip_id']+'_'+str(i)
            return trip
        self.links['trip_id'] = self.links['trip_id'].astype(str)
        self.links = self.links.sort_values(['trip_id', 'link_sequence'])
        self.links.index.name = 'index'
        tqdm.pandas(desc="Repairing sequences")
        l = self.links.groupby('trip_id').progress_apply(repair_sequence)\
            .reset_index(level=0, drop=True)
        l['index'] = l.index
        l = feed_links.clean_sequences(l, sequence='link_sequence', group_id='trip_id')
        self.links = l.set_index('index')
        
        self.integrity_fix_circular_lines(method='drop')
        #self.integrity_fix_circular_lines(method='duplicate', sep='_fix_circle_')
        #if 'circular_lines' in self.__dict__.keys() and len(self.circular_lines) > 0:
        #    print('Found {} circular lines and repaired them.'.format(len(self.circular_lines)))

        link_set = set(self.links['a']).union(set(self.links['b']))
        self.orphan_nodes = list(set(self.nodes.index).difference(link_set))
        self.missing_nodes = list(link_set.difference(set(self.nodes.index)))
        if 'orphan_nodes' in self.__dict__.keys() and len(self.orphan_nodes) > 0:
            print('Found {} orphan nodes. Dropping them...'.format(len(self.orphan_nodes)))
            self.nodes.drop(self.orphan_nodes, inplace=True)
        if 'missing_nodes' in self.__dict__.keys() and len(self.missing_nodes) > 0:
            print('Found {} missing nodes. Fixing...'.format(len(self.missing_nodes)))
            affected_trips = list(self.links.loc[(self.links['a'].isin(self.missing_nodes))
                                                    | (self.links['b'].isin(self.missing_nodes)), 'trip_id'])
            self.links = self.links.loc[~self.links['trip_id'].isin(affected_trips)]
            print('Dropped {} entire trips.'.format(len(affected_trips)))
            link_set = set(self.links['a']).union(set(self.links['b']))
            self.orphan_nodes = list(set(self.nodes.index).difference(link_set))
            self.nodes.drop(self.orphan_nodes, inplace=True)

        self.pt_links = self.links
        self.pt_nodes = self.nodes
    
    
    def generate_public_transport_network(self, path_to_gtfs, dates=None,
                                          maxfoot_distance=300, maxfoot_connections=3,
                                          min_change_time=2*60, max_waiting_time=30*60,
                                          ntleg_speed=3, remote_stations_for_headway=True,
                                          route_type_dict=None, fix_integrity=True,
                                          max_shortcut=False):
        """
        Generates a public transport network by processing a GTFS (General 
        Transit Feed Specification) feed. The network includes information on
        the type of day and daytime for each service for later filtering.
        It also includes information on the headway of similar services on
        one route, suitable for a headway model. The method appends the
        given GTFS feed to an existing network, if the method was run before.

        Parameters
        ----------
        path_to_gtfs : str
            The file path to the zipped GTFS feed
        dates : list [str]
            GTFS Feed is extracted for given dates ('YYYYMMDD') 
            if date = None compute feed for first full week (Mon-Sun)
        maxfoot_distance : int
            Maximum walking distance as-the-crow-flies between public transport
            stops that should be interconnected by footpaths. Defaults to 300 [m].
        maxfoot_connections: int
            Maximum number of links by foot per station.
        min_change_time: int
            Min time [seconds] that is needed to change stations 
        max_waiting_time: int 
            Maximum Value of the waiting time for each link [seconds].
            Only trips that arrive within the max_waiting_time are taken into 
            account for waiting time calculations.
            fallback value for waiting time.
        ntleg_speed: float
            speed [km/h] on walking legs between stations close by
        remote_stations_for_headway: bool
            Wether to take the arrivals at other stations close by 
            (up to maxfoot_distance) in to account when calulating the waiting 
            time of a trip. Takes significantly longer. Defaults to `True`.
        route_type_dict: dict
            Translation of GTFS-feed-specific route_type keys into strings.
            Should include all keys, but will retain initial values, if not
            matched in the given dictionary.
        fix_integrity: bool
            Test and fix the integrity of the PT network for the pathfinding
            step. Caution: this takes several minutes. Defaults to `True`.
        max_shortcut: bool or int
            Number of stops to consider while building links from stop_times.
            Caution: `False` takes several minutes. Defaults to `False`.
        
        Attributes
        ----------
        self.pt_links : geopandas.GeoDataFrame
            A GeoDataFrame containing the links in the public transport network.
            Each link belongs to a trip (`trip_id` column), having specified
            its `link_sequence`, origin and destination stop, temporal detail,
            and headway time before the next service on this route.

        self.pt_nodes : geopandas.GeoDataFrame
            A GeoDataFrame containing the nodes (stops or stations).
        
        self.pt_footpaths : geopandas GeoDataFrame
            A GeoDataFrame containing as-the-crow-flies connectors between
            nodes up to a distance specified in `maxfoot_distance`.
        """
        def timestr_to_sec(timestr):
            '''Calculates the number of seconds since midnight from string with format hh:mm:ss'''
            l = list(map(int, timestr.split(':')))
            return 3600*l[0]+60*l[1]+l[2]
        
        def spatial_restrict(feed, polygon):
            ''' With this function, we can restrict a feed spatially.
            Drop stops outside a given polygon '''
            stops = feed.stops.copy()
            stops['geometry'] = stops.apply(lambda r: geometry.Point([r['stop_lon'], r['stop_lat']]), axis=1)
            stops['included'] = stops.apply(lambda g: polygon.contains(g.geometry),axis = 1)
            # Restrict
            feed.stops = feed.stops.loc[stops['included'] == True]
            feed.stop_times = feed.stop_times.loc[
                feed.stop_times['stop_id'].isin(feed.stops['stop_id'])]
            feed.trips = feed.trips.loc[feed.trips['trip_id'].isin(feed.stop_times['trip_id'])]
            return feed
        
        def daybreak(link):
            '''Changes Times with  >23:59 from GTFS feed to next day
            Do it for departure and arrival time of each link.
            '''
            if link.dep_sec>= 3600*24:
                #Date as datetime Object
                d_obj = datetime.date(int(link.dep_date[0:4]), int(link.dep_date[4:6]), int(link.dep_date[6:8]))
                #Add one Day
                nextday = d_obj+datetime.timedelta(days = 1)
                #Calculate time [in sec] modulus 24 hrs
                link.dep_sec = link.dep_sec%(3600*24)
                #Format date
                link.dep_date = nextday.strftime("%Y%m%d")
                link.dep_wd = nextday.strftime("%a")
                link.daynumd = link.daynumd + 1
            if link.arr_sec>= 3600*24:
                d_obj = datetime.date(int(link.arr_date[0:4]), int(link.arr_date[4:6]), int(link.arr_date[6:8]))
                nextday = d_obj+datetime.timedelta(days = 1)
                link.arr_sec = link.arr_sec%(3600*24)
                nextday = d_obj+datetime.timedelta(days = 1)
                link.arr_date = nextday.strftime("%Y%m%d")
                link.arr_wd = nextday.strftime("%a")
                link.daynuma = link.daynuma + 1
            return link

        #import GTFS Feed
        f0 = importer.GtfsImporter(path=path_to_gtfs, dist_units='m')
        srfeed = spatial_restrict(f0, self.total_area)
        assert len(srfeed.stop_times) > 0, 'Spatial restriction went wrong; no stops are withing the polygon'
        print('Number of stops within the region ', len(srfeed.stops))
        #repair feed
        if not 'agency_id' in srfeed.routes.columns:
            srfeed.routes['agency_id'] = 0
            srfeed.agency = pd.DataFrame(
                {'agency_id':[0], 'agency_name':['default'], 'agency_url':[''], 'agency_timezone':['']})
        #detrmine relevant days to observe
        if dates is None:
            reldays = gk.calendar.get_first_week(feed = srfeed)
        else:
            reldays = dates
        links = gpd.GeoDataFrame({'geometry':[]})

        #srfeed = srfeed.restrict_to_dates(dates = reldays)
        print('Dates considered for PT network:', reldays)
        missing_dates = []
        for num, d in enumerate(tqdm(reldays, desc='Calendar days')):
            #restrict to single day
            feedday = srfeed.restrict_to_dates(dates = [d])
            if len(feedday.calendar_dates) > 0 and len(feedday.stop_times) > 0:
                #create links from gtfs feed
                links_temp = feed_links.link_from_stop_times(
                    stop_times=feedday.stop_times,
                    max_shortcut=max_shortcut,
                    stop_id = 'stop_id',
                    keep_origin_columns=['departure_time'],
                    keep_destination_columns=['arrival_time'],
                    stop_id_origin='origin',
                    stop_id_destination='destination',
                    out_sequence='link_sequence')

                #arrival & departure time in seconds
                if links_temp['departure_time'].isna().max() or links_temp['arrival_time'].isna().max():
                    links_temp.loc[links_temp['departure_time'].notna(), 'departure_time'] = \
                        links_temp.loc[links_temp['departure_time'].notna(), 'departure_time'].map(timestr_to_sec)
                    links_temp.loc[links_temp['arrival_time'].notna(), 'arrival_time'] = \
                        links_temp.loc[links_temp['arrival_time'].notna(), 'arrival_time'].map(timestr_to_sec)
                    # Interpolate time between stops linearly
                    links_temp['dep_sec'] = links_temp['departure_time'].astype(float).interpolate()
                    links_temp['arr_sec'] = links_temp['arrival_time'].astype(float).interpolate()
                else:
                    links_temp['dep_sec'] =  links_temp.departure_time.map(timestr_to_sec)
                    links_temp['arr_sec'] =  links_temp.arrival_time.map(timestr_to_sec)

                #write departure & arrival dates
                links_temp['dep_date'] =  d
                links_temp['arr_date'] =  d
                links_temp['daynumd'] = int(num)
                links_temp['daynuma'] = int(num)

                d_obj = datetime.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))
                links_temp['dep_wd'] =  d_obj.strftime("%a")
                links_temp['arr_wd'] =  d_obj.strftime("%a")

                links_temp = links_temp.apply(daybreak, axis = 1)
                links = pd.concat([links, links_temp])
            else:
                missing_dates.append(d)
        
        print('No links for dates ', d)
        assert len(links) > 0, 'No links generated for given dates'
        print('Number of links within the region ', len(links))
        
        trps = srfeed.trips.drop_duplicates('trip_id').set_index('trip_id')
        rts = srfeed.routes.drop_duplicates('route_id').set_index('route_id')
        stps = srfeed.stops.set_index('stop_id')
        #create geometry (point) for stops
        stps['geometry'] = gpd.points_from_xy(stps['stop_lon'],stps['stop_lat'])
        stps.set_geometry('geometry')
        ptnodes = gpd.GeoDataFrame(stps[['stop_name','geometry']],geometry='geometry', crs=4326)
        
         
        #add further information top links
        if 'direction_id' in trps.columns:
            if trps['direction_id'].notna().all():
                links['direction'] = (links['trip_id'].map(trps['direction_id'])).astype(int)
        links['route_id'] = links['trip_id'].map(trps['route_id'])
        links['route_type'] = links['route_id'].map(rts['route_type'])
        if isinstance(route_type_dict, dict):
            links['route_type'] = links['route_type'].replace(route_type_dict)

        #Create line between start and end point as geometry for links
        links['geometry'] = links.apply(
            lambda s:geometry.LineString((stps.loc[s.origin,'geometry'],stps.loc[s.destination,'geometry'])),
            axis = 1)
        #calculate total time in seconds from the start of the first day
        links['total_time_dep'] = links.dep_sec+links.daynumd*24*3600
        links['total_time_arr'] = links.arr_sec+links.daynuma*24*3600

        links= links.drop_duplicates()
        links = gpd.GeoDataFrame(links, geometry='geometry', crs=4326)

        #produce unique indices
        links = links.reset_index(drop = True)
        ptnodes.index = ptnodes.index.map(lambda x: 'node-pt_'+str(x))
        ptnodes.index = ptnodes.index.rename('index')
        links.index = links.index.map(lambda x: 'link-pt_'+str(x))
        links.index = links.index.rename('index')
        links = links.rename(columns={'origin':'a', 'destination':'b'})
        links.a = links.a.map(lambda x: 'node-pt_'+str(x))
        links.b = links.b.map(lambda x: 'node-pt_'+str(x))
        
        links['time'] = links.total_time_arr-links.total_time_dep

        #Get footpaths between Transit Sations within given Distance
        footlinks = engine.ntlegs_from_centroids_and_nodes(
            ptnodes, ptnodes, n_neighbors=maxfoot_connections, short_leg_speed= ntleg_speed)
        footlinks = footlinks[(footlinks['distance']>0)&(footlinks['distance']<maxfoot_distance)]
        footlinks.index = footlinks.index.map(lambda x: f'link-pt_foot_{int(x):012d}')
        footlinks.index = footlinks.index.rename('index')
        footlinks = gpd.GeoDataFrame(footlinks, geometry='geometry', crs=4326)

        #calculate individual headway for every link based on previous arrivals at the same Station:
        # - get the previous trip that leaves the sation wit the same next stop
        # - get all trips that arrive at the station between these two trips
        # - get the average waiting time for a change to the original trip by averaging over all waiting times 
        # (time difference between arrival times and departure time of original trips)
        #only get stations that are served by multiple routes (for performence reasons)
        gr = links.groupby(['a']).apply(lambda x: len(np.unique(x.route_id.values)))
        rel_stop_idx= np.unique(gr.index.values[gr > 1])
        #all stops:
        rel_stop_idx = np.unique(np.append(links.a.values, links.b.values))
        idx = []
        wtime =[]
        links['idx'] = links.index
        
        for stop_id in tqdm(rel_stop_idx, desc='Waiting time at stops'):
            #calculate waiting time for all stops
            allstart = links[links.a==stop_id]
            alldest = links[links.b == stop_id]

            connected_staions = np.unique(footlinks.a[(footlinks.b == stop_id)&(footlinks.a != stop_id)].values)
            if len(connected_staions)>0 and remote_stations_for_headway:
                #allstart = links[links.a==stop_id]
                #alldest = links[links.b == stop_id]
                routes_in_stat0 = np.unique(allstart.route_id.values)

                for stat in connected_staions:
                    arr_remote_stat = links[(links.b == stat)
                                            &(links.route_id.map(lambda x: x not in routes_in_stat0))]
                    alldest=pd.concat([alldest, arr_remote_stat])
            
            for c in (range(len(allstart))):
                #calculate  for each trip at each stop
                trip01 = allstart.iloc[c]
                #find the transit link before
                same_relation = allstart[(allstart.b == trip01.b)
                                            &(allstart.total_time_arr<trip01.total_time_dep)
                                            &(allstart.route_id == trip01.route_id)]
                if len(same_relation)==0:
                    #no trip earlier
                    idx.append(trip01.idx)
                    wtime.append(max_waiting_time)
                else:
                    transit_before = same_relation.loc[same_relation.total_time_arr.idxmax()]
                    #get all transit links that start in the time between

                    relevant_arrivals = alldest[(alldest.total_time_arr>transit_before.total_time_dep)
                                                &(alldest.total_time_arr<trip01.total_time_dep-min_change_time)]
                    #discard links that are traveling backwards
                    relevant_arrivals = relevant_arrivals[~(relevant_arrivals.a==trip01.b)]
                    waiting_time = trip01.total_time_dep-relevant_arrivals.total_time_arr
                    waiting_time =[x for x in waiting_time.values if x < max_waiting_time]
                    if len(waiting_time)>0:
                        mean_waiting_time = np.mean(waiting_time)
                        idx.append(trip01.idx)
                        wtime.append(mean_waiting_time)
                    else:
                        #no trips from other route inbetween.
                        #headway is time between links
                        idx.append(trip01.idx)
                        wtime.append(min((trip01.total_time_dep-transit_before.total_time_dep)/2,max_waiting_time))
        dct = {'link_id':idx, 'mean_headway':wtime}
        
        df = pd.DataFrame.from_dict(dct)
        df = df.set_index('link_id')
        links_hdwy = pd.merge(links, df, left_index=True, right_index=True, how = 'left')
        links_hdwy.mean_headway = links_hdwy.mean_headway.fillna(45)        
        links_hdwy.loc[links_hdwy.mean_headway > max_waiting_time, 'mean_headway'] = max_waiting_time

        links_hdwy  = links_hdwy.rename(columns={'origin':'a','destination':'b','mean_headway':'headway'})
        links_hdwy['time'] = links_hdwy.total_time_arr-links_hdwy.total_time_dep  
        #Waitingtime = Headway/2 :
        links_hdwy['headway'] = links_hdwy['headway']*2

        #####################################################################################
        # Headway END
        #####################################################################################

        # Add length
        links_hdwy['length'] = links_hdwy.to_crs(epsg=3857)['geometry'].length#/1000

        # Save
        links_hdwy.drop(['idx', 'direction', 'total_time_arr', 'total_time_dep', 'daynumd', 'daynuma'],
                      axis=1, errors='ignore', inplace=True)
        if len(self.pt_links) > 0:
            # Merge nodes
            duplicates = list(set(ptnodes.index).intersection(set(self.pt_nodes.index)))
            if len(duplicates) > 0:
                rename = dict(zip(duplicates, [i+'_sec' for i in duplicates]))
                links_hdwy['a'] = links_hdwy['a'].replace(rename)
                links_hdwy['b'] = links_hdwy['b'].replace(rename)
                ptnodes['idx'] = ptnodes.index
                ptnodes['idx'] = ptnodes['idx'].replace(rename)
                ptnodes.set_index('idx', drop=True, inplace=True)
                ptnodes.index = ptnodes.index.rename('index')
            self.pt_nodes = pd.concat([self.pt_nodes, ptnodes])
            # Merge and reindex links
            self.pt_links = pd.concat([self.pt_links, links_hdwy]).drop_duplicates(subset = ['a','b','dep_sec','arr_sec']).reset_index()
            self.pt_links.index = self.pt_links.index.map(lambda x: f'link-pt_{int(x):012d}')
            self.pt_footpaths = pd.concat([self.pt_footpaths, footlinks]).drop_duplicates(subset = ['a','b']).reset_index()
            self.pt_footpaths.index = self.pt_footpaths.index.map(lambda x: f'link-pt_foot_{int(x):012d}')
        else: 
            self.pt_links = links_hdwy
            self.pt_nodes = ptnodes
            self.pt_footpaths = footlinks
        
        if fix_integrity:
            self.fix_pt_network_integrity()
        # Transfor to min and km
        #self.pt_links.time = self.pt_links.time/60
        #self.pt_links['length'] = self.pt_links['length']/1000
        #self.pt_links.headway = self.pt_links.headway/60

    def pathfinding_auto(self, od_df, method='aon', num_iterations=1 , *args, **kwargs):
        """
        Runs a Dijkstra pathfinding algorithm to find the shortest paths
        across the road network for cars for a given list of origin-destination
        (OD) pairs at specified start times.

        Parameters
        ----------
        od_df : pandas.DataFrame
            A DataFrame containing the origin-destination data with the following columns:
            - `start_time`  : datetime
                The start time of the journey.
            - `origin_id`   : int or str
                Unique identifier of the origin
            - `destination_id`: int or str
                Unique identifier of the destination
            - `origin`      : shapely.geometry.Point
                The starting point of the journey, represented as a Shapely Point object.
            - `destination` : shapely.geometry.Point
                The destination point of the journey, represented as a Shapely Point object.
            - `activity`    : str
                The type of activity or purpose of the journey.
        method: str ['aon', 'msa','fw','bfw']
            Method for shortest-path finding. All-or-nothing (`aon`) find one
            shortest path (default). `msa`,'fw','bfw' is the iterative
            assignment, which requires road capacities and `num_iterations>1`.
        num_iterations : int
            Maximum number of iterations in case an iterative routing method is
            chosen. Defaults to 1.

        Attributes
        ----------
        self.auto_los : pandas.DataFrame
            A DataFrame containing the shortest paths for each OD pair with
            the following columns:
            - origin:       id of start zone
            - destination:  id of end zone
            - time:         trip duration [s]
            - path:         list of IDs from nodes used
            - link_path:    list of IDs from links used while driving
            - ntlegs:       list of tuples with IDs of start and end poitnts from non car legs used
        """
        #Set Start- & Endpoints as zones
        od_df['origin_id'] = od_df['origin_id'].astype(str)
        od_df['destination_id'] = od_df['destination_id'].astype(str)
        od_set = set(zip(od_df.origin_id, od_df.destination_id))
        new_zones = pd.concat([od_df.copy().rename(columns={'origin_id':'zone_id','origin':'geometry'})[['zone_id','geometry']],
                               od_df.copy().rename(columns={'destination_id':'zone_id','destination':'geometry'})[['zone_id','geometry']]])
        self.zones = new_zones.set_index('zone_id')
        self.zones = self.zones[~self.zones.index.duplicated(keep='first')]

        #create Links from Zone to Road-Network
        z2r = engine.ntlegs_from_centroids_and_nodes(self.zones,self.auto_nodes,n_neighbors=1)
        z2r.index = z2r.reset_index().index.map(lambda x: 'zone2road-link_auto_'+str(uuid.uuid1()))
        self.zone_to_road = pd.concat([self.zone_to_road, z2r]).drop_duplicates(subset = ['a','b'])
        #Set relevant zones
        self.road_links = self.auto_links.copy()
        self.road_nodes = self.auto_nodes.copy()

        #Set speed and calculate time
        #self.road_links.speed = self.road_links.apply(driving_speed, axis = 1)
        #self.road_links.time = self.road_links['length'] / (self.road_links['speed'])*60

        #pathfinder
        self.step_road_pathfinder(method = method ,od_set = od_set, maxiters =num_iterations, *args, **kwargs)
        self.auto_links = self.road_links.copy()
        #store LOS table
        self.car_los = self.car_los.loc[self.car_los['origin']!=self.car_los['destination']]
        self.car_los['length'] = self.car_los['link_path'].apply(
            lambda p: sum([self.road_links['length'][l] for l in p]))/1000
        self.auto_los = self.car_los.drop('gtime', axis=1).copy()

        # Add main modes
        self.auto_los['route_type'] = 'car'
        self.auto_los['route_types'] = [('car',) for _ in self.auto_los.index]

        # Add list of all Link indices to auto_los table
        all_links = pd.concat([self.zone_to_road, self.auto_links])
        all_links['idx'] = all_links.index
        all_links = all_links.set_index(['a','b'])

        self.auto_los['node_pairs_all'] =  self.auto_los.path.map(lambda x: list(zip(x[:-1],x[1:])))
        self.auto_los['link_path_all'] = self.auto_los.node_pairs_all.map(lambda x: all_links.loc[x].idx.values)
        self.auto_los['time'] = self.auto_los.link_path_all.map(lambda x: sum(all_links.set_index('idx').loc[x,'time'].values))/60


    

    def pathfinding_cycling(self, od_df, bike_type='normal', primary_road_penalty_thresh=1000, *args, **kwargs):
        """
        Runs a Dijkstra pathfinding algorithm to find the shortest paths
        across the road network for cars for a given list of origin-destination
        (OD) pairs at specified start times.

        Parameters
        ----------
        od_df : pandas.DataFrame
            A DataFrame containing the origin-destination data with the following columns:
            - `origin_id`   : int or str
                Unique identifier of the origin
            - `destination_id`: int or str
                Unique identifier of the destination
            - `origin` : shapely.geometry.Point
                The starting point of the journey, represented as a Shapely Point object.
            - `destination` : shapely.geometry.Point
                The destination point of the journey, represented as a Shapely Point object.
            - `activity` : str
                The type of activity or purpose of the journey.
        bike_type: str
            > 'normal':     standard bike
            > 'pedelec':    simple e-bike (support up to 25 km/h)
            > 's-pedelec':  lager e-bike (up to 45 km/h)
        primary_road_penalty_thresh: int
            Apply a penalty for using main roads (OSM key primary) without
            designated cycling lane for links above a certain distance
            threshold. The penalty is 1 hour to guide the pathfinder towards
            non-main roads, if possible. Default threshold is 1000 [meter].

        Attributes
        ----------
        self.cycle_los : pandas.DataFrame
            A DataFrame containing the shortest paths for each OD pair with
            the following columns:
            - origin:       id of start zone
            - destination:  id of end zone
            - time:         trip duration [s]
            - path:         list of IDs from nodes used
            - link_path:    list of IDs from links used while cycling
            - ntlegs:       list of tuples with IDs of start and end poitnts from non car legs used
        """
        #Set Start- & Endpoints as zones
        od_df['origin_id'] = od_df['origin_id'].astype(str)
        od_df['destination_id'] = od_df['destination_id'].astype(str)
        od_set = set(zip(od_df.origin_id, od_df.destination_id))
        new_zones = pd.concat([od_df.rename(columns={'origin_id':'zone_id','origin':'geometry'})[['zone_id','geometry']],
                               od_df.rename(columns={'destination_id':'zone_id','destination':'geometry'})[['zone_id','geometry']]])
        self.zones = new_zones.set_index('zone_id')
        self.zones = self.zones[~self.zones.index.duplicated(keep='first')]

        #create Links from Zone to Road-Network
        z2r = engine.ntlegs_from_centroids_and_nodes(self.zones,self.cycle_nodes,n_neighbors=1)
        z2r.index = z2r.reset_index().index.map(lambda x: f'zone2road-link_cycle_'+str(uuid.uuid1()))
        self.zone_to_road = pd.concat([self.zone_to_road, z2r]).drop_duplicates(subset = ['a','b'])
        self.road_links = self.cycle_links.copy()
        self.road_nodes = self.cycle_nodes.copy()

        #calculate time on segment
        self.road_links.speed = self.road_links.way_quality.apply(lambda s: biking_speed(s, bike_type))
        self.road_links.time = self.road_links['length'] / (self.road_links['speed'])*60

        # Apply a main-road penalty
        original_time = self.road_links['time'].to_dict()
        self.road_links.loc[(self.road_links['way_quality']=='primary')
                            & (self.road_links['length']>primary_road_penalty_thresh),
                            'time'] += 60
        #call pathfinder
        self.step_road_pathfinder(method = 'aon', od_set = od_set, *args, **kwargs)
        
        # Finish
        self.car_los = self.car_los.loc[self.car_los['origin']!=self.car_los['destination']]
        self.car_los['length'] = self.car_los['link_path'].apply(
            lambda p: sum([self.road_links['length'][l] for l in p]))/1000
        self.car_los['time'] = self.car_los['link_path'].apply(
            lambda p: sum([original_time[l] for l in p]))
        self.cycle_los = self.car_los.drop('gtime', axis=1).copy()

        # Add main modes
        self.cycle_los['route_type'] = 'bicycle'
        self.cycle_los['route_types'] = [('bicycle',) for _ in self.cycle_los.index]

        # Add list of all Link indices to cycle_los table
        all_links = pd.concat([self.zone_to_road, self.cycle_links])
        all_links['idx'] = all_links.index
        all_links = all_links.set_index(['a','b'])

        self.cycle_los['node_pairs_all'] =  self.cycle_los.path.map(lambda x: list(zip(x[:-1],x[1:])))
        self.cycle_los['link_path_all'] = self.cycle_los.node_pairs_all.map(lambda x: all_links.loc[x].idx.values)
        self.cycle_los['time'] = self.cycle_los.link_path_all.map(lambda x: sum(all_links.set_index('idx').loc[x,'time'].values))/60
    

    def pathfinding_walk(self, od_df, walking_speed=6, *args, **kwargs):
        """
        Runs a Dijkstra pathfinding algorithm to find the shortest paths
        across the road network for cars for a given list of origin-destination
        (OD) pairs at specified start times.

        Parameters
        ----------
        od_df : pandas.DataFrame
            A DataFrame containing the origin-destination data with the following columns:
            - `origin_id`   : int or str
                Unique identifier of the origin
            - `destination_id`: int or str
                Unique identifier of the destination
            - `origin` : shapely.geometry.Point
                The starting point of the journey, represented as a Shapely Point object.
            - `destination` : shapely.geometry.Point
                The destination point of the journey, represented as a Shapely Point object.
            - `activity` : str
                The type of activity or purpose of the journey.
        walking_speed: float
            Average walking speed on all paths in km/h. Defaults to 6.

        Attributes
        ----------
        self.walk_los : pandas.DataFrame
            A DataFrame containing the shortest paths for each OD pair with
            the following columns:
            - origin:       id of start zone
            - destination:  id of end zone
            - time:         trip duration [s]
            - path:         list of IDs from nodes used
            - link_path:    list of IDs from links used while driving
            - ntlegs:       list of tuples with IDs of start and end poitnts from access/egress links
        """
        #Set Start- & Endpoints as zones
        od_df['origin_id'] = od_df['origin_id'].astype(str)
        od_df['destination_id'] = od_df['destination_id'].astype(str)
        od_set = set(zip(od_df.origin_id, od_df.destination_id))
        new_zones = pd.concat([od_df.rename(columns={'origin_id':'zone_id','origin':'geometry'})[['zone_id','geometry']],
                               od_df.rename(columns={'destination_id':'zone_id','destination':'geometry'})[['zone_id','geometry']]])
        self.zones = new_zones.set_index('zone_id')
        self.zones = self.zones[~self.zones.index.duplicated(keep='first')]

        #create Links from Zone to Road-Network
        z2r = engine.ntlegs_from_centroids_and_nodes(self.zones,self.walk_nodes,n_neighbors=1, short_leg_speed=walking_speed)
        z2r.index = z2r.reset_index().index.map(lambda x: f'zone2road-link_walk_'+str(uuid.uuid1()))
        self.zone_to_road = pd.concat([self.zone_to_road, z2r]).drop_duplicates(subset = ['a','b'])

        self.road_links = self.walk_links.copy()
        self.road_nodes = self.walk_nodes.copy()
        #calculate time on link
        #self.road_links.speed = walking_speed
        #self.road_links.time = self.road_links['length'] / (self.road_links['speed'])*60

        #call pathfinder 
        self.step_road_pathfinder(method = 'aon', od_set = od_set, *args, **kwargs)
        
        self.car_los = self.car_los.loc[self.car_los['origin']!=self.car_los['destination']]
        self.car_los['length'] = self.car_los['link_path'].apply(
            lambda p: sum([self.road_links['length'][l] for l in p]))/1000
        self.walk_los = self.car_los.drop('gtime', axis=1).copy()

        # Add main modes
        self.walk_los['route_type'] = 'walk'
        self.walk_los['route_types'] = [('walk',) for _ in self.walk_los.index]

        # Add list of all Link indices to walk_los table
        all_links = pd.concat([self.zone_to_road, self.walk_links])
        all_links['idx'] = all_links.index
        all_links = all_links.set_index(['a','b'])

        self.walk_los['node_pairs_all'] =  self.walk_los.path.map(lambda x: list(zip(x[:-1],x[1:])))
        self.walk_los['link_path_all'] = self.walk_los.node_pairs_all.map(lambda x: all_links.loc[x].idx.values)
        self.walk_los['time'] = self.walk_los.link_path_all.map(lambda x: sum(all_links.set_index('idx').loc[x,'time'].values))/60

    

    def pathfinding_public_transport(self, od_df, weekday, start_time=0, end_time=86400,
                                     differentiate_modes=False, boarding_time=None, all_path_analysis=True):
        """
        Finds a headway based path through the GTFS-Feed restricted to times between start_time and end_time

        od_df: pandas.DataFrame
            Origin-destination table with the following columns:
            - `start_time`  : datetime
                The start time of the journey.
            - `origin_id`   : int or str
                Unique identifier of the origin
            - `destination_id`: int or str
                Unique identifier of the destination
            - `origin`      : shapely.geometry.Point
                The starting point of the journey, represented as a Shapely Point object.
            - `destination` : shapely.geometry.Point
                The destination point of the journey, represented as a Shapely Point object.
            - `activity`    : str
                The type of activity or purpose of the journey.
        weekday: str ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            Which day of the week to use as public transport service.
        start_time: int
            Seconds elapsed from 00:00h for filtering of links to specific time
            intervals. Defaults to 0, in which case no start time filter
            is applied.
        end_time: int
            Seconds elapsed from 00:00h for filtering of links to specific time
            intervals. Defaults to 24 * 3600, in which case no end time filter
            is applied.
        differentiate_modes: bool
            Wether to search shortest paths for each public transport mode
            and each possible combination of modes (True). Defaults to `False`,
            searching only one best path, independent of modes used.
        boarding_time: float
            time [s] that is added to total travel time as penalty for every
            interchange. Defaults to `None` to skip.
        all_path_analysis: bool
            Analyse the combination of all links (PT, ntlegs, footpaths) of every path.
            Defaults to `True`, takes hours even for small OD tables.

        Attributes
        ----------
        self.pt_los : pandas.DataFrame
            Dataframe containing the quickes Path for each OD-pair from the od_df and the following columns:
            - origin:           id of start zone
            - destination:      id of end zone
            - time:             total trip duration [s]
            - time_link_path:   Time [s] on moving transit
            - path:             list of IDs from nodes used
            - link_path:        list of IDs from links used on transit
            - ntlegs:           list of tuples with IDs of start and end poitnts from access/egress links
            - ntransfers:       number of transfers
        """
        #Set Start- & Endpoints as zones
        od_df['origin_id'] = od_df['origin_id'].astype(str)
        od_df['destination_id'] = od_df['destination_id'].astype(str)
        od_set = set(zip(od_df.origin_id, od_df.destination_id))
        new_zones = pd.concat([od_df.rename(columns={'origin_id':'zone_id','origin':'geometry'})[['zone_id','geometry']],
                               od_df.rename(columns={'destination_id':'zone_id','destination':'geometry'})[['zone_id','geometry']]])
        self.zones = new_zones.set_index('zone_id')
        self.zones = self.zones[~self.zones.index.duplicated(keep='first')]
        self.centroids = self.zones['geometry']

        #time restriction to links
        self.links = self.pt_links[(self.pt_links.dep_sec>=start_time)
                                   & (self.pt_links.arr_sec<end_time)
                                   & (self.pt_links.dep_wd==weekday)]
        assert len(self.links) > 0, 'No links found within the filter criteria'
        self.nodes = self.pt_nodes.loc[np.unique(np.append(self.links.a.values,self.links.b.values))]
        self.footpaths = self.pt_footpaths[(self.pt_footpaths.a.isin(self.nodes.index))
                                           | (self.pt_footpaths.b.isin(self.nodes.index))].drop_duplicates(subset = ['a','b'])
        #Connectorts between zone and transit
        self.zone_to_transit = pd.concat([self.zone_to_transit, engine.ntlegs_from_centroids_and_nodes(
                                            self.zones, self.nodes, n_neighbors=min([6,len(self.nodes)]))]).drop_duplicates(subset = ['a','b'])
        self.zone_to_transit.index = self.zone_to_transit.reset_index().index.map(lambda x: f'zone2transit-link_{x:08x}')

        # Pathfinding
        ppf = PublicPathFinder(self)
        if not differentiate_modes:
            ppf.find_best_path(od_set = od_set,
                            boarding_time = boarding_time,
                            ntlegs_penalty=1e9)
        else:
            ppf.find_best_paths(od_set = od_set,
                                broken_modes = differentiate_modes,
                                boarding_time = boarding_time,
                                ntlegs_penalty=1e9)
        
        # Analyse and filter the path
        self.pt_los = ppf.best_paths
        self.pt_los = self.pt_los.loc[self.pt_los['origin']!=self.pt_los['destination']]
        self.ptl = self.pt_los
        self.pt_los = analysis.path_analysis_od_matrix(
                od_matrix=self.pt_los,
                links=self.links,
                nodes=self.nodes,
                centroids=self.centroids)
        self.pt_los.drop(['pathfinder_session', 'reversed'], axis=1, errors='ignore', inplace=True)
        
        #Subtract waiting time from first link
        self.pt_los.gtime = self.pt_los.gtime - (self.pt_los.link_path.map(
            lambda x:  self.links.loc[x[0],'headway'] if len(x)> 0 else  0) * 0.5)

        # Add main modes
        self.pt_los['route_type'] = 'pt'
        self.pt_los['route_types'] = self.pt_los['link_path'].apply(
            lambda l: list(set(self.links.loc[l, 'route_type'])))
        
        
        # Finishing
        self.pt_los.rename(columns={'gtime':'time',
                                    'time_link_path': 'time_on_transit',
                                    'length_link_path': 'length'}, 
                                    inplace=True)
        self.pt_los['time_range'] = weekday+'_'+str(start_time)+'-'+str(end_time)

        # Time to min; Length to km
        self.pt_los.time = self.pt_los.time/60
        self.pt_los.time_on_transit = self.pt_los.time_on_transit/60
        self.pt_los.length = self.pt_los.length/1000
        
        # Make route only by foot improbable:
        self.pt_los.time = self.pt_los.all_walk*1e6+self.pt_los.time

        if all_path_analysis:
            # Add all list with Link ID's to pt_los table
            all_links = pd.concat([self.pt_links, self.footpaths, self.zone_to_transit])
            all_links['idx'] = all_links.index
            all_links = all_links.set_index(['a','b'])

            # build list with all used links
            footpaths  = pd.concat([self.zone_to_transit, self.footpaths])
            def get_link_path_all(p0,p1):
                if p0 in footpaths.a.values and p1 in footpaths.b.values:
                    return footpaths.loc[(footpaths.a == p0 ) & (footpaths.b == p1)].index[0]
                elif p0 in self.pt_links.index and p0 in self.pt_links.index:
                    return p0
                else:
                    return np.nan
            self.pt_los['link_path_all'] = self.pt_los.path.map(lambda x:pd.DataFrame(list(zip(x[:-1],x[1:])),columns = ['p0','p1']))
            self.pt_los['link_path_all'] = self.pt_los.link_path_all.map(
                                                lambda x: x.apply(lambda y: get_link_path_all(y.p0,y.p1), axis = 1).dropna().values)


    def pathfinding_park_and_ride(self, od_df, weekday, start_time=0, end_time=24*3600,
                                  pt_first=False, boarding_time=4*60, distance_factor=1.2):
        """
        Finds path from origin to destination by driving a car and using public transportation.
        Modes are changed exactly once per trip at station selected from the specified stations in self.pr_parking
        For both individual modes their respective pathfinding is used.
        Returns fastest path with mode change.

        od_df: pandas.DataFrame
            Origin-destination table with the following columns:
            - `start_time`  : datetime
                The start time of the journey.
            - `origin_id`   : int or str
                Unique identifier of the origin
            - `destination_id`: int or str
                Unique identifier of the destination
            - `origin`      : shapely.geometry.Point
                The starting point of the journey, represented as a Shapely Point object.
            - `destination` : shapely.geometry.Point
                The destination point of the journey, represented as a Shapely Point object.
            - `activity`    : str
                The type of activity or purpose of the journey.
        weekday: str ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            Which day of the week to use as public transport service.
        start_time: int
            Seconds elapsed from 00:00h for filtering of links to specific time
            intervals. Defaults to 0, in which case no start time filter
            is applied.
        end_time: int
            Seconds elapsed from 00:00h for filtering of links to specific time
            intervals. Defaults to 24 * 3600, in which case no end time filter
            is applied.
        boarding_time: float
            time [s] that is added to total travel time as penalty for every
            interchange. Defaults to `None` to skip.
        pt_first: bool
            if True: start with public transit
            if False: start with driving
        distance_factor: float
            factor how much longer the total distance (in a straight line) can be compared to the direct distance from start- to endpoint:
            distance (straight line) start point -> parking spot / station -> destination has to be smaller than
            distance facttor * distance (straight line) start->destination


        Attributes
        ----------
        self.pr_los: pandas.DataFrame
            Dataframe containing the quickes Path for each OD-pair from the od_df and the following columns:
            - origin:           id of start zone
            - destination:      id of end zone
            - time:             total trip duration [s]
            - parking_id:       id of interchange station between car and public transit
            - path_car:         analog to road-pathfinder for driving part of trip
            - link_path_car:    analog to road-pathfinder for driving part of trip        
            - time_car:         analog to road-pathfinder for driving part of trip
            - path_pt:          analog to pt-pathfinder for transit part of trip 
            - link_path_pt:     analog to pt-pathfinder for transit part of trip       
            - time_pt:          analog to pt-pathfinder for transit part of trip
            - transfers:        number of interchagnges while on public transit
            - time_on_transit:  net travelling time on public transit
            - trip_id:          unique id of trip

        """
        #Set Start- & Endpoints as zones
        od_df['origin_id'] = od_df['origin_id'].astype(str)
        od_df['destination_id'] = od_df['destination_id'].astype(str)
        new_zones = pd.concat([od_df.rename(columns={'origin_id':'zone_id','origin':'geometry'})[['zone_id','geometry']],
                               od_df.rename(columns={'destination_id':'zone_id','destination':'geometry'})[['zone_id','geometry']]])
        new_zones = new_zones.set_index('zone_id')
        self.pr_parking['park_id'] = self.pr_parking.index
        pidx = self.pr_parking[['geometry']].copy()
        pidx.index.name = 'zone_id'
        self.zones = pd.concat([new_zones, pidx])
        self.zones = self.zones[~self.zones.index.duplicated(keep='first')]
        self.zones.index = self.zones.index.astype(str)
        self.centroids = self.zones['geometry']
        
        od_df['trip_num'] = od_df.index
        #get all possible combinations of start-destination-pairs and parkingspots
        parkingspots = self.pr_parking.rename(columns={'geometry':'loc_parking'}) 
        all_options = pd.merge(left = od_df, right = parkingspots, how = 'cross')
        #caluclate distances between zones and parking spot
        all_options['dist0'] = distance(gpd.GeoSeries(data = all_options.origin.values,crs = 4326).to_crs(crs = 3035),
                            gpd.GeoSeries(data = all_options.destination.values,crs = 4326).to_crs(crs = 3035))
        all_options['dist1'] = distance(gpd.GeoSeries(data = all_options.origin.values,crs = 4326).to_crs(crs = 3035),
                                gpd.GeoSeries(data = all_options.loc_parking.values,crs = 4326).to_crs(crs = 3035))
        all_options['dist2'] = distance(gpd.GeoSeries(data = all_options.loc_parking.values,crs = 4326).to_crs(crs = 3035),
                                    gpd.GeoSeries(data = all_options.destination.values,crs = 4326).to_crs(crs = 3035))
        #drop options, where parking spot is to far away from start & destination
        all_options = all_options[(all_options.dist1<all_options.dist0)
                                  &(all_options.dist2<all_options.dist0)
                                  &((all_options.dist1+all_options.dist2)<distance_factor*all_options.dist0)]

        if len(all_options) == 0:
            self.pr_los = pd.DataFrame()
            return
        #Timerestiction on PT-links
        self.links = self.pt_links[(self.pt_links.dep_sec>=start_time)
                                   & (self.pt_links.arr_sec<end_time)
                                   & (self.pt_links.dep_wd==weekday)]
        assert len(self.links) > 0, 'No links found within the filter criteria'
        self.nodes = self.pt_nodes.loc[np.unique(np.append(self.links.a.values,self.links.b.values))]
        self.footpaths = self.pt_footpaths[(self.pt_footpaths.a.map(lambda x: x in self.nodes.index))
                |(self.pt_footpaths.b.map(lambda x: x in self.nodes.index))]
        
        self.road_links = self.auto_links.copy()
        self.road_nodes = self.auto_nodes.copy()
        #distinguish wether first leg of the journey is with Car or Public transport
        if pt_first == False:  
            #genereate od_set for all remaining options for street- and pt-pathfinder
            od_set_1 = set(zip(all_options.origin_id, all_options.park_id))
            od_set_2 = set(zip(all_options.park_id, all_options.destination_id)) 
            #Connectors to Road Network
            start2road = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.origin_id.values)], self.auto_nodes,n_neighbors=1)
            road2park = engine.ntlegs_from_centroids_and_nodes(self.pr_parking, self.auto_nodes, n_neighbors=1)
            z2r = pd.concat([start2road, road2park])
            z2r.index = z2r.reset_index().index.map(lambda x: f'zone2road_pr0-link_'+str(uuid.uuid1()))
            self.zone_to_road = pd.concat([self.zone_to_road, z2r]).drop_duplicates(subset = ['a','b'])
            
            #Connectors to PT-Network
            park2transit = engine.ntlegs_from_centroids_and_nodes(self.pr_parking, self.nodes, n_neighbors=1)
            transit2zone = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.destination_id.values)], self.nodes, n_neighbors=6)
            self.zone_to_transit=pd.concat([self.zone_to_transit, park2transit,transit2zone]).drop_duplicates(subset = ['a','b'])
            self.zone_to_transit.index = self.zone_to_transit.reset_index().index.map(lambda x: f'zone2transit_pt0-link_{x:08d}')

        elif pt_first == True:  
            #genereate od_set for all remaining options for street- and pt-pathfinder
            od_set_1 = set(zip(all_options.park_id, all_options.destination_id))
            od_set_2 = set(zip(all_options.origin_id, all_options.park_id)) 
            #Connectors to Road Network
            road2dest = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.destination_id.values)], self.auto_nodes,n_neighbors=1)
            road2park = engine.ntlegs_from_centroids_and_nodes(self.pr_parking, self.auto_nodes, n_neighbors=1)
            z2r = pd.concat([road2dest, road2park])
            z2r.index = z2r.reset_index().index.map(lambda x: f'zone2road_pr1-link_'+str(uuid.uuid1()))   
            self.zone_to_road = pd.concat([self.zone_to_road, z2r]).drop_duplicates(subset = ['a','b'])
                   
            #Connectors to PT-Network
            park2transit = engine.ntlegs_from_centroids_and_nodes(self.pr_parking, self.nodes, n_neighbors=1)
            start2transit = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.origin_id.values)], self.nodes, n_neighbors=6)
            self.zone_to_transit=pd.concat([self.zone_to_transit, park2transit,start2transit]).drop_duplicates(subset = ['a','b'])
            self.zone_to_transit.index = self.zone_to_transit.reset_index().index.map(lambda x: f'zone2transit_pt1-link_{x:08d}')

        #call pathfinder:
        self.step_road_pathfinder(method = 'aon',od_set = od_set_1)
        car_los_temp = self.car_los
        # Add list of all Link indices to auto_los table
        all_links_car = pd.concat([self.zone_to_road, self.auto_links])
        all_links_car['idx'] = all_links_car.index
        all_links_car = all_links_car.set_index(['a','b'])

        car_los_temp['node_pairs_all'] =  car_los_temp.path.map(lambda x: list(zip(x[:-1],x[1:])))
        car_los_temp['link_path_all'] = car_los_temp.node_pairs_all.map(lambda x: all_links_car.loc[x].idx.values)


        ppf = PublicPathFinder(self)
        ppf.find_best_path(od_set = od_set_2, boarding_time = boarding_time, ntlegs_penalty=1e9)
        pt_analysis = analysis.path_analysis_od_matrix(od_matrix=ppf.best_paths,
                                                        links=self.links,
                                                        nodes=self.nodes,
                                                        centroids=self.centroids,
                                                        agg={'link_path': ['time']})
        
        
        # build list with all used links
        footpaths  = pd.concat([self.zone_to_transit, self.footpaths])
        def get_link_path_all(p0,p1):
            if p0 in footpaths.a.values and p1 in footpaths.b.values:
                return footpaths.loc[(footpaths.a == p0 ) & (footpaths.b == p1)].index[0]
            elif p0 in self.pt_links.index and p0 in self.pt_links.index:
                return p0
            else:
                return np.nan
        pt_analysis['link_path_all'] = pt_analysis.path.map(lambda x:pd.DataFrame(list(zip(x[:-1],x[1:])),columns = ['p0','p1']))
        pt_analysis['link_path_all'] = pt_analysis.link_path_all.map(
                                            lambda x: x.apply(lambda y: get_link_path_all(y.p0,y.p1), axis = 1).dropna().values)

        
        # Add all list with Link ID's to pt_los table
        #all_links_pt = pd.concat([self.pt_links, self.footpaths, self.zone_to_transit])
        #all_links_pt['idx'] = all_links_pt.index
        #all_links_pt = all_links_pt.set_index(['a','b'])
        #Build node list
        #tmp = pt_analysis.explode('path')
        #tmp['path_nds'] = tmp.path.map(lambda x: self.pt_links.loc[x,'a'] if x in self.pt_links.index else x)
        #pt_analysis['node_path_all'] = tmp.groupby(['origin','destination'],as_index = False)['path_nds'].apply(list)['path_nds']
        #pt_analysis['node_path_all'] = pt_analysis.node_path_all.map(lambda x: pd.unique(pd.Series(x)))
        #Build Node pairs for links
        #pt_analysis['node_pairs_all'] = pt_analysis.node_path_all.map(lambda x: list(zip(x[:-1],x[1:])))
        #find Link indices and build list
        #pt_analysis['link_path_all'] = pt_analysis.node_pairs_all.map(lambda x:all_links_pt.loc[x,'idx'])

        #pt_analysis['link_path_all'] = pt_analysis.index.map(lambda i: [x if (x not in self.pt_links.index) 
        #                                            or x in pt_analysis.loc[0,'link_path']
        #                                            else None for x in pt_analysis.loc[i,'link_path_all']])
       # pt_analysis['link_path_all'] = pt_analysis.link_path_all.map(lambda x: list(pd.Series(x).dropna()))

        pt_analysis = pt_analysis.drop(columns=['origin','destination', 'pathfinder_session','reversed','path'])

        pt_los_temp = pd.merge(ppf.best_paths, pt_analysis, left_index=True, right_index=True, suffixes = (None,'_'))
        
        self.prptlos_debug = pt_los_temp
        pivott1 = all_options.pivot_table(index = 'origin_id', columns = 'park_id',
                                                 values =  'trip_num', aggfunc=list)
        pivott2 = all_options.pivot_table(index = 'park_id', columns = 'destination_id', 
                                                values =  'trip_num', aggfunc=list)
        #self.pt1 = pivott1
        #self.pt2 = pivott2
        if pt_first == False:
            car_los_temp['trip_id'] =car_los_temp.apply(lambda x: pivott1.loc[x.origin, x.destination],axis = 1)
            pt_los_temp['trip_id'] = pt_los_temp.apply(lambda x: pivott2.loc[x.origin, x.destination],axis = 1)
        elif pt_first == True:
            car_los_temp['trip_id'] =car_los_temp.apply(lambda x: pivott2.loc[x.origin, x.destination],axis = 1)
            pt_los_temp['trip_id'] = pt_los_temp.apply(lambda x: pivott1.loc[x.origin, x.destination],axis = 1)
        
        pt_los_temp = pt_los_temp.explode(column = 'trip_id')
        car_los_temp = car_los_temp.explode(column = 'trip_id')
        #produce combined los table
        assert isinstance(pt_first, bool)
        if pt_first == False:
            pr_los = pd.merge(left=car_los_temp, right=pt_los_temp,
                left_on = ['trip_id', 'destination'],
                right_on = ['trip_id', 'origin'], suffixes = ('_car','_pt'))
            pr_los=pr_los.rename(columns=
                {'origin_car':'origin','destination_pt':'destination', 'destination_car':'parking_id',
                 'time_link_path':'time_on_transit','gtime_car':'time_car','gtime_pt':'time_pt'})
            pr_los['link_path_all'] = pr_los.index.map(lambda x: list(pr_los.loc[x,'link_path_all_car'])
                                                                +list(pr_los.loc[x,'link_path_all_pt']))
        elif pt_first == True:
            pr_los = pd.merge(left = pt_los_temp, right  = car_los_temp,
                left_on = ['trip_id', 'destination'],
                right_on = ['trip_id', 'origin'], suffixes = ('_pt','_car'))
            pr_los=pr_los.rename(columns=
                {'origin_pt':'origin','destination_car':'destination', 'destination_pt':'parking_id',
                 'time_link_path':'time_on_transit','gtime_car':'time_car','gtime_pt':'time_pt'})
            pr_los['link_path_all'] =  pr_los.index.map(lambda x: list(pr_los.loc[x,'link_path_all_pt'])
                                                                +list(pr_los.loc[x,'link_path_all_car']))
        
        pr_los.time_car = pr_los.time_car/60
        pr_los.time_pt = pr_los.time_pt/60
        pr_los['time'] = pr_los.time_car+pr_los.time_pt
        pr_los = pr_los[['origin', 'parking_id','destination','trip_id',
                                'path_car', 'time_car','path_pt', 'time_pt',
                                'link_path_car','link_path_pt','time_on_transit',
                                'time','transfers','link_path_all','link_path_all_car','link_path_all_pt']]
        
        #select fastes route for each trip
        self.pr_los = pr_los.loc[pr_los.groupby('trip_id').time.idxmin()]
        self.pr_los = self.pr_los.loc[self.pr_los['origin']!=self.pr_los['destination']]
        self.pr_los = self.pr_los.loc[self.pr_los['link_path_pt'].apply(lambda l: len(l)>0)]
        self.pr_los = self.pr_los.set_index('trip_id')

        #Subtract waiting time from first PT-link 
        self.pr_los.time_pt = self.pr_los.time_pt - (self.pr_los.link_path_pt.map(
            lambda x:  self.links.loc[x[0],'headway'] if len(x)> 0 else  0)*0.5)

        # Compute distances
        self.pr_los['length_car'] = self.pr_los['link_path_car'].apply(
            lambda p: self.road_links.loc[p, 'length'].sum())/1000
        self.pr_los['length_pt'] = self.pr_los['link_path_pt'].apply(
            lambda p: self.links.loc[p, 'length'].sum())/1000
        self.pr_los['length'] = self.pr_los['length_car'] + self.pr_los['length_pt']

        # Add main modes
        self.pr_los['route_type'] = 'pr'
        self.pr_los['route_types'] = self.pr_los['link_path_pt'].apply(
            lambda l: ['car'] + list(set(self.links.loc[l, 'route_type'])))
        # Add time slot
        self.pr_los['time_range'] = weekday+'_'+str(start_time)+'-'+str(end_time)
    

    def pathfinding_cycle_and_ride(self, od_df, weekday, start_time=0, end_time=24*3600,
                                   pt_first=False, bike_type='normal',
                                   boarding_time=4*60, distance_factor=1.2):
        """
        Finds path from origin to destination by cycling and using public transportation.
        Modes are changed exactly once per trip at station selected from the specified stations in self.pr_parking
        For both individual modes their respective pathfinding is used.
        Returns fastest path with mode change.

        od_df: pandas.DataFrame
            Origin-destination table with the following columns:
            - `start_time`  : datetime
                The start time of the journey.
            - `origin_id`   : int or str
                Unique identifier of the origin
            - `destination_id`: int or str
                Unique identifier of the destination
            - `origin`      : shapely.geometry.Point
                The starting point of the journey, represented as a Shapely Point object.
            - `destination` : shapely.geometry.Point
                The destination point of the journey, represented as a Shapely Point object.
            - `activity`    : str
                The type of activity or purpose of the journey.
        weekday: str ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            Which day of the week to use as public transport service.
        start_time: int
            Seconds elapsed from 00:00h for filtering of links to specific time
            intervals. Defaults to 0, in which case no start time filter
            is applied.
        end_time: int
            Seconds elapsed from 00:00h for filtering of links to specific time
            intervals. Defaults to 24 * 3600, in which case no end time filter
            is applied.
        boarding_time: float
            time [s] that is added to total travel time as penalty for every
            interchange. Defaults to `None` to skip.
        pt_first: bool
            if True: start with public transit
            if False: start with driving
        distance_factor: float
            factor how much longer the total distance (in a straight line) can be compared to the direct distance from start- to endpoint:
            distance (straight line) start point -> parking spot / station -> destination has to be smaller than
            distance facttor * distance (straight line) start->destination


        Attributes
        ----------
        self.pr_los: pandas.DataFrame
            Dataframe containing the quickes Path for each OD-pair from the od_df and the following columns:
            - origin:           id of start zone
            - destination:      id of end zone
            - time:             total trip duration [s]
            - parking_id:       id of interchange station between car and public transit
            - path_cycle:       analog to road-pathfinder for driving part of trip
            - link_path_cycle:  analog to road-pathfinder for driving part of trip        
            - time_cycle:       analog to road-pathfinder for driving part of trip
            - path_pt:          analog to pt-pathfinder for transit part of trip 
            - link_path_pt:     analog to pt-pathfinder for transit part of trip       
            - time_pt:          analog to pt-pathfinder for transit part of trip
            - transfers:        number of interchagnges while on public transit
            - time_on_transit:  net travelling time on public transit
            - trip_id:          unique id of trip

        """
        #Set Start- & Endpoints as zones
        od_df['origin_id'] = od_df['origin_id'].astype(str)
        od_df['destination_id'] = od_df['destination_id'].astype(str)
        new_zones = pd.concat([od_df.copy().rename(columns={'origin_id':'zone_id','origin':'geometry'})[['zone_id','geometry']],
                               od_df.copy().rename(columns={'destination_id':'zone_id','destination':'geometry'})[['zone_id','geometry']]])
        new_zones = new_zones.set_index('zone_id')
        self.pc_stations['park_id'] = self.pc_stations.index
        pidx = self.pc_stations[['geometry']].copy()
        pidx.index.name = 'zone_id'
        self.zones = pd.concat([new_zones, pidx])
        self.zones = self.zones[~self.zones.index.duplicated(keep='first')]
        self.zones.index = self.zones.index.astype(str)
        self.centroids=self.zones['geometry']
        
        od_df['trip_num'] = od_df.index
        #get all possible combinations of start-destination-paris and Stations where transfer to cycling network is possible
        transferStations = self.pc_stations.rename(columns={'geometry':'loc_parking'}) 
        all_options = pd.merge(left = od_df, right = transferStations, how = 'cross')
        #caluclate distances between zones and parking spot
        all_options['dist0'] = distance(gpd.GeoSeries(all_options.origin.values, crs = 4326).to_crs(crs = 3035),
                            gpd.GeoSeries(all_options.destination.values, crs = 4326).to_crs(crs = 3035))
        all_options['dist1'] = distance(gpd.GeoSeries(all_options.origin.values, crs = 4326).to_crs(crs = 3035),
                                gpd.GeoSeries(all_options.loc_parking.values, crs = 4326).to_crs(crs = 3035))
        all_options['dist2'] = distance(gpd.GeoSeries(all_options.loc_parking.values, crs = 4326).to_crs(crs = 3035),
                                    gpd.GeoSeries(all_options.destination.values, crs = 4326).to_crs(crs = 3035))
        #drop options, where parking spot is to far away from start & destination
        all_options = all_options[(all_options.dist1<all_options.dist0)
                                  &(all_options.dist2<all_options.dist0)
                                  &((all_options.dist1+all_options.dist2)<distance_factor*all_options.dist0)]
        if len(all_options) == 0:
            self.pr_los = pd.DataFrame()
            return
        #Timerestiction on PT-links
        self.links = self.pt_links[(self.pt_links.dep_sec>=start_time)
                                   & (self.pt_links.arr_sec<end_time)
                                   & (self.pt_links.dep_wd==weekday)]
        assert len(self.links) > 0, 'No links found within the filter criteria'
        self.nodes = self.pt_nodes.loc[np.unique(np.append(self.links.a.values,self.links.b.values))]
        self.footpaths = self.pt_footpaths[(self.pt_footpaths.a.map(lambda x: x in self.nodes.index))
                |(self.pt_footpaths.b.map(lambda x: x in self.nodes.index))]
        
        self.road_links = self.cycle_links.copy()
        self.road_nodes = self.cycle_nodes.copy()
        #distinguish wether first leg of the journey is with Car or Public transport
        if pt_first == False:  
            #genereate od_set for all remaining options for street- and pt-pathfinder
            od_set_1 = set(zip(all_options.origin_id, all_options.park_id))
            od_set_2 = set(zip(all_options.park_id, all_options.destination_id)) 
            #Connectors to Road Network
            start2road = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.origin_id.values)], self.cycle_nodes,n_neighbors=1)
            road2park = engine.ntlegs_from_centroids_and_nodes(self.pc_stations, self.cycle_nodes, n_neighbors=1)
            z2r = pd.concat([start2road, road2park])
            z2r.index = z2r.reset_index().index.map(lambda x: f'zone2road_pc0-link_'+str(uuid.uuid1()))
            self.zone_to_road = pd.concat([self.zone_to_road, z2r]).drop_duplicates(subset = ['a','b'])
            #Connectors to PT-Network
            park2transit = engine.ntlegs_from_centroids_and_nodes(self.pc_stations, self.nodes, n_neighbors=1)
            transit2zone = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.destination_id.values)], self.nodes, n_neighbors=6)
            self.zone_to_transit=pd.concat([self.zone_to_transit, park2transit,transit2zone]).drop_duplicates(subset = ['a','b'])
            self.zone_to_transit.index = self.zone_to_transit.reset_index().index.map(lambda x: f'zone2transit_pc0-link_{x:08d}')
        elif pt_first == True:  
            #genereate od_set for all remaining options for street- and pt-pathfinder
            od_set_1 = set(zip(all_options.park_id, all_options.destination_id))
            od_set_2 = set(zip(all_options.origin_id, all_options.park_id)) 
            #Connectors to Road Network
            road2dest = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.destination_id.values)], self.cycle_nodes,n_neighbors=1)
            road2park = engine.ntlegs_from_centroids_and_nodes(self.pc_stations, self.cycle_nodes, n_neighbors=1)
            z2r = pd.concat([road2dest, road2park])
            z2r.index = z2r.reset_index().index.map(lambda x: f'zone2road_pc1-link_'+str(uuid.uuid1()))
            self.zone_to_road = pd.concat([self.zone_to_road, z2r]).drop_duplicates(subset = ['a','b'])
            #Connectors to PT-Network
            park2transit = engine.ntlegs_from_centroids_and_nodes(self.pc_stations, self.nodes, n_neighbors=1)
            start2transit = engine.ntlegs_from_centroids_and_nodes(self.zones.loc[np.unique(all_options.origin_id.values)], self.nodes, n_neighbors=6)
            self.zone_to_transit=pd.concat([self.zone_to_transit, park2transit,start2transit]).drop_duplicates(subset = ['a','b'])
            self.zone_to_transit.index = self.zone_to_transit.reset_index().index.map(lambda x: f'zone2transit_pc1-link_{x:08d}')

        
        #calculate time on segment
        self.road_links.speed = self.road_links.way_quality.apply(lambda s: biking_speed(s, bike_type))
        self.road_links.time = self.road_links['length'] / (self.road_links['speed']/3.6)
        #call pathfinder:
        self.step_road_pathfinder(method = 'aon',od_set = od_set_1)
        cycle_los_temp = self.car_los

        # Add list of all Link indices to auto_los table
        all_links_cycle = pd.concat([self.zone_to_road, self.cycle_links])
        all_links_cycle['idx'] = all_links_cycle.index
        all_links_cycle = all_links_cycle.set_index(['a','b'])

        cycle_los_temp['node_pairs_all'] =  cycle_los_temp.path.map(lambda x: list(zip(x[:-1],x[1:])))
        cycle_los_temp['link_path_all'] = cycle_los_temp.node_pairs_all.map(lambda x: all_links_cycle.loc[x].idx.values)



        ppf = PublicPathFinder(self)
        ppf.find_best_path(od_set = od_set_2, boarding_time = boarding_time, ntlegs_penalty=1e9)
        pt_analysis = analysis.path_analysis_od_matrix(od_matrix=ppf.best_paths,
                                                        links=self.links,
                                                        nodes=self.nodes,
                                                        centroids=self.centroids,
                                                        agg={'link_path': ['time']})

        # build list with all used links
        footpaths  = pd.concat([self.zone_to_transit, self.footpaths])
        def get_link_path_all(p0,p1):
            if p0 in footpaths.a.values and p1 in footpaths.b.values:
                return footpaths.loc[(footpaths.a == p0 ) & (footpaths.b == p1)].index[0]
            elif p0 in self.pt_links.index and p0 in self.pt_links.index:
                return p0
            else:
                return np.nan
        pt_analysis['link_path_all'] = pt_analysis.path.map(lambda x:pd.DataFrame(list(zip(x[:-1],x[1:])),columns = ['p0','p1']))
        pt_analysis['link_path_all'] = pt_analysis.link_path_all.map(
                                            lambda x: x.apply(lambda y: get_link_path_all(y.p0,y.p1), axis = 1).dropna().values)
        
        
        # # Add all list with Link ID's to pt_los table
        # all_links_pt = pd.concat([self.pt_links, self.footpaths, self.zone_to_transit])
        # all_links_pt['idx'] = all_links_pt.index
        # all_links_pt = all_links_pt.set_index(['a','b'])
        # #Build node list
        # tmp = pt_analysis.explode('path')
        # tmp['path_nds'] = tmp.path.map(lambda x: self.pt_links.loc[x,'a'] if x in self.pt_links.index else x)
        # pt_analysis['node_path_all'] = tmp.groupby(['origin','destination'],as_index = False)['path_nds'].apply(list)['path_nds']
        # pt_analysis['node_path_all'] = pt_analysis.node_path_all.map(lambda x: pd.unique(pd.Series(x)))
        # #Build Node pairs for links
        # pt_analysis['node_pairs_all'] = pt_analysis.node_path_all.map(lambda x: list(zip(x[:-1],x[1:])))
        # #find Link indices and build list
        # pt_analysis['link_path_all'] = pt_analysis.node_pairs_all.map(lambda x:all_links_pt.loc[x,'idx'])

        # pt_analysis['link_path_all'] = pt_analysis.index.map(lambda i: [x if (x not in self.pt_links.index) 
        #                                             or x in pt_analysis.loc[0,'link_path']
        #                                             else None for x in pt_analysis.loc[i,'link_path_all']])
        # pt_analysis['link_path_all'] = pt_analysis.link_path_all.map(lambda x: list(pd.Series(x).dropna()))
        
        pt_analysis = pt_analysis.drop(columns=['origin','destination', 'pathfinder_session','reversed','path'])
        pt_los_temp = pd.merge(ppf.best_paths, pt_analysis, left_index=True, right_index=True, suffixes = (None,'_'))
        
        pivott1 = all_options.pivot_table(index = 'origin_id', columns = 'park_id',
                                                 values =  'trip_num', aggfunc=list)
        pivott2 = all_options.pivot_table(index = 'park_id', columns = 'destination_id', 
                                                values =  'trip_num', aggfunc=list)
        #self.pt1 = pivott1
        #self.pt2 = pivott2
        if pt_first == False:
            cycle_los_temp['trip_id'] =cycle_los_temp.apply(lambda x: pivott1.loc[x.origin, x.destination],axis = 1)
            pt_los_temp['trip_id'] = pt_los_temp.apply(lambda x: pivott2.loc[x.origin, x.destination],axis = 1)
        elif pt_first == True:
            cycle_los_temp['trip_id'] =cycle_los_temp.apply(lambda x: pivott2.loc[x.origin, x.destination],axis = 1)
            pt_los_temp['trip_id'] = pt_los_temp.apply(lambda x: pivott1.loc[x.origin, x.destination],axis = 1)
        
        pt_los_temp = pt_los_temp.explode(column = 'trip_id')
        cycle_los_temp = cycle_los_temp.explode(column = 'trip_id')
        #produce combined los table
        assert isinstance(pt_first, bool)
        if pt_first == False:
            pr_los = pd.merge(left=cycle_los_temp, right=pt_los_temp,
                left_on = ['trip_id', 'destination'],
                right_on = ['trip_id', 'origin'], suffixes = ('_cycle','_pt'))
            pr_los=pr_los.rename(columns=
                {'origin_cycle':'origin','destination_pt':'destination', 'destination_cycle':'parking_id',
                 'time_link_path':'time_on_transit','gtime_cycle':'time_cycle','gtime_pt':'time_pt'})
            pr_los['link_path_all'] = pr_los.index.map(lambda x: list(pr_los.loc[x,'link_path_all_cycle'])
                                                                +list(pr_los.loc[x,'link_path_all_pt']))
        elif pt_first == True:
            pr_los = pd.merge(left = pt_los_temp, right  = cycle_los_temp,
                left_on = ['trip_id', 'destination'],
                right_on = ['trip_id', 'origin'], suffixes = ('_pt','_cycle'))
            pr_los=pr_los.rename(columns=
                {'origin_pt':'origin','destination_cycle':'destination', 'destination_pt':'parking_id',
                 'time_link_path':'time_on_transit','gtime_cycle':'time_cycle','gtime_pt':'time_pt'})
            pr_los['link_path_all'] =  pr_los.index.map(lambda x: list(pr_los.loc[x,'link_path_all_pt'])
                                                                +list(pr_los.loc[x,'link_path_all_cycle']))
        
        pr_los.time_cycle = pr_los.time_cycle/60
        pr_los.time_pt = pr_los.time_pt/60
        pr_los['time'] = pr_los.time_cycle+pr_los.time_pt
        pr_los = pr_los[['origin', 'parking_id','destination','trip_id',
                                'path_cycle', 'time_cycle','path_pt', 'time_pt',
                                'link_path_cycle','link_path_pt','time_on_transit',
                                'time','transfers','link_path_all']]
        
        #select fastes route for each trip
        self.pc_los = pr_los.loc[pr_los.groupby('trip_id').time.idxmin()]
        self.pc_los = self.pc_los.loc[self.pc_los['origin']!=self.pc_los['destination']]
        self.pc_los = self.pc_los.loc[self.pc_los['link_path_pt'].apply(lambda l: len(l)>0)]
        self.pc_los = self.pc_los.set_index('trip_id')

        #Subtract waiting time from first PT-link 
        self.pc_los.time_pt = self.pc_los.time_pt - (self.pc_los.link_path_pt.map(
            lambda x:  self.links.loc[x[0],'headway'] if len(x)> 0 else  0)*0.5)

        # Compute distances
        self.pc_los['length_cycle'] = self.pc_los['link_path_cycle'].apply(
            lambda p: self.road_links.loc[p, 'length'].sum())/1000
        self.pc_los['length_pt'] = self.pc_los['link_path_pt'].apply(
            lambda p: self.links.loc[p, 'length'].sum())/1000
        self.pc_los['length'] = self.pc_los['length_cycle'] + self.pc_los['length_pt']

        # Add main modes
        self.pc_los['route_type'] = 'pc'
        self.pc_los['route_types'] = self.pc_los['link_path_pt'].apply(
            lambda l: ['bicycle'] + list(set(self.links.loc[l, 'route_type'])))
        # Add time slot
        self.pc_los['time_range'] = weekday+'_'+str(start_time)+'-'+str(end_time)


    def mode_choice(self, time_range=None, pt_price_dict=None):
        """
        Runs mode choice models to estimate the probability of transport mode 
        selection for various demand segments based on level-of-service (LoS) 
        attributes and estimated parameters.

        Parameters
        ----------
        time_range : str, optional
            The time range for which the mode choice model should be run. If not 
            provided, the model will run for all time ranges in the data.
        pt_price_dict: dict, optional
            Dictionary of distance thresholds and prices. The key is
            the distance at which the value (price) applies, up to the
            next key. If None is given, use default prices.
        
        Attributes
        ----------
        self.los : pandas.DataFrame
            A concatenated DataFrame that includes the level-of-service data for 
            all transport modes (car, cycling, walking, as well as public 
            transport and others based on the time range).
        
        Requires:
        ---------
        self.segments
        self.los_attributes
        self.utility_values
        self.mode_nests
        self.logit_scales
        self.mode_utility 

        Notes
        -----
        - This method first prepares the level-of-service (LoS) data for all modes, 
        combining them into a unified DataFrame. 
        - If `time_range` is provided, it filters the public transport and other mode 
        data by the given time range; otherwise, the method runs for all available 
        time ranges.
        - The method processes the model estimation results to define the utility of 
        each mode, then applies logit models to predict mode choice probabilities 
        for each demand segment.
        """

        # Create level-of-service table for all modes of transport

        if time_range:
            self.los = pd.concat([
                self.auto_los, self.cycle_los, self.walk_los,
                self.pt_los.loc[self.pt_los['time_range']==time_range],
                self.pr_los.loc[self.pr_los['time_range']==time_range],
                self.pc_los.loc[self.pc_los['time_range']==time_range]
                ]).reset_index(drop=True)
        else: # Run this method for all time ranges after another
            ranges = set(list(self.pt_los['time_range'])
                        +list(self.pr_los['time_range'])
                        +list(self.pc_los['time_range'])
                        )
            if len(ranges) > 0:
                for time in ranges:
                    self.mode_choice(time)
            else: # No PT modes available
                self.los = pd.concat([
                    self.auto_los, self.cycle_los, self.walk_los
                    ]).reset_index(drop=True)

        
        #Add price to Los Table
        if not 'price' in self.los.columns or self.los['price'].isna().any():
            if pt_price_dict is None:
                self.los['price'] = self.los.apply(lambda x: self.calc_user_price(x), axis = 1)
            else:
                self.los['price'] = self.los.apply(lambda x: self.calc_user_price(x), args=(pt_price_dict,), axis = 1)
        
        # Make sure the LoS table has all perfomrance attributes
        for a in self.los_attributes:
            if not a in self.los.columns:
                for s in self.segments:
                    assert s+'_'+a in self.los.columns, \
                        'If LoS attributes are not in the los-table columns, '\
                        +'they must appear for each segment in the format of "seg_los"'
                    self.los[s+'_'+a] = self.los[s+'_'+a].fillna(0)
            else:
                self.los[a] = self.los[a].fillna(0)
                for s in self.segments:
                    if not s+'_'+a in self.los.columns:
                        self.los[s+'_'+a] = self.los[a]
        
        # Run the logit models
        self.analysis_mode_utility(how='main', segment=None)
        self.step_logit()


    def assignment(self, volumes):
        """
        Quick assignment of all volumes onto links. The volumes table must
        contain all segments of the mode choice model as columns with volumes.
        """
        # Preparations
        # Make path to tuples as workaround for pandas TypeError
        # (see issue: https://github.com/pandas-dev/pandas/issues/31177)
        self.los = self.los.loc[self.los['link_path'].notna()]
        self.los['link_path'] = self.los['link_path'].apply(lambda p: tuple(p))

        # Compute volumes for each path
        # Volumes must have segments as volume columns and IDs as origin and destination
        self.volumes = volumes
        self.compute_los_volume(keep_segments=True)

        # Assign volumes
        # First, merge all network links into one table
        links_list = [self.pt_links, self.auto_links, self.cycle_links, self.walk_links]
        all_indeces = sum([list(df.index) for df in links_list], [])
        assert len(set(all_indeces)) == len(all_indeces), \
            'Links of different networks must have different indices. There are duplicates.'
        links_df = pd.concat(links_list)
        links_df[self.segments + ['volume']] = links_df.get([self.segments + ['volume']],0)
        # Then, assign with vectorized pandas functions
        # By segment:
        #for seg in self.segments:
        #    links_df[seg] = self.los[['link_path', seg]].explode('link_path')\
        #        .groupby('link_path').sum()[seg]
        #    links_df[seg] = links_df[seg].fillna(0)
        #links_df['volume'] = links_df[self.segments].sum(axis=1)
        # Or all at once:
        links_df[['volume']+self.segments] = self.los[['link_path', 'volume']+self.segments].explode('link_path')\
            .groupby('link_path').sum()[['volume']+self.segments]
        links_df[['volume']+self.segments] = links_df[['volume']+self.segments].fillna(0)
        
        self.ldf = links_df

        # Now, copy link load in class attributes
        for links in links_list:
            links[self.segments + ['volume']] = \
                links_df.loc[links.index, self.segments + ['volume']]
   
    def save_networks_road(self, path_to_folder):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        
        try:    
            self.auto_links.to_file(path_to_folder+'auto_links.geojson',driver='GeoJSON')
            self.auto_nodes.to_file(path_to_folder+'auto_nodes.geojson',driver='GeoJSON')
            print('Stored car networt at: '+path_to_folder)
        except:
            pass
        try:
            self.cycle_links.to_file(path_to_folder+'cycle_links.geojson',driver='GeoJSON')
            self.cycle_nodes.to_file(path_to_folder+'cycle_nodes.geojson',driver='GeoJSON')
            print('Stored cycle networt at: '+path_to_folder)
        except:
            pass
        try:
            self.walk_links.to_file(path_to_folder+'walk_links.geojson',driver='GeoJSON')
            self.walk_nodes.to_file(path_to_folder+'walk_nodes.geojson',driver='GeoJSON')
            print('Stored walking networt at: '+path_to_folder)
        except:
            pass
    def load_networks_road(self, path_to_folder):
        self.auto_links = gpd.read_file(path_to_folder+'auto_links.geojson')
        self.auto_links = self.auto_links.set_index('index')
        self.auto_nodes = gpd.read_file(path_to_folder+'auto_nodes.geojson')
        self.auto_nodes = self.auto_nodes.set_index('index')

        self.cycle_links = gpd.read_file(path_to_folder+'cycle_links.geojson')
        self.cycle_links = self.cycle_links.set_index('index')
        self.cycle_nodes = gpd.read_file(path_to_folder+'cycle_nodes.geojson')
        self.cycle_nodes = self.cycle_nodes.set_index('index')

        self.walk_links = gpd.read_file(path_to_folder+'walk_links.geojson')
        self.walk_links = self.walk_links.set_index('index')
        self.walk_nodes = gpd.read_file(path_to_folder+'walk_nodes.geojson')
        self.walk_nodes = self.walk_nodes.set_index('index')
   
    def save_network_pt(self, path_to_folder):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        
        self.pt_nodes.to_file(path_to_folder+'pt_nodes.geojson', driver = 'GeoJSON')
        self.pt_links.to_file(path_to_folder+'pt_links.geojson', driver = 'GeoJSON')
        self.pt_footpaths.to_file(path_to_folder+'pt_footlinks.geojson', driver = 'GeoJSON')
    
    def load_network_pt(self, path_to_folder):
        self.pt_links = gpd.read_file(path_to_folder+'pt_links.geojson')
        self.pt_links = self.pt_links.set_index('index')
        self.pt_nodes = gpd.read_file(path_to_folder+'pt_nodes.geojson')
        self.pt_nodes = self.pt_nodes.set_index('index')
        self.pt_footpaths = gpd.read_file(path_to_folder+'pt_footlinks.geojson')
        self.pt_footpaths = self.pt_footpaths.set_index('index')
    
    def save_pr_data(self, path_to_folder):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        self.pr_parking.to_file(path_to_folder+'PR_parkingspots.geojson',driver = 'GeoJSON')
    
    def load_pr_data(self, path_to_folder):
        self.pr_parking = gpd.read_file(path_to_folder+'PR_parkingspots.geojson')
        self.pr_parking = self.pr_parking.set_index('node-parking_id')
    
        # -> TL!!!
    
    def calc_user_price(self, los_row, agent_list=None,
                        car_per_km=0.14, cs_per_km=0.27, cs_per_h=2.60,
                        pt_dict={0:3.4, 15:4.7, 20:6.9, 30:11.0}):
        """
        Calculates the cost for a trip with a given mode
        
        Parameters:
        ----------
        mode: str
        length: float [m]
        car_per_km: float
            cost of travel by car [Eur/km]
        user_pt_abo: bool
            Wether user has a public transit subscription
        pid: str
            Unique user id -> self.all_persons (not used right now)
        pt_dict: dict
            Dictionary of distance thresholds and prices. The key is
            the distance at which the value (price) applies, up to the
            next key.

        Returns:
        ---------
        price of trip [Eur]
        """
        def pt_price(abo, length):
            if abo:
                return 0.0
            for dist in sorted(list(pt_dict.keys()))[::-1]:
                if length >= dist:
                    return pt_dict[dist]
            
        mode = los_row.route_type
        length = los_row.length
        time = los_row.time
        if 'pid' in los_row.index:
            pid = los_row.pid
        else:
            pid = None

        if isinstance(agent_list, pd.DataFrame) and pid is not None:
            user_pt_abo = agent_list.loc[pid,'pt_abo']
        else:
            user_pt_abo  = False

        if mode == 'car':
            return car_per_km*length
        elif mode == 'carsharing':
            return cs_per_km*length+cs_per_h*time
        
        elif mode == 'pr' or mode == 'pr_rev':
            cc = car_per_km*los_row.length_car
            cp = pt_price(user_pt_abo, los_row.length_pt)
            return cc+cp
        elif mode == 'pc' or mode == 'pc_rev':
            cc = 0
            cp = pt_price(user_pt_abo,  los_row.length_pt)
            return cc+cp
        
        elif mode == 'bicycle' or mode == 'walk':
            return 0
        elif mode == 'pt':
            return pt_price(user_pt_abo,length)

        else:
            return None
        
    def load_estimation_parameters(self, path_to_file,
                                   mode_asc_0 = 'car',
                                   modes_list = ['pt','walk','bicycle','car'],
                                   los_attributes = ['price','time']):
        """
        Runs mode choice models to estimate the probability of transport mode 
        selection for various demand segments based on level-of-service (LoS) 
        attributes and estimated parameters.

        Parameters
        ----------
        estimation_file_path : str
            The file path to the estimation results file in Excel format. This file 
            contains mode choice model parameters, i.e. alternative-specific 
            constants (ASCs), nest parameters, and beta parameters for LoS attributes.
        
        mode_asc_0 : str, optional
            The mode that was estimated with a fixed ASC of zero (default `car`).
        
        modes_list : list of str, optional
            List of modes to account for in mode choice.
            Defaults to `['pt','walk','bicycle','car']`.
        
        los_attributes : list of str
            A list of level-of-service (LoS) attribute names that will be used 
            to estimate the utility values for each mode. These correspond to 
            columns in the LoS table.

        Attributes
        ----------
        self.los : pandas.DataFrame
            A concatenated DataFrame that includes the level-of-service data for 
            all transport modes (car, cycling, walking, as well as public 
            transport and others based on the time range).
        
        self.mode_nests : pandas.DataFrame
            A DataFrame representing the nesting structure of the logit model for 
            each mode and demand segment. If any nesting parameters are present in 
            the estimation file (starting with 'mu_'), they are automatically added.
        
        self.logit_scales : pandas.DataFrame
            A DataFrame storing the logit scale parameters for each mode and demand 
            segment. The nesting parameters correspond to the logit scales.
        
        self.mode_utility : pandas.DataFrame
            A table of alternative-specific constants (ASCs) for each mode and demand 
            segment. The ASCs are loaded from the estimation results and adjusted 
            for car availability when necessary.
        
        self.utility_values : pandas.DataFrame
            A DataFrame containing beta parameters for the LoS attributes. These 
            are used to calculate the utility values for each transport mode across 
            the different demand segments.

        Notes
        -----
        - The estimation file must contain specific parameter names:
            - Nesting parameters (e.g., 'mu_*') define mode nests for demand segments.
            - Alternative-specific constants (e.g., 'asc_*') set the utility constants 
            for modes, with the ASC of `mode_asc_0` being zero.
            - Beta parameters (e.g., 'b_*') define the contribution of each LoS attribute 
            to the utility value of a mode.
        """
        
        self.los_attributes = los_attributes
        excel_file = pd.ExcelFile(path_to_file)
        params_est = {}
        for s in excel_file.sheet_names:
            params_est[s] = excel_file.parse(s, index_col=0)
        # Demand segments are estimated models
        segments = list(params_est.keys())


        # Modes are the intersection of route_types listed in the LoS table
        # and modes with ASCs in the estimated parameters
        modes_los = set(modes_list)
        modes_param = set([mode_asc_0])
        for s in segments:
            modes_param = modes_param.union(set(
                [i[4:] for i in params_est[s].index if str(i).startswith('asc_')]))

        modes = list(modes_los.intersection(modes_param))
        # Prepare nesting of the choice models
        mode_nests = pd.DataFrame(index=modes + ['root'], data={seg: 'root' for seg in segments})
        mode_nests.loc['root'] = np.nan
        mode_nests.index.name = 'route_type'
        mode_nests.columns.name = 'segment'
        logit_scales = pd.DataFrame(index=modes + ['root'],data={seg: 1 for seg in segments})

        for seg in segments:
            for i, row in params_est[seg].iterrows():
                if str(i).startswith('mu_'):
                    nest = i[3:]
                    if not nest in mode_nests.index:
                        mode_nests.loc[nest] = ['root' for s in segments]
                        logit_scales.loc[nest] = [1 for s in segments]
                    for mode in nest.split('_'):
                        mode_nests.loc[mode, seg] = nest
                        logit_scales.loc[mode, seg] = row['Value']

        mode_utility = pd.DataFrame(index=modes + ['root'],
                                                data={seg: 0.0 for seg in segments})
        for seg in segments:
            for i, row in params_est[seg].iterrows():
                if str(i).startswith('asc_'):
                    mode = i[4:]
                    if mode in modes:
                        mode_utility.loc[mode, seg] = row['Value']

        # Create table with alternative-specific constants (ASCs).
        # Requirements for ASC parameters in the estimation results file:
        # They must start with 'asc_', followed by the respective mode.
        # This method assumes the ASC of cars to be zero (if car available).
        mode_utility = pd.DataFrame(index=modes + ['root'],
                                        data={seg: 0.0 for seg in segments})
        for seg in segments:
            for i, row in params_est[seg].iterrows():
                if str(i).startswith('asc_'):
                    mode = i[4:]
                    if mode in modes:
                        mode_utility.loc[mode, seg] = row['Value']
        # Set ASC 0 mode
        for seg in segments:
            mode_utility.loc[mode_asc_0, seg] = 0
        # Make cars unavailable in households without cars
        for seg in segments:
            if 'no_car' in seg or 'without_car' in seg:
                mode_utility.loc['car', seg] = -50


        # Create table with beta parameters.
        # Requirements for beta parameters in the estimation results file:
        # They must start with 'b_', followed by the name of the corresponding
        # LoS attribute, as given to this method.
        utility_values = pd.DataFrame(
            index=[seg+'_'+a for seg in segments for a in los_attributes],
            columns=segments, dtype=float)
        utility_values.loc['mode_utility'] = [1 for seg in segments]
        for a in los_attributes:
            for seg in segments:
                try:
                    utility_values.loc[seg+'_'+a, seg] = params_est[seg]\
                        .loc['b_'+a, 'Value']
                except KeyError:
                    utility_values.loc[seg+'_'+a, seg] = 0
        utility_values = utility_values.fillna(0)
        # Set attributes required for quetzal fuctions
        self.utility_values = utility_values
        self.segments = segments
        self.mode_nests = mode_nests
        self.logit_scales = logit_scales
        self.mode_utility  = mode_utility 
#######################################################
##Build Trips

    def prepare_trips_for_model(self, timeseg_df,all_trips, doy = None):
        """Assingns trips to correct time interval
        Parameters:
        ---------
        timeseg_df: pandas.DataFrame
            DataFrame that defines the time intervals for the networks 
            - index: Time segment Name/ID
            - starttime [s] seconds of current day 
            - endtime [s] seconds of current day
        all_trips: pandas.DataFrame
        


        Returns:
        ----------
        od_demands: pandas.DataFrame
            Dataframe with one row for each day and timesegment combination with columns vols and oddf:
            - vols : pandas.DataFrame with 
            - oddf : pandas.DataFrame 

        """

        def volumes_from_trips(trips, disag = False):
            """
            Returns a dataframe that gives the volumes for each origin-destination and user segement
            Parameters:
            -----------
            trips: Pandas.DataFrame
                Pandas Dataframe that contains all relevant trips as columns

            Returns:
            ----------
            oddf: Pandas.DataFrame
                Dataframe contianing origins and destinations and their geographic coordinates for all relevant trip relations
            vols: Pandas.DataFrame
                Dataframe containing the volumes each relevant origin-destination pair and user segement
            """
            v = trips.groupby(['origin','destination','segment']).count().rename({'dpt_sec':'volume'},axis = 1)['volume']
            vols = v.reset_index(drop = False,level = 2).pivot(columns = 'segment').volume.fillna(0.0).reset_index()
            vols.index.name = ''
            vols.columns.name = ''

            if disag == False:
                oddf = trips[['origin', 'destination', 'origin_coords', 'destination_coords']].rename({'origin':'origin_id',
                                                                                        'destination':'destination_id',
                                                                                        'origin_coords':'origin',
                                                                                        'destination_coords':'destination'},axis = 1)
                oddf = oddf.drop_duplicates().reset_index(drop = True)
            else:
                oddf = trips[['origin', 'destination', 'origin_coords', 'destination_coords','pid']].rename({'origin':'origin_id',
                                                                                        'destination':'destination_id',
                                                                                        'origin_coords':'origin',
                                                                                        'destination_coords':'destination'},axis = 1)
                oddf = oddf.reset_index(drop = True)
            return oddf, vols
        #restrict activities to given days
        if doy is not None:
            trips = all_trips.loc[all_trips.doy.map(lambda x: x in doy),:].copy()
        else:
            trips = all_trips.copy()

        #Assing correct time interval
        trips.loc[:,'time_seg'] = pd.cut(trips.dpt_sec,
                                    np.append(timeseg_df.starttime.values,timeseg_df.endtime.values[-1]),
                                    labels = timeseg_df.index.values, right = False)
        
        #Build volumes and oddf dataframes for all relevant time intervals
        temp = trips.groupby(['weekday', 'doy','time_seg'], observed = False).apply(lambda x: volumes_from_trips(x, disag = False))
        od_demands = pd.DataFrame()
        od_demands['vols'] = temp.map(lambda x: x[1])
        od_demands['oddf'] = temp.map(lambda x: x[0])
        return od_demands
    
    
    def build_journeys(self, all_trips,  days = [a for a in range(20,27)], agent_list = None):
        """
        Builds data on all trips conducted on given days.
        Individual Trips (from an origin to a destination) are grouped to journeys
        (from leaving home to returning back) to allow consistent Mode Choice.
        
        Parameters:
        ----------
        days: [int]
            List of days for wich routefinding is to be performed.
            Starting from 0 for 01. Jan.
        all_trips: pandas.DataFrame


        Requires:
        ----------
        From FullModel:
            self.timeseg_dict

        From RegionalModel:
            self.auto_links
            self.pt_links
            self.walk_links
            self.cycle_links



        Builds:
            self.trips_los: pd.DataFrame
                Index: Unique trip Id
                Columns:
                - origin str (id)
                - destination str (id)
                - time float ([s])
                - length  float ([m])
                - link_path_all [str] (list of link id's)
                - route_type str

            self.journeys: pd.DataFrame
                Index: (pid, trip_no): Uniqe Person id, Unique trip number for each person
                Columns:
                - origins [str] (List of origin id's of each segment)
                - single_trip_idx [str] (List of index for each link -> self.trip_los)
                - rel_mode_combinations [tuple] (List of tuple for every possible mode combination on journey)
        
        """
        def build_routes_for_timeseg(od_df, vols, dow, doy, from_time, to_time):
            """
            Routefinding for all origin/destination pairs within given timesegment
            Parameters:
            -----------
            od_df: pd.DataFrame
                quetzal Origin-Destination DataFrame
            vols: pd.DataFrame
                quetzal volumes DataFrame
            dow: str
                Weekday ( 'Mon','Tue',... )
            doy: int
                Day of year starting form 0 for 01. Jan.
            form_time: int 
                Start time [s] of interval considered
            to_time: int 
                End time [s] of interval considered

            Returns:
            ----------
            quetzal los table with all trips and modes for given time segement
            """
            #Add Car Volumes for all time segments
            vols['volume_car'] = vols.sum(axis=1, numeric_only=True)
            self.volumes = vols[['origin','destination','volume_car']]
            

            #pathfinding
            self.pathfinding_auto(od_df = od_df,method='aon', num_cores = 1, log = True)
            self.pathfinding_cycling(od_df = od_df)
            self.pathfinding_walk(od_df = od_df)
            self.pathfinding_public_transport(od_df = od_df, weekday = dow, 
                                        start_time = from_time,
                                        end_time = to_time)
            
            self.pathfinding_park_and_ride(od_df = od_df, weekday = dow, 
                                        start_time = from_time,
                                        end_time = to_time)
            pr_cf = self.pr_los
            pr_cf.route_type = 'pr'
            self.pathfinding_park_and_ride(od_df = od_df, weekday = dow, 
                                        start_time = from_time,
                                        end_time = to_time, pt_first=True)
            pr_pf = self.pr_los
            pr_pf.route_type = 'pr_rev'

            
            self.pathfinding_cycle_and_ride(od_df = od_df, weekday = dow, 
                                        start_time = from_time,
                                        end_time = to_time)
            pc_cf = self.pc_los
            pc_cf.route_type = 'pc'
            self.pathfinding_cycle_and_ride(od_df = od_df, weekday = dow, 
                                        start_time = from_time,
                                        end_time = to_time, pt_first=True)
            pc_pf = self.pc_los
            pc_pf.route_type = 'pc_rev'

            los_carsharing = self.auto_los.copy()
            los_carsharing['route_type'] = 'carsharing'
            l = pd.concat([self.auto_los,self.cycle_los,self.walk_los,self.pt_los, los_carsharing, pr_cf, pr_pf, pc_cf, pc_pf])
            l['dow'] = dow
            l['doy'] = doy
            l['from_time'] = from_time
            l['to_time'] = to_time
            
            l = l.reset_index(drop = False)
            l = l.groupby(['origin','destination', 'dow', 'doy','from_time','to_time']).apply(lambda x: pd.DataFrame(x).set_index('route_type',drop = False))
            return l[['origin','destination','time','length','link_path_all','route_type']]

        
        def get_mode_combinations(origins, av_modes = ['car','pt','walk','bicycle']):
            """
            Determines all logic combinations of different modes wihtin a journey: 
                Car and bicycle modes have to be take from the starting point and have to be brought back
                Car and bicycle can be parked for other modes when the journey loops back to this point
            Parameters:
            -----------
            origins: [str] 
                List of locations id's within journey as str. 
                origins[0] is the strating (and end) point; a trip back from origins[-1] to this poit is allways assumed
            av_nodes: [str]
                List of for this journey available traffic modes
            
            Returns: [tuple]
            -----------
                Retunrs a list with all possible mode cominations as tuples
            """
            # Build DataFrame for taking relations between individual nodes into account
            o = pd.DataFrame(origins, columns =['origin_id'])
            # Build unique id's for each stop in journey
            o['id'] = o.index.map(lambda x: o.loc[x,'origin_id']+f'_{x:02d}')
            o = o.set_index('id')
            # Identify nodes with multiple visits and link unique id's
            o['tia'] = o.index.map(lambda x: o.index[o.loc[x,'origin_id'] == o.origin_id].values)
            # Bulild order in which a location is visited
            o['cnt'] = o.groupby('origin_id').cumcount()
            o['tia2'] =  o.index.map(lambda x: o.loc[x,'tia'][0:o.loc[x,'cnt']])
            # Last Node of a loop before way back to start/destination
            o['mx'] = o.apply(lambda x:True if x.cnt == o.groupby('origin_id').cnt.max().loc[x.origin_id] and x.cnt >0 else False, axis = 1)
            
            df = pd.DataFrame(columns = o.index, data = [[[] for i in origins]])
            df.iat[0,0] = av_modes
            
            def get_next_modes(oid, av_modes, prev_mode, trip_data = None, flexible_modes =['walk','pt','carsharing'], return_modes =['car','bicycle']):
                """
                Determines the next possible modes based on previus modes and current location.
                Parameters:
                ----------
                    oid: str
                        unique id for location/position in current journey for current node
                    av_modes: [str]
                        List of for this journey available traffic modes
                    trip_data: pandas.DataFrame.Row
                        Relevant row of the df DataFrame representig the current state of calculation
                
                Requires:
                ---------
                df: pandas.DataFrame
                o: pandas.DataFrame
                
                Returns [str]:
                ---------
                    List of modes available for next trip within journey
                """
                # Normal Case: No loop/ return to this node:
                if (len(o.loc[oid,'tia'])<=1):
                    if prev_mode in return_modes:
                        return [prev_mode]
                    else:
                        return list(set(flexible_modes) & set(av_modes))
                # Loop Starting form here: parking of bike/car possible and use other modes:
                elif (len(o.loc[oid,'tia'])>1 and o.loc[oid,'mx']== False):
                    return list(set(flexible_modes+[prev_mode]) & set(av_modes))
                # loop joins back to its starting point: make sure to pick up parked modes:
                elif (len(o.loc[oid,'tia'])>1 and o.loc[oid,'mx']== True):
                    init_mode =  trip_data.iloc[df.columns.get_loc(o.loc[oid].tia2[0])-1] if o.loc[oid,'mx'] == True else None
                    if init_mode in return_modes:
                        return [init_mode]
                    else: 
                        return list(set(flexible_modes) & set(av_modes)) 
            # Go through all nodes after each other
            for idx, oid in enumerate(df.columns):
                # special case: starting point only use 
                if idx == 0:
                    df = df.explode(oid)
                    continue
                df = df.reset_index(drop = True)
                df[oid] = df.apply(lambda x: get_next_modes(oid, av_modes,
                                                prev_mode = x.iloc[idx-1],trip_data = x), axis = 1)
                df = df.explode(oid)
                
            df = df.reset_index(drop = True)
            ls = list(df.itertuples(index=False, name=None))
            if len(o) == 2 and 'pr' in av_modes:
                ls.append(('pr','pr_rev'))
            if len(o) == 2 and 'pc' in av_modes:
                ls.append(('pc','pc_rev'))
            return ls

        
        od_demands = self.prepare_trips_for_model(self.timeseg_dict,all_trips = all_trips, doy = days)
        #Calculate Trips
        od_demands['los'] = od_demands.apply(lambda x: build_routes_for_timeseg(od_df = x.oddf , vols = x.vols, 
                                                                    dow = x.name[0], doy = x.name[1],
                                                                    from_time = self.timeseg_dict.loc[x.name[2],'starttime'], 
                                                                    to_time = self.timeseg_dict.loc[x.name[2],'endtime']),axis = 1)

        #Build overal los table
        #total_los = pd.concat(od_demands.los.dropna().values).sort_index()
        total_los = pd.concat(od_demands.los.values).sort_index()
        

        #Assing Data from los table to each individual trip
        
        all_trips['timeseg'] = all_trips.dpt_sec.map(lambda x: self.timeseg_dict.index[[x>=t0 and x<t1 
                                                for t0,t1 in zip(self.timeseg_dict.starttime, self.timeseg_dict.endtime)
                                                            ]].values[0])
        trips = all_trips[(all_trips.doy>= min(days))&(all_trips.doy<=max(days))].copy()

        
        # LOS Table for each individual trip (one row per mode)
        trips['los'] = trips.apply(lambda x: total_los.loc[(x.origin,x.destination,x.weekday,x.doy,
            self.timeseg_dict.loc[x.timeseg,'starttime'],self.timeseg_dict.loc[x.timeseg,'endtime'])], axis = 1)

        trips.los = trips.los.map(lambda x: x.transpose())
        trips.los = trips.los.map(lambda x: [x.loc[:,j] for j in x.columns])

        trips = trips.reset_index(drop = True)
        trips.index = trips.index.map(lambda x: f'trip_{x:06x}')
        trips['idx'] = trips.index
        # Build LOS table for all trips and modes
        te = trips.explode(['los'])
        self.trips_debug = trips
        self.te_debug = te
        te['route_type'] = te.los.map(lambda x: x.route_type)
        telos = pd.DataFrame(te['los'].to_list(), columns = te.los.iloc[0].index, index = te.index)
        
        # Build trips_los: quetzal los-table with one row for each trip/mode combination 
        
        telos['idx'] = telos.index
        self.trip_los = pd.merge(te, telos,how =  'left', on = ['idx','route_type'],suffixes=('', '_y'))
        self.trip_los = self.trip_los.drop(['origin_y','destination_y','los','origin_coords', 'destination_coords'],axis = 1)

        self.trip_los['price'] = self.trip_los.apply(lambda x: self.calc_user_price(x,agent_list), axis = 1)
        #Goup trips to journeys (Home to home)
        self.journeys = pd.DataFrame()
        self.journeys['origins'] = trips.groupby(['pid','trip_no']).origin.apply(list)
        self.journeys['single_trip_idx'] = trips.groupby(['pid','trip_no']).idx.apply(list)
        #Get possible combinations of mode for each journey

        self.journeys['rel_mode_combinations'] = self.journeys.apply(lambda x: get_mode_combinations(origins=x.origins,
                                                                                        av_modes =agent_list.loc[x.name[0], 'av_modes'] if isinstance(agent_list, pd.DataFrame) 
                                                                                        else  ['car','pt','walk','bicycle']), axis = 1)

    def select_mode_combination(self, agent_list):
            """
            Calculates probabilities for all mode combination and select one for each journey

            Requires:
            ----------
            From FullModel:         
                self.trip_los
                self.segments
                self.los_attributes
                self.journeys
            

            Builds:
            ----------
                self.journeys: 
                    Adds columns:
                    - probs: dict with probability for each mode combination
                    - selected_modes: [str] list with all modes in selected combinmation
            """
            
            trip_los_data = self.trip_los.groupby(['idx','route_type']).apply(pd.DataFrame)
            trip_los_data = trip_los_data.droplevel(2)
            # Build journey informations and aggregate time and cost for all mode combinations
            je = self.journeys.explode('rel_mode_combinations')
            je = je.loc[je.apply(lambda x: all(x in trip_los_data.index for x in list(zip(x.single_trip_idx,x.rel_mode_combinations))),axis = 1)]
            je['jounrey_total_time'] = je.apply(lambda x:  trip_los_data.loc[list(zip(x.single_trip_idx,
                                                                        x.rel_mode_combinations)),'time'].sum(),
                                                                            axis = 1)
            je['jounrey_total_cost'] = je.apply(lambda x:  trip_los_data.loc[list(zip(x.single_trip_idx,
                                                                        x.rel_mode_combinations)),'price'].sum(),
                                                                            axis = 1)


            #Build psedeo quetzal los table for whole journeys
            self.los = je.rename({'jounrey_total_time':'time','jounrey_total_cost':'price'}, axis = 1)
            self.los = self.los.reset_index(drop = False)
            self.los = self.los.rename({'pid':'origin','trip_no':'destination'},axis = 1)
            self.los['route_type'] = self.los.rel_mode_combinations.map(lambda x: x[0])
            # Make sure the LoS table has all perfomrance attributes
            for a in self.los_attributes:
                if not a in self.los.columns:
                    for s in self.segments:
                        assert s+'_'+a in self.los.columns, \
                            'If LoS attributes are not in the los-table columns, '\
                            +'they must appear for each segment in the format of "seg_los"'
                        self.los[s+'_'+a] = self.los[s+'_'+a].fillna(0)
                else:
                    self.los[a] = self.los[a].fillna(0)
                    for s in self.segments:
                        if not s+'_'+a in self.los.columns:
                            self.los[s+'_'+a] = self.los[a]
            # Calculate Utility/Probability for each Mode/Trip combination from LOS table
            self.analysis_mode_utility(how='main', segment=None)
            self.step_logit()

            # Build dict with mode combinations and respective choice probabilities for each journey
            self.journeys['user_segment'] = self.journeys.index.map(lambda x: agent_list.loc[x[0],'segment'])
            los_prob = self.los.rename({'origin':'pid','destination':'trip_no'},axis = 1)
            los_prob = los_prob.set_index(['pid','trip_no','route_type'])
            [los_prob[(s,'probability')].fillna(0, inplace = True) for s in self.segments]
            self.journeys['probs'] = self.journeys.apply(lambda x: dict(los_prob.loc[
                                                x.name,['rel_mode_combinations',(x.user_segment, 'probability')]].values),axis = 1) 

            # Randomly select one mode combination for each journey according to the respective probabilities
            rng = np.random.default_rng()
            self.journeys['selected_modes'] = self.journeys.probs.map(lambda x: rng.choice(list(x.keys()), p = list(x.values())))


    def add_flows_to_links(self, bpr_alpha =0.15, bpr_beta = 4, bpr_limit = 20, bpr_penalty = 0):
        """
        Adds flow to each link. Builds a single column for each Weekay and Timesegment.
        Adds updated link travel time to self.auto_links
        Requires:
        ---------
        From RegionalModel:
            self.auto_links
            self.pt_links
            self.walk_links
            self.cycle_links
            self.footpaths
            self.zone_to_road
            self.zone_to_transit

        From FullModel:
            self.journeys (with selselected modes)
            self.trip_los

        """
        #assign modes to individual trips within journey
        j2 = self.journeys.drop(['rel_mode_combinations','probs'], axis = 1).copy()
        j2['combo'] = j2.apply(lambda x: list(zip(x['single_trip_idx'],x['selected_modes'])),axis = 1)
        j2 = j2[['combo']].explode('combo')
        
        #get all travelled routes and links
        self.trip_los['combo'] = self.trip_los.apply(lambda x: (x['idx'],x['route_type']),axis = 1)
        temp = pd.merge(left = j2, right = self.trip_los, on = 'combo',  how  = 'left')
        #Count how often a link is travelled within each time intervall
        temp = temp.explode('link_path_all')
        linkflows_all = temp.groupby(['doy','timeseg']).link_path_all.value_counts()

        lfa = linkflows_all.reset_index(drop = False)
        lfa['tid'] = lfa.apply(lambda x: f'doy_{x.doy:03d}-tseg_{x.timeseg}',axis =1)
        lfa_pv = lfa.pivot_table(index = 'link_path_all', values = 'count',columns = 'tid', fill_value = 0)
        #Write flows to Links
        self.auto_links = pd.merge(self.auto_links.drop(lfa_pv.columns,errors = 'ignore', axis=1), lfa_pv,
                                left_index=True, right_index=True, how = 'left').fillna(0)
        self.pt_links = pd.merge(self.pt_links.drop(lfa_pv.columns,errors = 'ignore', axis=1), lfa_pv,
                                left_index=True, right_index=True, how = 'left').fillna(0)
        self.walk_links = pd.merge(self.walk_links.drop(lfa_pv.columns,errors = 'ignore', axis=1), lfa_pv,
                                left_index=True, right_index=True, how = 'left').fillna(0)
        self.cycle_links = pd.merge(self.cycle_links.drop(lfa_pv.columns,errors = 'ignore', axis=1), lfa_pv,
                                left_index=True, right_index=True, how = 'left').fillna(0)
        self.pt_footpaths = pd.merge(self.pt_footpaths.drop(lfa_pv.columns,errors = 'ignore', axis=1), lfa_pv,
                                left_index=True, right_index=True, how = 'left').fillna(0)
        self.zone_to_road = pd.merge(self.zone_to_road.drop(lfa_pv.columns,errors = 'ignore', axis=1), lfa_pv,
                                left_index=True, right_index=True, how = 'left').fillna(0)
        self.zone_to_transit = pd.merge(self.zone_to_transit.drop(lfa_pv.columns,errors = 'ignore', axis=1), lfa_pv,
                                left_index=True, right_index=True, how = 'left').fillna(0)
        #update link times for auto links
        self.auto_links['alpha'] = bpr_alpha 
        self.auto_links['beta'] = bpr_beta
        self.auto_links['limit'] = bpr_limit
        self.auto_links['penalty'] = bpr_penalty
        for c in lfa_pv.columns:
            self.auto_links[c+'-link_time'] = default_bpr(self.auto_links.loc[:,['alpha','beta',
                                                                'limit',c,
                                                                'time','penalty',
                                                                'capacity']].values)

