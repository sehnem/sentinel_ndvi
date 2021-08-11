import fiona
import pyproj
import rasterio
import shapely
import pandas as pd
import numpy as np

from datetime import datetime
from functools import partial
from rasterio import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sentinelhub import AwsTile, AwsTileRequest, BBox, CRS, DataSource
from sentinelhub import get_area_info
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform, cascaded_union
from xml.dom import minidom


def plain_list(l):
    return [item for sublist in l for item in sublist]


def gen_geotif(data, transform, profile, path):
    profile.update({'transform': transform,
                    'width': data.shape[2],
                    'height': data.shape[1],
                    'dtype': data.dtype.name,
                    'blockxsize': 2**int(np.log2(data.shape[2])),
                    'blockysize': 2**int(np.log2(data.shape[1])),
                    'driver': 'GTiff',
                    'nodata': 0})
    with MemoryFile() as ds:
        with ds.open(**profile) as dst:
            dst.write(data)
        geo_data = UTM_transform(ds.open())
    create_gtif(geo_data.read(), path, geo_data.profile)


def UTM_transform(src):
    dst_crs = 'EPSG:4326'

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    profile = src.meta.copy()
    profile.update({'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'driver': 'GTiff'})

    ds = MemoryFile()
    with ds.open(**profile) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                num_threads=4)

    return ds.open()


def merge_tiles(srcs):
    profile = srcs[0].meta.copy()
    mosaic, out_trans = merge(srcs)

    profile.update({'crs': 'EPSG:4326',
                    'transform': out_trans,
                    'width': mosaic.shape[2],
                    'height': mosaic.shape[1],
                    'driver': 'GTiff'})

    ds = MemoryFile()
    with ds.open(**profile) as dst:
        dst.write(mosaic)

    return ds.open()


def get_area(polygon):
    s = shapely.geometry.shape(polygon)
    proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                   pyproj.Proj(init='epsg:3857'))

    return transform(proj, s).area/(100*100)


def read_shapes(files):
    if type(files) != list:
        files = [files]
    shapefiles = [fiona.open(file, "r") for file in files]
    shapes_l = []
    for shapes in shapefiles:
        for shape in shapes:
            shapes_l.append(Polygon(shape['geometry']['coordinates'][0]))
    shapes_ly = MultiPolygon(shapes_l)
    return shapes_ly


def title_to_path(s3URI):
    splited = s3URI.split('/')[4:-1]
    tile_UTM = ''.join(splited[:3])
    date = datetime(*tuple([int(s) for s in splited[3:-1]]))
    return tile_UTM + date.strftime(',%Y-%m-%d,' + splited[-1])


def filter_duplicated(t_df):
    for orbit in t_df['orbitNumber'].unique():
        t_orbit = t_df[t_df['orbitNumber'] == orbit]['shapes']
        sorted_t = t_orbit.apply(len).sort_values().index
        t_orbit = t_orbit.reindex(sorted_t)
        t_orbit = t_orbit[t_orbit.apply(lambda x: not x.is_empty)]

        for i in range(len(t_orbit)-1):
            orbits_len = list(range(len(t_orbit)))
            orbits_len.pop(i)
            polygons_set = []
            for j in orbits_len:
                polygons_set.extend([t for t in t_orbit[j]])
            un = [t for t in t_orbit[i] if t not in polygons_set]
            idx = t_orbit.index[i]
            t_df.loc[[idx], 'shapes'] = pd.Series([MultiPolygon(un)],
                                                  index=[idx])

    return t_df


def get_tiles_df(shapes, start, end):
    bbox = BBox(bbox=shapes.bounds, crs=CRS.WGS84)

    tiles = list(get_area_info(bbox, (start.isoformat(), end.isoformat())))

    cols_filter = ['cloudCover', 'completionDate', 'orbitNumber', 's3URI',
                   'sgsId', 'snowCover', 'spacecraft', 'title']

    t_df = pd.DataFrame([t['properties'] for t in tiles])
    t_df = t_df[cols_filter]
    t_poly = [t['geometry']['coordinates'][0][0] for t in tiles]
    t_df['completionDate'] = pd.to_datetime(t_df['completionDate'])

    t_df = pd.DataFrame({'geometry': [Polygon(t) for t in t_poly]}).join(t_df)

    for i, tile in t_df.iterrows():
        tmp = []
        for shape in shapes:
            if shape.within(tile['geometry']):
                tmp.append(shape)
        t_df.loc[[i], 'shapes'] = pd.Series([MultiPolygon(tmp)], index=[i])

    t_df = t_df[t_df.shapes.apply(len) != 0].set_index('title')

    t_df = filter_duplicated(t_df)

    t_df = t_df[t_df.shapes.apply(len) != 0]
    t_df['path'] = [title_to_path(s) for s in t_df['s3URI']]

    return t_df


def afine_to_extent(afine, height, width):
    x0 = afine[2]
    y0 = afine[5] + afine[4] * height
    x1 = afine[2] + afine[0] * width
    y1 = afine[5]
    return [x0, x1, y0, y1]


def get_tiles(titles, data_folder, bands, metafiles):
    for title in titles:
        tile_name, time, aws_index = AwsTile.tile_id_to_tile(title)

        request = AwsTileRequest(tile=tile_name, time=time, bands=bands,
                                 aws_index=aws_index, metafiles=metafiles,
                                 data_folder=data_folder,
                                 data_source=DataSource.SENTINEL2_L2A)
        request.save_data()


def read_clouds(file):
    gml = minidom.parse(file)

    items = gml.getElementsByTagName('gml:Polygon')

    clouds = {}
    clouds['CIRRUS'] = []
    clouds['OPAQUE'] = []

    for item in items:
        c_type = item.getAttribute('gml:id').split('.')[0]
        data = item.getElementsByTagName('gml:posList')[0].childNodes[0].data

        data = data.split(' ')

        array = []
        for i in range(0, len(data), 2):
            array.append((int(data[i]), int(data[i+1])))

        clouds[c_type].append(Polygon(array))

    for k in clouds.keys():
        clouds[k] = MultiPolygon(clouds[k])

    t_clouds = cascaded_union([clouds[c] for c in clouds.keys()])

    return t_clouds, clouds


def create_gtif(array, path, profile, dtype=None):
    if dtype == None:
        dtype = array.dtype.type
    profile.update(dtype=dtype,
                    driver='GTiff',
                    compress='lzw')

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array)
