import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from xml.dom import minidom
from shapely.geometry import Polygon, MultiPolygon

import os
import os.path
import rasterio
import shapely
import fiona
import numpy as np

from datetime import datetime
from fiona.transform import transform_geom
from glob import glob
from ndvitools import (read_shapes, get_tiles_df, get_tiles, create_gtif,
                       UTM_transform, gen_geotif, plain_list)
from rasterio import MemoryFile
from rasterio.mask import mask
from shapely.geometry import mapping, Polygon, MultiPolygon

def run_ndvi(files, start, end):

    bands = ['B04', 'B08', 'TCI']

    su = []

    for file in files:
        shape = fiona.open(file)
        crs = shape.crs
        if len(crs)==0:
            crs = 'epsg:4326'
        else:
            crs = crs['init']
        geoms = [shp['geometry']['coordinates'] for shp in shape if shp['geometry'] != None]
        poly_lists = plain_list(geoms)
        poly_lists = [item[0] if len(item)==1 else item for item in poly_lists]
        polys = []
        for p in poly_lists:
            if type(p[0])==list:
                polys.extend([Polygon(i) for i in p])
            else:
                polys.append(Polygon(p))
            multipol = MultiPolygon(polys)
        if crs != 'epsg:4326':
            poly = [transform_geom(crs, 'epsg:4326', mapping(p)) for p in multipol]
            poly = [Polygon(t['coordinates'][0]) for t in poly]
            multipol = MultiPolygon(poly)
        multipol = multipol.buffer(0)
        if type(multipol)==shapely.geometry.polygon.Polygon:
            multipol = MultiPolygon([multipol])

        produtor = file.split('/')[-5]
        fazenda = file.split('/')[-4]
        safra = file.split('/')[-3]
        area = file.split('/')[-1]

        try:
            t_df = get_tiles_df(multipol, start, end)
        except:
            continue

        bands_r = ['R10m/' + b for b in bands]
        metafiles = ['tileInfo', 'qi/MSK_CLOUDS_B00']

        get_tiles(t_df.index, './data', bands_r, metafiles)

        for orbit in t_df['orbitNumber'].unique():
            df = t_df[t_df['orbitNumber'] == orbit]

            tot_shape = []
            for i in range(len(df.shapes.values)):
                tot_shape.extend([shape for shape in df.shapes.values[i]])
            tot_shape = MultiPolygon(tot_shape).buffer(0)

            for i, row in df.iterrows():
                band_files = {}
                for band in bands:
                    f = './data/' + row.path + '/R10m/' + band + '.jp2'
                    band_files[band] = rasterio.open(f)

                proj = band_files[next(iter(band_files))].crs.get('init')

                new_shape = row.shapes
                square = mapping(shapely.affinity.scale(new_shape.envelope, 3, 3))
                talhoes = [mapping(shape) for shape in new_shape]
                square = transform_geom('epsg:4326', proj, square)
                square = Polygon(square['coordinates'][0])
                talhoes = [transform_geom('epsg:4326', proj, t) for t in talhoes]
                talhoes = [Polygon(t['coordinates'][0]) for t in talhoes]
                talhoes = MultiPolygon(talhoes).buffer(0)
                if type(talhoes)==shapely.geometry.polygon.Polygon:
                    talhoes = MultiPolygon([talhoes])

                profile = band_files[next(iter(band_files))].profile

                path = './saida/' + produtor + '/' + fazenda + '/' + safra + '/' + area + '/' + row.path[6:-2] + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

                RED, transf = mask(band_files['B04'], talhoes, crop=True)
                NIR, _ = mask(band_files['B08'], talhoes, crop=True)
                NDVI_crop = ((NIR - RED) / (NIR + RED)).astype(np.float32)
                NDVI_crop[NDVI_crop > 1] = 0
                NDVI_crop[NDVI_crop < -1] = 0
                
                max_ndvi = np.nanmax(NDVI_crop)
                try:
                    min_ndvi = np.nanmin(NDVI_crop[NDVI_crop > 0])
                    med_ndvi = np.nanmean(NDVI_crop[NDVI_crop > 0])
                except:
                    min_ndvi = 0
                    med_ndvi = 0
                t_area = talhoes.area/10000
                data = row['completionDate'].strftime('%d/%m/%Y')
                data_f = row['completionDate'].strftime('%d-%m-%Y')

                saida = path + 'dados.txt'
                with open(saida,'w') as f:
                    f.write(f'Produtor: {produtor}\n'
                            f'Fazenda: {fazenda}\n'
                            f'Área: {t_area:.1f} ha\n'
                            f'Data da imagem: {data}\n'
                            f'Pixel: 10m\n'
                            f'Média NDVI: {med_ndvi:.2f}\n'
                            f'Máximo NDVI: {max_ndvi:.2f}\n'
                            f'Mínimo NDVI: {min_ndvi:.2f}\n')

                saida = path + 'ndvi_' + data_f +'.tif'
                gen_geotif(NDVI_crop, transf, profile, saida)

                TCI_crop, transf = mask(band_files['TCI'], [square], crop=True, nodata=0)
                profile = band_files['TCI'].profile
                saida = path + 'TCI_' + data_f + '.tif'
                gen_geotif(TCI_crop, transf, profile, saida)
                su.append(file)
    return su
