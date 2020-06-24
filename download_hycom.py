import wget
import pandas as pd
import sys
import os

from os import path


for time in pd.date_range(start='2015-01-01', end='2015-12-31', freq='6H'):
    url = 'http://ncss.hycom.org/thredds/ncss/grid/GLBu0.08/expt_91.1/uv3z?var=water_u&var=water_v' \
         f'&north=80&west=0&east=359.92&south=-80&horizStride=1&time={time.strftime("%Y-%m-%dT%H")}' \
          '%3A00%3A00Z&vertCoord=0&accept=netcdf'
    filename = f'data/HYCOM_GLBu0.08_expt_91.1/uv_surf_{time.strftime("%Y-%m-%dT%H")}.nc'

    if not path.exists(filename):
        print(f'Downloading {filename}')
        sys.stdout = open(os.devnull, "w")
        wget.download(url, filename)
        sys.stdout = sys.__stdout__
    else:
        print(f'Skipping {filename}, already exists')
