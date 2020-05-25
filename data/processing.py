import xarray as xr
from tqdm import tqdm
import glob

def process_hycom_file(path='uv_2015_1_3d.nc'):
    """loads hycom data, assigns coords, changes the longitude from 0,360 to -180,180"""
    ds = xr.open_dataset(path)
    ds['lon'] = ((ds.lon+180) % 360) - 180
    ds = ds.assign_coords({'x': ds.lon, 'y': ds.lat, 'z': ds.depth})
    ds = ds.roll(x=len(ds.x) // 2, roll_coords=True)

    filename = path.split('/')[-1]
    ds.to_netcdf(f'{filename[:-3]}.formatted{filename[-3:]}')
    return ds


def process_all_hycom_files():
    for path in tqdm(glob.glob('../../trashtracker/utils/get hycom/nc/uv_2015*.nc')):
        process_hycom_file(path)
