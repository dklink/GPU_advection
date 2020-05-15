import xarray as xr


def process_hycom(path='uv_2015_1_3d.nc'):
    """loads hycom data, assigns coords, changes the longitude from 0,360 to -180,180"""
    ds = xr.open_dataset(path)
    ds['lon'] = ((ds.lon+180) % 360) - 180
    ds = ds.assign_coords({'x': ds.lon, 'y': ds.lat, 'z': ds.depth})
    ds = ds.roll(x=len(ds.x) // 2, roll_coords=True)
    ds.to_netcdf('hycom_formatted.nc')
    return ds
