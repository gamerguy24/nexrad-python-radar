@echo off
rem ------------------------------------------------------------------
rem  Builds dist\VortexRadar.exe (one file, no Python needed to run it)
rem  from the .py files in this folder.
rem
rem  To release an update:
rem    1. edit the .py files
rem    2. bump APP_VERSION in vortex_radar.py   (e.g. 1.1.0 -> 1.2.0)
rem    3. run this file
rem    4. on GitHub: Releases -> Draft a new release, tag v1.2.0,
rem       attach dist\VortexRadar.exe, publish
rem  Running copies of older versions then show a "New version" button.
rem ------------------------------------------------------------------
cd /d "%~dp0"

python -m pip install -r requirements.txt pyinstaller || goto :fail
python us_map.py --build-cache || goto :fail

python -m PyInstaller --noconfirm --clean --onefile --windowed --name VortexRadar ^
  --workpath build --specpath build --distpath dist ^
  --add-data "%~dp0basemap.pkl.gz;." ^
  --exclude-module cartopy --exclude-module matplotlib --exclude-module IPython ^
  --exclude-module pandas --exclude-module dask --exclude-module xarray ^
  --exclude-module netCDF4 --exclude-module h5py --exclude-module h5netcdf ^
  --exclude-module metpy --exclude-module cfgrib --exclude-module pint ^
  --exclude-module numba --exclude-module pyarrow --exclude-module sympy ^
  --exclude-module torch --exclude-module jax --exclude-module cupy ^
  --exclude-module sparse --exclude-module zarr --exclude-module distributed ^
  --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide6 ^
  vortex_radar.py || goto :fail

echo.
echo Built dist\VortexRadar.exe
exit /b 0

:fail
echo.
echo BUILD FAILED
exit /b 1
