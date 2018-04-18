### accessing GeoTiff files:
https://nsidc.org/support/how/how-do-i-convert-snodas-binary-files-geotiff-or-netcdf

unzip each .gz archive

2nd file is 24-hr snow

remove filename.Hdr files

make new filename.hdr file with contents:
 - ENVI
 - samples = 6935
 - lines   = 3351
 - bands   = 1
 - header offset = 0
 - file type = ENVI Standard
 - data type = 2
 - interleave = bsq
 - byte order = 1

bash:
$ gdal_translate -of GTiff -a_srs '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' -a_nodata -9999 -a_ullr  -124.73333333 52.87500000 -66.94166667 24.95000000 <input.dat> <output.tif>

### Common Error:
A common error that can be thrown after issuing the 'gdal_translate ...' command is "Error 4: <input.dat> not recognized as a supported file format. Depending on your system, GDAL might be confused between the .hdr you created and the .Hdr which came with the data. If this occurs, try storing the .Hdr files outside of the working directory and trying again.

### For Unmasked data:
Appendix 1. Conversion example for unmasked SNODAS data.

gdal_translate -of GTiff -a_srs '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' -a_nodata -9999 -a_ullr -130.516666666661 58.2333333333310 -62.2499999999975 24.0999999999990 34.dat 34.tif

and header file should be:

ENVI
samples=8192
lines=4096
bands=1
header offset=0
file type=ENVI Standard
data type=2
interleave=bsq
byte order=1

### pre/post Oct 1 2013
Appendix 2. Spatial bounds to feed into GDAL -ullr flag for pre and post Oct 01 2013.

Pre Oct 01 2013: -124.73375000 52.87458333 -66.94208333 24.87458333

Post Oct 01 2013: -124.73333333 52.87500000 -66.94166667 24.95000000
