# # # # #
# Example of a single tile surfclass pipeline - Vertical processing
# # # # #

BBOX="727000 6171000 728000 6172000"
PREFIX=1km_6171_727
MODEL=randomforestndvi.sav

start_time=`date +%s`
surfclass prepare --help
# Use LidarGrid to turn .las into rasters
surfclass -v DEBUG prepare lidargrid -b $BBOX -r 0.4 -d Amplitude -d "Pulse width" -d ReturnNumber --prefix ${PREFIX}_ ${PREFIX}.las ./prepare

# NDVI Calculations using gdal_calc.py
gdalwarp ./prepare/2019_${PREFIX}.tif ./prepare/2019_${PREFIX}_40cm.tif -tr 0.4 0.4
gdal_calc.py -A ./prepare/2019_${PREFIX}_40cm.tif -B ./prepare/2019_${PREFIX}_40cm.tif --A_band=4 --B_band=1 --overwrite --creation-option COMPRESS=DEFLATE --creation-option PREDICTOR=3 --type="Float32" --calc="(A.astype(float)-B)/(A.astype(float)+B)" --outfile=./prepare/${PREFIX}_NDVI.tif

# Extract mean and variance for Ampltide, Pulsewith and NDVI
surfclass -v DEBUG prepare extractfeatures -b ${BBOX} -n 5 -c reflect --prefix ${PREFIX}_Amplitude_ -f mean -f var ./prepare/${PREFIX}_Amplitude.tif ./prepare
surfclass -v DEBUG prepare extractfeatures -b ${BBOX} -n 5 -c reflect --prefix ${PREFIX}_Pulsewidth_ -f mean -f var ./prepare/${PREFIX}_Pulsewidth.tif ./prepare
surfclass -v DEBUG prepare extractfeatures -b ${BBOX} -n 5 -c reflect --prefix ${PREFIX}_NDVI_ -f mean -f var ./prepare/${PREFIX}_NDVI.tif ./prepare

# Run Classification
surfclass -v DEBUG classify randomforestndvi -b ${BBOX} -f1 ./prepare/${PREFIX}_Amplitude.tif \
                                                                     -f2 ./prepare/${PREFIX}_Amplitude_mean.tif \
                                                                     -f3 ./prepare/${PREFIX}_Amplitude_var.tif \
                                                                     -f4 ./prepare/${PREFIX}_NDVI.tif \
                                                                     -f5 ./prepare/${PREFIX}_NDVI_mean.tif \
                                                                     -f6 ./prepare/${PREFIX}_NDVI_var.tif \
                                                                     -f7 ./prepare/${PREFIX}_Pulsewidth.tif \
                                                                     -f8 ./prepare/${PREFIX}_Pulsewidth_mean.tif \
                                                                     -f9 ./prepare/${PREFIX}_Pulsewidth_var.tif \
                                                                     -f10 ./prepare/${PREFIX}_ReturnNumber.tif \
                                                                     --prefix ${PREFIX}_0_21_3 \
                                                                     ${MODEL} \
                                                                     ./extract
# Extract step
surfclass -v DEBUG extract denoise -b ${BBOX} ./extract/${PREFIX}_0_21_3classification.tif ./extract/${PREFIX}_denoised.tif
surfclass -v DEBUG extract count --in ${PREFIX}_polygon.geojson --out ./extract/${PREFIX}_polygon_out.geojson --format geojson --clip --classrange 0 5 ./extract/${PREFIX}_denoised.tif
end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.