import subprocess
from pathlib import Path
import shutil

# ------------------------------------------------------------------------------------------
# This is an example of how to classify multiple tiles horizontally, by creating .vrt files
# for features. This example classifies sixteen 1km tiles in a 4km x 4km block using an
# already trained model. For examples of how to train a model look in examples/training.py
#
# Tiles used:
# Eastings 722, 723, 724, 725
# Northings 6177, 6178, 6179, 6180
# -----------------------------------------------------------------------------------------

e_range = range(722, 726)
n_range = range(6177, 6181)
tiles = []
for e in e_range:
    for n in n_range:
        tiles.append((n, e))


out_dir = Path("./tmp4")
las_dir = Path(
    "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/trænings_las"
)
orto_dir = Path(
    "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/ortofoto"
)
modelfile = Path(
    "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/train_test/randomforestndvi2.model"
)

dimensions = ["Amplitude", "Pulsewidth", "ReturnNumber"]
geodkdb = "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/geodk.gpkg"


# Do all lidar gridding
def process_lidar_tile(t):
    n, e = t
    kvnet = "1km_%s_%s" % (n, e)
    if (out_dir / ("%s_Amplitude.tif" % kvnet)).exists():
        print("Existing grids found for %s. Skipping" % kvnet)
        return
    bbox = (e * 1000, n * 1000, e * 1000 + 1000, n * 1000 + 1000)
    args = ["surfclass", "prepare", "lidargrid"]
    args += ["--srs", "epsg:25832"]
    args += ["-r", "0.4"]
    args += ["--bbox"] + [str(x) for x in bbox]
    for d in ["Amplitude", "Pulse width", "ReturnNumber"]:
        args += ["-d", d]
    args += ["--prefix", "%s_" % kvnet]
    args += [las_dir / (kvnet + ".las")]
    args += [out_dir]
    print("Running: ", args)
    subprocess.run(args, check=True)


print("Grid lidar files")
for t in tiles:
    process_lidar_tile(t)

print("Make GDAL vrts")
for d in dimensions:
    args = ["gdalbuildvrt"]
    args += ["-resolution", "user"]
    # Cover entire DK + margin
    args += ["-tap"]
    args += ["-tr", "0.4", "0.4"]
    args += ["-te", "440000", "6048000", "895000", "6404000"]
    args += ["%s/%s.vrt" % (out_dir, d)]  # Output vrt
    args += ["%s/*_%s.tif" % (out_dir, d)]  # Input files
    print("Running: ", args)
    subprocess.Popen(" ".join(args), shell=True).wait()


print("Process NDVI")
for t in tiles:
    kvnet = "1km_%s_%s" % t
    srcfile = orto_dir / ("2019_%s.tif" % kvnet)
    tmpfile = out_dir / ("tmp_%s.tif" % kvnet)
    dstfile = out_dir / ("%s_ndvi.tif" % kvnet)
    if dstfile.exists():
        print("Existing NDVI found for %s. Skipping" % kvnet)
        continue
    # Resample to 0.4m
    args = ["gdal_translate"]
    args += ["-co", "tiled=yes", "-co", "compress=deflate"]
    args += ["-tr", "0.4", "0.4"]
    args += [srcfile, tmpfile]
    print("Running: ", args)
    subprocess.run(args, check=True)
    # Calculate ndvi
    args = ["gdal_calc.py"]
    args += ["-A", tmpfile, "--A_band=4"]
    args += ["-B", tmpfile, "--B_band=1"]
    args += ["--creation-option", "compress=deflate"]
    args += ["--creation-option", "tiled=true"]
    args += ["--type", "Float32"]
    args += ["--calc", "(A.astype(float)-B)/(A.astype(float)+B)"]
    args += ["--outfile", dstfile]
    print("Running: ", args)
    subprocess.run(args, check=True)
    tmpfile.unlink()

print("Make GDAL vrts for NDVI")
args = ["gdalbuildvrt"]
args += ["-resolution", "user"]
# Cover entire DK + margin
args += ["-tap"]
args += ["-tr", "0.4", "0.4"]
args += ["-te", "440000", "6048000", "895000", "6404000"]
args += [str(out_dir / "ndvi.vrt")]  # Output vrt
args += [str(out_dir / "1km_*_ndvi.tif")]  # Input files
print("Running: ", args)
subprocess.Popen(" ".join(args), shell=True).wait()


def process_derived(t):
    n, e = t
    kvnet = "1km_%s_%s" % (n, e)
    # Caculate bbox including edge for kernel
    bbox = (e * 1000 - 0.8, n * 1000 - 0.8, e * 1000 + 1000.8, n * 1000 + 1000.8)
    for d in ["Amplitude", "Pulsewidth", "ndvi"]:
        if (out_dir / ("%s_%s_mean.tif" % (kvnet, d))).exists():
            print("Existing derived features found for %s_%s. Skipping" % (kvnet, d))
            continue
        args = ["surfclass", "prepare", "extractfeatures"]
        args += ["--bbox"] + [str(x) for x in bbox]
        args += ["-n", "5"]
        args += ["-c", "crop"]
        args += ["--prefix", "%s_%s_" % (kvnet, d)]
        args += ["-f", "mean"]
        args += ["-f", "var"]
        args += ["-f", "diffmean"]
        args += ["%s/%s.vrt" % (out_dir.resolve(), d)]
        args += [out_dir]
        print("Running: ", args)
        subprocess.run(args, check=True)


print("Calculate derived features")
for t in tiles:
    process_derived(t)

print("Make GDAL vrts for derived features")
for d in ("Amplitude", "Pulsewidth", "ndvi"):
    for m in ("mean", "var", "diffmean"):
        args = ["gdalbuildvrt"]
        args += ["-resolution", "user"]
        # Cover entire DK + margin
        args += ["-tap"]
        args += ["-tr", "0.4", "0.4"]
        args += ["-te", "440000", "6048000", "895000", "6404000"]
        args += ["%s/%s_%s.vrt" % (out_dir, d, m)]  # Output vrt
        args += ["%s/*_%s_%s.tif" % (out_dir, d, m)]  # Input files
        print("Running: ", args)
        subprocess.Popen(" ".join(args), shell=True).wait()

print("Run classification")
for t in tiles:
    n, e = t
    kvnet = "1km_%s_%s" % (n, e)
    bbox = (e * 1000, n * 1000, e * 1000 + 1000, n * 1000 + 1000)
    dstfile = out_dir / ("%s_classification.tif" % kvnet)
    probfile = out_dir / ("%s_classification_prob.tif" % kvnet)
    if dstfile.exists():
        print("%s exists. Skipping" % dstfile)
        continue
    args = ["surfclass", "classify", "randomforestndvi"]
    args += ["--bbox"] + [str(x) for x in bbox]
    args += ["-f1", out_dir / "Amplitude_diffmean.vrt"]
    args += ["-f2", out_dir / "Amplitude_mean.vrt"]
    args += ["-f3", out_dir / "Amplitude_var.vrt"]
    args += ["-f4", out_dir / "ndvi_diffmean.vrt"]
    args += ["-f5", out_dir / "ndvi_mean.vrt"]
    args += ["-f6", out_dir / "ndvi_var.vrt"]
    args += ["-f7", out_dir / "pulsewidth_diffmean.vrt"]
    args += ["-f8", out_dir / "pulsewidth_mean.vrt"]
    args += ["-f9", out_dir / "pulsewidth_var.vrt"]
    args += ["-f10", out_dir / "returnnumber.vrt"]
    args += ["--prob", probfile]
    args += [modelfile]
    args += [dstfile]
    print("Running: ", args)
    subprocess.run(args, check=True)


print("Make GDAL vrts for classified")
args = ["gdalbuildvrt"]
args += ["-resolution", "user"]
args += ["-tap"]
args += ["-tr", "0.4", "0.4"]
args += ["-te", "440000", "6048000", "895000", "6404000"]
args += [str(out_dir / "classification.vrt")]  # Output vrt
args += [str(out_dir / "1km_*_classification.tif")]  # Input files
print("Running: ", args)
subprocess.Popen(" ".join(args), shell=True).wait()
# Probability
args = ["gdalbuildvrt"]
args += ["-resolution", "user"]
args += ["-tap"]
args += ["-tr", "0.4", "0.4"]
args += ["-te", "440000", "6048000", "895000", "6404000"]
args += [str(out_dir / "classification_prob.vrt")]  # Output vrt
args += [str(out_dir / "1km_*_classification_prob.tif")]  # Input files
print("Running: ", args)
subprocess.Popen(" ".join(args), shell=True).wait()

print("Denoise")
for t in tiles:
    n, e = t
    kvnet = "1km_%s_%s" % (n, e)
    bbox = (e * 1000, n * 1000, e * 1000 + 1000, n * 1000 + 1000)
    # Add buffer to reduce nearest neighbor artefacts
    bbox_buffer = (bbox[0] - 20, bbox[1] - 20, bbox[2] + 20, bbox[3] + 20)
    srcfile = out_dir / ("classification.vrt")
    tmpfile = out_dir / ("tmp_%s_classification_denoised.tif" % kvnet)
    dstfile = out_dir / ("%s_classification_denoised.tif" % kvnet)
    if dstfile.exists():
        print("%s exists. Skipping" % dstfile)
        continue
    args = ["surfclass", "extract", "denoise"]
    args += ["--bbox"] + [str(x) for x in bbox_buffer]
    args += [srcfile, tmpfile]
    print("Running: ", args)
    subprocess.run(args, check=True)
    # Crop away edges
    args = ["gdal_translate"]
    args += ["-co", "tiled=true"]
    args += ["-co", "compress=deflate"]
    args += ["-projwin", str(bbox[0]), str(bbox[3]), str(bbox[2]), str(bbox[1])]
    args += [tmpfile]
    args += [dstfile]
    print("Running: ", args)
    subprocess.run(args, check=True)
    tmpfile.unlink()

print("Make GDAL vrts for denoised")
args = ["gdalbuildvrt"]
args += ["-resolution", "user"]
# Cover entire DK + margin
args += ["-tap"]
args += ["-tr", "0.4", "0.4"]
args += ["-te", "440000", "6048000", "895000", "6404000"]
args += [str(out_dir / "classification_denoised.vrt")]  # Output vrt
args += [str(out_dir / "1km_*_classification_denoised.tif")]  # Input files
print("Running: ", args)
subprocess.Popen(" ".join(args), shell=True).wait()

print("Burn buildings and lakes")
for t in tiles:
    kvnet = "1km_%s_%s" % t
    n, e = t
    bbox = (e * 1000, n * 1000, e * 1000 + 1000, n * 1000 + 1000)
    srcfile = out_dir / ("%s_classification_denoised.tif" % kvnet)
    tmpfile = out_dir / ("tmp_%s_classification_denoised_burn.tif" % kvnet)
    dstfile = out_dir / ("%s_classification_denoised_burn.tif" % kvnet)
    if dstfile.exists():
        print("%s exists. Skipping" % dstfile)
        continue
    shutil.copy(srcfile, tmpfile)
    for classid, layername in [
        (8, "geodanmark_60_nohist.bygning"),
        (9, "geodanmark_60_nohist.soe"),
    ]:
        # gdal_rasterize is is slow with large input vector datasets
        # Extract bbox from database
        tmpgeom = out_dir / ("tmp_%s.geojson" % kvnet)
        args = ["ogr2ogr"]
        args += ["-f", "GeoJSON"]
        args += ["-spat"] + [str(x) for x in bbox]
        args += [tmpgeom, geodkdb, layername]
        print("Running: ", " ".join(map(str, args)))
        subprocess.run(args, check=True)
        # Now use subset
        args = ["gdal_rasterize"]
        args += ["-burn", str(classid)]
        # args += ["-l", layername]
        args += [tmpgeom]
        args += [tmpfile]
        print("Running: ", " ".join(map(str, args)))
        subprocess.run(args, check=True)
        tmpgeom.unlink()
    shutil.copy(tmpfile, dstfile)
    tmpfile.unlink()

print("Make GDAL vrts for denoised burned")
args = ["gdalbuildvrt"]
args += ["-resolution", "user"]
# Cover entire DK + margin
args += ["-tap"]
args += ["-tr", "0.4", "0.4"]
args += ["-te", "440000", "6048000", "895000", "6404000"]
args += [str(out_dir / "classification_denoised_burn.vrt")]  # Output vrt
args += [str(out_dir / "1km_*_classification_denoised_burn.tif")]  # Input files
print("Running: ", args)
subprocess.Popen(" ".join(args), shell=True).wait()
