# ----------------------------------------------------------------------------------
# This is an example of how to train a model
# Most of the lines below are actually creating input data. There is no difference
# in creating input data for training and classification, so they are more or less
# an exact copy of the classification pipelines.
#
# Note, however, that we only need to process input tiles where we have training
# polygons. The tiles listed below are spatially disjoint and distributed over a
# large area. Using VRTs these tiles are collected into single rasters covering the
# entire area (with nodata where there are no tiles).
# ----------------------------------------------------------------------------------

import subprocess
from pathlib import Path
import shutil

tiles = [(6167, 729), (6171, 727), (6176, 724), (6184, 720), (6211, 689), (6220, 717)]

out_dir = Path("tmp3")
dimensions = ["Amplitude", "Pulsewidth", "ReturnNumber"]

las_dir = Path(
    "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/trænings_las"
)
orto_dir = Path(
    "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/ortofoto"
)

train_ds = "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/train_test/train_polys_all.gpkg"
train_lyr = "train_polys_all"
train_class_attribute = "class"


def process_lidar(tiles, las_dir, out_dir):
    print("Grid lidar files")
    for t in tiles:
        process_lidar_tile(t, las_dir, out_dir)


# Do all lidar gridding
def process_lidar_tile(t, las_dir, out_dir):
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
    args += [str(las_dir / (kvnet + ".las"))]
    args += [str(out_dir)]
    print("Running: ", args)
    subprocess.run(args, check=True)


def gdal_vrt_lidar(dimensions, out_dir):
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


def process_ndvi(tiles, orto_dir, out_dir):
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


def gdal_vrt_ndvi(out_dir):
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


def process_derived(tiles, out_dir):
    print("Calculate derived features")
    for t in tiles:
        process_derived_tile(t, out_dir)


def process_derived_tile(t, out_dir):
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
        args += [str(out_dir)]
        print("Running: ", args)
        subprocess.run(args, check=True)


def gdal_vrt_derived(out_dir):
    for d in ("Amplitude", "Pulsewidth", "ndvi"):
        for m in ("mean", "var", "diffmean"):
            args = ["gdalbuildvrt"]
            args += ["-resolution", "user"]
            # Cover entire DK + margin
            args += ["-tap"]
            args += ["-tr", "0.4", "0.4"]
            args += ["-te", "440000", "6048000", "895000", "6404000"]
            args += [str(out_dir / ("%s_%s.vrt" % (d, m)))]  # Output vrt
            args += [str(out_dir / ("*_%s_%s.tif" % (d, m)))]  # Input files
            print("Running: ", args)
            subprocess.Popen(" ".join(args), shell=True).wait()


def prep_train_data(out_dir):
    dstfile = out_dir / "traindata.npz"
    if dstfile.exists():
        print("Traindata %s exists. Skipping" % dstfile)
        return
    args = ["surfclass", "prepare", "traindata"]
    args += ["--in", train_ds]
    args += ["--inlyr", train_lyr]
    args += ["-a", train_class_attribute]
    for f in [
        "Amplitude",
        "Amplitude_mean",
        "Amplitude_var",
        "ndvi",
        "ndvi_mean",
        "ndvi_var",
        "Pulsewidth",
        "Pulsewidth_mean",
        "Pulsewidth_var",
        "ReturnNumber",
    ]:
        args += ["-f", out_dir / ("%s.vrt" % f)]
    args += [dstfile]
    print("Running: ", args)
    subprocess.run(args, check=True)


def train_model(out_dir):
    datafile = out_dir / "traindata.npz"
    modelfile = out_dir / "trained.model"
    if modelfile.exists():
        print("Model %s found. Skipping" % modelfile)
        return
    args = ["surfclass", "train", "randomforestndvi", datafile, modelfile]
    print("Running: ", args)
    subprocess.run(args, check=True)


if __name__ == "__main__":
    """ This is executed when run from the command line """

    # Create all the data
    process_lidar(tiles, las_dir, out_dir)
    gdal_vrt_lidar(dimensions, out_dir)

    process_ndvi(tiles, orto_dir, out_dir)
    gdal_vrt_ndvi(out_dir)

    process_derived(tiles, out_dir)
    gdal_vrt_derived(out_dir)

    prep_train_data(out_dir)
    train_model(out_dir)
