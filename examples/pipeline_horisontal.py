import subprocess
from pathlib import Path

# In lidar dir we have sixteen 1km tiles in a 4km x 4km block
# Eastings 722, 723, 724, 725
# Northings 6177, 6178, 6179, 6180
e_range = range(722, 726)
n_range = range(6177, 6181)
tiles = []
for e in e_range:
    for n in n_range:
        tiles.append((n, e))


las_dir = Path(
    "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/trænings_las"
)
orto_dir = Path(
    "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/ortofoto"
)
out_dir = Path("./tmp")
dimensions = ["Amplitude", "Pulsewidth", "ReturnNumber"]

# Do all lidar gridding
def process_lidar_tile(t):
    n, e = t
    kvnet = "1km_%s_%s" % (n, e)
    bbox = (e * 1000, n * 1000, e * 1000 + 1000, n * 1000 + 1000)
    args = ["surfclass", "prepare", "lidargrid"]
    args += ["--srs", "epsg:25832"]
    args += ["-r", "0.4"]
    args += ["--bbox"] + [str(x) for x in bbox]
    for d in dimensions:
        args += ["-d", d]
    args += ["--prefix", "%s_" % kvnet]
    args += [las_dir / (kvnet + ".las")]
    args += [out_dir]
    print("Running: ", args)
    # subprocess.run(args, check=True)


print("Grid lidar files")
for t in tiles:
    process_lidar_tile(t)
    pass

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
    # subprocess.Popen(" ".join(args), shell=True).wait()


def process_derived(t):
    n, e = t
    kvnet = "1km_%s_%s" % (n, e)
    # Caculate bbox including edge for kernel
    bbox = (e * 1000 - 0.8, n * 1000 - 0.8, e * 1000 + 1000.8, n * 1000 + 1000.8)
    for d in ["Amplitude", "Pulsewidth"]:
        args = ["surfclass", "prepare", "extractfeatures"]
        args += ["--bbox"] + [str(x) for x in bbox]
        args += ["-n", "5"]
        args += ["-c", "crop"]
        args += ["--prefix", "%s_%s_" % (kvnet, d)]
        args += ["-f", "mean"]
        args += ["-f", "var"]
        args += ["%s/%s.vrt" % (out_dir.resolve(), d)]
        args += [out_dir]
        print("Running: ", args)
        # subprocess.run(args, check=True)


print("Calculate derived features")
for t in tiles:
    process_derived(t)

print("Make GDAL vrts for derived features")
for d in ("Amplitude", "Pulsewidth"):
    for m in ("mean", "var"):
        args = ["gdalbuildvrt"]
        args += ["-resolution", "user"]
        # Cover entire DK + margin
        args += ["-tap"]
        args += ["-tr", "0.4", "0.4"]
        args += ["-te", "440000", "6048000", "895000", "6404000"]
        args += ["%s/%s_%s.vrt" % (out_dir, d, m)]  # Output vrt
        args += ["%s/*_%s_%s.tif" % (out_dir, d, m)]  # Input files
        print("Running: ", args)
        # subprocess.Popen(" ".join(args), shell=True).wait()

print("Process NDVI")
for t in tiles:
    kvnet = "1km_%s_%s" % t
    srcfile = orto_dir / ("2019_%s.tif" % kvnet)
    tmpfile = out_dir / ("tmp_%s.tif" % kvnet)
    dstfile = out_dir / ("2019_%s_ndvi.tif" % kvnet)
    # Resample to 0.4m
    args = ["gdal_translate"]
    args += ["-co", "tiled=yes", "-co", "compress=deflate"]
    args += ["-tr", "0.4", "0.4"]
    args += [srcfile, tmpfile]
    print("Running: ", args)
    # subprocess.run(args, check=True)
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
    # subprocess.run(args, check=True)
    # tmpfile.unlink()

print("Make GDAL vrts for NDVI")
args = ["gdalbuildvrt"]
args += ["-resolution", "user"]
# Cover entire DK + margin
args += ["-tap"]
args += ["-tr", "0.4", "0.4"]
args += ["-te", "440000", "6048000", "895000", "6404000"]
args += [str(out_dir / "ndvi.vrt")]  # Output vrt
args += [str(out_dir / "2019_1km_*_ndvi.tif")]  # Input files
print("Running: ", args)
# subprocess.Popen(" ".join(args), shell=True).wait()

print("Run classification")
print("...not implemented yet...")
