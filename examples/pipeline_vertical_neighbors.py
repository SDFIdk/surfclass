import subprocess
from pathlib import Path

# This pipeline processes the tile 1km_6178_724 including a small buffer overlapping the neighbor tiles
tile_e = 724
tile_n = 6178
las_dir = "/Volumes/Macintosh HD/Volumes/GoogleDrive/My Drive/Septima - Ikke synkroniseret/Projekter/SDFE/Befæstelse/data/trænings_las"
kernel_size = 5
resolution = 0.4
dimensions = ["Amplitude", "Pulse width", "ReturnNumber"]
out_dir = "."

tile_kvnet = "1km_%s_%s" % (tile_n, tile_e)
tile_bbox = (tile_e * 1000, tile_n * 1000, tile_e * 1000 + 1000, tile_n * 1000 + 1000)
tile_cells = (
    (tile_bbox[2] - tile_bbox[0]) / resolution,
    (tile_bbox[3] - tile_bbox[1]) / resolution,
)
buffer_cells = (kernel_size - 1) / 2
buffer = buffer_cells * resolution
buffered_tile_bbox = (
    tile_bbox[0] - buffer,
    tile_bbox[1] - buffer,
    tile_bbox[2] + buffer,
    tile_bbox[3] + buffer,
)

# LAS file names of tile and neighbor tiles
las_files = []
for n in range(tile_n - 1, tile_n + 1):
    for e in range(tile_e - 1, tile_e + 2):
        las_files.append("%s/1km_%s_%s.las" % (las_dir, n, e))


print("Grid Lidar data with buffer to allow calculation of kernel features")
args = ["surfclass", "prepare", "lidargrid"]
args += ["--srs", "epsg:25832"]
args += ["-r", str(resolution)]
args += ["--bbox"] + [str(x) for x in buffered_tile_bbox]
for d in dimensions:
    args += ["-d", d]
args += ["--prefix", "%s_" % tile_kvnet]
args += las_files
args += [out_dir]
print("Running: ", args)
subprocess.run(args, check=True)

print("Calculate kernel features")
for d in ["Amplitude", "Pulsewidth"]:
    args = ["surfclass", "prepare", "extractfeatures"]
    args += ["--bbox"] + [str(x) for x in buffered_tile_bbox]
    args += ["-n", str(kernel_size)]
    args += ["-c", "crop"]
    args += ["--prefix", "%s_%s_" % (tile_kvnet, d)]
    args += ["-f", "mean"]
    args += ["-f", "var"]
    args += ["%s/%s_%s.tif" % (out_dir, tile_kvnet, d)]
    args += [out_dir]
    print("Running: ", args)
    subprocess.run(args, check=True)

print("Crop away buffer from gridded lidar")
x = y = buffer_cells
width, height = tile_cells
for d in ["Amplitude", "Pulsewidth", "ReturnNumber"]:
    f = "%s/%s_%s.tif" % (out_dir, tile_kvnet, d)
    tmpfile = "%s/tmp_%s_%s.tif" % (out_dir, tile_kvnet, d)
    args = ["gdal_translate"]
    args += ["-co", "tiled=true"]
    args += ["-co", "compress=deflate"]
    args += ["-srcwin", str(x), str(y), str(width), str(height)]
    args += [f]
    args += [tmpfile]
    print("Running: ", args)
    subprocess.run(args, check=True)
    Path(f).unlink()
    Path(tmpfile).rename(f)

print("Run classification")
print("...not implemented yet...")
