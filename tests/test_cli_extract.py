from surfclass.scripts.cli import cli
from surfclass.vectorize import FeatureReader


def test_cli_extract(cli_runner):
    result = cli_runner.invoke(cli, ["extract"], catch_exceptions=False)
    assert result.exit_code == 0


def test_cli_extract_count_help(cli_runner):
    result = cli_runner.invoke(
        cli, ["extract", "count", "--help"], catch_exceptions=False
    )
    assert result.exit_code == 0


def test_cli_extract_count(
    cli_runner, classraster_filepath, polygons_filepath, tmp_path
):
    outfile = tmp_path / "count.geojson"
    args = (
        f"extract count --in {polygons_filepath} --out {outfile} --format geojson --clip"
        f" --classrange 0 5 {classraster_filepath}"
    )

    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0
    assert outfile.is_file()

    expected_classes = ["class_%s" % x for x in range(6)]
    out_reader = FeatureReader(outfile)

    out_features = list(out_reader)
    assert len(out_features) == 83
    for outf in out_features:
        total = sum([outf[x] for x in expected_classes])
        assert total == outf["total_count"]

    # Check a couple features
    values = [out_features[3][x] for x in expected_classes]
    assert values == [0, 1, 0, 15, 3, 0]
    values = [out_features[59][x] for x in expected_classes]
    assert values == [151, 3, 0, 1, 18, 0]
