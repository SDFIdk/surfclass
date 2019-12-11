import pickle
from surfclass.train import load_training_data
from surfclass.scripts.cli import cli


def test_cli_train(cli_runner):
    result = cli_runner.invoke(cli, ["train"], catch_exceptions=False)
    assert result.exit_code == 0


def test_cli_train_testmodel1(cli_runner, tmp_path, testmodel1_traindata_filepath):
    outfile = tmp_path / "tmp_model.sav"
    args = f"train testmodel1 -n 50 {testmodel1_traindata_filepath} {outfile}"
    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0

    # Check the file exists
    assert outfile.is_file()
    # Sanity check, load the model and predict some sample data
    loaded_model = pickle.load(open(outfile, "rb"))
    (_, classes, features) = load_training_data(testmodel1_traindata_filepath)
    result = loaded_model.predict(features)

    assert result.shape[0] == features.shape[0] == classes.shape[0]
    # sanity checks


def test_cli_train_genericmodel(cli_runner, tmp_path, testmodel1_traindata_filepath):
    outfile = tmp_path / "tmp_model.sav"
    args = f"train genericmodel -n 50 {testmodel1_traindata_filepath} {outfile}"
    result = cli_runner.invoke(cli, args.split(" "), catch_exceptions=False)
    assert result.exit_code == 0

    # Check the file exists
    assert outfile.is_file()
    # Sanity check, load the model and predict some sample data
    loaded_model = pickle.load(open(outfile, "rb"))
    (_, classes, features) = load_training_data(testmodel1_traindata_filepath)
    result = loaded_model.predict(features)

    assert result.shape[0] == features.shape[0] == classes.shape[0]
    # sanity checks
