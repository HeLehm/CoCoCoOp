import pathlib

def data_dir_path():
    """Returns the path to the data directory."""
    return pathlib.Path(__file__).parent.resolve().joinpath("data")