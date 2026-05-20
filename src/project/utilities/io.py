from pathlib import Path
import pandas as pd
import numpy as np
from project.utilities.simulation import ExperimentResults
import matplotlib as mpl

def save_results(experiment_results: ExperimentResults, filename: str, folder: str = "data"):
    """
    Saves the results of an experiment to a csv file. The summary statistics are saved in a csv file. If trial data is returned, save in a separate csv
    file.

    Parameters
    -----------------
    experiment_results: ExperimentResults
        - Dataclass containing the results of the experiment. See ExperimentResults for more details.
    filename: str
        - The name of the file to which the results should be saved. The summary statistics will be saved to 'filename_summary.csv', 
        and the trial data will be saved to 'filename_trials.csv'.
    folder: str, optional
        - The folder to which the results should be saved. Default is "data". If the folder does not exist, it will be created.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    experiment_results.summary_stats.to_csv(folder / f"{filename}_summary.csv", index = True)
    print(f"Saved: {folder / f'{filename}_summary.csv'}")

    if experiment_results.trial_data is not None:
        experiment_results.trial_data.to_csv(folder / f"{filename}_trials.csv", index = True)
        print(f"Saved: {folder / f'{filename}_trials.csv'}")


def load_results(filename, folder="data", index_col='exp_id'):
    """
    Function for loading csv file containing summary statistics of experiments.

    Parameters
    ----------
    filename: str
        Name of the CSV file.
    folder: str or Path
        Folder where the CSV file is located.
    index_col: str, optional
        Column to set as index. Default is 'exp_id'.
        
    Returns
    -------
    summary_stats: pd.DataFrame
        Dataframe containing the summary statistics.
    """

    folder = Path(folder)
    filename = filename if filename.endswith(".csv") else f"{filename}.csv"
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"No CSV file found at: {path}")

    df = pd.read_csv(path, index_col = index_col)
    print(f"Loaded: {path}")

    return df



#============== For plots ===============#
def use_latex_fonts(font_size=14):
    """
    Configure Matplotlib to use LaTeX fonts.

    Requires a working LaTeX installation.
    """

    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],

        "font.size": font_size,
        "figure.titlesize": font_size + 2,
        "axes.labelsize": font_size - 2,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 3 ,

        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amssymb}
        """,
    })


def figure_size(preset:str):
    """
    For determining figure size based on preset. Presets are 'single', 'double', 'triple', referring to the number of rows. The last is 'wrap',
    for when the figure should be wrapped around the text.

    Should generally only be used if there are multiple columns.
    """

    presets = {
        'single': (10, 4),
        'double': (10, 7),
        'triple': (10, 10),
        'wrap': (6, 3)
    }

    if preset not in presets:
        raise ValueError(f"Preset must be one of {presets.keys()}")

    return presets[preset]

def save_plot(
    fig,
    filename,
    folder="figures",
    formats=("pdf",),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
):
    """
    Save a matplotlib figure to one or more formats.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        The figure object to save.
    filename: str
        File name without extension
    folder: str | Path
        Output folder.
    formats: tuple[str, ...]
        Output formats, e.g. ("pdf",) or ("pdf", "png").
    dpi: int
        Resolution for raster formats such as PNG.
    bbox_inches: str
        Use "tight" to remove excess whitespace.
    pad_inches: float
        Padding around the saved figure.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    filename = Path(filename).stem

    for fmt in formats:
        path = folder / f"{filename}.{fmt}"
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        print(f"Saved: {path}")

