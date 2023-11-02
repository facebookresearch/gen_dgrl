# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import collections
import csv
import fnmatch
import os
import re
from functools import reduce
from itertools import chain, repeat
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
Usage:
    Plot grid of games:
    >>> python -m utils.plot \
            --grid  \
            -r xpid_prefix1 xpid_prefix2 \
            -l method_name1 method_name2  \
            -xi 5000000 -xts M --save_width 190 --save_height 225 \
            --linewidth=0.5 \
            -a 0.1 --fontsize 6 -yl 'Mean test episode return' \
            --savename full_procgen_results


    Comparing mean curves:
    >>> python -m utils.plot \
            --avg_procgen \
            -r \
            xpid_prefix1 \
            xpid_prefix2 \
            -l \
            method_name1 \
            method_name2 \
            -a 0.1 -xi 5000000 -xts M --save_width 200 --save_height 200 --savename 'savename'
"""

"""
Example usage:

    >>> python -m utils.plot \
            --grid  \
            -r "./data_path/offlinerl/bc" \
            --prefix easy-200-bc-p1.0-lr0.0001-bs512 \
            -l l200 \
            --x_axis=epoch \
            --y_axis=test_rets_mean \
            -xi 5000000 -xts M --save_width 190 --save_height 225 \
            --linewidth=0.5 \
            -a 0.1 --fontsize 6 -yl 'Mean test episode return' \
            --save_path . \
            --savename 'bc_test_returns_grid'

    >>> python -m utils.plot \
            --avg_procgen \
            -r "./data_path/offlinerl/bc" \
            --prefix easy-200-bc-p1.0-lr0.0001-bs512 \
            -l l200 \
            --x_axis=epoch \
            --y_axis=test_rets_mean \
            -yl 'Mean normalized test episode return' \
            -a 0.1 -xi 5000000 -xts M --save_width 200 --save_height 200 \
            --save_path . \
            --savename 'bc_test_returns_mean'
"""


class OuterZipStopIteration(Exception):
    pass


def outer_zip(*args):
    """
    https://stackoverflow.com/questions/13085861/outerzip-zip-longest-function-with-multiple-fill-values
    """
    count = len(args) - 1

    def sentinel(default):
        nonlocal count
        if not count:
            raise OuterZipStopIteration
        count -= 1
        yield default

    iters = [chain(p, sentinel(default), repeat(default)) for p, default in args]
    try:
        while iters:
            yield tuple(map(next, iters))
    except OuterZipStopIteration:
        pass


def islast(itr):
    old = next(itr)
    for new in itr:
        yield False, old
        old = new
    yield True, old


def file_index_key(f):
    pattern = r"\d+$"
    key_match = re.findall(pattern, Path(f).stem)
    if len(key_match):
        return int(key_match[0])
    return f


def reformat_large_tick_values(tick_val, pos=None):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).

    From: https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    """
    if tick_val >= 1_000_000_000:
        val = round(tick_val / 1_000_000_000, 1)
        new_tick_format = "{:}B".format(val)
    elif tick_val >= 1_000_000:
        val = round(tick_val / 1_000_000, 1)
        new_tick_format = "{:}M".format(val)
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        new_tick_format = "{:}K".format(val)
    # elif tick_val < 1000 and tick_val >= 0.1:
    #    new_tick_format = round(tick_val, 1)
    elif tick_val >= 10:
        new_tick_format = round(tick_val, 1)
    elif tick_val >= 1:
        new_tick_format = round(tick_val, 2)
    elif tick_val >= 1e-4:
        # new_tick_format = '{:}m'.format(val)
        new_tick_format = round(tick_val, 3)
    elif tick_val >= 1e-8:
        # val = round(tick_val*10000000, 1)
        # new_tick_format = '{:}μ'.format(val)
        new_tick_format = round(tick_val, 8)
    else:
        new_tick_format = tick_val

    new_tick_format = str(new_tick_format)
    new_tick_format = new_tick_format if "e" in new_tick_format else new_tick_format[:6]
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0" and (tick_val >= 10 or tick_val <= -10 or tick_val == 0.0):
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2 :]

    # FIXME: manual hack
    if new_tick_format == "-0.019":
        new_tick_format = "-0.02"
    elif new_tick_format == "-0.039":
        new_tick_format = "-0.04"

    return new_tick_format


def gather_results_for_prefix(args, results_path, prefix, env_name: str, point_interval):
    pattern = f"*{prefix}*"

    xpids = fnmatch.filter(os.listdir(os.path.join(results_path, env_name)), pattern)
    xpids.sort(key=file_index_key)

    assert len(xpids) > 0, f"Results for {pattern} not found."

    pd_series = []

    nfiles = 0
    for i, f in enumerate(xpids):
        print(f"xpid: {f}")
        if int(f[-1]) > args.max_index:
            print("skipping xpid... ", f)
            continue
        f_in = open(os.path.join(results_path, env_name, f, args.log_filename), "rt")
        reader = csv.reader((line.replace("\0", " ") for line in f_in))
        headers = next(reader, None)
        # print(f)
        if len(headers) < 2:
            raise ValueError("result is malformed")
        headers[0] = headers[0].replace("#", "").strip()  # remove comment hash and space

        xs = []
        ys = []
        last_x = -1

        double_x_axis = False
        if "-kl" in f:
            double_x_axis = True

        # debug = False
        # if f == 'lr-ucb-hard-fruitbot-random-s500_3':
        # 	debug = True
        # 	import pdb; pdb.set_trace()

        for row_index, (is_last, row) in enumerate(islast(reader)):
            # if debug:
            # print(row_index, is_last)

            if len(row) != len(headers):
                continue

            # print(row_index)
            if args.max_lines and row_index > args.max_lines:
                break
            if row_index % point_interval == 0 or is_last:
                row_dict = dict(zip(headers, row))
                x = int(row_dict[args.x_axis])
                if args.x_axis == "step" and double_x_axis:
                    x *= 2

                if x < last_x:
                    # print(f, x, row_index)
                    continue
                last_x = x

                if args.max_x is not None and x > args.max_x:
                    print("broke here")
                    break
                if args.gap:
                    y = float(row_dict["train_rets_mean"]) - float(row_dict["test_rets_mean"])
                else:
                    try:
                        y_value = ast.literal_eval(row_dict[args.y_axis])
                        y = float(y_value[0] if isinstance(y_value, collections.abc.Container) else y_value)
                    except Exception:
                        print("setting y=None")
                        y = None

                xs.append(x)
                ys.append(y)

        pd_series.append(pd.Series(ys, index=xs).sort_index(axis=0))
        nfiles += 1

    return nfiles, pd_series


def plot_results_for_prefix(args, ax, results_path, prefix: str, label, env_name: str = None, tag=""):
    if not env_name:
        env_name = prefix.split("-")[0]

    assert env_name in PROCGEN.keys(), f"{env_name} is not a valid Procgen game!"

    nfiles, pd_series = gather_results_for_prefix(args, results_path, prefix, env_name, args.point_interval)

    for i, series in enumerate(pd_series):
        pd_series[i] = series.loc[~series.index.duplicated(keep="first")]

    try:
        df = pd.concat(pd_series, axis=1).interpolate(method="linear") * args.scale
    except Exception:
        df = pd.concat(pd_series, axis=1) * args.scale

    # TODO: HACK To prevent unwanted lines
    df.drop(df.index[-1], inplace=True)

    ewm = df.ewm(alpha=args.alpha, ignore_na=True).mean()

    all_x = np.array([i for i in df.index])
    max_x = max(all_x)
    plt_x = all_x
    plt_y_avg = np.array([y for y in ewm.mean(axis=1)])
    plt_y_std = np.array([std for std in ewm.std(axis=1)])
    # for plt_y_row in range(1, len(plt_x)):
    # 	if plt_x[plt_y_row] <= plt_x[plt_y_row-1]:
    # 		print("Error is at row",plt_y_row)
    # 		print(plt_x[plt_y_row], plt_x[plt_y_row-1])
    # print(plt_y_avg.shape)

    # max_y = max(plt_y_avg + plt_y_std)
    # y_ticks = [0, np.floor(max_y/2.), max_y]
    # ax.set_yticks(y_ticks)

    # import pdb; pdb.set_trace()

    ax.plot(plt_x, plt_y_avg, linewidth=args.linewidth, label=label)
    ax.fill_between(plt_x, plt_y_avg - plt_y_std, plt_y_avg + plt_y_std, alpha=0.1)

    if args.grid:
        ax.set_title(env_name, fontsize=args.fontsize)
    else:
        ax.title(env_name, fontsize=args.fontsize)

    info = {"max_x": max_x, "all_x": all_x, "avg_y": plt_y_avg, "std_y": plt_y_std, "df": ewm, "tag": tag}
    return info


def format_subplot(subplt):
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3, prop={'size': args.fontsize})
    # tick_fontsize = 4
    tick_fontsize = 6
    subplt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    subplt.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    subplt.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(reformat_large_tick_values))
    subplt.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mpl.ticker.FormatStrFormatter("%d")))
    subplt.tick_params(axis="y", which="major", pad=-1)
    subplt.tick_params(axis="x", which="major", pad=0)
    subplt.grid(linewidth=0.5)


def format_plot(args, fig, plt):
    ax = plt.gca()

    if args.legend_inside:
        fig.legend(loc="lower right", prop={"size": args.fontsize})
    else:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, prop={"size": args.fontsize})
        # fig.legend(loc='upper center', bbox_to_anchor=(0.405, 0.915), ncol=1, prop={'size': args.fontsize})
        # fig.legend(loc='upper center', bbox_to_anchor=(0.65, 0.41), ncol=1, prop={'size': args.fontsize})
        # fig.legend(loc='upper center', bbox_to_anchor=(0.29, 0.95), ncol=1, prop={'size': 8})
        # fig.legend(loc='upper right', bbox_to_anchor=(1.26, 0.85), ncol=1, prop={'size': args.fontsize})
        # ax.set_title('ninja', fontsize=8)
        if args.title:
            ax.set_title(args.title, fontsize=8)

        # pass
    # ax.set_ylim([0,1.0])

    ax.set_xlabel(args.x_label, fontsize=args.fontsize)
    ax.set_ylabel(args.y_label, fontsize=args.fontsize)

    tick_fontsize = args.fontsize
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(reformat_large_tick_values))

    if args.max_y is not None:
        ax.set_ylim(top=args.max_y)

    if args.min_y is not None:
        ax.set_ylim(bottom=args.min_y)


PROCGEN = {
    "bigfish": {"easy": (1, 40), "hard": (0, 40)},
    "bossfight": {"easy": (0.5, 13), "hard": (0.5, 13)},
    "caveflyer": {"easy": (3.5, 12), "hard": (2, 13.4)},
    "chaser": {"easy": (0.5, 13), "hard": (0.5, 14.2)},
    "climber": {"easy": (2, 12.6), "hard": (1, 12.6)},
    "coinrun": {"easy": (5, 10), "hard": (5, 10)},
    "dodgeball": {"easy": (1.5, 19), "hard": (1.5, 19)},
    "fruitbot": {"easy": (-1.5, 32.4), "hard": (-0.5, 27.2)},
    "heist": {"easy": (3.5, 10), "hard": (2, 10)},
    "jumper": {"easy": (1, 10), "hard": (1, 10)},
    "leaper": {"easy": (1.5, 10), "hard": (1.5, 10)},
    "maze": {"easy": (5, 10), "hard": (4, 10)},
    "miner": {"easy": (1.5, 13), "hard": (1.5, 20)},
    "ninja": {"easy": (3.5, 10), "hard": (2, 10)},
    "plunder": {"easy": (4.5, 30), "hard": (3, 30)},
    "starpilot": {"easy": (2.5, 64), "hard": (1.5, 35)},
}


if __name__ == "__main__":
    """
    Arguments:
            --prefix: filename prefix of result files. Results from files with shared filename prefix will be averaged.
            --results_path: path to directory with result files
            --label: labels for each curve
            --max_index: highest index i to consider in computing curves per prefix, where filenames are of form "^(prefix).*_(i)$"
            --alpha: Polyak averaging smoothing coefficient
            --x_axis: csv column name for x-axis data, defaults to "epoch"
            --y_axis: csv column name for y-axis data, defaults to "loss"
            --threshold: show a horizontal line at this y value
            --threshold_label: label for threshold
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--base_path", type=str, default="~/logs/ppo", help="base path to results directory per prefix"
    )
    parser.add_argument("-r", "--results_path", type=str, nargs="+", default=[""], help="path to results directory")
    parser.add_argument(
        "--prefix", type=str, nargs="+", default=[""], help="Plot each xpid group matching this prefix per game"
    )
    parser.add_argument(
        "--log_filename", type=str, default="logs.csv", help="Name of log output file in each result directory"
    )
    parser.add_argument("-lines", "--max_lines", type=int, default=None, help="only plot every this many points")
    parser.add_argument("--grid", action="store_true", help="Plot all prefix tuples per game in a grid")
    # parser.add_argument('--xpid_prefix', type=str, default='lr-ppo', help='Prefix of xpid folders if plotting curves aggregated by subfolders')
    parser.add_argument(
        "--xpid_prefix",
        type=str,
        nargs="+",
        default=[],
        help="Prefix of xpid folders if plotting curves aggregated by subfolders",
    )

    parser.add_argument("-s", "--scale", type=float, default=1.0, help="scale all values by this constant")
    parser.add_argument("-l", "--label", type=str, nargs="+", default=[None], help="labels")
    parser.add_argument("-m", "--max_index", type=int, default=10, help="max index of prefix match to use")

    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="alpha for emwa")
    parser.add_argument("-x", "--x_axis", type=str, default="epoch", help="csv column name of x-axis data")
    parser.add_argument(
        "-y", "--y_axis", type=str, default="test:mean_episode_return", help="csv column name of y-axis data"
    )
    parser.add_argument("-yr", "--y_range", type=float, default=[], help="y range")
    parser.add_argument("-xl", "--x_label", type=str, default="Steps", help="x-axis label")
    parser.add_argument("-yl", "--y_label", type=str, default="Mean test episode return", help="y-axis label")
    parser.add_argument("-xi", "--x_increment", type=int, default=1, help="x-axis increment")
    parser.add_argument("-xts", "--x_tick_suffix", type=str, default="M", help="x-axis tick suffix")
    parser.add_argument("-pi", "--point_interval", type=int, default=1, help="only plot every this many points")
    parser.add_argument("--max_x", type=float, default=None, help="max x-value")
    parser.add_argument("--max_y", type=float, default=None, help="max y-value")
    parser.add_argument("--min_y", type=float, default=None, help="min y-value")
    parser.add_argument("--x_values_as_axis", action="store_true", help="Show exactly x-values in data along x-axis")
    parser.add_argument(
        "--ignore_x_values_in_axis", type=float, nargs="+", default=[], help="Ignore these x-values in axis"
    )
    parser.add_argument("--linewidth", type=float, default=1.0, help="line width")
    parser.add_argument("--linestyle", type=str, default="-", help="line style")

    parser.add_argument("--threshold", type=float, default=None, help="show a horizontal line at this y value")
    parser.add_argument("--threshold_label", type=str, default="", help="label for threshold")

    parser.add_argument("--save_path", type=str, default="figures/", help="Path to save image")
    parser.add_argument("--savename", type=str, default=None, help="Name of output image")
    parser.add_argument("--dpi", type=int, default=72, help="dpi of saved image")
    parser.add_argument("--save_width", type=int, default=800, help="pixel width of saved image")
    parser.add_argument("--save_height", type=int, default=480, help="pixel height of saved image")
    parser.add_argument("--fontsize", type=int, default=6, help="pixel height of saved image")
    parser.add_argument("--legend_inside", action="store_true", help="show legend inside plot")
    parser.add_argument("--title", type=str, help="title for single plot")

    parser.add_argument("--gap", action="store_true", default=False, help="Whether to plot the generalization gap")
    parser.add_argument("--avg_procgen", action="store_true", help="Average all return-normalized curves")

    parser.add_argument("--procgen_mode", type=str, default="easy", choices=["easy", "hard"], help="Procgen env mode")

    args = parser.parse_args()

    sns.set_style("whitegrid", {"grid.color": "#EFEFEF"})

    # Create an array with the colors you want to use
    # TODO: check num_colors
    num_colors = max(len(args.prefix), len(args.results_path))

    num_colors = 7
    # num_colors = 5
    # num_colors = 3
    colors = sns.husl_palette(num_colors, h=0.1)

    # tmp = colors[1]
    # colors[1] = colors[-2]
    # TODO: Why?
    tmp = colors[2]
    colors[2] = colors[-1]

    # colors[1] = colors[-1]
    # colors[2] = colors[-2]

    # colors[-1] = tmp
    # colors[1] = colors[-1]
    # colors[0] = colors[-2]

    # colors = [
    # 	(0.8859561388376407, 0.5226505841897354, 0.195714831410001), # Orange
    # 	(1., 0.19215686, 0.19215686), # TSCL red
    # 	(1, 0.7019607843137254, 0.011764705882352941), # mixreg yellow
    # 	# (0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green
    # 		(0.20964485513246672, 0.6785281560863642, 0.6309437466865638), # Teal
    # 		(0.9615698478167679, 0.3916890619185551, 0.8268671491444017), # Pink
    # 	(0.3711152842731098, 0.6174124752499043, 0.9586047646790773), # Blue
    # ]

    # colors = [
    # 	(0.8859561388376407, 0.5226505841897354, 0.195714831410001), # Orange
    # 	# (0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green
    # 	# (1, 0.7019607843137254, 0.011764705882352941), # mixreg yellow
    # 	# (1., 0.19215686, 0.19215686), # TSCL red
    # 	# (0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green
    # 	(0.9615698478167679, 0.3916890619185551, 0.8268671491444017), # Pink
    # 		(0.20964485513246672, 0.6785281560863642, 0.6309437466865638), # Teal
    # 	# sub, # Blue
    # ]

    # colors =[
    # 	(0.5019607843, 0.5019607843, 0.5019607843), # gray DR
    # 	(0.8859561388376407, 0.5226505841897354, 0.195714831410001), # Orange "ROBIN":
    # 	(0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green "OG":
    # 	(1, 0.7019607843137254, 0.011764705882352941), # mixreg yellow "Constant":
    # 	(0.9615698478167679, 0.3916890619185551, 0.8268671491444017), # Pink "Big":
    #  (0.3711152842731098, 0.6174124752499043, 0.9586047646790773), # Blue "DPD":
    # ]

    colors = [
        (0.407, 0.505, 0.850),
        (0.850, 0.537, 0.450),
        (0.364, 0.729, 0.850),
        (0.850, 0.698, 0.282),
        (0.321, 0.850, 0.694),
        (0.850, 0.705, 0.717),
        (0.4, 0.313, 0.031),
        (0.858, 0, 0.074),
        (0.098, 1, 0.309),
        (0.050, 0.176, 1),
        (0.580, 0.580, 0.580),
        (0.9615698478167679, 0.3916890619185551, 0.8268671491444017),
    ]

    # (0.20964485513246672, 0.6785281560863642, 0.6309437466865638), # Teal "Reinit":

    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))

    dpi = args.dpi
    if args.grid:
        num_plots = len(PROCGEN.keys())
        subplot_width = 4
        subplot_height = int(np.ceil(num_plots / subplot_width))
        fig, ax = plt.subplots(subplot_height, subplot_width, sharex="col", sharey=False)
        ax = ax.flatten()
        fig.set_figwidth(args.save_width / dpi)
        fig.set_figheight(args.save_height / dpi)
        fig.set_dpi(dpi)
        # fig.tight_layout()
        plt.subplots_adjust(left=0.025, bottom=0.10, right=0.97, top=0.80, wspace=0.05, hspace=0.3)
    else:
        ax = plt
        fig = plt.figure(figsize=(args.save_width / dpi, args.save_height / dpi), dpi=dpi)

    plt_index = 0
    max_x = 0

    print(f"========= Final {args.y_axis} ========")
    results_path = args.results_path
    if args.base_path:
        base_path = os.path.expandvars(os.path.expanduser(args.base_path))
        results_path = [os.path.join(base_path, p) for p in results_path]

    results_metas = list(outer_zip((results_path, results_path[-1]), (args.prefix, args.prefix[-1]), (args.label, "")))

    infos_dict = {f"{p}_{str(i)}": [] for i, p in enumerate(args.results_path)}
    for i, meta in enumerate(results_metas):
        rp, p, label = meta
        print(f"Results Path: {rp}, Prefix: {p}, Label: {label}")

        if args.grid:
            for j, env in enumerate(PROCGEN.keys()):
                if j > 0:
                    label = None
                info = plot_results_for_prefix(args, ax[j], rp, p, label, env_name=env, tag=env)
                # TODO: remove results_path[i]
                infos_dict[f"{results_path[i]}_{i}"].append(info)
                max_x = max(info["max_x"], max_x)

        elif args.avg_procgen:
            all_series = []
            for j, env in enumerate(PROCGEN.keys()):
                print(j, env)
                if j > 0:
                    label = None

                _, pd_series = gather_results_for_prefix(args, rp, p, env_name=env, point_interval=args.point_interval)

                if not args.gap:
                    R_min, R_max = PROCGEN[env][args.procgen_mode]
                    # This is the normalization part
                    pd_series = [p.add(-R_min).divide(R_max - R_min) for p in pd_series]

                all_series.append(pd_series)

            all_series_pd = []
            min_length = float("inf")
            all_series_updated = []
            for series in all_series:
                updated_series = [s[~s.index.duplicated(keep="first")] for s in series]
                all_series_updated.append([s[~s.index.duplicated(keep="first")] for s in series])
                min_length = min(np.min([len(s) for s in updated_series]), min_length)
                print(f"Length: {min(np.min([len(s) for s in updated_series]), min_length)}")

            min_length = int(min_length)
            all_series = all_series_updated

            for series in all_series:
                trunc_series = [s[:min_length] for s in series]
                all_series_pd.append(pd.concat(trunc_series, axis=1).interpolate(method="linear") * args.scale)

            df = reduce(lambda x, y: x.add(y, fill_value=0), all_series_pd) / len(PROCGEN.keys())
            # try:
            # 	import pdb; pdb.set_trace()
            # 	df = pd.concat(avg_series, axis=1).interpolate(method='linear')*args.scale
            # except:
            # 	df = pd.concat(avg_series, axis=1)*args.scale
            ewm = df.ewm(alpha=args.alpha, ignore_na=True).mean()

            all_x = np.array([i for i in df.index])
            max_x = max(all_x)
            plt_x = all_x
            plt_y_avg = np.array([y for y in ewm.mean(axis=1)])
            plt_y_std = np.array([std for std in ewm.std(axis=1, ddof=1)])

            # plot
            ax.plot(plt_x, plt_y_avg, linewidth=args.linewidth, label=meta[-1], linestyle=args.linestyle)
            ax.fill_between(plt_x, plt_y_avg - plt_y_std, plt_y_avg + plt_y_std, alpha=0.1)

            info = {
                "max_x": max_x,
                "all_x": all_x,
                "avg_y": plt_y_avg,
                "std_y": plt_y_std,
                "df": ewm,
                "tag": results_metas[i][-1],
            }
            infos_dict[f"{results_path[i]}_{i}"].append(info)
        else:
            info = plot_results_for_prefix(args, plt, rp, p, label)
            max_x = max(info["max_x"], max_x)

            # print(f"{p}: {round(info['avg_y'][-1], 2)} +/- {round(info['std_y'][-1], 2)}")

    all_x = info["all_x"]

    all_ax = ax if args.grid else [plt]
    for subax in all_ax:
        if args.threshold is not None:
            threshold_x = np.linspace(0, max_x, 2)
            subax.plot(
                threshold_x,
                args.threshold * np.ones(threshold_x.shape),
                zorder=1,
                color="k",
                linestyle="dashed",
                linewidth=args.linewidth,
                alpha=0.5,
                label=args.threshold_label,
            )

    if args.grid:
        handles, labels = all_ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1), prop={"size": args.fontsize})
        fig.text(0.5, 0.01, args.x_label, ha="center", fontsize=args.fontsize)
        fig.text(0.0, 0.5, args.y_label, va="center", rotation="vertical", fontsize=args.fontsize)
        for ax in all_ax:
            format_subplot(ax)
    else:
        format_plot(args, fig, plt)

    # Render plot
    if args.savename:
        plt.savefig(os.path.join(args.save_path, f"{args.savename}.pdf"), bbox_inches="tight", pad_inches=0, dpi=dpi)
    else:
        # plt.subplot_tool()
        plt.show()
