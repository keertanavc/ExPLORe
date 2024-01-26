# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import seaborn as sns
import os
from filter_data import get_runs_with_metrics
from wandb_dl import Data

if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    import argparse
    parser = argparse.ArgumentParser(
        prog='Plot results',
        description='Generate paper plots'
    )

    parser.add_argument('--domain', type=str, help="one of [antmaze, adroit, cog, all]")
    parser.add_argument('--num_seeds', type=int, default=3)
    args = parser.parse_args()

    #if args.domain == "deepsea" or args.domain == "all":
    with open('data/deepsea.pkl', 'rb') as handle:
        deepsea_raw_data = pkl.load(handle)
    steps = deepsea_raw_data._runs[0]['env_step']
    returns = deepsea_raw_data._runs[0]['evaluation/return']
    plt.plot(returns[0], returns[1])
    plt.show()
