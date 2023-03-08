#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This script exemplifies a way to load and analyze the produced training logs,
it plots the cross-validation F1 score as a function of training, and prints
the maximum found.
"""


import os
# For omegaconf
from dataclasses import dataclass
from typing import Optional, List
#
from omegaconf import OmegaConf, MISSING
import pandas as pd
#
from ov_piano.logging import ColorLogger


# ##############################################################################
# # GLOBALS
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar LOG_PATH: Path to the training log.
    :cvar PLOT_RANGE: If given, string in the form ``'[a, b]``, with the range
      of performances to plot. If nothing given, no plot will be produced.
    """
    LOG_PATH: str = MISSING
    PLOT_RANGE: Optional[List[float]] = None


# ##############################################################################
# # MAIN LOOP INITIALIZATION
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)

    txt_logger = ColorLogger(os.path.basename(__file__))
    txt_logger.info("\n\nCONFIGURATION:\n" + OmegaConf.to_yaml(CONF))

    df = pd.DataFrame(
        pd.read_json(CONF.LOG_PATH, lines=True).iloc[:, 1].tolist())
    # ['PARAMETERS', 'WARNING', 'MODEL_INFO', 'TRAIN', 'XV_PROCESSING',
    # 'XV_BEST_ONSET', 'XV_BEST_ONSET_VEL', 'SAVED_MODEL']
    df_types = df[0].unique().tolist()
    df_by_type = {t: pd.DataFrame(df[df[0] == t][1].tolist())
                  for t in df_types}

    params = df_by_type["PARAMETERS"].to_dict("records")[0]
    txt_logger.info("\n".join(f"{k}: {v}" for k, v in params.items()))

    f1o = [float(x.split("\n")[-3].split(" ")[-1])
           for x in df_by_type["XV_BEST_ONSET"].iloc[:, 0]]
    f1v = [float(x.split("\n")[-3].split(" ")[-1])
           for x in df_by_type["XV_BEST_ONSET_VEL"].iloc[:, 0]]
    best_f1o = max(f1o)
    best_f1v = max(f1v)
    txt_logger.debug(f"best config: {(best_f1o, best_f1v)}")
    if CONF.PLOT_RANGE is not None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(f1o, label=f"F1 Onsets (max={best_f1o})")
        ax.plot(f1v, label=f"F1 O+V (max={best_f1v})")
        ax.legend()
        ax.set_ylim(CONF.PLOT_RANGE)
        fig.show()
        fig.suptitle(CONF.LOG_PATH)
        breakpoint()
