#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import os
# For omegaconf
from dataclasses import dataclass
from typing import Optional, List
#
from omegaconf import OmegaConf, MISSING
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
#
from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import load_model
from ov_piano.logging import ColorLogger
from ov_piano.data.maestro import MetaMAESTROv1, MetaMAESTROv2, MetaMAESTROv3
from ov_piano.data.maestro import MelMaestro
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder
from ov_piano.eval import GtLoaderMaestro
from ov_piano.eval import threshold_eval_single_file


# change plot font to latex
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"


# ##############################################################################
# # PLOT
# ##############################################################################
def qualitative_plot(gt_mel, gt_roll, pred_ons, pred_vel, mel_cmap="binary",
                     roll_cmap="binary", figsize=(10, 40), threshold=0.75,
                     tn_rgb=(255, 255, 255), tp_rgb=(0, 0, 0),
                     fp_rgb=(179, 179, 255), fn_rgb=(255, 51, 153),
                     secs_per_frame=0.024,
                     title_size=38, label_size=32, tick_size=25,
                     ev_title="Ground Truth vs. Thresholded Onset Predictions",
                     min_idx=None, max_idx=None, invert_yaxis=True):
    """
    :param gt_mel: Log-mel spectrogram of shape ``(f, t)``
    :param gt_roll: Ground truth boolean piano roll of shape ``(k, t)``
    :pred_ons: Predicted onsets of shape ``(k, t)`` between 0 and 1
    :pred_vel: Predicted velocities of shape ``(k, t)`` between 0 and 1
    :returns: figure and axes.
    """
    if max_idx is not None:
        gt_mel = gt_mel[:, :max_idx]
        gt_roll = gt_roll[:, :max_idx]
        pred_ons = pred_ons[:, :max_idx]
        pred_vel = pred_vel[:, :max_idx]
    if min_idx is not None:
        gt_mel = gt_mel[:, min_idx:]
        gt_roll = gt_roll[:, min_idx:]
        pred_ons = pred_ons[:, min_idx:]
        pred_vel = pred_vel[:, min_idx:]
    #
    fig, (mel_ax, v_ax, o_ax, eval_ax) = plt.subplots(
        nrows=4, figsize=figsize, sharex=True)
    #
    mel_ax.imshow(gt_mel, cmap=mel_cmap, aspect="auto")
    v_ax.imshow(pred_vel, cmap=roll_cmap, aspect="auto")
    o_ax.imshow(pred_ons, cmap=roll_cmap, aspect="auto")
    #
    pred_mask = (pred_ons >= threshold)
    tp_mask = (gt_roll & pred_mask)
    fp_mask = (~gt_roll & pred_mask)
    fn_mask = (gt_roll & ~pred_mask)
    #
    eval_arr = np.zeros(gt_roll.shape + (3,), dtype=np.uint8)
    eval_arr[:] = tn_rgb
    eval_arr[tp_mask.nonzero()] = tp_rgb
    eval_arr[fp_mask.nonzero()] = fp_rgb
    eval_arr[fn_mask.nonzero()] = fn_rgb
    eval_ax.imshow(eval_arr, aspect="auto")
    # appearance
    mel_ax.set_ylabel("Frequency", fontsize=label_size)
    v_ax.set_ylabel("Key", fontsize=label_size)
    o_ax.set_ylabel("Key", fontsize=label_size)
    eval_ax.set_ylabel("Key", fontsize=label_size)
    eval_ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: f"{x * secs_per_frame:g}"))
    eval_ax.set_xlabel("Time (s)", fontsize=label_size)
    #
    mel_ax.tick_params(labelsize=tick_size)
    v_ax.tick_params(labelsize=tick_size)
    o_ax.tick_params(labelsize=tick_size)
    eval_ax.tick_params(labelsize=tick_size)
    #
    axtitle_pad = 20
    mel_ax.set_title("Input Spectrogram", fontsize=title_size,
                     pad=axtitle_pad)
    v_ax.set_title("Predicted Onset Velocities", fontsize=title_size,
                   pad=axtitle_pad)
    o_ax.set_title("Predicted Onset Probabilities", fontsize=title_size,
                   pad=axtitle_pad)
    eval_ax.set_title(ev_title,
                      fontsize=title_size, pad=axtitle_pad)
    #
    if invert_yaxis:
        mel_ax.invert_yaxis()
        v_ax.invert_yaxis()
        o_ax.invert_yaxis()
        eval_ax.invert_yaxis()
    #
    return fig, (mel_ax, o_ax, v_ax, eval_ax)


def make_triple_onsets(onsets):
    """
    :param onsets: boolean array of shape ``(k, t)``
    :returns: boolean array of same shape, but every true entry at time
      ``t``is also extended to ``t+1, t+2``.
    """
    result = onsets.copy()
    result[:, 1:] |= result[:, :-1]
    result[:, 1:] |= result[:, :-1]
    return result


# ##############################################################################
# # GLOBALS
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar str DEVICE: For the PyTorch operations. Can be ``cpu`` or ``cuda``
      if a GPU is present.
    :cvar MAESTRO_PATH: Path to the root directory of the MAESTRO version
    :cvar int MAESTRO_VERSION: Currently 1, 2, 3 supported. 3 recommended.
    :cvar str OUTPUT_DIR: Optional directory to store instead of show plots.

    :cvar HDF5_MEL_PATH: Path to the HDF5 mel file previously generated.
    :cvar HDF5_ROLL_PATH: Path to the HDF5 piano roll file previously
      generated, must be compatible with the corresponding mel file.
    :cvar SNAPSHOT_INPATH: Optional input path to a pre-trained model, used
      to make predictions

    :cvar INFERENCE_CHUNK_SIZE: In this module, full files are processed, which
      may be too large for memory and have to be processed in strided chunks.
      This is the chunk size in seconds, it doesn't affect performance as long
      as it is large enough.
    :cvar INFERENCE_CHUNK_OVERLAP: See ``INFERENCE_CHUNK_SIZE``. This is the
      overlap among consecutive chunks. It doesn't affect performance as long
      as it is large enough to avoid boundary artifacts.
    """
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAESTRO_PATH: str = os.path.join("datasets", "maestro", "maestro-v3.0.0")
    MAESTRO_VERSION: int = 3
    #
    HDF5_MEL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")
    SNAPSHOT_INPATH: Optional[str] = None
    #
    INFERENCE_CHUNK_SIZE: float = 400
    INFERENCE_CHUNK_OVERLAP: float = 11
    INFERENCE_THRESHOLD: float = 0.74
    #
    CONV1X1: List[int] = (200, 200)
    LEAKY_RELU_SLOPE: float = 0.1
    #
    OUTPUT_DIR: Optional[str] = None  # "out/plots"
    FIGSIZE: List[float] = (20, 20)
    DPI: int = 350
    MEL_CMAP: str = "bone_r"  # cividis
    ROLL_CMAP: str = "bone_r" # binary
    TN_RGB: List[int] = (255, 255, 255)
    TP_RGB: List[int] = (0, 0, 0)
    FP_RGB: List[int] = (179, 179, 255)
    FN_RGB: List[int] = (255, 51, 153)
    TITLE_SIZE: int = 38
    LABEL_SIZE: int = 32
    TICK_SIZE: int = 25


CONF = OmegaConf.structured(ConfDef())
cli_conf = OmegaConf.from_cli()
CONF = OmegaConf.merge(CONF, cli_conf)

# derivative globals + parse HDF5 filenames and ensure they are consistent
(DATASET_NAME, SAMPLERATE, WINSIZE, HOPSIZE,
 MELBINS, FMIN, FMAX) = HDF5PathManager.parse_mel_hdf5_basename(
    os.path.basename(CONF.HDF5_MEL_PATH))
roll_params = HDF5PathManager.parse_roll_hdf5_basename(
    os.path.basename(CONF.HDF5_ROLL_PATH))
SECS_PER_FRAME = HOPSIZE / SAMPLERATE
#
CHUNK_SIZE = round(CONF.INFERENCE_CHUNK_SIZE / SECS_PER_FRAME)
CHUNK_OVERLAP = round(CONF.INFERENCE_CHUNK_OVERLAP / SECS_PER_FRAME)
#
assert DATASET_NAME == roll_params[0], "Inconsistent HDF5 datasets?"
assert SECS_PER_FRAME == roll_params[1], "Inconsistent roll quantization?"
assert (CHUNK_OVERLAP % 2) == 0, f"Only even overlap allowed! {CHUNK_OVERLAP}"
#
METAMAESTRO_CLASS = {1: MetaMAESTROv1, 2: MetaMAESTROv2,
                     3: MetaMAESTROv3}[CONF.MAESTRO_VERSION]
if CONF.OUTPUT_DIR:
    TXT_LOG_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "txt_logs")
    os.makedirs(TXT_LOG_OUTDIR, exist_ok=True)
else:
    TXT_LOG_OUTDIR = None


# ##############################################################################
# # MAIN LOOP INITIALIZATION
# ##############################################################################
if __name__ == "__main__":
    txt_logger = ColorLogger(os.path.basename(__file__), TXT_LOG_OUTDIR)
    txt_logger.info("\n\nCONFIGURATION:\n" + OmegaConf.to_yaml(CONF) + "\n\n")

    txt_logger.info("Loading test set")
    metamaestro_test = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["test"], years=METAMAESTRO_CLASS.ALL_YEARS)


    txt_logger.warning("SHORTENING TEST SET LENGTH")
    metamaestro_test.data = metamaestro_test.data[::5]


    maestro_test = MelMaestro(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_test.data),
        as_torch_tensors=False)
    onsets_beg, onsets_end = maestro_test.ONSETS_RANGE
    key_beg, key_end = PIANO_MIDI_RANGE
    #
    txt_logger.info("Loading test ground truths")
    test_gts = GtLoaderMaestro(maestro_test, metamaestro_test)

    # instantiate and load trained NN model
    txt_logger.info("Loading NN")
    num_mels = maestro_test[0][0].shape[0]
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg
    #
    model = OnsetsAndVelocities(
        in_chans=2,  # X and time_derivative(X)
        in_height=num_mels, out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=0).to(CONF.DEVICE)
    if CONF.SNAPSHOT_INPATH is not None:
        load_model(
            model, CONF.SNAPSHOT_INPATH, eval_phase=True, to_cpu=True)


    ##############
    # INFERENCE
    ##############
    def model_inference(x):
        """
        Convenience wrapper around the DNN to ensure output and input sequences
        have same length.
        """
        probs, vels = model(x)
        probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
        vels = F.pad(torch.sigmoid(vels), (1, 0))
        return probs, vels

    test_results = []
    test_results_vel = []
    len_test = len(maestro_test)
    for i, (mel, roll, md) in enumerate(maestro_test, 1):
        txt_logger.info(f"[{i}/{len_test} (test set)] {md}")
        onsets = (roll[onsets_beg:onsets_end][key_beg:key_end] > 0)
        triple_onsets = make_triple_onsets(onsets)
        #
        with torch.no_grad():
            tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
            onset_pred, vel_pred = strided_inference(
                model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP)
            onset_pred = onset_pred.cpu().numpy().squeeze()
            vel_pred = vel_pred.cpu().numpy().squeeze()
        #
        def qplot_ranged(min_idx=None, max_idx=None):
            """
            Closure to inspect ranges flexibly via one-liners like::
              qplot_ranged(0, 1000)[0].show()

            Note that the bottom plot has been adjusted to show only the GT.
            """
            fig, axes = qualitative_plot(mel, triple_onsets,
                                         onset_pred, vel_pred,
                                         mel_cmap=CONF.MEL_CMAP,
                                         roll_cmap=CONF.ROLL_CMAP,
                                         figsize=CONF.FIGSIZE,
                                         threshold=CONF.INFERENCE_THRESHOLD,
                                         # note that we only plot the GT here
                                         tn_rgb=CONF.TN_RGB, tp_rgb=CONF.TP_RGB,
                                         fp_rgb=CONF.TN_RGB, fn_rgb=CONF.TP_RGB,
                                         secs_per_frame=SECS_PER_FRAME,
                                         title_size=CONF.TITLE_SIZE,
                                         label_size=CONF.LABEL_SIZE,
                                         tick_size=CONF.TICK_SIZE,
                                         ev_title="Onset Ground Truth",
                                         min_idx=min_idx, max_idx=max_idx)
            return fig, axes
        #


        figtitle = md[0] + "\n" + " ".join(str(x) for x in md[1:])
        txt_logger.debug(figtitle)
        # fig = qplot_ranged(None, None)[0]
        # fig.suptitle(figtitle)
        # fig.tight_layout()
        # fig.show()
        # breakpoint()

        if i == 7:
            fig = qplot_ranged(5300, 7400)[0]
            fig.tight_layout()
            if CONF.OUTPUT_DIR is None:
                fig.show()
            else:
                fig.savefig(os.path.join(CONF.OUTPUT_DIR, md[0] + ".png"),
                            dpi=CONF.DPI)
            breakpoint()
