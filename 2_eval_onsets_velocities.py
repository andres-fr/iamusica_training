#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Assuming a pretrained model to detect piano key onsets and velocities, this
script uses the MAESTRO validation split to find its optimal detection
threshold and delay via a grid search, and then the MAESTRO test split to
compute the evaluation results (in the form of precision, recall and F1) for
the corresponding optimal threshold and delay. Specifically:

1. loads cross-validation and test datasets
2. loads ground truth annotations and convert into event format
3. instantiates model and decoder to predict logmels into event format
4. performs model inference on the full XV dataset
5. performs grid search XV eval to find best threshold and delay hyperpars
6. performs full test evaluation with XV-optimal threshold and delay

Note that, to minimize the chances of overfitting, the optimal hyperparameters
are being searched on the cross-validation set. Then, they are kept constant
and the test evaluation is performed only once.
"""


import os
# For omegaconf
from dataclasses import dataclass
from typing import List
#
from omegaconf import OmegaConf, MISSING
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
#
from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import load_model
from ov_piano.logging import ColorLogger
from ov_piano.data.maestro import MetaMAESTROv1, MetaMAESTROv2, MetaMAESTROv3
from ov_piano.data.maestro import MelMaestro
import ov_piano.models
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder
from ov_piano.eval import GtLoaderMaestro
from ov_piano.eval import threshold_eval_single_file

# import matplotlib.pyplot as plt


# ##############################################################################
# # GLOBALS
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar str DEVICE: For the PyTorch operations. Can be ``cpu`` or ``cuda``
      if a GPU is present. GPU is highly recommended.
    :cvar MAESTRO_PATH: Path to the root directory of the MAESTRO version
    :cvar int MAESTRO_VERSION: Currently 1, 2, 3 supported. 3 recommended.
    :cvar str OUTPUT_DIR: Where to store model snapshots and text logs.
      Created if non-existing.

    :cvar HDF5_MEL_PATH: Path to the HDF5 mel file previously generated.
    :cvar HDF5_ROLL_PATH: Path to the HDF5 piano roll file previously
      generated, must be compatible with the corresponding mel file.
    :cvar SNAPSHOT_INPATH: Optional input path to a pre-trained model, used
      to intialize and resume training from.

    :cvar XV_TAKE_ONE_EVERY: Since we are doing a (likely inefficient)
      grid search on the cross-validation set, and the size is considerable,
      we can use this parameter to take only one file from every N in the set.
      Experiments show that taking 1 of 5 doesn't alter results significantly.
    :cvar SEARCH_THRESHOLDS: Before running the test, several thresholds are
      being searched via grid search on the cross-validation split. This list
      determines said thresholds.
    :cvar SEARCH_SHIFTS: Analogous to the thresholds, but determines what delay
      offset, in seconds, is applied to the predictions.

    :cvar DECODER_GAUSS_STD: The decoder on top of the DNN predictions performs
      a Gaussian time-convolution to smoothen detections. This is the standard
      deviation, in time-frames.
    :cvar DECODER_GAUSS_KSIZE: The window size, in time-frames, for the
      smoothening Gaussian time-convolution.

    :cvar TOLERANCE_SECS: The maximum absolute error between onset prediction
      and ground truth, in seconds, to consider the prediction correct.
    :cvar TOLERANCE_VEL: The maximum absolute error between velocity prediction
      and ground truth, in ratio between 0 and 1, to consider the prediction
      correct. To better understand this ratio, see the official documentation
      for ``mir_eval.transcription_velocity``.

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
    OUTPUT_DIR: str = "out"
    #
    HDF5_MEL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")
    SNAPSHOT_INPATH: str = MISSING
    #
    CONV1X1: List[int] = (200, 200)
    #
    XV_TAKE_ONE_EVERY: int = 5
    SEARCH_THRESHOLDS: List[float] = (0.70, 0.71, 0.72, 0.73, 0.74, 0.75,
                                      0.76, 0.77, 0.78, 0.79, 0.80)
    SEARCH_SHIFTS: List[float] = (-0.01,)
    #
    DECODER_GAUSS_STD: float = 1
    DECODER_GAUSS_KSIZE: int = 11
    #
    TOLERANCE_SECS: float = 0.05
    TOLERANCE_VEL: float = 0.1
    #
    INFERENCE_CHUNK_SIZE: float = 300
    INFERENCE_CHUNK_OVERLAP: float = 11



# ##############################################################################
# # MAIN LOOP INITIALIZATION
# ##############################################################################
if __name__ == "__main__":
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
    TXT_LOG_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "txt_logs")
    os.makedirs(TXT_LOG_OUTDIR, exist_ok=True)

    txt_logger = ColorLogger(os.path.basename(__file__), TXT_LOG_OUTDIR)
    txt_logger.info("\n\nCONFIGURATION:\n" + OmegaConf.to_yaml(CONF) + "\n\n")

    txt_logger.info("Loading datasets")
    metamaestro_xv = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["validation"],
        years=METAMAESTRO_CLASS.ALL_YEARS)
    maestro_xv = MelMaestro(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_xv.data),
        as_torch_tensors=False)
    metamaestro_test = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["test"], years=METAMAESTRO_CLASS.ALL_YEARS)
    maestro_test = MelMaestro(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_test.data),
        as_torch_tensors=False)

    # shorten xv set to speed up cross validation times
    if CONF.XV_TAKE_ONE_EVERY != 1:
        txt_logger.critical("SHORTENING XV SPLIT FOR FASTER CROSSVALIDATION!")
        maestro_xv.data = maestro_xv.data[::CONF.XV_TAKE_ONE_EVERY]
        metamaestro_xv.data = metamaestro_xv.data[::CONF.XV_TAKE_ONE_EVERY]
    #
    txt_logger.info("Loading XV ground truths")
    xv_gts = GtLoaderMaestro(maestro_xv, metamaestro_xv)

    txt_logger.info("Loading test ground truths")
    test_gts = GtLoaderMaestro(maestro_test, metamaestro_test)

    # instantiate and load trained NN model
    txt_logger.info("Loading NN")
    num_mels = maestro_xv[0][0].shape[0]
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg
    #
    model = OnsetsAndVelocities(
        in_chans=2,  # X and time_derivative(X)
        in_height=num_mels, out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=CONF.BATCH_NORM,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=CONF.DROPOUT).to(CONF.DEVICE)
    load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True)
    # instantiate decoder
    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys, nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1, vel_pad_right=1)

    ##############
    # XV INFERENCE
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

    xv_dataframes = []
    len_xv = len(maestro_xv)
    for i, (mel, roll, md) in enumerate(maestro_xv, 1):
        txt_logger.info(f"[{i}/{len_xv}] XV inference: {md}")
        with torch.no_grad():
            tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
            onset_pred, vel_pred = strided_inference(
                model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP)
            del tmel
            pred_df = decoder(
                onset_pred, vel_pred, pthresh=min(CONF.SEARCH_THRESHOLDS))
            gt_df = xv_gts(md)[0]
            xv_dataframes.append((gt_df, pred_df))

    ###############
    # XV GRIDSEARCH
    ###############
    xv_gridsearch = {}
    xv_gridsearch_vel = {}
    for thresh in CONF.SEARCH_THRESHOLDS:
        for shift in CONF.SEARCH_SHIFTS:
            this_eval = []
            this_eval_vel = []
            for i, (gtdf, preddf) in enumerate(xv_dataframes, 1):
                txt_logger.info(f"[{i}/{len_xv} (xv set)]: {(thresh, shift)}")
                prf1, prf1_v = threshold_eval_single_file(
                    gtdf, preddf, SECS_PER_FRAME, key_beg,
                    thresh=thresh, shift_preds=shift,
                    tol_secs=CONF.TOLERANCE_SECS, tol_vel=CONF.TOLERANCE_VEL)
                this_eval.append(prf1)
                this_eval_vel.append(prf1_v)
            xv_gridsearch[(thresh, shift)] = this_eval
            xv_gridsearch_vel[(thresh, shift)] = this_eval_vel
    xv_summary = {k: np.mean(v, axis=0) for k, v in xv_gridsearch.items()}
    xv_summary_vel = {k: np.mean(v, axis=0)
                      for k, v in xv_gridsearch_vel.items()}
    ((best_t, best_s), (best_p, best_r, best_f1)) = max(
        xv_summary.items(), key=lambda elt: elt[1][2])
    #
    xv_summary_df = pd.DataFrame(
        ((t, s, p, r, f1) for ((t, s), (p, r, f1)) in xv_summary.items()),
        columns=["threshold", "shift", "P", "R", "F1"])

    xv_summary_df_vel = pd.DataFrame(
        ((t, s, p, r, f1) for ((t, s), (p, r, f1)) in xv_summary_vel.items()),
        columns=["threshold", "shift", "P", "R", "F1"])

    txt_logger.warning("XV HYPERPARAMETER SEARCH:")
    txt_logger.warning("Summary (without velocity):\n" + str(xv_summary_df))
    txt_logger.warning("Summary (with velocity):\n" + str(xv_summary_df_vel))

    ###############
    # TEST
    ###############
    test_results = []
    test_results_vel = []
    len_test = len(maestro_test)
    for i, (mel, roll, md) in enumerate(maestro_test, 1):
        txt_logger.info(f"[{i}/{len_test} (test set)] {md}")
        with torch.no_grad():
            tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
            onset_pred, vel_pred = strided_inference(
                model_inference, tmel, CHUNK_SIZE, CHUNK_OVERLAP)
            del tmel
            pred_df = decoder(
                onset_pred, vel_pred, pthresh=min(CONF.SEARCH_THRESHOLDS))
            gt_df = test_gts(md)[0]
        prf1, prf1_v = threshold_eval_single_file(
            gt_df, pred_df, SECS_PER_FRAME, key_beg,
            thresh=best_t, shift_preds=best_s,
            tol_secs=CONF.TOLERANCE_SECS, tol_vel=CONF.TOLERANCE_VEL)
        test_results.append((md[0], *prf1))
        test_results_vel.append((md[0], *prf1_v))
    #
    test_results_df = pd.DataFrame(
        test_results, columns=["Filename", "P", "R", "F1"])
    averages = [f"AVERAGES (t={best_t}, s={best_s})",
                *test_results_df.iloc[:, 1:].mean().tolist()]
    test_results_df.loc[len(test_results_df)] = averages
    #
    test_results_df_vel = pd.DataFrame(
        test_results_vel, columns=["Filename", "P", "R", "F1"])
    averages_vel = [f"AVERAGES (t={best_t}, s={best_s})",
                    *test_results_df_vel.iloc[:, 1:].mean().tolist()]
    test_results_df_vel.loc[len(test_results_df_vel)] = averages_vel
    #
    txt_logger.warning("TEST RESULTS WITH BEST XV HYPERPARS " +
                       f"(MAESTROv{CONF.MAESTRO_VERSION}, " +
                       f"{CONF.SNAPSHOT_INPATH})\n")
    txt_logger.warning(
        "ONSETS:\n" + str(test_results_df))
    txt_logger.warning(
        "ONSETS+VELOCITIES:\n" + str(test_results_df_vel))
