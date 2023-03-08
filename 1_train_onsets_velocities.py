#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module instantiates, trains and cross-validates a (potentially pre-loaded)
deep learning model for piano key onset+velocity detection on the MAESTRO
dataset.

It is structured in 3 parts:
1. Fetching and preparing global parameters
2. Instantiating required parts (dataloader, model, decoder, optimizer...)
3. Training loop, featuring an inner cross-validation loop
"""


import os
import random
# For omegaconf
from dataclasses import dataclass
from typing import Optional, List
#
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
#
from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import ModelSaver, load_model, breakpoint_json, set_seed
from ov_piano.logging import JsonColorLogger
from ov_piano.data.maestro import MetaMAESTROv1, MetaMAESTROv2, MetaMAESTROv3
from ov_piano.data.maestro import MelMaestro, MelMaestroChunks
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import MaskedBCEWithLogitsLoss
from ov_piano.optimizers import AdamWR
from ov_piano.inference import strided_inference, OnsetVelocityNmsDecoder
from ov_piano.eval import GtLoaderMaestro, eval_note_events

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

    :cvar TRAIN_BS: Train batch size. Reduce if insufficient memory.
    :cvar TRAIN_BATCH_SECS: Time duration of the chunks used for training,
      reduce if insufficient memory.

    :cvar OPTIMIZER: Supported are SGDR and AdamWR (default)
    :cvar LR_MAX: Initial learning rate for the optimizer
    :cvar LR_PERIOD: Number of steps per LR cycle for the optimizer
    :cvar LR_DECAY: Each LR cycle, the max and min LR are multiplied by this
    :cvar LR_SLOWDOWN: Each LR cycle, the duration is multiplied by this
    :cvar MOMENTUM: Gradient momentum for the optimizer
    :cvar WEIGHT_DECAY: L2 regularization factor for the optimizer

    :cvar BATCH_NORM: Momentum for the (batch, spectral) normalization layers
    :cvar DROPOUT: Probability of dropping a weight
    :cvar LEAKY_RELU_SLOPE: Slope for the negative part of leaky ReLU

    :cvar ONSET_POSITIVES_WEIGHT: The loss function for the piano rolls will
      multiply the positive examples by this constant (used to compensate the
      fact that onsets are less than 50% of frames)
    :cvar VEL_LOSS_LAMBDA: total loss is
      ``onset_loss + LAMBDA * velocity loss`` for this lambda.
    :cvar TRAINABLE_ONSETS: If false, only the velocity-specific parameters are
      being trained. Useful e.g. for fine-tuning a model that already performs
      good onset detection.

    :cvar DECODER_GAUSS_STD: The decoder on top of the DNN predictions performs
      a Gaussian time-convolution to smoothen detections. This is the standard
      deviation, in time-frames.
    :cvar DECODER_GAUSS_KSIZE: The window size, in time-frames, for the
      smoothening Gaussian time-convolution.

    :cvar XV_TOLERANCE_SECS: The maximum absolute error between onset pred
      and ground truth, in seconds, to consider the prediction correct. Used
      during cross-vlaidation
    :cvar XV_TOLERANCE_VEL: The maximum absolute error between velocity pred
      and ground truth, in ratio between 0 and 1, to consider the prediction
      correct. To better understand this ratio, see the official documentation
      for ``mir_eval.transcription_velocity``. Used during cross-validation.

    :cvar XV_CHUNK_SIZE: For cross-validation, full files are processed, which
      may be too large for memory and have to be processed in strided chunks.
      This is the chunk size in seconds, it doesn't affect performance as long
      as it is large enough.
    :cvar XV_CHUNK_OVERLAP: See ``XV_CHUNK_SIZE``. This is the overlap among
      consecutive chunks. It doesn't affect performance as long as it is large
      enough to avoid boundary artifacts.
    :cvar XV_THRESHOLDS: List of thresholds to perform cross-validation on.
      Note that XV will be performed once per threshold, so the more, the
      slower training, but also better chances of assessing performance right.
    """
    # general
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED: Optional[int] = None
    # I/O
    OUTPUT_DIR: str = "out"
    MAESTRO_PATH: str = os.path.join("datasets", "maestro", "maestro-v3.0.0")
    MAESTRO_VERSION: int = 3
    HDF5_MEL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5")
    HDF5_ROLL_PATH: str = os.path.join(
        "datasets",
        "MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5")
    SNAPSHOT_INPATH: Optional[str] = None
    # data loader
    TRAIN_BS: int = 40
    TRAIN_BATCH_SECS: float = 5.0
    DATALOADER_WORKERS: int = 8
    # model/optimizer
    CONV1X1: List[int] = (200, 200)
    # optimizer
    LR_MAX: float = 0.008
    LR_WARMUP: float = 0.5
    LR_PERIOD: int = 1000
    LR_DECAY: float = 0.975
    LR_SLOWDOWN: float = 1.0
    MOMENTUM: float = 0.95
    WEIGHT_DECAY: float = 0.0003
    BATCH_NORM: float = 0.95
    DROPOUT: float = 0.15
    LEAKY_RELU_SLOPE: Optional[float] = 0.1
    # loss
    ONSET_POSITIVES_WEIGHT: float = 8.0
    VEL_LOSS_LAMBDA: float = 10.0
    TRAINABLE_ONSETS: bool = True
    # decoder
    DECODER_GAUSS_STD: float = 1
    DECODER_GAUSS_KSIZE: int = 11
    # training loop
    NUM_EPOCHS: int = 10
    TRAIN_LOG_EVERY: int = 10
    XV_EVERY: int = 1000
    XV_CHUNK_SIZE: float = 600
    XV_CHUNK_OVERLAP: float = 2.5
    XV_THRESHOLDS: List[float] = (0.7, 0.725, 0.75, 0.775, 0.8)
    # xv tolerances
    XV_TOLERANCE_SECS: float = 0.05
    XV_TOLERANCE_VEL: float = 0.1


# ##############################################################################
# # MAIN LOOP INITIALIZATION
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)

    # if no seed is given, take a random one
    if CONF.RANDOM_SEED is None:
        CONF.RANDOM_SEED = random.randint(0, 1e7)
    set_seed(CONF.RANDOM_SEED)

    # derivative globals + parse HDF5 filenames and ensure they are consistent
    (DATASET_NAME, SAMPLERATE, WINSIZE, HOPSIZE,
     MELBINS, FMIN, FMAX) = HDF5PathManager.parse_mel_hdf5_basename(
        os.path.basename(CONF.HDF5_MEL_PATH))
    roll_params = HDF5PathManager.parse_roll_hdf5_basename(
        os.path.basename(CONF.HDF5_ROLL_PATH))
    SECS_PER_FRAME = HOPSIZE / SAMPLERATE
    CHUNK_LENGTH = round(CONF.TRAIN_BATCH_SECS / SECS_PER_FRAME)
    CHUNK_STRIDE = round(CHUNK_LENGTH / CONF.TRAIN_BATCH_SECS)
    #
    assert DATASET_NAME == roll_params[0], "Inconsistent HDF5 datasets?"
    assert SECS_PER_FRAME == roll_params[1], "Inconsistent roll quantization?"
    #
    XV_CHUNK_SIZE = round(CONF.XV_CHUNK_SIZE / SECS_PER_FRAME)
    XV_CHUNK_OVERLAP = round(CONF.XV_CHUNK_OVERLAP / SECS_PER_FRAME)
    #
    METAMAESTRO_CLASS = {1: MetaMAESTROv1, 2: MetaMAESTROv2,
                         3: MetaMAESTROv3}[CONF.MAESTRO_VERSION]
    # output dirs
    MODEL_SNAPSHOT_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "model_snapshots")
    TXT_LOG_OUTDIR = os.path.join(CONF.OUTPUT_DIR, "txt_logs")
    os.makedirs(MODEL_SNAPSHOT_OUTDIR, exist_ok=True)
    os.makedirs(TXT_LOG_OUTDIR, exist_ok=True)

    txt_logger = JsonColorLogger(
        f"[{os.path.basename(__file__)}]", TXT_LOG_OUTDIR)
    txt_logger.loj("PARAMETERS", OmegaConf.to_container(CONF))

    # datasets and dataloaders
    metamaestro_train = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["train"], years=METAMAESTRO_CLASS.ALL_YEARS)
    maestro_train = MelMaestroChunks(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        CHUNK_LENGTH, CHUNK_STRIDE,
        *(x[0] for x in metamaestro_train.data),
        with_oob=True, logmel_oob_pad_val="min",
        as_torch_tensors=False)
    train_dl = torch.utils.data.DataLoader(
        maestro_train, batch_size=CONF.TRAIN_BS, shuffle=True,
        num_workers=CONF.DATALOADER_WORKERS,
        pin_memory=False, persistent_workers=False)
    #
    metamaestro_xv = METAMAESTRO_CLASS(
        CONF.MAESTRO_PATH, splits=["validation"],
        years=METAMAESTRO_CLASS.ALL_YEARS)
    # shorten xv set to speed up cross validation times
    txt_logger.loj("WARNING",
                   "shortening xv split for faster crossvalidation!")
    metamaestro_xv.data = metamaestro_xv.data[::5]
    #
    maestro_xv = MelMaestro(
        CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
        *(x[0] for x in metamaestro_xv.data),
        as_torch_tensors=False)
    xv_gt_loader = GtLoaderMaestro(maestro_xv, metamaestro_xv)

    # data-specific constants
    batches_per_epoch = len(train_dl)
    num_mels = maestro_train[0][0].shape[0]
    key_beg, key_end = PIANO_MIDI_RANGE
    num_piano_keys = key_end - key_beg

    # DNN (instantiation+serialization)
    model = OnsetsAndVelocities(
        in_chans=2,  # X and time_derivative(X)
        in_height=num_mels, out_height=num_piano_keys,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=CONF.BATCH_NORM,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=CONF.DROPOUT).to(CONF.DEVICE)
    if CONF.SNAPSHOT_INPATH is not None:
        load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=False)
    model_saver = ModelSaver(
        model, MODEL_SNAPSHOT_OUTDIR,
        log_fn=lambda msg: txt_logger.loj("SAVED_MODEL", msg))

    # decoder
    decoder = OnsetVelocityNmsDecoder(
        num_piano_keys, nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1, vel_pad_right=1)  # this module stays on cpu

    # loss
    ons_pos_weights = torch.FloatTensor(
        [CONF.ONSET_POSITIVES_WEIGHT]).to(CONF.DEVICE)
    ons_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=ons_pos_weights)
    vel_loss_fn = MaskedBCEWithLogitsLoss()

    # optimizer
    trainable_params = model.parameters() if CONF.TRAINABLE_ONSETS else \
        model.velocity_stage.parameters()

    opt_hpars = {
        "lr_max": CONF.LR_MAX, "lr": CONF.LR_MAX,
        "lr_period": CONF.LR_PERIOD, "lr_decay": CONF.LR_DECAY,
        "lr_slowdown": CONF.LR_SLOWDOWN, "cycle_end_hook_fn": model_saver,
        "cycle_warmup": CONF.LR_WARMUP, "weight_decay": CONF.WEIGHT_DECAY,
        "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}
    opt = AdamWR(trainable_params, **opt_hpars)


    # ##########################################################################
    # # XV HELPERS
    # ##########################################################################
    def model_inference(x):
        """
        Convenience wrapper around the DNN to ensure output and input sequences
        have same length.
        """
        probs, vels = model(x)
        probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
        vels = F.pad(torch.sigmoid(vels), (1, 0))
        return probs, vels


    def xv_file(mel, md, thresholds=[0.5], verbose=False):
        """
        Convenience function to perform cross-validation on a single file:
        1. Loads ground-truth event sequence from given MIDI
        2. Performs strided inference on given mel, and extracts predicted
          event sequence
        3. Computes XV metrics for every given threshold, once for onsets only
          and once for onsets+velocities
        4. Returns ``(o_results, ov_results)`` as lists with one element per
          threshold
        """
        # gather ground truth
        df_gt = xv_gt_loader(md)[0]
        # gather onset predictions
        tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
        onset_pred, vel_pred = strided_inference(
            model_inference, tmel, XV_CHUNK_SIZE, XV_CHUNK_OVERLAP)
        del tmel
        df_pred = decoder(onset_pred, vel_pred, pthresh=min(thresholds))
        # evaluate for all thresholds, without taking velocity into account
        results = []
        for t in thresholds:
            # prob must be above threshold, unless velocity score high enough
            df_pred_t = df_pred[df_pred["prob"] >= t]
            # evaluate
            prec, rec, f1 = eval_note_events(
                df_gt["onset"].to_numpy(),
                df_gt["key"].to_numpy(),
                df_pred_t["t_idx"].to_numpy(),
                df_pred_t["key"].to_numpy(),
                #
                tol_secs=CONF.XV_TOLERANCE_SECS, pitch_tolerance=0.1,
                pred_key_shift=key_beg,
                pred_onset_mul=SECS_PER_FRAME,
                pred_shift=0)
            results.append((md[0], prec, rec, f1))
            if verbose:
                txt_logger.loj(
                    "XV_ONSET", {"threshold": t, "P": prec, "R": rec, "F1": f1})
        # evaluate for all thresholds, taking velocity into account
        results_vel = []
        for t in thresholds:
            # threshold predictions
            df_pred_t = df_pred[df_pred["prob"] >= t]
            # evaluate
            prec, rec, f1 = eval_note_events(
                df_gt["onset"].to_numpy(),
                df_gt["key"].to_numpy(),
                df_pred_t["t_idx"].to_numpy(),
                df_pred_t["key"].to_numpy(),
                #
                gt_vels=df_gt["vel"].to_numpy(),
                pred_vels=df_pred_t["vel"].to_numpy(),
                #
                tol_secs=CONF.XV_TOLERANCE_SECS, pitch_tolerance=0.1,
                velocity_tolerance=CONF.XV_TOLERANCE_VEL,
                pred_key_shift=key_beg,
                pred_onset_mul=SECS_PER_FRAME,
                pred_shift=0)
            results_vel.append((md[0], prec, rec, f1))
            if verbose:
                txt_logger.loj(
                    "XV_ONSET_VEL",
                    {"threshold": t, "P": prec, "R": rec, "F1": f1})
        #
        return results, results_vel


    # ##########################################################################
    # # TRAINING LOOP
    # ##########################################################################
    txt_logger.loj("MODEL_INFO", {"class": model.__class__.__name__})
    global_step = 1
    onsets_beg, onsets_end = maestro_train.ONSETS_RANGE
    frames_beg, frames_end = maestro_train.FRAMES_RANGE
    for epoch in range(1, CONF.NUM_EPOCHS + 1):
        for i, (logmels, rolls, metas) in enumerate(train_dl):
            # ##################################################################
            # # CROSS VALIDATION
            # ##################################################################
            if (global_step % CONF.XV_EVERY) == 0:
                model.eval()
                #
                torch.cuda.empty_cache()
                with torch.no_grad():
                    xv_results = []
                    xv_results_vel = []
                    len_xv = len(maestro_xv)
                    for ii, (mel, roll, md) in enumerate(maestro_xv, 1):
                        txt_logger.loj(
                            "XV_PROCESSING",
                            {"idx": ii, "len_xv": len_xv, "filename": md[0]})
                        xv_result, xv_result_vel = xv_file(
                            mel, md, CONF.XV_THRESHOLDS)
                        xv_results.append(xv_result)
                        xv_results_vel.append(xv_result_vel)
                # compare non-vel results and report best
                xv_dfs = [(t, pd.DataFrame(
                    x, columns=["filename", "P", "R", "F1"]))
                          for t, x in zip(CONF.XV_THRESHOLDS,
                                          zip(*xv_results))]
                f1_avgs = []
                for t, df in xv_dfs:
                    averages = [f"AVERAGES (t={t})",
                                *df.iloc[:, 1:].mean().tolist()]
                    df.loc[len(df)] = averages
                    f1_avgs.append(averages[-1])
                best_f1_idx = np.argmax(f1_avgs)
                best_f1 = f1_avgs[best_f1_idx]
                # compare vel results and report best
                xv_dfs_vel = [
                    (t, pd.DataFrame(x, columns=["filename", "P", "R", "F1"]))
                    for t, x in zip(CONF.XV_THRESHOLDS, zip(*xv_results_vel))]
                f1_avgs_vel = []
                for t, df in xv_dfs_vel:
                    averages = [f"AVERAGES (t={t})",
                                *df.iloc[:, 1:].mean().tolist()]
                    df.loc[len(df)] = averages
                    f1_avgs_vel.append(averages[-1])
                best_f1_idx_vel = np.argmax(f1_avgs_vel)
                best_f1_vel = f1_avgs_vel[best_f1_idx_vel]
                # report results, save model, resume training
                txt_logger.loj("XV_BEST_ONSET", str(xv_dfs[best_f1_idx][1]))
                txt_logger.loj("XV_BEST_ONSET_VEL",
                               str(xv_dfs_vel[best_f1_idx_vel][1]))
                txt_logger.loj("XV_SUMMARY", {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_f1_o_thresh": CONF.XV_THRESHOLDS[
                        int(best_f1_idx)],
                    "best_f1_o": best_f1,
                    "best_f1_v_thresh": CONF.XV_THRESHOLDS[
                        int(best_f1_idx_vel)],
                    "best_f1_v": best_f1_vel})
                model_saver(
                    f"step={global_step}_f1={best_f1:.4f}__{best_f1_vel:.4f}")
                #
                torch.cuda.empty_cache()
                model.train()

            # ##################################################################
            # # TRAINING
            # ##################################################################
            with torch.no_grad():
                logmels = logmels.to(CONF.DEVICE)
                rolls = rolls[:, :, 1:].to(CONF.DEVICE)
                onsets = rolls[:, onsets_beg:onsets_end][:, key_beg:key_end]
                # frames = rolls[:, frames_beg:frames_end][:, key_beg:key_end]

                ################################################################
                double_onsets = onsets.clone()
                torch.maximum(onsets[..., :-1], onsets[..., 1:],
                              out=double_onsets[..., 1:])
                triple_onsets = double_onsets.clone()
                torch.maximum(double_onsets[..., :-1], double_onsets[..., 1:],
                              out=triple_onsets[..., 1:])
                #
                onsets_clip = triple_onsets.clip(0, 1)
                onsets_norm = triple_onsets / 127.0
                del onsets
                del double_onsets
                del triple_onsets
                # idx = 0; plt.clf(); plt.imshow(logmels[idx].cpu().numpy()[::-1]); plt.show()
                # idx = 0; plt.clf(); plt.imshow(onsets[idx].cpu().numpy()[::-1]); plt.show()
                # idx = 0; plt.clf(); plt.imshow(double_onsets[idx].cpu().numpy()[::-1]); plt.show()

                ##################################################################

            # zero the parameter gradients
            opt.zero_grad()
            onset_stages, velocities = model(logmels, CONF.TRAINABLE_ONSETS)

            vel_loss = CONF.VEL_LOSS_LAMBDA * vel_loss_fn(
                velocities, onsets_norm, mask=onsets_clip)
            loss = vel_loss
            if CONF.TRAINABLE_ONSETS:
                ons_loss = sum(ons_loss_fn(ons, onsets_clip)
                               for ons in onset_stages) / len(onset_stages)
                loss += ons_loss
            if breakpoint_json("breakpoint.json", global_step):
                onsets = rolls[:, onsets_beg:onsets_end][:, key_beg:key_end]
                breakpoint()
                # idx=0; vel_t=0.1; ons=torch.sigmoid(onset_stages[-1][idx]); plt.clf(); plt.imshow(torch.cat([onsets_clip[idx], onsets_norm[idx], ons, torch.sigmoid(velocities[idx]) * (ons > vel_t)], dim=0).detach().cpu().numpy()[::-1, :1000]); plt.show()
                # idx=0; vel_t=0.1; ons=torch.sigmoid(onset_stages[-1][idx]); plt.clf(); plt.imshow(torch.cat([onsets_norm[idx], torch.sigmoid(velocities[idx]) * (ons > vel_t)], dim=0).detach().cpu().exp().numpy()[::-1, :1000]); plt.show()
                # idx=0; plt.clf(); plt.imshow(torch.cat([onsets_norm[idx], torch.sigmoid(velocities[idx])], dim=0).detach().cpu().exp().numpy()[::-1, :1000]); plt.show()
            #
            loss.backward()
            opt.step()
            #
            if (global_step % CONF.TRAIN_LOG_EVERY) == 0:
                losses = [vel_loss.item()]
                if CONF.TRAINABLE_ONSETS:
                    losses.append(ons_loss.item())
                txt_logger.loj("TRAIN", {"epoch": epoch,
                                         "step": i,
                                         "global_step": global_step,
                                         "batches_per_epoch": batches_per_epoch,
                                         "losses": losses,
                                         "LR": opt.get_lr()})
            #
            global_step += 1
