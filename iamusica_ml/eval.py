#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module hosts re-usable evaluation code for automated piano transcription,
including:
* Convenience classes to load MIDI files from datasets, and convert them to
  Pandas dataframes storing events with onset, offset... this helps in two
  ways: first, multiple things can be processed in parallel; second, the full
  dataset is processed at once, avoiding redundancies and speeding up training.
* Event-base evaluation that makes use of official ``mir_eval`` implementations
  to ensure rigor, reproducibility and compatibility with previous literature.
"""


import os
#
import numpy as np
from concurrent.futures import ProcessPoolExecutor
#
from .data.key_model import KeyboardStateMachine
from .data.midi import SingletrackMidiParser, MaestroMidiParser
from .data.midi import MidiToPianoRoll
from mir_eval.transcription import precision_recall_f1_overlap as prf1o
from mir_eval.transcription_velocity import precision_recall_f1_overlap \
    as prf1o_v


# ##############################################################################
# # GROUND TRUTH CONVENIENCE LOADERS
# ##############################################################################
class GtLoaderMaps:
    """
    Auxiliary class to the main eval class (in order to avoid multiple forward
    passes during evaluation and to leverage optimal F1 threshold from roll
    evaluation).
    During construction, it parses and stores the MIDI files (also to avoid
    doing this multiple times). Then, it offers a series of convenience methods
    that can be used by the main eval class.
    """
    PARSER = SingletrackMidiParser
    MIDI_EXT = ".mid"
    MIN_NOTE_DUR = 0.001  # in seconds

    @classmethod
    def get_metadata_path(cls, data_md, meta_dataset):
        """
        :param dataset_md: along with the logmels and rolls, datasets provide
          metadata. This method reconstructs the complete path from this
          given metadata, such that it can be found in the meta_dataset.
        """
        basename, instr, cat = data_md
        path = os.path.join(meta_dataset.rootpath, instr, cat, basename)
        return path + cls.MIDI_EXT

    @classmethod
    def get_midi_eventdata(cls, abspath):
        """
        """
        # load and check midi
        mid = cls.PARSER.load_midi(abspath)
        msgs, meta_msgs = cls.PARSER.parse_midi(mid)
        MidiToPianoRoll._check_midi(msgs, meta_msgs)
        # convert midi to events with onset and offset
        (key_events, sus_states, ten_states, soft_states,
         largest_ts) = cls.PARSER.ksm_parse_midi_messages(
             msgs, KeyboardStateMachine(
                 MidiToPianoRoll.SUS_PEDAL_THRESH,
                 MidiToPianoRoll.TEN_PEDAL_THRESH,
                 ignore_redundant_keypress=True,
                 ignore_redundant_keylift=True))
        #
        return (key_events, sus_states, ten_states, soft_states, largest_ts)

    def __init__(self, dataset, meta_dataset):
        """
        """
        self.dataset, self.meta_dataset = dataset, meta_dataset
        self.midi_abspaths = [self.get_metadata_path(md, meta_dataset)
                              for _, _, md in dataset]
        with ProcessPoolExecutor() as executor:
            midi_eventdata = executor.map(
                self.get_midi_eventdata, self.midi_abspaths)
        self.midi_eventdata = {ap: data for ap, data
                               in zip(self.midi_abspaths, midi_eventdata)}
        # all onset-offset intervals must be >0, so add epsilon if needed:
        for (key_evts, _, _, _, _) in self.midi_eventdata.values():
            diffs = key_evts["offset"] - key_evts["onset"]
            key_evts.loc[diffs == 0, "offset"] += self.MIN_NOTE_DUR

    def __call__(self, data_md):
        """
        :param data_md: The metadata output of the dataset. It is also the
          input to ``get_metadata_path``.
        """
        md_path = self.get_metadata_path(data_md, self.meta_dataset)
        result = self.midi_eventdata[md_path]
        return result


class GtLoaderMaestro(GtLoaderMaps):
    """
    Extension of ``GtLoaderMaps`` for MAESTRO.
    """
    PARSER = MaestroMidiParser
    MIDI_EXT = ".midi"

    @classmethod
    def get_metadata_path(cls, data_md, meta_dataset):
        """
        :param dataset_md: along with the logmels and rolls, datasets provide
          metadata. This method reconstructs the complete path from this
          given metadata, such that it can be found in the meta_dataset.
        """
        basename, year, _, _, _, _ = data_md
        path = os.path.join(meta_dataset.rootpath, str(year), basename)
        return path + cls.MIDI_EXT


# ##############################################################################
# # EVENT-BASED EVALUATION
# ##############################################################################
def eval_note_events(gt_onsets, gt_keys,
                     pred_onsets, pred_keys,
                     gt_vels=None, pred_vels=None,
                     tol_secs=0.05, pitch_tolerance=0.1,
                     velocity_tolerance=0.1,
                     pred_key_shift=0, pred_onset_mul=1.0,
                     pred_shift=0):
    """
    Given sets of ground truth and predicted note events (with their onsets,
    keys, and optionally velocities), as well as the potential shift of onset
    keys and scale+shift of onset times, computes+returns the precision, recall
    and F1 score. Predictions are considered correct if they are within given
    onset time and pitch tolerances (and also velocity if given). Check the
    ``precision_recall_f1_overlap`` functions from ``mir_eval.transcription``
    and ``mir_eval.transcription_velocity`` for more details.

    :param gt_onsets: Numpy 1D array with onset timestamps in seconds.
    :param gt_keys: Numpy 1D array with same shape as gt_onsets designing the
      corresponding keys.
    :param pred_onsets: Predicted onsets, they can be of different length
      than the ground truth but needs to be a numpy array.
    :param pred_keys: see gt_keys
    :param gt_vels: Numpy 1D array with GT velocities (usually MIDI 0-127, but
      will be rescaled to 0-1 during evaluation).
    :param pred_vels: Numpy 1D array with predicted velocities. During
      evaluation, it will be scaled+shifted to the gt_vels, so scale and shift
      are not important.
    :param velocity_tolerance: Once ``pred_vels`` are scaled+shifted to best
      fit the GT, a prediction is considered true if within this tolerance.
    :param tol_secs: Tolerance for considering a true prediction, in seconds
    :param pred_onset_mul: Given pred_onsets will be multiplied by this
    :param pred_shift: **After** onsets are multiplied by ``pred_onset_mul``,
      they will be added this shift.
    """
    if len(pred_onsets) == 0:
        # if model didn't predict any onsets gather 0 for all metrics
        prec, rec, f1 = 0, 0, 0
    else:
        #
        if pred_key_shift != 0:
            pred_keys = pred_keys + pred_key_shift
        if pred_onset_mul != 1.0:
            pred_onsets = pred_onsets * pred_onset_mul
        if pred_shift != 0:
            pred_onsets = pred_onsets + pred_shift
        # mir_eval code needs offsets, even when ignored
        gt_offsets = gt_onsets + 1
        pred_offsets = pred_onsets + 1
        # eval predictions using the mir_eval lib
        if (gt_vels is not None) and (pred_vels is not None):
            prec, rec, f1, _ = prf1o_v(
                np.stack((gt_onsets, gt_offsets)).T, gt_keys, gt_vels,
                np.stack((pred_onsets, pred_offsets)).T, pred_keys, pred_vels,
                onset_tolerance=tol_secs, pitch_tolerance=pitch_tolerance,
                velocity_tolerance=velocity_tolerance,
                offset_ratio=None,  # ignore offsets
                offset_min_tolerance=tol_secs)
        else:
            prec, rec, f1, _ = prf1o(
                np.stack((gt_onsets, gt_offsets)).T, gt_keys,
                np.stack((pred_onsets, pred_offsets)).T, pred_keys,
                onset_tolerance=tol_secs, pitch_tolerance=pitch_tolerance,
                offset_ratio=None,  # ignore offsets
                offset_min_tolerance=tol_secs)
    #
    return prec, rec, f1


def threshold_eval_single_file(
        gt_df, pred_df, secs_per_frame, pred_key_offset,
        thresh=0.5, shift_preds=0, tol_secs=0.05, tol_vel=0.1):
    """
    Given a set of ground truths, predictions and tolerances, the imported
    function ``eval_note_events`` returns their precision, recall and F1 score
    for the given time/velocity tolerances.

    This wrapper function receives ground truths and predictions as Pandas
    dataframes, and runs ``eval_note_events`` twice; once for the onsets alone
    and once for onsets+velocities. Also, it thresholds the predictions first
    by their onset probability, and shifts+scales the onset time by a constant.

    See the evaluation script for an usage example.

    :returns: A tuple ``((p, r, f1), (p_v, r_v, f1_v))``, where the first
      triple contains the onsets-only evaluation, and the second
      onsets+velocities.
    """
    pred_df = pred_df[pred_df["prob"] >= thresh]
    pred_t = (pred_df["t_idx"].to_numpy() *
              float(secs_per_frame)) + shift_preds
    pred_k = pred_df["key"].to_numpy() + float(pred_key_offset)
    pred_v = pred_df["vel"].to_numpy()
    #
    gt_t = gt_df["onset"].to_numpy()
    gt_k = gt_df["key"].to_numpy()
    gt_v = gt_df["vel"].to_numpy()
    # without velocity
    prec, rec, f1 = eval_note_events(
        gt_t, gt_k, pred_t, pred_k,
        tol_secs=tol_secs, pitch_tolerance=0.1)
    # with velocity
    prec_v, rec_v, f1_v = eval_note_events(
        gt_t, gt_k, pred_t, pred_k,
        gt_vels=gt_v, pred_vels=pred_v,
        tol_secs=tol_secs, pitch_tolerance=0.1,
        velocity_tolerance=tol_vel)
    #
    return (prec, rec, f1), (prec_v, rec_v, f1_v)
