#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module loads the MAPS filenames and corresponding WAV+MIDI files,
computes the log-mel spectrograms from WAV and piano rolls from MIDI, and
stores them into 2 separate HDF5 files as consecutive matrices to be
efficiently queried for DL training and evaluation.

The roll file is a vertical stack of [onsets; frames; pedals] having a
dimensionality equal to 3+2*num_midi_notes.
It also stores the metadata and relevant parameters in the filenames, so data
can be fully traced to its origins.
"""


import os
# For omegaconf
from dataclasses import dataclass
#
from omegaconf import OmegaConf
import torch
import numpy as np
# import matplotlib.pyplot as plt
#
from ov_piano import HDF5PathManager
from ov_piano.utils import IncrementalHDF5
from ov_piano.utils import TorchWavToLogmel, torch_load_resample_audio
from ov_piano.data.maps import MetaMAPS
from ov_piano.data.midi import MidiToPianoRoll


# ##############################################################################
# # GLOBALS
# ##############################################################################
@dataclass
class ConfDef:
    """
    :cvar str MAPS_INPATH: path to the MAPS root directory
    :cvar str OUTPUT_DIR: Where to store HDF5 results. Created if non-existing.

    :cvar int TARGET_SR: Regardless of the original samplerate of the WAV
      files, they will be loaded and converted to this samplerate
    :cvar int STFT_WINSIZE: Window size for the STFT (normally a power of 2)
    :cvar int STFT_HOPSIZE: Hop size for the STFT, determines time resolution
    :cvar int MELBINS: Number of filters in the mel filterbank
    :cvar int MEL_FMIN: Lowest freq for the mel filterbank
    :cvar int MEL_FMAX: Highest freq for the mel filterbank

    :cvar bool MIDI_SUS_EXTEND: If true, MIDI notes in piano roll will be
      extended whenever the sustain pedal is pressed.

    :cvar int HDF5_CHUNKLEN_SECONDS: This parameter affects speed of the
      HDF5 read/write operations. Should be close to the chunk length used in
      training (ideally a bit larger), but it isn't crucial to tune.
    :cvar str DEVICE: For the PyTorch operations. Can be ``cpu`` or ``cuda``
      if a GPU is present, but CPU seems to be faster anyway.
    :cvar bool IGNORE_MEL: whether to compute only the piano rolls.
    """
    MAPS_INPATH: str = os.path.join("datasets", "MAPS")
    OUTPUT_DIR: str = "datasets"
    #
    TARGET_SR: int = 16_000
    STFT_WINSIZE: int = 2048
    STFT_HOPSIZE: int = 384
    MELBINS: int = 229
    MEL_FMIN: int = 50
    MEL_FMAX: int = 8_000
    #
    MIDI_SUS_EXTEND: bool = True
    #
    HDF5_CHUNKLEN_SECONDS: float = 8.0
    DEVICE: str = "cpu"
    IGNORE_MEL: bool = False


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n\n")

    # derivative globals
    HDF5_CHUNKLEN = round(CONF.HDF5_CHUNKLEN_SECONDS /
                          (CONF.STFT_HOPSIZE / CONF.TARGET_SR))
    ROLL_HEIGHT = 3 + MidiToPianoRoll.NUM_MIDI_VALUES * 2
    MIDI_QUANT_SECS = CONF.STFT_HOPSIZE / CONF.TARGET_SR

    # output paths
    os.makedirs(CONF.OUTPUT_DIR, exist_ok=True)
    if not CONF.IGNORE_MEL:
        HDF5_MEL_OUTPATH = os.path.join(
            CONF.OUTPUT_DIR,
            HDF5PathManager.get_mel_hdf5_basename(
                f"MAPS", CONF.TARGET_SR, CONF.STFT_WINSIZE, CONF.STFT_HOPSIZE,
                CONF.MELBINS, CONF.MEL_FMIN, CONF.MEL_FMAX))
    HDF5_ROLL_OUTPATH = os.path.join(
        CONF.OUTPUT_DIR,
        HDF5PathManager.get_roll_hdf5_basename(
            f"MAPS", MIDI_QUANT_SECS,
            MidiToPianoRoll.NUM_MIDI_VALUES, CONF.MIDI_SUS_EXTEND))

    all_maps = MetaMAPS(CONF.MAPS_INPATH,
                        include_instr={"StbgTGd2", "AkPnBsdf", "AkPnBcht",
                                       "AkPnCGdD", "AkPnStgb", "SptkBGAm",
                                       "SptkBGCl", "ENSTDkAm", "ENSTDkCl"},
                        include_cat={"ISOL", "MUS", "RAND", "UCHO"},
                        handle_redundant_mus=0)
    if not CONF.IGNORE_MEL:
        # functor to create logmels from wavs
        logmel_fn = TorchWavToLogmel(
            CONF.TARGET_SR, CONF.STFT_WINSIZE,
            CONF.STFT_HOPSIZE, CONF.MELBINS,
            CONF.MEL_FMIN, CONF.MEL_FMAX).to(CONF.DEVICE)
    # functor to create piano rolls from MIDI
    pianoroll_fn = MidiToPianoRoll()

    # corresponding HDF5 file handles
    if not CONF.IGNORE_MEL:
        h5mel = IncrementalHDF5(
            HDF5_MEL_OUTPATH, CONF.MELBINS, dtype=np.float32,
            compression="lzf", data_chunk_length=HDF5_CHUNKLEN,
            metadata_chunk_length=HDF5_CHUNKLEN, err_if_exists=True)
    h5roll = IncrementalHDF5(
        HDF5_ROLL_OUTPATH, ROLL_HEIGHT, dtype=np.float32,
        compression="lzf", data_chunk_length=HDF5_CHUNKLEN,
        metadata_chunk_length=HDF5_CHUNKLEN, err_if_exists=True)

    print("Computing features...")
    if not CONF.IGNORE_MEL:
        print("Logmels stored into", HDF5_MEL_OUTPATH)
    print("Piano rolls stored into", HDF5_ROLL_OUTPATH)
    loop_length = len(all_maps.data)

    for i, (abspath, instr, cat) in enumerate(all_maps.data, 1):
        basepath = os.path.basename(abspath)
        metadata = str((basepath, instr, cat))

        if not CONF.IGNORE_MEL:
            # compute logmel and add to corresponding HDF5
            with torch.no_grad():
                wave = torch_load_resample_audio(
                    abspath + ".wav", CONF.TARGET_SR, mono=True,
                    normalize_wav=True, device=CONF.DEVICE)
                logmel = logmel_fn(wave).to("cpu").numpy()
            h5mel.append(logmel, metadata)

        # compute piano roll and add to corresponding HDF5
        midipath = abspath + ".mid"
        (onset_roll, offset_roll, frame_roll,
         sus_roll, soft_roll, ten_roll, key_events) = pianoroll_fn(
             midipath, quant_secs=MIDI_QUANT_SECS,
             extend_offsets_sus=CONF.MIDI_SUS_EXTEND)
        roll = np.vstack([onset_roll, frame_roll,
                          sus_roll, soft_roll, ten_roll])
        #
        _, len_roll = roll.shape
        if not CONF.IGNORE_MEL:
            _, len_logmel = logmel.shape
            assert len_logmel >= len_roll, \
                "Wav isn't expected to be shorter than MIDI!"
            if len_logmel > len_roll:
                # print("WARNING: wav is longer than MIDI.",
                #       "Padding MIDI end with zeros")
                roll = np.pad(roll, ((0, 0), (0, len_logmel - len_roll)))
            assert len_logmel == roll.shape[1], \
                "Logmel and roll have different length?"
        # plt.clf(); plt.imshow(logmel[::-1]); plt.show()
        # plt.clf(); plt.imshow(onset_roll[::-1]); plt.show()
        #
        h5roll.append(roll, metadata)
        #
        if (i % 100) == 0:
            print(f"[{i}/{loop_length}]", abspath)
    if not CONF.IGNORE_MEL:
        h5mel.close()
    h5roll.close()
    print("Done!")
