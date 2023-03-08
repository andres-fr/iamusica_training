#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module contains package-wide constants and conventions.
"""


from parse import parse


# ##############################################################################
# # GLOBALS
# ##############################################################################
# Tested on MAESTRO val and also consistent with
# https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
PIANO_MIDI_RANGE = (21, 109)  # 21 included, 109 excluded


# ##############################################################################
# # I/O
# ##############################################################################
class HDF5PathManager:
    """
    Class to map from HDF5 parameters into filenames, and back.
    """

    MEL_FSTRING = ("{dataset_name}_logmel_sr={samplerate}_" +
                   "stft={stft_winsize}w{stft_hopsize}h_" +
                   "mel={num_melbins}({mel_fmin}-{mel_fmax}).h5")
    ROLL_FSTRING = ("{dataset_name}_roll_quant={quant_secs}_" +
                    "midivals={num_midi_vals}_extendsus={extendsus}.h5")

    @classmethod
    def get_mel_hdf5_basename(cls, dataset_name, samplerate, stft_winsize,
                              stft_hopsize, num_melbins, mel_fmin, mel_fmax):
        """
        """
        return cls.MEL_FSTRING.format(
            dataset_name=dataset_name, samplerate=samplerate,
            stft_winsize=stft_winsize, stft_hopsize=stft_hopsize,
            num_melbins=num_melbins, mel_fmin=mel_fmin, mel_fmax=mel_fmax)

    @classmethod
    def get_roll_hdf5_basename(cls, dataset_name, quant_secs, num_midi_vals,
                               extendsus):
        """
        """
        return cls.ROLL_FSTRING.format(
            dataset_name=dataset_name, quant_secs=quant_secs,
            num_midi_vals=num_midi_vals, extendsus=extendsus)

    @classmethod
    def parse_mel_hdf5_basename(cls, basename):
        """
        """
        d = parse(cls.MEL_FSTRING, basename)
        return (d["dataset_name"], int(d["samplerate"]),
                int(d["stft_winsize"]), int(d["stft_hopsize"]),
                int(d["num_melbins"]),
                float(d["mel_fmin"]), float(d["mel_fmax"]))

    @classmethod
    def parse_roll_hdf5_basename(cls, basename):
        """
        """
        d = parse(cls.ROLL_FSTRING, basename)
        return (d["dataset_name"], float(d["quant_secs"]),
                int(d["num_midi_vals"]), bool(d["extendsus"]))
