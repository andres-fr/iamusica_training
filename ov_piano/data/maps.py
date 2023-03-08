#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Our dataloading mechanism for piano onset detection is structured in 3 tiers:
1. Meta-dataset: handles the filenames of a given dataset. This is important
  to handle relevant splits (e.g. by year, train/test...).
2. Dataset: Assuming we pre-processed the wav/midi dataset into logmel/roll,
  this class asserts logmel/roll consistency and provides full-file dataloading
  capabilities (useful to perform validation/test tasks)
3. Chunked dataset: For training, it is impractical and undesired to load a
  single, full file at a time. Rather, we would like to split files in smaller
  chunks and shuffle them. Chunked datasets extend datasets with this
  functionality
"""


import os
from collections import defaultdict
import random
from ast import literal_eval
#
import torch
import numpy as np
import h5py
#
from ..utils import IncrementalHDF5
from .midi import MidiToPianoRoll


# ##############################################################################
# #  META (paths etc)
# ##############################################################################
class MetaMAPS:
    """
    This class parses the filesystem tree for the MAPS dataset and, based on
    the given filters, stores a list of file paths, instruments and categories,
    as well as comprehensive information as given by the filesystem and MAPS
    paper.

    It can be used to manage MAPS files and to create custom dataloaders.
    """

    INSTRUMENTS = {
        "StbgTGd2": ("software", "Hybrid", "software default",
                     "The grand 2 (Steinberg)"),
        "AkPnBsdf": ("software", "Boesendorfer 290 Imperial", "church",
                     "Akoustik Piano (Native Instruments)"),
        "AkPnBcht": ("software", "Bechstein D 280", "concert hall",
                     "Akoustik Piano (Native Instruments)"),
        "AkPnCGdD": ("software", "Concert Grand D", "studio",
                     "Akoustik Piano (Native Instruments)"),
        "AkPnStgb": ("software", "Steingraeber 130 (upright)",
                     "jazz club", "Akoustik Piano (Native Instruments)"),
        "SptkBGAm": ("software", "Steinway D", "ambient",
                     "The Black Grand (Sampletekk)"),
        "SptkBGCl": ("software", "Steinway D", "close",
                     "The Black Grand (Sampletekk)"),
        # Real
        "ENSTDkAm": ("real", "Yamaha Disklavier Mark III",
                     "ambient", "Real piano (Disklavier)"),
        "ENSTDkCl": ("real", "Yamaha Disklavier Mark III",
                     "close", "Real piano (Disklavier)")}
    ISOL_SUBCATS = {"NO": "2s notes", "ST": "staccato", "RE": "repeated",
                    "CH": "chromatic", "TR1": "half tone trill",
                    "TR2": "whole tone trill"}

    @staticmethod
    def helper_parse_isol(basename, subcat):
        """
        :returns: ``(instr, cat, subcat, loudness, pedal, pitch, note_len)``
        """
        if subcat in ("NO", "ST", "RE", "TR1", "TR2"):
            _, cat, sub, loudness, pedal, pitch, instr = basename.split("_")
            assert sub == subcat, "Mismatching subcat!"
            pedal = (pedal == "S1")
            pitch = int(pitch.replace("M", ""))
            return (instr, cat, sub, loudness, pedal, pitch, None)
        # CH has no pedal, and no pitch is specified
        elif subcat == "CH":
            _, cat, ch_len, loudness, instr = basename.split("_")
            assert ch_len[:2] == "CH", "Mismatching subcat!"
            note_len = float(ch_len[2:])
            return (instr, cat, "CH", loudness, False, None, note_len)
        else:
            raise RuntimeError(f"Unknown ISOL subcat: {subcat}")

    @classmethod
    def parse_isol(cls, isol_path):
        """
        The MAPS dataset has quite structured subdivisions (see paper).
        Information is provided through the filesystem, and this method
        extracts such information.

        Specifically, it handles the ISOL categorie (isolated notes): for each
        one of the ISOL subcategories, it gathers ``(abspath, *metadata)``,
        where metadata is given by ``helper_parse_isol``.
        """
        result = []
        for subcat in cls.ISOL_SUBCATS:
            subcat_path = os.path.join(isol_path, subcat)
            subcat_files = os.listdir(subcat_path)
            subcat_basenames = {os.path.splitext(p)[0] for p in subcat_files}
            assert len(subcat_basenames) * 3 == len(subcat_files), \
                "Dataset must have one [.wav, .txt, .mid] per file!"
            for n in subcat_basenames:
                abspath = os.path.join(subcat_path, n)
                metadata = cls.helper_parse_isol(n, subcat)
                result.append((abspath, *metadata))
        return result

    @classmethod
    def parse_rand(cls, rand_path):
        """
        The MAPS dataset has quite structured subdivisions (see paper).
        Information is provided through the filesystem, and this method
        extracts such information.

        Specifically, it handles the RAND categorie (random chords).

        For each level of polyphony, pitch range, velocity range and sustain
        pedal, this category contains 'n' random chords.
        :returns: collection of ``(abspath, instr, cat, poly, velrange, pedal,
          pitchrange, idx)``.
        """
        result = []
        for root, dirs, files in os.walk(rand_path):
            if files:
                basenames = {os.path.splitext(p)[0] for p in files}
                assert len(files) == len(basenames) * 3, \
                    "Dataset must have one [.wav, .txt, .mid] per file!"
                for f in basenames:
                    abspath = os.path.join(root, f)
                    (_, cat, poly,
                     pitchrange, velrange, pedal, idx, instr) = f.split("_")
                    #
                    assert cat == "RAND", "Mismatching category!"
                    #
                    assert poly[0] == "P", "Poly must be in format 'P1'!"
                    poly = int(poly[1:])
                    #
                    assert velrange[0] == "I", \
                        "Velocity must be in format I..."
                    velrange = tuple(int(v) for v in velrange[1:].split("-"))
                    #
                    assert pitchrange[0] == "M", "Pitch must be in format M..."
                    pitchrange = tuple(int(p)
                                       for p in pitchrange[1:].split("-"))
                    #
                    pedal = (pedal == "S1")
                    #
                    assert idx[0] == "n", "Index must be in format n10"
                    idx = int(idx[1:])
                    result.append((abspath, instr, cat, poly, velrange,
                                  pedal, pitchrange, idx))
        return result

    @classmethod
    def parse_ucho(cls, ucho_path):
        """
        The MAPS dataset has quite structured subdivisions (see paper).
        Information is provided through the filesystem, and this method
        extracts such information.

        Specifically, it handles the UCHO categorie (usual chords).
        :returns: collection of ``(abspath, instr, cat, chord, velrange,
          pedal,  idx)``.
        """
        result = []
        for root, dirs, files in os.walk(ucho_path):
            if files:
                basenames = {os.path.splitext(p)[0] for p in files}
                assert len(files) == len(basenames) * 3, \
                    "Dataset must have one [.wav, .txt, .mid] per file!"
                for f in basenames:
                    abspath = os.path.join(root, f)
                    _, cat, chord, velrange, pedal, idx, instr = f.split("_")
                    #
                    assert cat == "UCHO", "Mismatching category!"
                    #
                    assert chord[0] == "C", "Chord must be in format C0-5"
                    chord = tuple(int(c) for c in chord[1:].split("-"))
                    #
                    assert velrange[0] == "I", \
                        "Velocity must be in format I..."
                    velrange = tuple(int(v) for v in velrange[1:].split("-"))
                    #
                    pedal = (pedal == "S1")
                    #
                    assert idx[0] == "n", "Index must be in format n10"
                    idx = int(idx[1:])
                    result.append((abspath, instr, cat, chord, velrange,
                                   pedal, idx))
        return result

    @classmethod
    def parse_mus(cls, mus_path):
        """
        The MAPS dataset has quite structured subdivisions (see paper).
        Information is provided through the filesystem, and this method
        extracts such information.

        Specifically, it handles the MUS categorie (musical pieces).
        :returns: collection of ``(abspath, instr, cat, description)
        """
        result = []
        files = os.listdir(mus_path)
        basenames = {os.path.splitext(p)[0] for p in files}
        assert len(files) == len(basenames) * 3, \
            "Dataset must have one [.wav, .txt, .mid] per file!"
        #
        for f in basenames:
            abspath = os.path.join(mus_path, f)
            assert f[:9] == "MAPS_MUS-", "Mismatching category!"
            instr = f[9:].split("_")[-1]
            description = f[9:-len(instr)-1]
            result.append((abspath, instr, "MUS", description))
        return result

    @classmethod
    def parse_maps(cls, rootpath):
        """
        Given the root path to MAPS, parses all its filename contents and
        returns ``(isol_data, rand_data, ucho_data, mus_data)``.
        """
        isol_data, rand_data, ucho_data, mus_data = [], [], [], []
        for instr in cls.INSTRUMENTS:
            isol_path = os.path.join(rootpath, instr, "ISOL")
            rand_path = os.path.join(rootpath, instr, "RAND")
            ucho_path = os.path.join(rootpath, instr, "UCHO")
            mus_path = os.path.join(rootpath, instr, "MUS")
            #
            isol_data.extend(cls.parse_isol(isol_path))
            rand_data.extend(cls.parse_rand(rand_path))
            ucho_data.extend(cls.parse_ucho(ucho_path))
            mus_data.extend(cls.parse_mus(mus_path))
        #
        return isol_data, rand_data, ucho_data, mus_data

    def __init__(self, rootpath,
                 include_instr={"StbgTGd2", "AkPnBsdf", "AkPnBcht", "AkPnCGdD",
                                "AkPnStgb", "SptkBGAm", "SptkBGCl", "ENSTDkAm",
                                "ENSTDkCl"},
                 include_cat={"ISOL", "MUS", "RAND", "UCHO"},
                 handle_redundant_mus=1):
        """
        :param int handle_redundant_mus: Some MUS files are the same tune in
          different instruments. If this parameter is 0, nothing is done. If
          it is 1, only one version will be kept, chosen at random. If it is 2,
          all versions will be ignored, leaving none left.

        Given the rootpath to the MAPS dataset, parse all its contained
        filenames (after applying the given ``include`` filters and handling
        redundancies. Results are stored in ``self.full_data``.
        """
        self.rootpath = rootpath
        self.include_instr = include_instr
        self.include_cat = include_cat

        # gather all entries
        isol_data, rand_data, ucho_data, mus_data = self.parse_maps(rootpath)

        # optionally remove MUS duplicates
        mus_histogram = defaultdict(list)
        for abspath, _, _, mus_description in mus_data:
            mus_histogram[mus_description].append(abspath)
        mus_duplicates = {k for k, v in mus_histogram.items() if len(v) > 1}
        if handle_redundant_mus == 0:
            # do nothing
            pass
        elif handle_redundant_mus == 1:
            # for each rendundant song, remove all entries except one
            to_remove = set()
            for v in mus_histogram.values():
                random.shuffle(v)
                to_remove.update(v[1:])
            mus_data = [x for x in mus_data if x[0] not in to_remove]
        elif handle_redundant_mus == 2:
            # remove ALL redundant entries
            mus_data = [x for x in mus_data if x[-1] not in mus_duplicates]
        else:
            raise RuntimeError("handle_redundant_mus must be 0, 1 or 2!")
        # gather all filtered (abspath, instr, cat) entries
        self.data = []
        if "ISOL" in include_cat:
            isol = [x[:3] for x in isol_data if x[1] in include_instr]
            self.data.extend(isol)
        if "RAND" in include_cat:
            rand = [x[:3] for x in rand_data if x[1] in include_instr]
            self.data.extend(rand)
        if "UCHO" in include_cat:
            ucho = [x[:3] for x in ucho_data if x[1] in include_instr]
            self.data.extend(ucho)
        if "MUS" in include_cat:
            mus = [x[:3] for x in mus_data if x[1] in include_instr]
            self.data.extend(mus)
        # gather comprehensive entries
        self.full_data = {"ISOL": isol_data, "RAND": rand_data,
                          "UCHO": ucho_data, "MUS": mus_data}

    def get_file_abspath(self, basename):
        """
        :param basename: Base name of the corresponding MIDI file without
         extension, e.g.
         MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--4'
        :returns: Unique absolute path for that basename
        """
        matches = [bn for bn, _, _ in self.data if basename in bn]
        assert len(matches) == 1, "Expected exactly 1 match!"
        return matches[0]


# ##############################################################################
# #  PYTORCH DATASETS
# ##############################################################################
class MelMaps(torch.utils.data.Dataset):
    """
    Assuming we computed the log-mels of the MAPS dataset and the MIDIs as
    pianorolls into 2 synchronized IncrementalHDF5 files, this dataset
    retrieves the log-mel and piano roll features of a given set of files.
    """

    NUM_MIDI_VALUES = MidiToPianoRoll.NUM_MIDI_VALUES
    NUM_MIDI_VALUES_TWICE = MidiToPianoRoll.NUM_MIDI_VALUES * 2
    ONSETS_RANGE = (0, MidiToPianoRoll.NUM_MIDI_VALUES)
    FRAMES_RANGE = (MidiToPianoRoll.NUM_MIDI_VALUES, -3)
    SUS_IDX = -3
    SOFT_IDX = -2
    TEN_IDX = -1

    @classmethod
    def _init_helper(cls, hdf5_logmels_path, hdf5_pianorolls_path,
                     basenames):
        """
        Given the paths to the HDF5 files and the desired audio files by their
        basename, opens the HDF5s and checks for consistency before returning.

        :returns: ``(mel_hdf5_handle, roll_hdf5_handle, metadata, file_idxs)``,
          where the HDF5 handles contain all files but the metadata and idxs
          contain only the provided ``basenames``.
        """
        # load HDF5 files and sanity check
        h5m = h5py.File(hdf5_logmels_path, "r")
        h5r = h5py.File(hdf5_pianorolls_path, "r")
        assert (h5m[IncrementalHDF5.IDXS_NAME][:] ==
                h5r[IncrementalHDF5.IDXS_NAME][:]).all(), \
            "Unequal data indexes between logmel and piano roll?"
        assert (h5m[IncrementalHDF5.METADATA_NAME][:] ==
                h5r[IncrementalHDF5.METADATA_NAME][:]).all(), \
            "Unequal metadata between logmel and piano roll?"
        assert (h5m[IncrementalHDF5.DATA_NAME].shape[1] ==
                h5r[IncrementalHDF5.DATA_NAME].shape[1]), \
            "Unequal data length between logmel and piano roll?"
        assert (h5r[IncrementalHDF5.DATA_NAME].shape[0] ==
                cls.NUM_MIDI_VALUES_TWICE + 3), \
            "Unexpected number of columns in pianoroll HDF5 file!"
        # extract metadata and file idxs matching given basenames
        metadata = [literal_eval(t.decode("utf-8"))
                    for t in h5m[IncrementalHDF5.METADATA_NAME]]
        chosen_metadata, chosen_file_idxs = zip(
            *[(x, i) for i, x in enumerate(metadata) if x[0] in basenames])
        assert len(chosen_file_idxs) == len(basenames), \
            "HDF5 database is missing files?"
        # return HDF5 file handles and chosen idxs/metadata
        return h5m, h5r, chosen_metadata, chosen_file_idxs

    def __init__(self, hdf5_logmels_path, hdf5_pianorolls_path,
                 *abspaths, as_torch_tensors=True):
        """
        Given the file paths, loads and checks the files before storing them
        into ``self.data``.

        :param hdf5_logmels_path: Path to the HDF5 containing log-mel
          spectrograms, as generated by the preprocessing scripts.
        :param hdf5_pianorolls_path: Path to the HDF5 containing piano rolls,
          as generated by the preprocessing scripts. Expected to be compatible
          with the log-mels (e.g. same time-quantization, same lengths, same
          files...), normally generated in the exact same preprocessing run.
        :param abspaths: strings with the filenames without extension.
          Only these files will be taken into account.
        :param as_torch_tensors: If false, ``getitem`` returns numpy arrays.
        """
        self.h5_mel_path = hdf5_logmels_path
        self.h5_roll_path = hdf5_pianorolls_path
        self.as_tensors = as_torch_tensors
        self.basenames = {os.path.basename(x) for x in abspaths}
        #
        self.h5m, self.h5r, metadata, file_idxs = self._init_helper(
            hdf5_logmels_path, hdf5_pianorolls_path, self.basenames)
        # for each chosen idx/metadata, retrieve beg:end range
        self.data = []
        for meta, file_idx in zip(metadata, file_idxs):
            beg, end = self.h5m[IncrementalHDF5.IDXS_NAME][:, file_idx]
            self.data.append((beg, end, meta))

    def cleanup(self):
        """
        Close HDF5 file handles
        """
        self.h5m.close()
        self.h5r.close()

    def __len__(self):
        """
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        :returns: ``(logmel, roll, meta)``, where the first 2 are matrices of
          same width containing the log-mel spectrogram and piano roll for the
          file with index ``idx``, and meta is a string containing the
          corresponding file metadata.
        """
        beg, end, meta = self.data[idx]
        logmel = self.h5m[IncrementalHDF5.DATA_NAME][:, beg:end]
        roll = self.h5r[IncrementalHDF5.DATA_NAME][:, beg:end]
        #
        if self.as_tensors:
            logmel = torch.from_numpy(logmel)
            roll = torch.from_numpy(roll)
        #
        return logmel, roll, meta


class MelMapsChunks(MelMaps):
    """
    Like its parent class, but instead of retrieving full files, it retrieves
    smaller strided chunks.

    :cvar ROLL_PAD_VAL: Whenever a chunk is out-of-bounds, the corresponding
      piano roll regions will be padded with this value (usually zero, meaning
      no activity).
    """

    ROLL_PAD_VAL = 0

    @staticmethod
    def split_range(rng_beg, rng_end, chunk_length, chunk_stride,
                    with_oob=True):
        """
        Helper method to extract chunks from a given range, making sure that
        out-of-bounds chunks are also present if desired, and properly handled.
        """
        dur = rng_end - rng_beg
        oob = (chunk_length - chunk_stride) if with_oob else 0
        # compute relative chunk beginnings and ends
        chunk_begs = np.arange(-oob, (dur - chunk_length) + oob, chunk_stride)
        chunk_ends = chunk_begs + chunk_length
        # compute corresponding zero-pads (negative=left, positive=right)
        zero_pads = np.zeros_like(chunk_begs)
        zero_pads[chunk_begs < 0] = chunk_begs[chunk_begs < 0]
        zero_pads[chunk_ends >= dur] = chunk_ends[chunk_ends >= dur] - dur
        # translate relative beg/ends to absolute
        chunk_begs += rng_beg
        chunk_ends += rng_beg
        #
        result = np.vstack([chunk_begs, chunk_ends, zero_pads])  # (3, N)
        return result

    def __init__(self, hdf5_logmels_path, hdf5_pianorolls_path,
                 chunk_length, chunk_stride, *abspaths,
                 with_oob=True, logmel_oob_pad_val="min",
                 as_torch_tensors=True):
        """
        :param with_oob: if false, all extracted chunks belong fully to an
          existing file. Otherwise, they can be partially outside (padded with
          oob_pad_val).
        :param logmel_oob_pad_val: This can be a scalar or 'min', in which case
          the minimum (non-excluded) logmel value will be used. See with_oob
          for more details.

        See parent method for more information.
        """
        self.h5_mel_path = hdf5_logmels_path
        self.h5_roll_path = hdf5_pianorolls_path
        self.chunk_length = chunk_length
        self.chunk_stride = chunk_stride
        self.with_oob = with_oob
        self.logmel_oob_pad_val = logmel_oob_pad_val
        self.as_tensors = as_torch_tensors
        #
        self.basenames = {os.path.basename(x) for x in abspaths}
        self.h5m, self.h5r, self.metadata, file_idxs = self._init_helper(
            hdf5_logmels_path, hdf5_pianorolls_path, self.basenames)
        # cut each chosen file into chunks and gather
        self.data = []
        self.metadata_chunks = []
        for meta_i, file_idx in enumerate(file_idxs):
            beg, end = self.h5m[IncrementalHDF5.IDXS_NAME][:, file_idx]
            idxs = self.split_range(
                beg, end, chunk_length, chunk_stride, with_oob)
            self.data.append(idxs)
            self.metadata_chunks.extend(meta_i for _ in range(idxs.shape[1]))
        self.data = np.concatenate(self.data, axis=1).T  # (n, 3)
        self.metadata_chunks = np.uint64(self.metadata_chunks)
        # aliases for convenience
        self.mels = self.h5m[IncrementalHDF5.DATA_NAME]
        self.rolls = self.h5r[IncrementalHDF5.DATA_NAME]
        #
        self.mel_buffer = np.zeros((self.mels.shape[0], chunk_length),
                                   dtype=self.mels.dtype)
        self.roll_buffer = np.zeros((self.rolls.shape[0], chunk_length),
                                    dtype=self.rolls.dtype)

    def __getitem__(self, idx):
        """
        :returns: ``(logmel, roll, meta)``, where the first 2 are matrices of
          same width containing the log-mel spectrogram and piano roll for the
          chunk with index ``idx``, and meta is a string containing the
          corresponding file metadata. Note that chunks with out-of-bounds
          regions still belong to exactly one file.
        """
        beg, end, pad = self.data[idx]
        meta = (*self.metadata[self.metadata_chunks[idx]], beg, end, pad)
        #
        if pad < 0:  # pad the beginning
            pad = abs(pad)
            self.roll_buffer[:, :pad] = self.ROLL_PAD_VAL
            self.roll_buffer[:, pad:] = self.rolls[:, (beg + pad):end]
            #
            self.mel_buffer[:, pad:] = self.mels[:, (beg+pad):end]
            mel_pad_val = (self.mel_buffer[:, pad:].min()
                           if self.logmel_oob_pad_val == "min"
                           else self.logmel_oob_pad_val)
            self.mel_buffer[:, :pad] = mel_pad_val

        elif pad > 0:  # pad the end
            self.roll_buffer[:, -pad:] = self.ROLL_PAD_VAL
            self.roll_buffer[:, :-pad] = self.rolls[:, beg:(end-pad)]
            #
            self.mel_buffer[:, :-pad] = self.mels[:, beg:(end-pad)]
            mel_pad_val = (self.mel_buffer[:, :-pad].min()
                           if self.logmel_oob_pad_val == "min"
                           else self.logmel_oob_pad_val)
            self.mel_buffer[:, -pad:] = mel_pad_val
        else:  # no pad
            self.roll_buffer = self.rolls[:, beg:end]
            self.mel_buffer = self.mels[:, beg:end]
        # return buffer copy (optionally convert to tensor)
        if self.as_tensors:
            logmel = torch.from_numpy(self.mel_buffer)
            roll = torch.from_numpy(self.roll_buffer)
        else:
            logmel = np.array(self.mel_buffer)
            roll = np.array(self.roll_buffer)
        #
        return logmel, roll, meta
