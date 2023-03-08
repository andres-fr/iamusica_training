#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module is analogous to ``maps``, but adapted for the MAESTRO dataset.
"""


import os
#
import pandas as pd
#
from .maps import MelMaps, MelMapsChunks


# ##############################################################################
# #  META (paths etc)
# ##############################################################################
class MetaMAESTROv3:
    """
    This class parses the filesystem tree for the MAESTRO dataset and, based on
    the given filters, stores a list of file paths.

    It can be used to manage MAESTRO files and to create custom dataloaders.
    """

    CSV_NAME = "maestro-v3.0.0.csv"
    ALL_SPLITS = {"train", "validation", "test"}
    ALL_YEARS = {2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018}
    AUDIO_EXT = ".wav"
    MIDI_EXT = ".midi"

    def __init__(self, rootpath, splits=None, years=None):
        """
        """
        self.rootpath = rootpath
        self.meta_path = os.path.join(rootpath, self.CSV_NAME)
        # filter sanity check
        if splits is None:
            splits = self.ALL_SPLITS
        if years is None:
            years = self.ALL_YEARS
        assert (s in self.ALL_SPLITS for s in splits), \
            f"Unknown split in {splits}"
        assert (y in self.ALL_YEARS for y in years), f"Unknown year in {years}"
        # load and filter csv
        df = pd.read_csv(self.meta_path)
        df = df[df["split"].isin(splits)]
        df = df[df["year"].isin(years)]
        # reformat into DATA_COLUMNS + metadata_str and gather
        columns = ["audio_filename", "year", "split", "duration",
                   "canonical_composer", "canonical_title"]
        self.data = []
        for i, (path, y, s, dur, comp, title) in df[columns].iterrows():
            basepath = os.path.splitext(path)[0]
            meta = (y, s, dur, comp, title)
            self.data.append((basepath, meta))
        self.full_data = df

    def get_file_abspath(self, basename):
        """
        :param basename: Base name of the corresponding MIDI file without
         extension, e.g.
         MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--4'
        :returns: Unique absolute path for that basename
        """
        matches = [fn for fn, _ in self.data if basename in fn]
        assert len(matches) == 1, "Expected exactly 1 match!"
        path = os.path.join(self.rootpath, matches[0])
        return path


class MetaMAESTROv1(MetaMAESTROv3):
    """
    Identical to parent class, except for ``CSV_NAME`` pointing to a different
    CSV file, and ``ALL_YEARS`` containing different years.
    """
    CSV_NAME = "maestro-v1.0.0.csv"
    ALL_YEARS = {2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017}


class MetaMAESTROv2(MetaMAESTROv3):
    """
    Identical to parent class, except for ``CSV_NAME`` pointing to a different
    CSV file, and ``ALL_YEARS`` containing different years.
    """
    CSV_NAME = "maestro-v2.0.0.csv"
    ALL_YEARS = {2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017}


# ##############################################################################
# #  PYTORCH DATASETS
# ##############################################################################
class MelMaestro(MelMaps):
    """
    Identical to parent class
    """
    pass


class MelMaestroChunks(MelMapsChunks):
    """
    Identical to parent class
    """
    pass
