#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module contains functionality to parse MIDI piano files, making use of
the ``mido`` Python library, and also to convert parsed MIDI files into
piano roll representation, making use of functionality implemented in the
``key_model`` module.
"""


from collections import defaultdict, OrderedDict
#
import mido
import numpy as np
import pandas as pd
#
from .key_model import KeyboardStateMachine


# ##############################################################################
# # PIANO MIDI PARSERS
# ##############################################################################
class SingletrackMidiParser:
    """
    Static class providing functionality to load and parse a MIDI file, as
    well as converting it to an event-based representation via
    ``KeyboardStatemachine``.

    The MIDI files provided in the MAPS dataset are structured as single-track
    sequences (code 0), and this class is tailored for that.
    """

    SINGLETRACK_MIDI_CODE = 0
    DEFAULT_MIDI_TEMPO = 500_000  # MIDI standard
    # https://cmtext.indiana.edu/MIDI/chapter3_controller_change2.php
    CONTROL_CODES = {64: "sustain_pedal", 66: "sostenuto_pedal",
                     67: "soft_pedal"}

    @classmethod
    def load_midi(cls, midi_path):
        """
        :param str midi_path: Path to a MIDI file, assumed single-track.
        :returns: ``mido.MidiFile`` instance with the loaded file.
        """
        mid = mido.MidiFile(midi_path)
        assert mid.type == cls.SINGLETRACK_MIDI_CODE, \
            "Only single-track MIDI files expected!"
        return mid

    @classmethod
    def dispatch_msg(cls, msg):
        """
        Given a MIDI message, extract its relevant contents. Dispatcher
        required because different messages have different structures, but we
        want a normalized one.

        :returns: A triple ``(msg_type, msg_value, msg_channel)``, where
          value can also be a composite type.
        """
        if msg.type in ("note_on", "note_off"):
            return (msg.type, (msg.note, msg.velocity), msg.channel)
        elif msg.type == "control_change":
            msg_name = cls.CONTROL_CODES[msg.control]
            return (msg_name, msg.value, msg.channel)
        else:
            raise RuntimeError(f"Unhandled message! {msg}")

    @classmethod
    def parse_midi(cls, midi_file):
        """
        https://stackoverflow.com/a/34174936

        MIDI is a sequential format designed for real-time, including changes
        in tempo and relative delays between messages. For this reason, it
        doesn't provide the global timestamp of each message.

        This method parses the given ``mido.MidiFile``, extracting the relevant
        information from each message and also converting the timestamps to
        global frame, in seconds.
        """
        assert len(midi_file.tracks) == 1, "Only single-track MIDI supported!"
        tpb = midi_file.ticks_per_beat  # given in file
        tempo = cls.DEFAULT_MIDI_TEMPO
        seconds_counter = 0
        messages = []
        meta_messages = []
        for msg in midi_file.tracks[0]:
            # all messages: convert delta to microseconds and add to counter
            delta_beats = msg.time / tpb
            delta_microseconds = tempo * delta_beats
            seconds_counter += delta_microseconds / 1_000_000
            # if message is not meta: simply gather
            if not msg.is_meta:
                msg_data = cls.dispatch_msg(msg)
                messages.append((seconds_counter, msg_data))
            # meta messages may require special handling
            else:
                meta_messages.append((seconds_counter, msg))
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                elif msg.type == "end_of_track":
                    pass
                else:
                    raise RuntimeError(
                        f"Unhandled MIDI meta-message?: {msg.type}")
        assert msg.type == "end_of_track", "Last message must be end_of_track!"
        assert np.isclose(midi_file.length, meta_messages[-1][0]), \
            "MIDI length should equal end_of_track timestamp! " + \
            f"{(midi_file.length, meta_messages[-1][0])}"
        return messages, meta_messages

    @staticmethod
    def ksm_parse_midi_messages(msgs, ksm):
        """
        This method iterates over the midi messages given by``parse_midi`` and
        feeds them to the given keyboard state machine, so that they are
        converted from real-time events to note events with onset and offset.
        It also performs some sanity checks.
        :returns: The tuple ``(key_events, sus, ten, soft, largest_ts)`` where
          the elements are pandas DataFrames, except for ``largest_ts``, which
          is the largest timestamp among all other dataframes.
        """
        sus_states = []
        ten_states = []
        soft_states = []
        #
        onsets = []
        offsets = []
        sounding = []

        msgs_ts = [ts for ts, _ in msgs]
        assert msgs_ts == sorted(msgs_ts), "msgs expected to be sorted by ts!"
        for ts, m in msgs:
            m_type = m[0]
            if m_type == "sustain_pedal":
                offs = ksm.update_sus(m[1], ts)
                sus_states.append((ts, m[1]))
            elif m_type == "sostenuto_pedal":
                offs = ksm.update_ten(m[1], ts)
                ten_states.append((ts, m[1]))
            elif m_type == "soft_pedal":
                offs = ksm.update_soft(m[1])
                soft_states.append((ts, m[1]))
            #
            elif (m_type == "note_on") and (m[1][1] > 0):
                key, vel = m[1]
                offs = ksm.key_pressed((key, vel), timestamp=ts)
                onsets.append((ts, {key: vel}))
            elif ((m_type == "note_off") or
                  ((m_type == "note_on") and (m[1][1] == 0))):
                key, vel = m[1]
                offs = ksm.key_lifted(key, timestamp=ts)
            else:
                raise RuntimeError(f"Unhandled MIDI event! {(ts, m)}")
            #
            if offs:
                offsets.append((ts, offs))
            #
            down, reson = ksm()[:2]
            sounding.append((ts, {**down, **reson}))

        # we must have as many onsets as offsets+sounding notes
        last_sounding = len(sounding[-1][1])
        all_onsets = sum([len(x) for _, x in onsets])
        all_offsets = sum([len(x) for _, x in offsets])
        assert all_onsets == (all_offsets + last_sounding), \
            "Mismatching number of onsets and offsets+sounding notes!"

        # unfortunately, MAPS seems to have simultaneous pedal events with
        # different values. we handle this by picking the last one and ignoring
        # the rest
        sus_states = [(k, v) for k, v in OrderedDict(sus_states).items()]
        ten_states = [(k, v) for k, v in OrderedDict(ten_states).items()]
        soft_states = [(k, v) for k, v in OrderedDict(soft_states).items()]
        assert len(sus_states) == len(dict(sus_states)), \
            "Simultaneous sus pedal events not allowed!"
        assert len(ten_states) == len(dict(ten_states)), \
            "Simultaneous ten pedal events not allowed!"
        assert len(soft_states) == len(dict(soft_states)), \
            "Simultaneous soft pedal events not allowed!"

        # we can't have 2 simultaneous same-key onsets or offsets
        onsets_check = defaultdict(list)
        offsets_check = defaultdict(list)
        for k, v in onsets:
            onsets_check[k].extend(v.keys())
        for k, v in offsets:
            offsets_check[k].extend(v.keys())
        assert all([len(v) == len(set(v)) for v in onsets_check.values()]), \
            "Simultaneous same-key onsets not allowed!"
        assert all([len(v) == len(set(v)) for v in offsets_check.values()]), \
            "Simultaneous same-key offsets not allowed!"
        # at this point, no downpressed notes are expected, but some resonant
        # notes may be left
        assert not ksm()[0], "Notes left downpressed at end of MIDI?"

        # handle resonant leftovers by adding them to offsets
        largest_ts = max(
            offsets[-1][0],  # simply find largest timestamp
            max(sus_states[-1:], key=lambda elt: elt[0], default=[-1])[0],
            max(sus_states[-1:], key=lambda elt: elt[0], default=[-1])[0],
            max(sus_states[-1:], key=lambda elt: elt[0], default=[-1])[0])
        for ke in ksm()[1].values():
            ke.offset_ts = largest_ts
        offsets.append((largest_ts, ksm()[1].copy()))

        # After all sanity checks, build the pandas dataframes
        key_events = pd.DataFrame(
            ((ke.onset_ts, ke.offset_ts, ke.key, ke.vel) for _, ke_dict
             in offsets for _, ke in ke_dict.items()),
            columns=("onset", "offset", "key", "vel")).sort_values(by="onset")
        sus_states = pd.DataFrame(sus_states, columns=["ts", "val"])
        ten_states = pd.DataFrame(ten_states, columns=["ts", "val"])
        soft_states = pd.DataFrame(soft_states, columns=["ts", "val"])
        # return key events and pedal sequences
        return key_events, sus_states, ten_states, soft_states, largest_ts


class MaestroMidiParser(SingletrackMidiParser):
    """
    An extension of SingleTrackMidiParser, with an added extra step of
    converting MAESTRO MIDI files into single track, to ensure downstream
    compatibility with other single-instrument MIDI data.
    """

    @staticmethod
    def convert_maestro_midi_to_singletrack(mid):
        """
        :param mid: A ``mido.MidiFile`` from the MAESTRO dataset.
        :returns: None, it modifies the MIDI in-place

        MAPS files have all messages on the same track, but MAESTRO files have
        3 specific meta-messages on a separate track. This function merges them
        onto the main track, so that the resulting MIDI has a MAPS-alike format
        """
        # sanity check: MAESTRO MIDI has 2 tracks
        assert mid.type == 1, "Unexpected MAESTRO MIDI type!"

        if len(mid.tracks[0]) != 3:
            # in MAESTRO, at least 1 track has repeated time_signature. We
            # handle this by simply ignoring the repeated entry, and treating
            # tracks[0] as 3-event
            assert len(mid.tracks[0]) == 4, "Not 3, not 4??"
            expected_types = ["set_tempo", "time_signature",
                              "time_signature", "end_of_track"]
            assert expected_types == [t.type for t in mid.tracks[0]], \
                "unexpected MAESTRO MIDI track [0]!"
            mid.tracks[0].pop(2)

        # check that 1st track only has 3 messages at time 0
        tempo_msg, sig_msg, end_msg = mid.tracks[0]
        assert (tempo_msg.type == "set_tempo") and (tempo_msg.time == 0), \
            "Unexpected first message in MAESTRO MIDI track[0]!"
        assert (sig_msg.type == "time_signature") and (sig_msg.time == 0), \
            "Unexpected second message in MAESTRO MIDI track[0]!"
        assert (end_msg.type == "end_of_track") and (end_msg.time == 1), \
            "Unexpected third message in MAESTRO MIDI track[0]!"

        # The 2nd track may contain a single "track_name" message at the beg
        name_msgs = [(i, x) for i, x in enumerate(mid.tracks[1])
                     if x.type == "track_name"]
        if name_msgs:
            assert len(name_msgs) == 1, "More than 1 track_name message!"
            assert name_msgs[0][0] == 0, "Track_name wasn't the 1st in track!"
            assert name_msgs[0][1].time == 0, "track_name doesn't have time=0!"
            # if all is good, remove the message
            mid.tracks[1].pop(0)

        # once handling track_name, check that 2nd track always contains a
        # single prog=0,ch=0,t=0 msg at the beginning
        prog_msgs = [(i, x) for i, x in enumerate(mid.tracks[1])
                     if x.type == "program_change"]
        assert len(prog_msgs) == 1, "More than 1 program_change message!"
        assert prog_msgs[0][0] == 0, "Program_change wasn't the 1st in track!"
        assert prog_msgs[0][1].channel == 0, "program_change channel != 0"
        assert prog_msgs[0][1].program == 0, "program_change program != 0"
        assert prog_msgs[0][1].time == 0, "program_change time != 0"
        # Merge both tracks into 1: replace the program_change message with
        # the tempo_message at beginning, ignore signature, and append
        # end_message at the end. Note the in-place modifications
        mid.type = 0
        mid.tracks[1][0] = tempo_msg
        mid.tracks = mid.tracks[1:]

    @classmethod
    def load_midi(cls, midi_path):
        """
        Like parent method, but we call ``convert_maestro_midi_to_singletrack``
        before proceeding further.
        """
        mid = mido.MidiFile(midi_path)
        cls.convert_maestro_midi_to_singletrack(mid)  # in-place conversion!
        assert mid.type == cls.SINGLETRACK_MIDI_CODE, \
            "Only single-track MIDI files expected!"
        return mid


# ##############################################################################
# # PIANO ROLL CONVERTER
# ##############################################################################
class MidiToPianoRoll:
    """
    This class makes use of the MIDI parsers to fully convert from a MIDI path
    (expected to contain a piano score) into a piano roll.
    """

    NUM_MIDI_VALUES = 128  # from 0 to 127
    # pedals are considered active if strictly above this threshold
    SUS_PEDAL_THRESH = 7
    TEN_PEDAL_THRESH = 0

    @staticmethod
    def _check_midi(msgs, meta_msgs):
        """
        :raises: Exception if any of the assumptions on the data aren't met.
        """
        # Check that all messages are on channel 0 (single-instrument)
        msg_channels = [x[1][-1] for x in msgs]
        assert all(x == msg_channels[0] for x in msg_channels), \
            "Only single-channel MIDI supported!"
        # Check that meta messages are simply set tempo at beginning and
        # end of track at end
        assert len(meta_msgs) == 2, \
            "Expected only 2 meta msgs. If multiple tempo changes, fix this!"
        assert meta_msgs[0][1].type == "set_tempo", \
            "First meta message expected is set_tempo"
        assert meta_msgs[1][1].type == "end_of_track", \
            "Second meta message expected is end_of_track"

    @classmethod
    def __call__(cls, midi_path,
                 midi_parser=SingletrackMidiParser,
                 quant_secs=0.01,
                 extend_offsets_sus=True,
                 ignore_redundant_keypress=False,
                 ignore_redundant_keylift=False):
        """
        :param midi_parser: Use ``SingletrackMidiParser`` for single-track
          midi piano files (e.g. MAPS), and ``MaestroMidiParser`` for MAESTRO
          (and likely other Disklavier-based) files.
        :param float quant_secs: Quantization unit in seconds, representing
          the distance between two consecutive columns of the resulting
          piano-roll matrix.
        :param bool extend_offsets_sus: If true, note offsets will be delayed
          if the sus pedal is pressed (specifically, if the sus value at the
          offset is >= class.SUS_PEDAL_THRESH). The offset will occur either
          when the pedal is lifted (i.e. value below thresh), or if the note
          is repeated, in which case there will be no gap between extended
          and new note (only velocity may change). This is not a problem
          because the onset map specifically provides all onsets.
        :param bool ignore_redundant_keypress: If false, pressign down a note
          that is already pressed raises an exception. Otherwise, the
          preexisting note is "lifted", and a new note is created.
        :param bool ignore_redundant_keylift: If false, lifting a note that
          hasn't been pressed raises an exception. Otherwise, a warning is
          printed instead.
        """
        # load and check midi
        mid = midi_parser.load_midi(midi_path)
        msgs, meta_msgs = midi_parser.parse_midi(mid)
        cls._check_midi(msgs, meta_msgs)
        # convert midi to events with onset and offset
        sus_t = (cls.SUS_PEDAL_THRESH if extend_offsets_sus else float("inf"))
        (key_events, sus_states, ten_states, soft_states,
         largest_ts) = midi_parser.ksm_parse_midi_messages(
             msgs, KeyboardStateMachine(
                 sus_t, cls.TEN_PEDAL_THRESH,
                 ignore_redundant_keypress, ignore_redundant_keylift))
        # prepare time-quantized output
        frame_ts = np.arange(0, largest_ts, quant_secs)[:-1]
        num_frames = len(frame_ts)
        # create output datastructures
        onset_roll = np.zeros((cls.NUM_MIDI_VALUES, num_frames),
                              dtype=np.uint8)
        offset_roll = np.zeros((cls.NUM_MIDI_VALUES, num_frames),
                               dtype=np.uint8)
        frame_roll = np.zeros_like(onset_roll)
        sus_roll = np.zeros_like(onset_roll[0, :])
        ten_roll = np.zeros_like(sus_roll)
        soft_roll = np.zeros_like(sus_roll)

        # write pedal sequences onto quantized rolls
        if not sus_states.empty:
            sus_idxs = np.searchsorted(
                frame_ts, sus_states["ts"], side="right") - 1
            for beg, end, v in zip(sus_idxs[:-1], sus_idxs[1:],
                                   sus_states["val"][:-1]):
                sus_roll[beg:end] = v
            sus_roll[sus_idxs[-1]:] = sus_states["val"].iloc[-1]
        #
        if not ten_states.empty:
            ten_idxs = np.searchsorted(
                frame_ts, ten_states["ts"], side="right") - 1
            for beg, end, v in zip(ten_idxs[:-1], ten_idxs[1:],
                                   ten_states["val"][:-1]):
                ten_roll[beg:end] = v
            ten_roll[ten_idxs[-1]:] = ten_states["val"].iloc[-1]
        #
        if not soft_states.empty:
            soft_idxs = np.searchsorted(
                frame_ts, soft_states["ts"], side="right") - 1
            for beg, end, v in zip(soft_idxs[:-1], soft_idxs[1:],
                                   soft_states["val"][:-1]):
                soft_roll[beg:end] = v
            soft_roll[soft_idxs[-1]:] = soft_states["val"].iloc[-1]

        # write key events onto quantized rolls
        beg_idxs = np.searchsorted(
            frame_ts, key_events["onset"], side="right") - 1
        end_idxs = np.searchsorted(
            frame_ts, key_events["offset"], side="right") - 1
        for beg, end, key, vel in zip(
                beg_idxs, end_idxs, key_events["key"], key_events["vel"]):
            is_collision = ((onset_roll[key, beg] != 0) or
                            (offset_roll[key, end] != 0))
            if is_collision:
                print("WARNING: onset/offset collision. To avoid this,",
                      "increase time-quant resolution:", (beg, end, key, vel))
            if is_collision:
                print("WARNING: beg==end, note ignored. To avoid this,",
                      "increase time-quant resolution:", (beg, end, key, vel))
            onset_roll[key, beg] = vel
            offset_roll[key, end] = vel
            frame_roll[key, beg:end] = vel
        # import matplotlib.pyplot as plt
        # plt.clf(); plt.plot(sus_roll); plt.show()
        # plt.clf(); plt.plot(soft_roll); plt.show()
        # plt.clf(); plt.plot(ten_roll); plt.show()
        # plt.clf(); plt.plot(sus_states["ts"], sus_states["val"]); plt.show()
        # plt.clf(); plt.imshow(np.vstack([ten_roll, sus_roll, offset_roll, frame_roll, onset_roll])[::-1, -5000:]); plt.show()
        return (onset_roll, offset_roll, frame_roll,
                sus_roll, soft_roll, ten_roll, key_events)
