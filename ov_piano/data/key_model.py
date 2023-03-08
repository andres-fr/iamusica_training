#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Pure-Python module implementing the state of a musical keyboard.

Useful to convert MIDI-style messages (designed for real-time processing) into
key events with onset, offset, velocity and key_id.
It can be also used to check/sanitize a MIDI file (e.g. if a key is pressed,
and then pressed again without being released this leads to an inconsistent
state).
"""


# ##############################################################################
# # KEY EVENT
# ##############################################################################
class KeyEvent:
    """
    Building block to convert from real-time midi events to onset-offset
    events.
    Each ``KeyEvent`` represents the act of pressing (and eventually releasing)
    a key note with a specific ID, using a specific velocity.

    Create an instance when a key is pressed, and update the ``offset_ts``
    timestamp when released.
    """
    def __init__(self, key_id, velocity, onset_ts=None):
        """
        """
        self.key = key_id
        self.vel = velocity
        #
        self.onset_ts = onset_ts
        self.offset_ts = None

    def __repr__(self):
        """
        """
        fstr = (f"(key={self.key}, vel={self.vel}, " +
                f"on_ts={self.onset_ts}, off_ts={self.offset_ts})")
        return fstr

    def __eq__(self, other_ke):
        """
        """
        assert isinstance(other_ke, self.__class__), \
            f"{other_ke} must be an instance of type {self.__class__}!"
        cond = ((self.key == other_ke.key) and
                (self.vel == other_ke.vel) and
                (self.onset_ts == other_ke.onset_ts) and
                (self.offset_ts == other_ke.offset_ts))
        return cond


# ##############################################################################
# # KEYBOARD STATE MACHINE
# ##############################################################################
class KeyboardStateMachine:
    """
    This class handles the inner resonant state of a 3-pedal keyboard in a
    callback-style that is convenient to build an interface between MIDI
    sequences and other formats, like piano rolls or interval sequences. Its
    logic can be summarized as follows:

    * The ``downpressed`` dictionary registers the keys that are currently
      pressed down, with their velocities. Only the ``key_pressed, key_lifted``
      methods alter this dictionary.
    * Changes in the 3 pedals get registered in the ``sus, soft, ten`` instance
      variables. When ``ten`` is activated, any downpressed notes are
      registered in the ``tenuto`` internal set. When ``ten`` is deactivated,
      ``tenuto`` is cleared.
    * ``key_pressed`` always registers onsets and creates KeyEvents. If no
      pedal is active, ``key_lifted`` registers offsets, but if a sus/ten pedal
      is affecting a lifted note, it becomes ``resonant`` (i.e. sound without
      being pressed).
    * An offset happens when a (downpressed or resonant) note stops sounding.
      This can happen for 4 reasons:
        1. A key is unaffected by pedals, and lifted
        2. A resonant key is being pressed again (so the resonant gets offset)
        3. The sus pedal is lifted (all resonant, non-ten notes get offset)
        4. The ten pedal is lifted (all resonant, non-sus notes get offset)
      This is handled in this class consistently: the ``key_pressed``,
      ``key_lifted``, ``update_sus`` and ``update_ten`` methods return a
      dictionary in the form``{key: KeyEvent}`` containing the notes that were
      offset as a result of calling the method.
    * At any point, instances of this class can be called to check the internal
      state comprehensively.
    """

    def __init__(self, sus_thresh=7, ten_thresh=0,
                 ignore_redundant_keypress=False,
                 ignore_redundant_keylift=False,
                 pre_lifting_epsilon=0.001):
        """
        :param sus_thresh: Any number strictly grater than this is considered
          an active pedal, otherwise inactive.
        :param bool ignore_redundant_keypress: If false, pressign down a note
          that is already pressed raises an exception. Otherwise, the
          preexisting note is "lifted", and a new note is created.
        :param bool ignore_redundant_keylift: If false, lifting a note that
          hasn't been pressed raises an exception. Otherwise, a warning is
          printed instead.
        :param pre_lifting_epsilon: If we ignore redundant keypresses, the
          preexisting keyevent is finished and a new one is created. To avoid
          strict overlapping, the preexisting event is finished at timepoint
          ``t-epsilon``
        """
        self.sus_thresh = sus_thresh
        self.ten_thresh = ten_thresh
        #
        self.downpressed = {}  # key is currently pressed down
        self.resonant = {}  # key is sounding but not pressed down
        self.tenuto = set()  # key is currently hooked by the sostenuto pedal
        #
        self.ignore_redundant_keypress = ignore_redundant_keypress
        self.ignore_redundant_keylift = ignore_redundant_keylift
        #
        self.pre_lifting_epsilon = pre_lifting_epsilon
        #
        self.sus = 0
        self.soft = 0
        self.ten = 0

    # PEDAL GETTERS
    def _sus_active(self):
        """
        :returns: A boolean telling if the sus pedal is currently above thresh.
        """
        return (self.sus > self.sus_thresh)

    def _sus_activation(self, old_val, new_val):
        """
        :returns: One of 3 integers: 0 (no change in activation), 1 (ten pedal
          was inactive and is now active), or -1 (pedal was active and is now
          inactive). The threshold for active is given at construction.
        """
        old_active = int(old_val > self.sus_thresh)
        new_active = int(new_val > self.sus_thresh)
        return (new_active - old_active)

    def _ten_activation(self, old_val, new_val):
        """
        :returns: One of 3 integers: 0 (no change in activation), 1 (ten pedal
          was inactive and is now active), or -1 (pedal was active and is now
          inactive). The threshold for active is given at construction.
        """
        old_active = int(old_val > self.ten_thresh)
        new_active = int(new_val > self.ten_thresh)
        return (new_active - old_active)

    # PEDAL UPDATES
    def update_soft(self, val):
        """
        """
        self.soft = val
        return {}  # no offsets can happen here

    def update_sus(self, val, timestamp=None):
        """
        :returns: None if activated, dict with offsets if deactivated.
        """
        sus_act = self._sus_activation(self.sus, val)
        self.sus = val
        #
        offsets = {}
        if sus_act == -1:
            offsets = self._update_offsets(timestamp)
        return offsets

    def update_ten(self, val, timestamp=None):
        """
        If tenuto pedal gets activated, downpressed notes are registered as
        tenuto. If it gets deactivated, all tenuto notes are removed.
        :returns: None if activated, dict with offsets if deactivated.
        """
        ten_act = self._ten_activation(self.ten, val)
        self.ten = val
        #
        offsets = {}
        if ten_act == -1:
            self.tenuto.clear()
            offsets = self._update_offsets(timestamp)
        elif ten_act == 1:
            # unlike sus, in ten we need to keep track of activated notes
            self.tenuto.update(self.downpressed)
        return offsets

    # KEYS
    def key_pressed(self, *key_vel_pairs, timestamp=None):
        """
        Add (or update) given key as pressed with given velocity. Pedals don't
        affect this operation.
        :param key_vel_pairs: Collection of ``(k, v)`` pairs to be pressed.
        :returns: A dictionary of offsets. Any notes that were in a resonant
          state are being re-triggered, therefore the previous note is
          considered to be offset at the same time.
        """
        offsets = {}
        # check if any of the given keys was already pressed
        for key, _ in key_vel_pairs:
            if key in self.downpressed:
                err_msg = f"Pressing a pressed key: {key}"
                if self.ignore_redundant_keypress:
                    # in this case, "lift" preexisting key and print a warning
                    print(f"WARNING: {err_msg}. Simulating lifting...")
                    end_ts = timestamp - self.pre_lifting_epsilon
                    lifted = self.key_lifted(key, timestamp=end_ts)
                    offsets.update(lifted)
                else:
                    raise RuntimeError(err_msg)

        for key, velocity in key_vel_pairs:
            ke = KeyEvent(key, velocity, onset_ts=timestamp)
            self.downpressed[key] = ke
            # if note was resonant, remove from there and return as offset
            if key in self.resonant:
                prev_ke = self.resonant[key]
                prev_ke.offset_ts = timestamp
                offsets[key] = prev_ke
                del self.resonant[key]
        #
        return offsets

    def key_lifted(self, *keys, timestamp=None):
        """
        :param keys: Collection of keys to be lifted
        :returns: A dictionary of ``{key: vel}`` pairs containing the offsets.
          Resonant keys will not be contained.
        """
        offsets = {}
        for key in keys:
            if key not in self.downpressed:
                # this triggers an exception, or a warning
                err_msg = f"Lifted key wasn't downpressed! {key}"
                if self.ignore_redundant_keylift:
                    print(f"WARNING: {err_msg}. Ignoring...")
                    continue  # do not handle this key further
                else:
                    raise RuntimeError(err_msg)
            # in this case, the key was downpressed, handle normally
            else:
                prev_ke = self.downpressed[key]
                del self.downpressed[key]
                #
                if (self._sus_active()) or (key in self.tenuto):
                    # in this case the key becomes resonant
                    assert key not in self.resonant, \
                        "Key was already resonant?"
                    self.resonant[key] = prev_ke
                else:
                    # in this case keylift is an offset, return
                    prev_ke.offset_ts = timestamp
                    offsets[key] = prev_ke
        #
        return offsets

    # OFFSET UPDATE
    def _update_offsets(self, timestamp=None):
        """
        This method is called when either sus or ten is lifted.
        It removes any resonant notes that aren't supported by any pedal.
        """
        sus_inactive = not self._sus_active()
        # offsets are any resonant notes that aren't supported by pedals
        offsets = {}
        for k, ke in self.resonant.items():
            assert k not in self.downpressed, \
                "Bug! resonant key can't be downpressed"
            # for each note, check if both tenuto and sus are missing
            if (k not in self.tenuto) and sus_inactive:
                ke.offset_ts = timestamp
                offsets[k] = ke
        # delete offsets from resonant
        for k in offsets:
            del self.resonant[k]
        #
        return offsets

    # MAIN GETTER
    def __call__(self):
        """
        :returns: A tuple representing the current state of the keyboard:
          ``(downpressed_dict, resonant_dict, sus_val, ten_val, soft_val)``
        """
        result = (self.downpressed.copy(), self.resonant.copy(),
                  self.sus, self.ten, self.soft)
        return result
