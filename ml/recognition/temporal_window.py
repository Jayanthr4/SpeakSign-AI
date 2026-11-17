"""Bounded per-signer landmark sequence buffer for future local sign models.

This module does not claim to classify ASL signs; it provides the temporal
input interface needed for trained sequence classifiers.
"""
from collections import deque

class TemporalWindow:
    def __init__(self, length=32, max_tracks=12):
        if length < 2 or max_tracks < 1:
            raise ValueError("length must be >=2 and max_tracks >=1")
        self.length = length
        self.max_tracks = max_tracks
        self._tracks = {}

    def update(self, track_id, hands, pose=None, face=None):
        if track_id not in self._tracks:
            if len(self._tracks) >= self.max_tracks:
                del self._tracks[next(iter(self._tracks))]
            self._tracks[track_id] = deque(maxlen=self.length)
        self._tracks[track_id].append({
            "hands": hands, "pose": pose or [], "face": face or [],
        })
        return {"ready": len(self._tracks[track_id]) == self.length,
                "frames": len(self._tracks[track_id]),
                "required_frames": self.length}

    def sequence(self, track_id):
        return list(self._tracks.get(track_id, ()))

    def clear(self, track_id=None):
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)
