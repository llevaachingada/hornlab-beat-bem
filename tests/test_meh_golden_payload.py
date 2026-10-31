"""The one-tag, no-matrix request and result are frozen.

M2 adds multi-source fields to the same code path. The hard requirement is that
the default stays byte-identical: one tag and no ``impedance_matrix`` must put
exactly today's keys on the wire and return exactly today's arrays. The payload
below is copied from the package before the change (a literal, not a value
recomputed from the implementation), so a rename, a new key or a changed number
fails here rather than in a consumer.

The second half pins the one-channel result path: the parser now stacks channel
rows, and at C == 1 that must still produce the same pressure, SPL and
impedance, with every new field left ``None``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hornlab_beat_bem import SolveConfig
from hornlab_beat_bem import sweep
from hornlab_beat_bem.sweep import _WARMUP_TETRAHEDRON, _request_payload, solve_frequencies

#: Exactly what ``_request_payload`` produced for a default config on the
#: pinned single-source package (760a1ec), with one 1 kHz frequency.
_GOLDEN_DEFAULT_PAYLOAD = {
    "schema_version": 2,
    "config": {
        "mesh_file": "mesh.msh",
        "scale_factor": 1.0,
        "meshes": [
            {
                "name": "mesh",
                "file": "mesh.msh",
                "scale_factor": 1.0,
                "translation_m": [0.0, 0.0, 0.0],
            }
        ],
        "distance": 2.0,
        "axial_offset": 0.0,
        "step_size": 5.0,
        "min_angle": 0.0,
        "max_angle": 180.0,
        "freq_min": 1000.0,
        "freq_max": 1000.0,
        "freq_count": 1,
        "tag_throat": 2,
        "rho": 1.2041,
        "sound_speed": 343.0,
        "symmetry": "off",
        "flat_target_normalization_enabled": False,
        "spherical_sampling_enabled": False,
        "source_motion": "normal",
        "quadrature_order": 4,
        "singular_order": 4,
    },
    "frequencies_hz": [1000.0],
}


def test_the_default_request_is_the_frozen_golden_payload():
    payload = _request_payload(
        "mesh.msh", np.asarray([1000.0]), SolveConfig(), (0.0, 0.0, 0.0)
    )

    assert payload == _GOLDEN_DEFAULT_PAYLOAD
    # Byte-level too: a value that compares equal but serialises differently
    # (1 vs 1.0, a tuple that became a list) is a payload change.
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        _GOLDEN_DEFAULT_PAYLOAD, sort_keys=True
    )


def _one_channel_result_event(frequency_hz: float) -> dict:
    row = [[1.0, 0.25]]
    block = {"real": row, "imag": [[0.0, -0.5]]}
    return {
        "type": "result",
        "result": {
            "freq_hz": frequency_hz,
            "channel_names": ["main"],
            "horizontal_pressure": block,
            "vertical_pressure": block,
            # [Re(F)/2, -Im(F)/2] with F = 10 * area * p_velocity.
            "impedance": [[5.0, -2.5]],
            "timings": {"assembly_s": 0.1, "solve_s": 0.2, "field_s": 0.3},
        },
    }


class _FakeSession:
    def __init__(self, events, metadata):
        self._events = events
        self.metadata = metadata
        self.closed = 0
        self.cancel_requests = 0

    def events(self):
        yield from self._events

    def request_cancel(self) -> None:
        self.cancel_requests += 1

    def close(self) -> None:
        self.closed += 1


def _run_one_channel(monkeypatch, tmp_path):
    mesh_path = tmp_path / "tetra.msh"
    mesh_path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")
    session = _FakeSession(
        [_one_channel_result_event(1000.0), {"type": "completed"}],
        {"polar_angle_deg": [0.0, 90.0]},
    )
    monkeypatch.setattr(sweep, "discover_julia", lambda *a, **k: "/fake/julia")
    monkeypatch.setattr(sweep, "BeatSolveSession", lambda *a, **k: session)
    return solve_frequencies(mesh_path, [1000.0], SolveConfig())


def test_the_one_channel_result_path_is_unchanged_and_new_fields_are_none(
    monkeypatch, tmp_path
):
    result = _run_one_channel(monkeypatch, tmp_path)
    omega = 2.0 * math.pi * 1000.0
    acceleration_scale = 1.0 / (-1j * omega)

    expected_pressure = np.asarray(
        [[1.0 + 0.0j, 0.25 - 0.5j], [1.0 + 0.0j, 0.25 - 0.5j]]
    ) * acceleration_scale
    assert result.pressure_complex.shape == (1, 2, 2)
    assert np.allclose(result.pressure_complex[0], expected_pressure)
    assert np.allclose(
        result.spl_db[0],
        20.0 * np.log10(np.abs(expected_pressure) / 20e-6),
    )

    # F = 10 * area * p_velocity. The 0.08 m warmup tetrahedron's tag 2 has
    # three right faces of 0.5*0.08^2 and one equilateral face of side 0.08*sqrt(2).
    area = 3.0 * 0.5 * 0.08 * 0.08 + math.sqrt(3.0) / 4.0 * (0.08 * math.sqrt(2.0)) ** 2
    expected_impedance = (10.0 + 5.0j) / (10.0 * area) * acceleration_scale
    assert result.impedance[0] == pytest.approx(expected_impedance)

    assert result.channel_tags is None
    assert result.channel_pressure is None
    assert result.channel_sphere_pressure is None
    assert result.impedance_matrix is None
    assert result.tag_areas_m2 is None
