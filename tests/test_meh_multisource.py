"""MEH multi-source request and result parsing, with no Julia anywhere.

These pin the Python half of C4: the payload shape for 2..8 tags, the
zero-padded channels the engine's lexicographic sort sees, the per-channel
arrays, and the [f, i, j] unpacking of the radiator force matrix. The Julia
side (superposition, reciprocity, single-tag parity, symmetry) is written in
``hornlab_beat_bem/julia/tests/impedance_matrix_tests.jl`` and runs in M8.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hornlab_beat_bem import ObservationConfig, SolveConfig
from hornlab_beat_bem import sweep
from hornlab_beat_bem.sweep import _request_payload, solve_frequencies

# Two triangles, areas 0.005 m^2 each, on tags 2 and 10.
_TWO_TAG_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 2 "entry_mid"
2 10 "entry_mid_right"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 0.1 0.0 0.0
3 0.0 0.1 0.0
4 0.1 0.1 0.0
$EndNodes
$Elements
2
1 2 2 2 2 1 2 3
2 2 2 10 10 2 4 3
$EndElements
"""

_ANGLES = [0.0, 90.0]


def _block(real, imag=None) -> dict:
    real = np.asarray(real, dtype=float)
    imag = np.zeros_like(real) if imag is None else np.asarray(imag, dtype=float)
    return {"real": real.tolist(), "imag": imag.tolist()}


def _two_channel_result(frequency_hz: float = 1000.0) -> dict:
    return {
        "type": "result",
        "result": {
            "freq_hz": frequency_hz,
            "channel_names": ["tag002", "tag010"],
            "horizontal_pressure": _block(
                [[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.25], [0.75, 1.25]]
            ),
            "vertical_pressure": _block(
                [[5.0, 6.0], [7.0, 8.0]], [[0.0, -0.1], [-0.2, -0.3]]
            ),
            "impedance": [[5.0, 2.5], [15.0, 7.5]],
            "impedance_matrix": [
                [[5.0, 2.5], [15.0, 7.5]],
                [[15.0, 7.5], [25.0, 12.5]],
            ],
            "sphere_pressure": _block(
                [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
            ),
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


def _metadata_with_sphere() -> dict:
    theta_deg = np.repeat(np.asarray([0.0, 180.0]), 3)
    phi_deg = np.tile(np.asarray([0.0, 120.0, 240.0]), 2)
    return {
        "polar_angle_deg": list(_ANGLES),
        "sphere_metadata": {
            "theta_polar_rad": np.radians(theta_deg).tolist(),
            "phi_azimuth_rad": np.radians(phi_deg).tolist(),
        },
    }


def _run(monkeypatch, mesh_text, config, events, metadata=None):
    import tempfile
    from pathlib import Path

    directory = Path(tempfile.mkdtemp(prefix="meh-ms-"))
    mesh_path = directory / "mesh.msh"
    mesh_path.write_text(mesh_text, encoding="utf-8")
    session = _FakeSession(
        events,
        _metadata_with_sphere() if metadata is None else metadata,
    )
    monkeypatch.setattr(sweep, "discover_julia", lambda *a, **k: "/fake/julia")
    monkeypatch.setattr(sweep, "BeatSolveSession", lambda *a, **k: session)
    return solve_frequencies(mesh_path, [1000.0], config), session


def _config(**kwargs) -> SolveConfig:
    defaults = dict(
        velocity_sources={2: 1.0, 10: 1.0},
        impedance_matrix=True,
        observation=ObservationConfig(
            planes=["horizontal", "vertical"],
            distance_m=1.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=2,
            sphere_grid=(2, 3),
        ),
    )
    defaults.update(kwargs)
    return SolveConfig(**defaults)


# ---------------------------------------------------------------- config shape


def test_source_tags_are_sorted_and_the_legacy_tag_is_the_smallest():
    config = SolveConfig(velocity_sources={10: 1.0, 2: 1.0})
    assert config.source_tags == (2, 10)
    assert config.source_tag == 2


def test_two_to_eight_tags_are_accepted_and_nine_are_not():
    for count in range(2, 9):
        tags = [2] + list(range(10, 10 + count - 1))
        assert SolveConfig(velocity_sources={tag: 1.0 for tag in tags}).source_tags == tuple(tags)
    with pytest.raises(NotImplementedError, match="one and eight"):
        SolveConfig(velocity_sources={tag: 1.0 for tag in range(2, 11)})
    with pytest.raises(NotImplementedError, match="one and eight"):
        SolveConfig(velocity_sources={})


def test_every_tag_keeps_the_unit_and_real_amplitude_refusals():
    with pytest.raises(NotImplementedError, match="unit source amplitude"):
        SolveConfig(velocity_sources={2: 1.0, 10: 0.5})
    with pytest.raises(NotImplementedError, match="complex"):
        SolveConfig(velocity_sources={2: 1.0, 10: 1.0 + 0.5j})
    with pytest.raises(ValueError, match="real number"):
        SolveConfig(velocity_sources={2: 1.0, 10: "1.0j"})


# ------------------------------------------------------------------- payload


def test_multi_tag_payload_carries_zero_padded_radiators_in_tag_order():
    config = SolveConfig(velocity_sources={10: 1.0, 2: 1.0})
    config_block = _request_payload(
        "mesh.msh", np.asarray([1000.0]), config, (0.0, 0.0, 0.0)
    )["config"]

    assert config_block["tag_throat"] == 2
    assert config_block["radiators"] == [
        {"name": "tag002", "tag": 2, "channel": "tag002"},
        {"name": "tag010", "tag": 10, "channel": "tag010"},
    ]
    assert "channels" not in config_block
    assert "impedance_matrix_enabled" not in config_block


def test_the_matrix_flag_is_opt_in_and_adds_one_key():
    default = _request_payload(
        "mesh.msh", np.asarray([1000.0]), SolveConfig(), (0.0, 0.0, 0.0)
    )["config"]
    enabled = _request_payload(
        "mesh.msh",
        np.asarray([1000.0]),
        SolveConfig(impedance_matrix=True),
        (0.0, 0.0, 0.0),
    )["config"]
    assert "impedance_matrix_enabled" not in default
    assert enabled["impedance_matrix_enabled"] is True


def test_one_tag_puts_no_radiators_key_on_the_wire():
    config = SolveConfig(velocity_sources={10: 1.0})
    config_block = _request_payload(
        "mesh.msh", np.asarray([1000.0]), config, (0.0, 0.0, 0.0)
    )["config"]
    assert "radiators" not in config_block
    assert config_block["tag_throat"] == 10


# ------------------------------------------------------------------- parsing


def test_two_channel_reply_unpacks_pressure_matrix_areas_and_legacy_fields(
    monkeypatch,
):
    events = [_two_channel_result(), {"type": "completed"}]
    result, session = _run(monkeypatch, _TWO_TAG_MESH, _config(), events)
    assert session.closed == 1

    assert result.channel_tags == (2, 10)
    assert result.channel_pressure.shape == (1, 2, 2, 2)
    assert result.channel_sphere_pressure.shape == (1, 2, 6)
    assert result.impedance_matrix.shape == (1, 2, 2)
    assert result.tag_areas_m2 == {2: pytest.approx(0.005), 10: pytest.approx(0.005)}

    omega = 2.0 * math.pi * 1000.0
    acceleration_scale = 1.0 / (-1j * omega)

    # Channel 0 is tag 2 and channel 1 is tag 10, in the engine's sorted order.
    assert result.channel_pressure[0, 0, 0] == pytest.approx(
        np.asarray([1.0 + 0.5j, 2.0 + 0.25j]) * acceleration_scale
    )
    assert result.channel_pressure[0, 1, 0] == pytest.approx(
        np.asarray([3.0 + 0.75j, 4.0 + 1.25j]) * acceleration_scale
    )
    # Legacy fields are the smallest tag's channel.
    assert result.pressure_complex[0] == pytest.approx(result.channel_pressure[0, 0])
    assert result.sphere_pressure_complex[0] == pytest.approx(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * acceleration_scale
    )
    assert result.channel_sphere_pressure[0, 1] == pytest.approx(
        np.asarray([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]) * acceleration_scale
    )

    # [f, i, j]: i is the receiver tag, j the driven channel. Each wire pair is
    # [Re(F)/2, -Im(F)/2] with F = 10 * symmetry * reduced_area_i * p_velocity.
    expected = np.asarray(
        [
            [10.0 - 5.0j, 30.0 - 15.0j],
            [30.0 - 15.0j, 50.0 - 25.0j],
        ]
    ) / (10.0 * 0.005) * acceleration_scale
    assert result.impedance_matrix[0] == pytest.approx(expected)
    # Legacy impedance is channel 0's own-tag matrix diagonal.
    assert result.impedance[0] == pytest.approx(result.impedance_matrix[0, 0, 0])
    assert result.impedance[0] == pytest.approx(
        (10.0 - 5.0j) / (10.0 * 0.005) * acceleration_scale
    )


def test_a_wrongly_named_channel_row_is_refused(monkeypatch):
    event = _two_channel_result()
    event["result"]["channel_names"] = ["tag002", "tag011"]
    with pytest.raises(RuntimeError, match="zero-padded channel names"):
        _run(monkeypatch, _TWO_TAG_MESH, _config(), [event, {"type": "completed"}])


def test_a_missing_tag_area_is_refused_before_a_session_is_built(
    monkeypatch, tmp_path
):
    # The one-tag tetrahedron has no tag 10.
    from hornlab_beat_bem.sweep import _WARMUP_TETRAHEDRON

    mesh_path = tmp_path / "one_tag.msh"
    mesh_path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")

    def refuse_to_build(*args, **kwargs):
        raise AssertionError("a session was built before tag areas were checked")

    monkeypatch.setattr(sweep, "discover_julia", lambda *a, **k: "/fake/julia")
    monkeypatch.setattr(sweep, "BeatSolveSession", refuse_to_build)
    with pytest.raises(ValueError, match="no triangles with source tag 10"):
        solve_frequencies(mesh_path, [1000.0], _config())


def test_a_requested_matrix_that_the_solver_omits_is_refused(monkeypatch):
    event = _two_channel_result()
    del event["result"]["impedance_matrix"]
    with pytest.raises(RuntimeError, match="returned none"):
        _run(monkeypatch, _TWO_TAG_MESH, _config(), [event, {"type": "completed"}])


def test_one_tag_with_a_matrix_populates_the_same_fields_at_c_one(
    monkeypatch, tmp_path
):
    from hornlab_beat_bem.sweep import _WARMUP_TETRAHEDRON

    mesh_path = tmp_path / "one_tag.msh"
    mesh_path.write_text(_WARMUP_TETRAHEDRON, encoding="utf-8")
    row = [[1.0, 2.0]]
    event = {
        "type": "result",
        "result": {
            "freq_hz": 1000.0,
            "channel_names": ["main"],
            "horizontal_pressure": _block(row),
            "vertical_pressure": _block(row),
            "impedance": [[5.0, -2.5]],
            "impedance_matrix": [[[5.0, -2.5]]],
            "timings": {},
        },
    }
    session = _FakeSession([event, {"type": "completed"}], {"polar_angle_deg": list(_ANGLES)})
    monkeypatch.setattr(sweep, "discover_julia", lambda *a, **k: "/fake/julia")
    monkeypatch.setattr(sweep, "BeatSolveSession", lambda *a, **k: session)

    config = SolveConfig(
        impedance_matrix=True,
        observation=ObservationConfig(
            planes=["horizontal"],
            distance_m=1.0,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=2,
        ),
    )
    result = solve_frequencies(mesh_path, [1000.0], config)

    assert result.channel_tags == (2,)
    assert result.channel_pressure.shape == (1, 1, 1, 2)
    assert result.impedance_matrix.shape == (1, 1, 1)
    assert result.impedance[0] == pytest.approx(result.impedance_matrix[0, 0, 0])
    area = 3.0 * 0.5 * 0.08 * 0.08 + math.sqrt(3.0) / 4.0 * (0.08 * math.sqrt(2.0)) ** 2
    assert result.tag_areas_m2 == {2: pytest.approx(area)}
