# MEH multi-source radiator impedance matrix.
#
# Written by M2 but NOT executed there (the task's cloud environment has no
# Julia/CUDA). The owner runs this with `julia/tests/runtests.jl` in M8. It
# covers the four properties C4 asks for: superposition, reciprocity,
# single-tag parity with the legacy impedance, and reduced-vs-full symmetry.

# The wire format nests one block per channel, and inside a block the pressure
# array is itself nested (planes x angles). Flatten each channel to a vector so
# the tests compare numbers, not shapes; the tolerances stay as written.
_flatten(x::Number) = Float32[x]
_flatten(x::AbstractArray) = reduce(vcat, (_flatten(v) for v in x); init = Float32[])
wire_rows(block) = [
    ComplexF32.(_flatten(re), _flatten(im)) for
    (re, im) in zip(block["real"], block["imag"])
]
force_from_pair(pair) = ComplexF64(2.0 * pair[1]) - 2.0im * ComplexF64(pair[2])

@testset "MEH radiator impedance matrix (helper algebra)" begin
    T = Float32
    vertices = [
        SVector{3,T}(0.0, 0.0, 0.0),
        SVector{3,T}(0.1, 0.0, 0.0),
        SVector{3,T}(0.0, 0.1, 0.0),
        SVector{3,T}(0.0, 0.0, 0.1),
    ]
    mesh = BoundaryMesh(
        vertices,
        [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
        [2, 10, 1, 1],
    )
    element_mesh_ids = ones(Int, length(mesh.faces))
    radiators = [
        Dict{String,Any}(
            "name" => "tag002", "tag" => 2, "mesh_id" => 1,
            "channel" => "tag002", "velocity_offset_db" => 0.0,
        ),
        Dict{String,Any}(
            "name" => "tag010", "tag" => 10, "mesh_id" => 1,
            "channel" => "tag010", "velocity_offset_db" => 0.0,
        ),
    ]
    p1 = Complex{T}[1.0 + 0.1im, 2.0 - 0.2im, 0.5 + 0.3im, -0.7 + 0.4im]
    p2 = Complex{T}[0.3 - 0.1im, -0.4 + 0.2im, 1.1 + 0.05im, 0.2 - 0.6im]

    # Single-tag parity: with one channel and drive 1 the matrix entry is
    # exactly the legacy impedance pair (same force scale and x10 packing).
    legacy = impedance_for_radiators(
        mesh, element_mesh_ids, p1, [radiators[1]], Complex{T}[1.0], T;
        symmetry_mode=:off,
    )
    matrix_one = impedance_matrix_for_channels(
        mesh, element_mesh_ids, [p1], [radiators[1]], T; symmetry_mode=:off,
    )
    @test matrix_one[1][1] == legacy[1]

    # Superposition: the matrix is linear in the boundary pressure.
    m1 = impedance_matrix_for_channels(
        mesh, element_mesh_ids, [p1], radiators, T; symmetry_mode=:off,
    )
    m2 = impedance_matrix_for_channels(
        mesh, element_mesh_ids, [p2], radiators, T; symmetry_mode=:off,
    )
    m_sum = impedance_matrix_for_channels(
        mesh, element_mesh_ids, [p1 .+ p2], radiators, T; symmetry_mode=:off,
    )
    for i in 1:2
        @test m_sum[1][i][1] ≈ m1[1][i][1] + m2[1][i][1] rtol = 1.0f-6
        @test m_sum[1][i][2] ≈ m1[1][i][2] + m2[1][i][2] rtol = 1.0f-6
    end
end

@testset "MEH radiator impedance matrix (symmetry scaling)" begin
    T = Float32
    # Two triangles per side, mirrored across x = 0. Tag 2 is the first pair,
    # tag 10 the second. The full model sees both mirror copies; the reduced
    # model keeps the x >= 0 half and asks for the :x symmetry factor.
    a = [SVector{3,T}(0.1, 0.0, 0.0), SVector{3,T}(0.2, 0.0, 0.0), SVector{3,T}(0.1, 0.1, 0.0)]
    am = [SVector{3,T}(-p[1], p[2], p[3]) for p in a]
    b = [SVector{3,T}(0.1, 0.0, 0.3), SVector{3,T}(0.2, 0.0, 0.3), SVector{3,T}(0.1, 0.1, 0.3)]
    bm = [SVector{3,T}(-p[1], p[2], p[3]) for p in b]
    full = BoundaryMesh(
        vcat(a, am, b, bm),
        [(1, 2, 3), (4, 6, 5), (7, 8, 9), (10, 12, 11)],
        [2, 2, 10, 10],
    )
    reduced = BoundaryMesh(
        vcat(a, b),
        [(1, 2, 3), (4, 5, 6)],
        [2, 10],
    )
    full_ids = ones(Int, length(full.faces))
    reduced_ids = ones(Int, length(reduced.faces))
    radiators = [
        Dict{String,Any}("name" => "tag002", "tag" => 2, "mesh_id" => 1,
                         "channel" => "tag002", "velocity_offset_db" => 0.0),
        Dict{String,Any}("name" => "tag010", "tag" => 10, "mesh_id" => 1,
                         "channel" => "tag010", "velocity_offset_db" => 0.0),
    ]
    # Mirror-symmetric pressure: the two halves carry the same values.
    p_half = Complex{T}[1.0 - 0.3im, -0.5 + 0.2im, 0.4 + 0.6im]
    q_half = Complex{T}[0.2 + 0.1im, 0.7 - 0.4im, -0.1 + 0.05im]
    p_full = vcat(p_half, p_half, q_half, q_half)
    p_reduced = vcat(p_half, q_half)

    full_matrix = impedance_matrix_for_channels(
        full, full_ids, [p_full, p_full], radiators, T; symmetry_mode=:off,
    )
    reduced_matrix = impedance_matrix_for_channels(
        reduced, reduced_ids, [p_reduced, p_reduced], radiators, T; symmetry_mode=:x,
    )
    for j in 1:2, i in 1:2, part in 1:2
        @test full_matrix[j][i][part] ≈ reduced_matrix[j][i][part] rtol = 1.0f-6
    end
end

@testset "MEH radiator impedance matrix (driver-level reciprocity)" begin
    T = Float32
    # A regular tetrahedron of edge 0.2 m. Every face has the same area, and
    # two of them are driven on separate channels; the remaining two are rigid.
    s = 0.2
    h = sqrt(3.0) / 2.0 * s
    z = sqrt(2.0 / 3.0) * s
    vertices = [
        (0.0, 0.0, 0.0),
        (s, 0.0, 0.0),
        (s / 2.0, h, 0.0),
        (s / 2.0, h / 3.0, z),
    ]
    # Outward-oriented faces: base, (1,2,4) tag 2, (2,3,4) tag 10, (3,1,4).
    faces = [(1, 3, 2, 1), (1, 2, 4, 2), (2, 3, 4, 10), (3, 1, 4, 1)]

    function write_two_tag_tetrahedron(path)
        open(path, "w") do io
            println(io, "\$MeshFormat")
            println(io, "2.2 0 8")
            println(io, "\$EndMeshFormat")
            println(io, "\$PhysicalNames")
            println(io, "3")
            println(io, "2 1 \"wall\"")
            println(io, "2 2 \"entry\"")
            println(io, "2 10 \"entry_right\"")
            println(io, "\$EndPhysicalNames")
            println(io, "\$Nodes")
            println(io, length(vertices))
            for (index, vertex) in enumerate(vertices)
                println(io, "$index $(vertex[1]) $(vertex[2]) $(vertex[3])")
            end
            println(io, "\$EndNodes")
            println(io, "\$Elements")
            println(io, length(faces))
            for (index, (a, b, c, tag)) in enumerate(faces)
                println(io, "$index 2 2 $tag $tag $a $b $c")
            end
            println(io, "\$EndElements")
        end
        return path
    end

    function driver_result(mesh_path, radiators)
        request = Dict{String,Any}(
            "schema_version" => 2,
            "beat_engine_backend" => "cpu",
            "frequencies_hz" => [100.0],
            "config" => Dict{String,Any}(
                "mesh_file" => mesh_path,
                "scale_factor" => 1.0,
                "distance" => 1.0,
                "axial_offset" => 0.0,
                "step_size" => 90.0,
                "min_angle" => 0.0,
                "max_angle" => 90.0,
                "freq_min" => 100.0,
                "freq_max" => 100.0,
                "freq_count" => 1,
                "tag_throat" => 2,
                "rho" => 1.2041,
                "sound_speed" => 343.0,
                "symmetry" => "off",
                "source_motion" => "normal",

                "flat_target_normalization_enabled" => false,
                "impedance_matrix_enabled" => true,
                "radiators" => radiators,
            ),
        )
        return mktempdir() do directory
            event_path = joinpath(directory, "events.jsonl")
            open(event_path, "w") do io
                redirect_stdout(io) do
                    solve_request(request)
                end
            end
            events = [JSON.parse(line) for line in eachline(event_path) if startswith(strip(line), "{")]
            results = [event for event in events if get(event, "type", "") == "result"]
            length(results) == 1 ||
                error("Expected exactly one result event, got $(length(results)).")
            results[1]["result"]
        end
    end

    radiators_two = [
        Dict{String,Any}("name" => "tag002", "tag" => 2, "channel" => "tag002"),
        Dict{String,Any}("name" => "tag010", "tag" => 10, "channel" => "tag010"),
    ]
    radiators_one = [
        Dict{String,Any}("name" => "tag002", "tag" => 2, "channel" => "main"),
        Dict{String,Any}("name" => "tag010", "tag" => 10, "channel" => "main"),
    ]

    mktempdir() do directory
        mesh_path = write_two_tag_tetrahedron(joinpath(directory, "two_tags.msh"))
        pair = driver_result(mesh_path, radiators_two)
        combined = driver_result(mesh_path, radiators_one)
        one = driver_result(mesh_path, [radiators_two[1]])

        @test pair["channel_names"] == ["tag002", "tag010"]

        # Superposition: the two channel rows sum to the single combined row,
        # both in the observed pressure and in the force matrix.
        pair_h = reduce(+, wire_rows(pair["horizontal_pressure"]))
        combined_h = reduce(+, wire_rows(combined["horizontal_pressure"]))
        @test pair_h ≈ combined_h rtol = 1.0f-5
        pair_v = reduce(+, wire_rows(pair["vertical_pressure"]))
        combined_v = reduce(+, wire_rows(combined["vertical_pressure"]))
        @test pair_v ≈ combined_v rtol = 1.0f-5

        pair_forces = [
            [force_from_pair(entry) for entry in row] for row in pair["impedance_matrix"]
        ]
        combined_forces = [
            [force_from_pair(entry) for entry in row] for row in combined["impedance_matrix"]
        ]
        for i in 1:2
            @test pair_forces[1][i] + pair_forces[2][i] ≈ combined_forces[1][i] rtol = 1.0f-5
        end

        # Reciprocity: F_ij == F_ji for equal-area patches on a rigid body.
        scale = max(maximum(abs, reduce(vcat, pair_forces)), eps(Float64))
        @test abs(pair_forces[1][2] - pair_forces[2][1]) / scale < 1.0f-3

        # Single-tag parity: the matrix diagonal is the legacy impedance pair.
        @test one["impedance_matrix"][1][1] ≈ one["impedance"][1] rtol = 1.0f-6
    end
end
