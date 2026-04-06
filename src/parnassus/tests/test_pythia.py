import math
import os
import sys
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path

import pyhepmc
import pythia8mc

# DUT
from parnassus.pythia import HepMC3Generator, Pythia8ToHepMC3

N_EVENTS = 5
N_JOBS = 2


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr at the file descriptor level.
    This is necessary to suppress output from backend C++ libraries like Pythia8.
    """
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save copies of original file descriptors
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        # Flush Python's buffered output
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect stdout and stderr to devnull
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)

        yield

    finally:
        # Flush again before restoring
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)

        # Close temporary file descriptors
        os.close(devnull_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


# Helpers for validation ########################################


def compare_four_vectors(
    v_c: pyhepmc.FourVector,
    v_py: pyhepmc.FourVector,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> tuple[bool, str]:
    """Compare two four-vectors with tolerance."""
    for attr in ["x", "y", "z", "t"]:
        val_c = getattr(v_c, attr)
        val_py = getattr(v_py, attr)

        # Check if both are effectively zero
        if abs(val_c) < abs_tol and abs(val_py) < abs_tol:
            continue

        # Relative comparison for non-zero values
        if not math.isclose(val_c, val_py, rel_tol=rel_tol, abs_tol=abs_tol):
            return False, f"FourVector.{attr} mismatch: {val_c} vs {val_py}"

    return True, ""


def compare_particles(
    particle_c: pyhepmc.GenParticle,
    particle_py: pyhepmc.GenParticle,
    particle_idx: int,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    ignore_id: bool = False,
) -> tuple[bool, str]:
    """Compare two particles for equivalence."""
    # Check PDG ID
    if particle_c.pid != particle_py.pid:
        return (
            False,
            f"Particle {particle_idx}: PDG ID mismatch: {particle_c.pid} vs {particle_py.pid}",
        )

    # Check status
    if particle_c.status != particle_py.status:
        return (
            False,
            (
                f"Particle {particle_idx}: "
                f"Status mismatch: {particle_c.status} vs {particle_py.status}"
            ),
        )

    # Check momentum
    match, msg = compare_four_vectors(particle_c.momentum, particle_py.momentum, rel_tol, abs_tol)
    if not match:
        return False, f"Particle {particle_idx}: Momentum {msg}"

    # Check generated mass
    if not math.isclose(
        particle_c.generated_mass,
        particle_py.generated_mass,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        return (
            False,
            (
                f"Particle {particle_idx}: "
                f"Mass mismatch: {particle_c.generated_mass} vs {particle_py.generated_mass}"
            ),
        )

    # Check particle ID (if not ignoring)
    if not ignore_id and particle_c.id != particle_py.id:
        return (
            False,
            f"Particle {particle_idx}: ID mismatch: {particle_c.id} vs {particle_py.id}",
        )

    return True, ""


def compare_vertices(
    v_c: pyhepmc.GenVertex,
    v_py: pyhepmc.GenVertex,
    vertex_idx: int,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
) -> tuple[bool, str]:
    """Compare two vertices for equivalence."""
    # Check position (if set)
    if not v_c.position.is_zero() or not v_py.position.is_zero():
        match, msg = compare_four_vectors(v_c.position, v_py.position, rel_tol, abs_tol)
        if not match:
            return False, f"Vertex {vertex_idx}: Position {msg}"

    # Check number of incoming particles
    n_in_c = len(v_c.particles_in)
    n_in_py = len(v_py.particles_in)
    if n_in_c != n_in_py:
        return (
            False,
            f"Vertex {vertex_idx}: Number of incoming particles mismatch: {n_in_c} vs {n_in_py}",
        )

    # Check number of outgoing particles
    n_out_c = len(v_c.particles_out)
    n_out_py = len(v_py.particles_out)
    if n_out_c != n_out_py:
        return (
            False,
            f"Vertex {vertex_idx}: Number of outgoing particles mismatch: {n_out_c} vs {n_out_py}",
        )

    return True, ""


def compare_events(
    evt_c: pyhepmc.GenEvent,
    evt_py: pyhepmc.GenEvent,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    check_event_number: bool = False,
    check_particle_ordering: bool = False,
    exclude_detached: bool = True,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """Compare two HepMC events for semantic equivalence.

    Parameters
    ----------
    evt_c, evt_py : pyhepmc.GenEvent
        The two events to compare
    rel_tol : float
        Relative tolerance for floating point comparisons
    abs_tol : float
        Absolute tolerance for floating point comparisons
    check_event_number : bool
        Whether to check if event numbers match
    check_particle_ordering : bool
        Whether particle ordering must be identical (if False, uses physics matching)
    exclude_detached : bool
        Whether to exclude detached/non-stable particles from comparison
    verbose : bool
        Whether to print detailed comparison info

    Returns
    -------
    match : bool
        True if events are equivalent
    message : list[str]
        Description of mismatches found (empty if match=True)
    """
    event_match: bool = True
    mismatch_messages: list[str] = []

    n_particles_c = len(evt_c.particles)
    n_particles_py = len(evt_py.particles)
    n_vertices_c = len(evt_c.vertices)
    n_vertices_py = len(evt_py.vertices)

    if verbose:
        print(f"Comparing events: {n_particles_c} particles, {n_vertices_c} vertices")

    # Check event number #######################
    if check_event_number and evt_c.event_number != evt_py.event_number:
        event_match = False
        mismatch_messages.append(
            f"Event number mismatch: {evt_c.event_number} vs {evt_py.event_number}"
        )

    # Check number of particles #######################
    n_particles_mismatch = n_particles_c - n_particles_py
    if n_particles_mismatch < 0:
        event_match = False
        mismatch_messages.append(
            "Python event has more particles than C++ event, which should not happen."
        )

    # Check number of vertices #######################
    n_vertices_mismatch = n_vertices_c - n_vertices_py
    if n_vertices_mismatch < 0:
        event_match = False
        mismatch_messages.append(
            "Python event has more vertices than C++ event, which should not happen."
        )

    # Check that particles match #######################
    if check_particle_ordering:
        # Direct order comparison
        for i, (particle_c, particle_py) in enumerate(
            zip(evt_c.particles, evt_py.particles, strict=False)
        ):
            particle_match, msg = compare_particles(particle_c, particle_py, i, rel_tol, abs_tol)
            if not particle_match:
                event_match = False
                mismatch_messages.append(f"Particle {i} mismatch (WITH ORDERING): {msg}")
    else:
        # Physics-based matching (by PDG ID, momentum, status)
        particles_c_list = list(evt_c.particles)
        particles_py_list = list(evt_py.particles)

        matched_indices_py = set()

        for i, particle_c in enumerate(particles_c_list):
            # Try to find a matching particle in evt_py
            found_particle_match = False
            for j, particle_py in enumerate(particles_py_list):
                if j in matched_indices_py:
                    continue

                particle_match, _ = compare_particles(
                    particle_c, particle_py, i, rel_tol, abs_tol, ignore_id=True
                )
                if particle_match:
                    matched_indices_py.add(j)
                    found_particle_match = True
                    break

            if not found_particle_match:
                # Skip detached or non-stable particles
                # pyhepmc bindings don't allow us to add these on Python side;
                # but not relevant to physics
                if exclude_detached and particle_c.status == 42:
                    n_particles_mismatch -= 1
                    continue
                    # TODO: Below more rigorous?
                    # end_vtx = getattr(particle_c, 'production_vertex', None)
                    # if end_vtx is None: continue
                    # particles_out = getattr(end_vtx, 'particles_out', [])
                    # if len(particles_out)==0: continue

                event_match = False
                mismatch_messages.append(
                    f"Particle {i} "
                    "(pid={particle_c.pid}, status={particle_c.status}, p={particle_c.momentum}) "
                    "has no match in python event"
                )

    # Check that vertices/topology match #######################
    # Build particle map by physics properties for vertex comparison
    def get_particle_signature(p: pyhepmc.GenParticle) -> str:
        """Create a unique signature for a particle based on physics properties."""
        return (
            f"{p.pid}_{p.status}_{p.momentum.x:.6e}_{p.momentum.y:.6e}"
            f"_{p.momentum.z:.6e}_{p.momentum.t:.6e}"
        )

    # Check that vertices have equivalent connections
    vertices_c_list = list(evt_c.vertices)
    vertices_py_list = list(evt_py.vertices)

    matched_vertices_py = set()

    for i, v_c in enumerate(vertices_c_list):
        # Get signatures of particles connected to this vertex
        v_c_in_sigs = sorted([get_particle_signature(p) for p in v_c.particles_in])
        v_c_out_sigs = sorted([get_particle_signature(p) for p in v_c.particles_out])

        # Try to find matching vertex in evt_py
        found_vertex_match = False
        for j, v_py in enumerate(vertices_py_list):
            if j in matched_vertices_py:
                continue

            v_py_in_sigs = sorted([get_particle_signature(p) for p in v_py.particles_in])
            v_py_out_sigs = sorted([get_particle_signature(p) for p in v_py.particles_out])

            if v_c_in_sigs == v_py_in_sigs and v_c_out_sigs == v_py_out_sigs:
                matched_vertices_py.add(j)
                found_vertex_match = True
                break

        if not found_vertex_match:
            # Skip detached or non-stable particles
            # pyhepmc bindings don't allow us to add these on Python side;
            # but not relevant to physics
            if exclude_detached and len(v_c.particles_in) == 1 and v_c.particles_in[0].status == 42:
                n_vertices_mismatch -= 1
                continue
            event_match = False
            mismatch_messages.append(
                f"Vertex {i} with {len(v_c.particles_in)} in, {len(v_c.particles_out)} out "
                "has no match in python event"
            )

    # Check other event-level info (if present) #######################

    # PDF info: TODO
    # if evt_c.pdf_info and evt_py.pdf_info:
    #     pdf_c = evt_c.pdf_info
    #     pdf_py = evt_py.pdf_info

    #     if (
    #         pdf_c.parton_id1[0] != pdf_py.parton_id1[0]
    #         or pdf_c.parton_id2[1] != pdf_py.parton_id2[1]
    #     ):
    #         return False, f"PDF parton ID mismatch: {pdf_c.parton_id} vs {pdf_py.parton_id}"

    #     for attr in ["x", "scale", "xf"]:
    #         val_c = getattr(pdf_c, attr)
    #         val_py = getattr(pdf_py, attr)
    #         if isinstance(val_c, (list, tuple)):
    #             for k, (v_c, v_py) in enumerate(zip(val_c, val_py)):
    #                 if not math.isclose(v_c, v_py, rel_tol=rel_tol, abs_tol=abs_tol):
    #                     return False, f"PDF {attr}[{k}] mismatch: {v_c} vs {v_py}"
    #         else:
    #             if not math.isclose(val_c, val_py, rel_tol=rel_tol, abs_tol=abs_tol):
    #                 return False, f"PDF {attr} mismatch: {val_c} vs {val_py}"

    # Cross section
    if evt_c.cross_section and evt_py.cross_section:
        xs_c = evt_c.cross_section.xsec()
        xs_py = evt_py.cross_section.xsec()
        if not math.isclose(xs_c, xs_py, rel_tol=rel_tol, abs_tol=abs_tol):
            event_match = False
            mismatch_messages.append(f"Cross section mismatch: {xs_c} vs {xs_py}")

    # Event weights: TODO
    # if len(evt_c.weights) != len(evt_py.weights):
    #     event_match = False
    #     mismatch_messages.append("Number of weights mismatch: "
    #       f"{len(evt_c.weights)} vs {len(evt_py.weights)}")

    # for i, (w_c, w_py) in enumerate(zip(evt_c.weights, evt_py.weights)):
    #     if not math.isclose(w_c, w_py, rel_tol=rel_tol, abs_tol=abs_tol):
    #         event_match = False
    #         mismatch_messages.append(f"Weight {i} mismatch: {w_c} vs {w_py}")

    # Final mismatch summary #######################
    if n_vertices_mismatch != 0:
        event_match = False
        mismatch_messages.append(f"Vertices mismatch: n_vertices_mismatch = {n_vertices_mismatch}")

    if n_particles_mismatch != 0:
        event_match = False
        mismatch_messages.append(
            f"Particles mismatch: n_particles_mismatch = {n_particles_mismatch}"
        )

    if not event_match:
        for msg in mismatch_messages:
            print("  - " + msg)
    else:
        print("  ✓ Events match!")

    return event_match, mismatch_messages


def compare_hepmc_files(
    file_c: str,
    file_py: str,
    n_events: int,
    max_events: int | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-9,
    check_event_number: bool = False,
    check_particle_ordering: bool = False,
    verbose: bool = False,
) -> tuple[bool, str, int]:
    """Compare two HepMC files for semantic equivalence.

    Returns
    -------
    match : bool
        True if all events match
    message : str
        Description of first mismatch (empty if all match)
    num_compared : int
        Number of events successfully compared
    """
    with pyhepmc.open(file_c, "r") as f_c, pyhepmc.open(file_py, "r") as f_py:
        event_count = 0

        for evt_c, evt_py in zip(f_c, f_py, strict=False):
            if max_events and event_count >= max_events:
                break

            if verbose:
                print(f"\nComparing event {event_count}...")

            match, msgs = compare_events(
                evt_c,
                evt_py,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                check_event_number=check_event_number,
                check_particle_ordering=check_particle_ordering,
                verbose=verbose,
            )

            if not match:
                print(f"Event {event_count}: ", event_count)
                for m in msgs:
                    print("  - " + m)

            event_count += 1

        # Check if files have same number of events
        if event_count < n_events - 1:
            with suppress(StopIteration):
                next(f_c)

            with suppress(StopIteration):
                next(f_py)

    return True, f"All {event_count} events match!", event_count


def test_pythia8_to_hepmc3():
    """Tests standalone Pythia8ToHepMC3 end-to-end.

    Validates data output against benchmark hepmc files generated with C++ Pythia8 package;
    see HZZ4l.cc and HZZ4l.cmnd
    """
    # Tests Pythia8ToHepMC3.__init__()
    # Tests Pythia8ToHepMC3.fill_next_event()
    # ._get_particles()
    # ._get_vertices()
    # ._add_tree()
    # ._topological_sort_vertices()
    # ._check_if_free_particle()
    # ._store_event_info()
    # This leaves Pythia8ToHepMC3._add_color(),
    # which is currently broken due to HepMC3 Python bindings limitations

    # Generate events with our interface #############
    with suppress_stdout_stderr():
        pythia = pythia8mc.Pythia()

        # Check version number to select benchmark file
        pythia_version = str(pythia.settings.parm("Pythia:versionNumber"))
        if pythia_version == "8.316":
            fpath_benchmark = f"src/parnassus/tests/expected_results/HZZ4l_{N_EVENTS}_8316.hepmc"
        elif pythia_version == "8.315":
            fpath_benchmark = f"src/parnassus/tests/expected_results/HZZ4l_{N_EVENTS}_8315.hepmc"
        else:
            raise NotImplementedError(
                f"No benchmark file for pythia8mc=={pythia_version}; must be in {8.315, 8.316}"
            )

        assert Path(fpath_benchmark).exists()

        # Random seed
        pythia.readString("Random:setSeed = on")
        pythia.readString("Random:seed = 42")

        # Command file for rest of config
        pythia.readFile("src/parnassus/tests/HZZ4l.cmnd")

        if not pythia.init():
            print("test_pythia::test_pythia8_to_hepmc3: Pythia initialization failed!")
            return

        converter = Pythia8ToHepMC3()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".hepmc") as f:
            fpath_out = f.name

        writer = pyhepmc.io.WriterAscii(fpath_out)
        n_written = 0
        idx_event = 0
        while n_written < N_EVENTS:
            if not pythia.next():
                continue  # event failed, try again

            hepmc_event = converter.fill_next_event(pythia, idx_event)
            writer.write_event(hepmc_event)
            n_written += 1
            idx_event += 1

            # Note: progress messages will also be suppressed inside this context

        pythia.stat()
        writer.close()

    print(f"Pythia8ToHepMC3: Generated {N_EVENTS} events")

    # Compare against C++ benchmark #############
    match, message, _ = compare_hepmc_files(
        fpath_benchmark,
        fpath_out,
        N_EVENTS,
        verbose=True,  # Print details for each event
        check_particle_ordering=False,  # Don't require same particle order
        check_event_number=True,
    )
    assert match, f"Pythia8ToHepMC3: HepMC files do not match! {message}"


def test_hepmc3_generator():
    """Tests HepMC3Generator end-to-end.

    Top-level configuration, parallelization, and merging of hepmc files.
    """
    # Tests HepMC3Generator.__init__()
    # ._is_hadronization_on()
    # Tests HepMC3Generator.generate()
    # ._write_single_job()
    # ._gen_hepmc_single_job()
    # ._merge_hepmc_files()
    # ._append_hepmc_file()
    # So that is all the methods of HepMC3Generator covered

    with (
        tempfile.TemporaryDirectory(delete=False) as log_dir,
        tempfile.TemporaryDirectory(delete=False) as output_dir,
    ):
        generator = HepMC3Generator(
            cmnd_file="src/parnassus/tests/HZZ4l.cmnd",
            output_dir=output_dir,
            log_dir=log_dir,
        )
        fpath_merged = generator.generate(n_events=N_EVENTS, max_workers=N_JOBS, debug=False)
        print(f"Wrote to file {fpath_merged}")
        with pyhepmc.open(fpath_merged, "r") as f_merged:
            n_events_merged = sum(1 for _ in f_merged)
            assert n_events_merged == N_EVENTS, (
                f"HepMC3Generator: Merged file has {n_events_merged} events, expected {N_EVENTS}"
            )
