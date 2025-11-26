import hashlib

import pyhepmc
import pythia8mc

from parnassus.utils.logger import setup_logger

LOG = setup_logger()


class HashableGenVertex:
    """Hashable wrapper for pyhepmc.GenVertex objects.

    This wrapper implements __hash__ and __eq__ to allow HepMC GenVertex
    objects to be used as dictionary keys and set members by hashing a
    string representation of the kinematic and identification attributes
    of incoming and outgoing particles.

    Warning: hash depends on particle ordering!

    Attributes
    ----------
    vertex : pyhepmc.GenVertex
        The underlying HepMC3 GenVertex object being wrapped.
    """

    def __init__(self, pyhepmc_vertex: pyhepmc.GenVertex):
        """Example usage.

        test_map = {}
        for v in all_vertices_sorted:
            v_hashable = HashableGenVertex(v)
            print(f"v_hashable: {v_hashable}")
            test_map[v_hashable] = 0

        # Assert uniqueness of each hash
        assert(len(test_map.keys()) == len(all_vertices_sorted))

        for v in all_vertices_sorted:
            v_hashable = HashableGenVertex(v)
            k_of_v = test_map[v_hashable]
            print(f"Successfully indexed dictionary based on HashableGenVertex")

        """
        self.vertex = pyhepmc_vertex

    def _particle_to_str(self, particle: pyhepmc.GenParticle) -> str:
        string_out = ""
        for momentum_attribute in ["x", "y", "z", "px", "py", "pz", "t"]:
            string_out += f"{getattr(particle.momentum, momentum_attribute):.5e}_"

        string_out += f"{particle.generated_mass}_"
        string_out += f"{particle.pid}_"
        string_out += f"{particle.status}"
        return string_out

    def __hash__(self) -> int:
        particles = []
        if hasattr(self.vertex, "particles_in"):
            particles += list(self.vertex.particles_in)
        if hasattr(self.vertex, "particles_out"):
            particles += list(self.vertex.particles_out)
        if len(particles) == 0:
            raise ValueError("Vertex has no incoming or outgoing particles!")

        particles = [self._particle_to_str(p) for p in particles]
        hash_argument = "__".join(list(particles))
        return int(hashlib.md5(hash_argument.encode("utf-8")).hexdigest(), 16)  # noqa: S324

    def __eq__(self, other_object: object) -> bool:
        if not isinstance(other_object, HashableGenVertex):
            return False

        return self.__hash__() == other_object.__hash__()


class Pythia8ToHepMC3:
    """Convert Pythia8 events to HepMC3 GenEvent objects.

    This class converts a Pythia8 event representation (particles,
    vertices, and event-level information) into a pyhepmc.GenEvent,
    while optionally performing checks and storing metadata such as
    PDF info, cross-sections, and event weights.

    Parameters
    ----------
    m_hadronization_on : bool, optional
        Whether hadronization checks are enabled (default True).
    m_internal_event_number : int, optional
        Starting internal event number (default 0).
    m_print_inconsistency : bool, optional
        Whether to print inconsistency warnings (default True).
    m_free_parton_warnings : bool, optional
        Whether to warn about free partons (default True).
    m_crash_on_problem : bool, optional
        Whether to raise on problematic events (default False).
    m_convert_gluon_to_0 : bool, optional
        Convert gluon PDF ID (21) to 0 in stored PDF info (default False).
    m_store_pdf : bool, optional
        Whether to store PDF info (default True).
    m_store_proc : bool, optional
        Whether to store process-level attributes (default True).
    m_store_xsec : bool, optional
        Whether to store cross-section info (default True).
    m_store_weights : bool, optional
        Whether to store event weights (default True).
    """

    def __init__(
        self,
        m_hadronization_on: bool = True,
        m_internal_event_number: int = 0,
        m_print_inconsistency: bool = True,
        m_free_parton_warnings: bool = True,
        m_crash_on_problem: bool = False,
        m_convert_gluon_to_0: bool = False,
        m_store_pdf: bool = True,
        m_store_proc: bool = True,
        m_store_xsec: bool = True,
        m_store_weights: bool = True,
    ):
        self.m_hadronization_on = m_hadronization_on
        self.m_internal_event_number = m_internal_event_number
        self.m_print_inconsistency = m_print_inconsistency
        self.m_free_parton_warnings = m_free_parton_warnings
        self.m_crash_on_problem = m_crash_on_problem
        self.m_convert_gluon_to_0 = m_convert_gluon_to_0
        self.m_store_pdf = m_store_pdf
        self.m_store_proc = m_store_proc
        self.m_store_xsec = m_store_xsec
        self.m_store_weights = m_store_weights

    def fill_next_event(self, pythia: pythia8mc.Pythia, evt_num: int) -> pyhepmc.GenEvent:
        # 1. Initalize HepMC event #################

        hepmc_event = pyhepmc.GenEvent()
        hepmc_event.event_number = evt_num
        hepmc_event.set_units(pyhepmc.Units.GEV, pyhepmc.Units.MM)

        # 2. Fill particle information #################
        hepevt_particles = self._get_particles(pythia.event)

        # 3. Fill vertex information and find beam particles #################
        vertex_cache, beam_particles = self._get_vertices(pythia.event, hepevt_particles)

        # Reserve memory for the event
        hepmc_event.reserve(len(hepevt_particles), len(vertex_cache))

        # Add particles and vertices in topological order
        self._add_tree(hepmc_evt=hepmc_event, beam_particles=beam_particles)

        # Add color attributes to particles AFTER adding them to event
        # self._add_color(pythia.event, hepevt_particles)  # noqa: ERA001
        # TODO: Causes segmentation fault; requires custom HepMC3 bindings
        # (otherwise thread-locking not accessible)

        # 4. Check for particles which come from nowhere, #################
        # i.e. are without mothers or daughters. These need to be attached
        # to a vertex, or else they will never become part of the event.

        for i in range(pythia.event.size()):
            if not hepevt_particles[i].in_event:
                LOG.warning(
                    f"Pythia8ToHepMC3: Found detached particle; "
                    f"status = {hepevt_particles[i].status}, pid = {hepevt_particles[i].pid}"
                )
                # TODO: Below causes error; too few vertices when reading from hepmc file
                # prod_vtx = pyhepmc.GenVertex()  # noqa: ERA001
                # prod_vtx.add_particle_out(hepevt_particles[i])  # noqa: ERA001
                # hepmc_event.add_vertex(prod_vtx)  # noqa: ERA001

        # 5. Check for free partons #################

        if self.m_hadronization_on:
            for i in range(pythia.event.size()):
                if hepevt_particles[i].pid == 21 and self._check_if_free_particle(
                    hepevt_particles[i]
                ):
                    if self.m_crash_on_problem:
                        raise RuntimeError(
                            f"Error: Found final-state gluon with no end vertex! "
                            f"event {evt_num}, particle {i}"
                        )
                    if self.m_free_parton_warnings:
                        LOG.warning(
                            f"Pythia8ToHepMC3: Found final-state gluon with no end vertex! "
                            f"event {evt_num}, particle {i}"
                        )

                if abs(hepevt_particles[i].pid) <= 6 and self._check_if_free_particle(
                    hepevt_particles[i]
                ):
                    if self.m_crash_on_problem:
                        raise RuntimeError(
                            f"Error: Found final-state quark with no end vertex! "
                            f"event {evt_num}, particle {i}"
                        )
                    if self.m_free_parton_warnings:
                        LOG.warning(
                            f"Pythia8ToHepMC3: Found final-state quark with no end vertex! "
                            f"event {evt_num}, particle {i}"
                        )

        # 6. Store PDF, weight, cross section and other event information. #################
        self._store_event_info(pythia, hepmc_event)

        return hepmc_event

    def _get_particles(self, pythia_event: pythia8mc.Event) -> list[pyhepmc.GenParticle]:
        hepevt_particles = []
        for particle_idx in range(pythia_event.size()):
            pythia_particle = pythia_event[particle_idx]
            hepmc_particle = pyhepmc.GenParticle(
                pyhepmc.FourVector(
                    pythia_particle.px(),
                    pythia_particle.py(),
                    pythia_particle.pz(),
                    pythia_particle.e(),
                ),
                pythia_particle.id(),
                pythia_particle.statusHepMC(),
            )
            hepmc_particle.generated_mass = pythia_particle.m()
            hepevt_particles.append(hepmc_particle)
        return hepevt_particles

    def _get_vertices(
        self, pythia_event: pythia8mc.Event, hepevt_particles: list[pyhepmc.GenParticle]
    ) -> tuple[list[pyhepmc.GenVertex], list[pyhepmc.GenParticle]]:
        vertex_cache = []
        beam_particles = []
        for particle_idx in range(pythia_event.size()):
            pythia_particle = pythia_event[particle_idx]

            mother_indices = pythia_particle.motherList()
            mother_indices = sorted(mother_indices)

            while len(mother_indices) > 0 and mother_indices[0] == 0:
                mother_indices.pop(0)

            # If it has mother particle, produce a GenVertex for it
            if len(mother_indices) > 0:
                prod_vtx = hepevt_particles[mother_indices[0]].end_vertex

                if prod_vtx is None:
                    prod_vtx = pyhepmc.GenVertex()
                    for mother_idx in mother_indices:
                        prod_vtx.add_particle_in(hepevt_particles[mother_idx])
                    vertex_cache.append(prod_vtx)

                prod_pos = pyhepmc.FourVector(
                    pythia_particle.xProd(),
                    pythia_particle.yProd(),
                    pythia_particle.zProd(),
                    pythia_particle.tProd(),
                )

                # Update vertex position if necessary
                if (not prod_pos.is_zero()) and (prod_vtx.position.is_zero()):
                    prod_vtx.position = prod_pos

                prod_vtx.add_particle_out(hepevt_particles[particle_idx])
                """
                To confirm in-place-ness
                (i.e. confirm that updating prod_vtx indeed updates vertex_cache[-1]):

                import pyhepmc
                prt = pyhepmc.GenParticle(pyhepmc.FourVector(1,2,3,4), 6, 1)
                vtx = pyhepmc.GenVertex()

                vtx_list = []
                vtx_list.append(vtx)

                print(f"BEFORE updating vtx outside-of-list: \n"
                      f"{vtx_list=}\n{vtx_list[0].particles_in=}\n{vtx_list[0].position=}"
                )

                vtx.add_particle_in(prt)
                vtx.position = pyhepmc.FourVector(1,2,3,4)

                print(f"AFTER updating vtx outside-of-list: \n"
                      f"{vtx_list=}\n{vtx_list[0].particles_in=}\n{vtx_list[0].position=}"
                )
                """

            else:  # Otherwise, it's a beam particle
                beam_particles.append(hepevt_particles[particle_idx])

            if len(beam_particles) != 2 and self.m_crash_on_problem:
                raise NotImplementedError(f"Error: len(beam_particles) = {len(beam_particles)}")

        return vertex_cache, beam_particles

    def _add_tree(
        self,
        hepmc_evt: pyhepmc.GenEvent,
        beam_particles: list[pyhepmc.GenParticle],
    ):
        all_vertices_sorted = self._topological_sort_vertices(beam_particles)
        for v in all_vertices_sorted:
            hepmc_evt.add_vertex(v)

        # TODO: Validate root-vertex handling; requires custom HepMC3 bindings
        # (otherwise no attribute pyhepmc.GenEvent.m_root_vertex)

    def _topological_sort_vertices(
        self, beam_particles: list[pyhepmc.GenParticle]
    ) -> list[pyhepmc.GenVertex]:
        all_vertices_sorted = []
        vertices_processed = []  # Track which vertices we've already

        # Find all starting vertices (end vertices of particles that have no production vertex)
        vertex_queue = []
        for p in beam_particles:
            v_prod = getattr(p, "production_vertex", None)

            if not v_prod:  # If it has no production vertex
                v_end = getattr(p, "end_vertex", None)
                if v_end:
                    vertex_queue.append(v_end)

            else:  # Or if it does have production vertex
                v_prod_particles_in = getattr(v_prod, "particles_in", None)
                if (
                    not v_prod_particles_in or len(v_prod_particles_in) == 0
                ):  # But that production vertex has no input particles
                    v_end = getattr(p, "end_vertex", None)
                    if v_end:
                        vertex_queue.append(v_end)

        # Add vertices to the event in topological order
        while vertex_queue:
            current_vertex = vertex_queue.pop(0)
            if current_vertex in vertices_processed:
                continue

            # Add mothers to front of queue
            prerequisites_satisfied = True
            for p_in in current_vertex.particles_in:
                v_prod = p_in.production_vertex

                if (
                    (v_prod is not None)
                    and (v_prod not in vertices_processed)
                    and (v_prod not in vertex_queue)
                ):
                    vertex_queue = [v_prod, *vertex_queue]  # add to front of queue
                    prerequisites_satisfied = False

            # If we have added at least one production vertex,
            # our vertex is not the first one on the list
            if not prerequisites_satisfied:
                continue

            #  If vertex is not yet added
            if current_vertex not in vertices_processed:
                all_vertices_sorted.append(current_vertex)
                vertices_processed.append(current_vertex)

                # Add children to back of queue
                for p_out in current_vertex.particles_out:
                    v_end = p_out.end_vertex
                    if (
                        (v_end is not None)
                        and (v_end not in vertices_processed)
                        and (v_end not in vertex_queue)
                    ):
                        vertex_queue.append(v_end)

        return all_vertices_sorted

    def _check_if_free_particle(self, hepevt_particle: pyhepmc.GenParticle) -> bool:
        end_vertex = getattr(hepevt_particle, "end_vertex", None)
        if end_vertex is None:
            return True

        particles_out = getattr(end_vertex, "particles_out", None)
        if particles_out is None:
            return True
        return len(particles_out) == 0

    def _add_color(self, pythia_event: pythia8mc.Event, hepmc_particles: list[pyhepmc.GenParticle]):
        """To check in-place-ness.

        import pyhepmc
        p1 = pyhepmc.GenParticle()
        p2 = pyhepmc.GenParticle()
        test_evt = pyhepmc.GenEvent()
        test_evt.add_particle(p1)
        test_evt.add_particle(p2)
        print(f"BEFORE inplace-op: {test_evt.particles[0].attributes=}")
        p1.attributes["flow1"] = 0
        print(f"AFTER inplace-op: {test_evt.particles[0].attributes=}")
        """
        for i in range(pythia_event.size()):
            col_type = pythia_event[i].colType()
            if col_type in {-1, 1, 2}:
                flow1 = 0
                flow2 = 0
                if col_type in {1, 2}:
                    flow1 = pythia_event[i].col()
                if col_type in {-1, 2}:
                    flow2 = pythia_event[i].acol()

                hepmc_particles[i].attributes["flow1"] = flow1
                hepmc_particles[i].attributes["flow2"] = flow2

    def _store_event_info(self, pythia: pythia8mc.Pythia, hepmc_event: pyhepmc.GenEvent):
        pyinfo = pythia.infoPython()

        # PDF information
        id1pdf = pyinfo.id1pdf()
        id2pdf = pyinfo.id2pdf()
        if self.m_store_pdf:
            if self.m_convert_gluon_to_0:
                if id1pdf == 21:
                    id1pdf = 0
                if id2pdf == 21:
                    id2pdf = 0

            pdf_info = pyhepmc.GenPdfInfo(
                id1pdf,
                id2pdf,
                pyinfo.x1pdf(),
                pyinfo.x2pdf(),
                pyinfo.QFac(),
                pyinfo.pdf1(),
                pyinfo.pdf2(),
            )
            hepmc_event.pdf_info = pdf_info

        # Process code, scale, alpha_em, alpha_s
        if self.m_store_proc:
            hepmc_event.attributes["mpi"] = pyinfo.nMPI()
            hepmc_event.attributes["signal_process_id"] = pyinfo.code()
            hepmc_event.attributes["event_scale"] = pyinfo.QRen()
            hepmc_event.attributes["alphaQCD"] = pyinfo.alphaS()
            hepmc_event.attributes["alphaQED"] = pyinfo.alphaEM()

        # Cross-section information
        if self.m_store_xsec:
            xsec = pyhepmc.GenCrossSection()
            xsec.set_cross_section(pyinfo.sigmaGen() * 1e9, pyinfo.sigmaErr() * 1e9)
            hepmc_event.cross_section = xsec

        # Event weights
        # TODO: This doesn't save to file!
        if self.m_store_weights:
            hepmc_event.weights.clear()
            for i in range(pyinfo.nWeights()):
                hepmc_event.weights.append(pyinfo.weight(i))
