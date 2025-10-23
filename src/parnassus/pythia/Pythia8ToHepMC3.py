import hashlib

import pyhepmc


class HashableGenVertex:
    def __init__(self, pyhepmc_vertex: pyhepmc.GenVertex):
        """Example usage:

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

    def _particle_to_str(self, particle):
        string_out = ""
        for momentum_attribute in ["x", "y", "z", "px", "py", "pz", "t"]:
            string_out += f"{getattr(particle.momentum, momentum_attribute):.5e}_"

        string_out += f"{particle.generated_mass}_"
        string_out += f"{particle.pid}_"
        string_out += f"{particle.status}"
        return string_out

    def __hash__(self):
        particles = []
        if hasattr(self.vertex, "particles_in"):
            particles += list(self.vertex.particles_in)
        if hasattr(self.vertex, "particles_out"):
            particles += list(self.vertex.particles_out)
        if len(particles) == 0:
            raise ValueError("Vertex has no incoming or outgoing particles!")

        particles = [self._particle_to_str(p) for p in particles]
        hash_argument = "__".join(list(particles))
        hash_ = int(hashlib.md5(hash_argument.encode("utf-8")).hexdigest(), 16)

        return hash_

    def __eq__(self, other_object):
        if not isinstance(other_object, HashableGenVertex):
            return False

        return self.__hash__() == other_object.__hash__()


class Pythia8ToHepMC3:
    def __init__(
        self,
        m_hadronization_on=True,
        m_internal_event_number=0,
        m_print_inconsistency=True,
        m_free_parton_warnings=True,
        m_crash_on_problem=False,
        m_convert_gluon_to_0=False,
        m_store_pdf=True,
        m_store_proc=True,
        m_store_xsec=True,
        m_store_weights=True,
        m_detect_cycles=False,
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
        self.m_detect_cycles = m_detect_cycles

    def fill_next_event(self, pythia, evt_num):
        # 1. Initalize HepMC event #################

        # Error if no event passed
        if not pythia.next():
            print(f"Skipping event {evt_num}; no Pythia event generated")
            return None

        hepmc_event = pyhepmc.GenEvent()
        hepmc_event.event_number = evt_num
        hepmc_event.set_units(pyhepmc.Units.GEV, pyhepmc.Units.MM)

        # 2. Fill particle information #################
        hepevt_particles = self.get_particles(pythia.event)

        # 3. Fill vertex information and find beam particles #################
        vertex_cache, beam_particles = self.get_vertices(pythia.event, hepevt_particles)

        # Reserve memory for the event
        hepmc_event.reserve(len(hepevt_particles), len(vertex_cache))

        # Add particles and vertices in topological order
        self.add_tree(pythia.event, hepmc_event, beam_particles)

        # Add color attributes to particles AFTER adding them to event
        # self.add_color(pythia.event, hepevt_particles)  # TODO: Causes segmentation fault; requires custom HepMC3 bindings (otherwise thread-locking not accessible)

        # 4. Check for particles which come from nowhere, #################
        # i.e. are without mothers or daughters. These need to be attached
        # to a vertex, or else they will never become part of the event.

        # TODO: Below causes error; too few vertices when reading from hepmc file
        for i in range(pythia.event.size()):
            if not hepevt_particles[i].in_event:
                print(
                    f"Found detached particle; status = {hepevt_particles[i].status}, pid = {hepevt_particles[i].pid}"
                )
                # prod_vtx = pyhepmc.GenVertex()
                # prod_vtx.add_particle_out(hepevt_particles[i])
                # hepmc_event.add_vertex(prod_vtx)

        # 5. Check for free partons #################

        if self.m_hadronization_on:
            for i in range(pythia.event.size()):
                if hepevt_particles[i].pid == 21 and self._check_if_free_particle(
                    hepevt_particles[i]
                ):
                    if self.m_crash_on_problem:
                        raise RuntimeError(
                            f"Error: Found final-state gluon with no end vertex! event {evt_num}, particle {i}"
                        )
                    if self.m_free_parton_warnings:
                        print(
                            f"Warning: Found final-state gluon with no end vertex! event {evt_num}, particle {i}"
                        )

                if abs(hepevt_particles[i].pid) <= 6 and self._check_if_free_particle(
                    hepevt_particles[i]
                ):
                    if self.m_crash_on_problem:
                        raise RuntimeError(
                            f"Error: Found final-state quark with no end vertex! event {evt_num}, particle {i}"
                        )
                    if self.m_free_parton_warnings:
                        print(
                            f"Warning: Found final-state quark with no end vertex! event {evt_num}, particle {i}"
                        )

        # 6. Store PDF, weight, cross section and other event information. #################
        self.store_event_info(pythia, hepmc_event)

        return hepmc_event

    def get_particles(self, pythia_event):
        hepevt_particles = [None for particle_idx in range(pythia_event.size())]
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
            hepevt_particles[particle_idx] = hepmc_particle
        return hepevt_particles

    def get_vertices(self, pythia_event, hepevt_particles):
        vertex_cache = []
        beam_particles = []
        for particle_idx in range(pythia_event.size()):
            pythia_particle = pythia_event[particle_idx]

            mother_indeces = pythia_particle.motherList()

            # If it has mother particle, produce a GenVertex for it
            if len(mother_indeces) > 0:
                hepmc_first_mother = hepevt_particles[mother_indeces[0]]

                prod_vtx = None
                if hepmc_first_mother.end_vertex is not None:
                    prod_vtx = hepmc_first_mother.end_vertex

                if prod_vtx is None:
                    prod_vtx = pyhepmc.GenVertex()
                    for mother_idx in mother_indeces:
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
                To confirm in-place-ness (i.e. confirm that updating prod_vtx indeed updates vertex_cache[-1]):

                import pyhepmc
                prt = pyhepmc.GenParticle(pyhepmc.FourVector(1,2,3,4), 6, 1)
                vtx = pyhepmc.GenVertex()

                vtx_list = []
                vtx_list.append(vtx)

                print(f"BEFORE updating vtx outside-of-list: \n    vtx_list: {vtx_list}\n        vtx_list[0].particles_in: {vtx_list[0].particles_in}\n        vtx_list[0].position: {vtx_list[0].position}")

                vtx.add_particle_in(prt)
                vtx.position = pyhepmc.FourVector(1,2,3,4)

                print(f"AFTER updating vtx outside-of-list: \n    vtx_list: {vtx_list}\n        vtx_list[0].particles_in: {vtx_list[0].particles_in}\n        vtx_list[0].position: {vtx_list[0].position}")
                """

            else:  # Otherwise, it's a beam particle
                beam_particles.append(hepevt_particles[particle_idx])

            if len(beam_particles) != 2:
                if self.m_crash_on_problem:
                    raise NotImplementedError(f"Error: len(beam_particles) = {len(beam_particles)}")

        return vertex_cache, beam_particles

    def visit_children(self, vertex_visit_map, current_vertex):
        # Traverse all outgoing particles from this vertex
        for p_out in current_vertex.vertex.particles_out:
            if getattr(p_out, "end_vertex", None):
                end_vertex = HashableGenVertex(p_out.end_vertex)

                # CYCLE DETECTION: If we've already visited this vertex, then we've found a cycle
                if end_vertex in vertex_visit_map:
                    if vertex_visit_map[end_vertex] != 0:
                        return True

                # Mark this vertex as visited
                if end_vertex not in vertex_visit_map:
                    vertex_visit_map[end_vertex] = 0
                else:
                    vertex_visit_map[end_vertex] = vertex_visit_map[end_vertex] + 1

                # Recursively check children of this end_vertex
                if self.visit_children(vertex_visit_map, end_vertex):
                    return True

        # If we make it here, then no cycles found
        return False

    def detect_cycles(self, pythia_evt, hepmc_evt, beam_particles):
        # Check if cycles attribute already exists
        existing_hc = getattr(pythia_evt, "cycles", None)
        has_cycles = False
        vertex_visit_map = {}
        starting_vertices = []

        # If cycles attribute exists and is non-zero, we already know there are cycles
        if existing_hc:
            if existing_hc != 0:
                has_cycles = True

        # If no cycles attribute exists, we need to detect them ourselves
        if not existing_hc:
            # First pass: collect all vertices and identify starting point
            for p_in in beam_particles:
                prod_vtx = getattr(p_in, "production_vertex", None)

                # Add production vertex to our tracking map
                if prod_vtx:
                    prod_vtx_hashable = HashableGenVertex(prod_vtx)
                    vertex_visit_map[prod_vtx_hashable] = 0

                # If particle has no production vertex OR production vertex
                # has no incoming particles, then its end vertex is a potential starting point for cycle detection
                if not prod_vtx:
                    end_vtx = getattr(p_in, "end_vertex", None)
                    if end_vtx:
                        starting_vertices.append(end_vtx)
                        end_vtx_hashable = HashableGenVertex(end_vtx)
                        vertex_visit_map[end_vtx_hashable] = 0

                prod_vtx_particles_in = getattr(prod_vtx, "particles_in", None)
                if not (prod_vtx_particles_in) or len(prod_vtx_particles_in) == 0:
                    end_vtx = getattr(p_in, "end_vertex", None)
                    if end_vtx:
                        starting_vertices.append(end_vtx)
                        end_vtx_hashable = HashableGenVertex(end_vtx)
                        vertex_visit_map[end_vtx_hashable] = 0

            # Second pass: check for cycles starting from each starting vertex
            for start_vtx in starting_vertices:
                temp_vertex_visit_map = {k: v for k, v in vertex_visit_map.items()}
                start_vtx_hashable = HashableGenVertex(start_vtx)
                found_cycles = self.visit_children(temp_vertex_visit_map, start_vtx_hashable)
                has_cycles = has_cycles or found_cycles

        if has_cycles:
            hepmc_evt.attributes["cycles"] = 1

    def topological_sort_vertices(self, beam_particles):
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

    def add_tree(self, pythia_evt, hepmc_evt, beam_particles):
        if self.m_detect_cycles:
            self.detect_cycles(pythia_evt, hepmc_evt, beam_particles)

        all_vertices_sorted = self.topological_sort_vertices(beam_particles)
        for v in all_vertices_sorted:
            hepmc_evt.add_vertex(v)

        # TODO: Validate root-vertex handling; requires custom HepMC3 bindings (otherwise no attribute pyhepmc.GenEvent.m_root_vertex)

    def add_color(self, pythia_event, hepmc_particles):
        """To check in-place-ness:
        import pyhepmc
        p1 = pyhepmc.GenParticle()
        p2 = pyhepmc.GenParticle()
        test_evt = pyhepmc.GenEvent()
        test_evt.add_particle(p1)
        test_evt.add_particle(p2)
        print(f"test_evt.particles[0].attributes BEFORE inplace-op: {test_evt.particles[0].attributes}")
        p1.attributes["flow1"] = 0
        print(f"test_evt.particles[0].attributes AFTER inplace-op: {test_evt.particles[0].attributes}")
        """
        for i in range(pythia_event.size()):
            colType = pythia_event[i].colType()
            if colType in {-1, 1, 2}:
                flow1 = 0
                flow2 = 0
                if colType in {1, 2}:
                    flow1 = pythia_event[i].col()
                if colType in {-1, 2}:
                    flow2 = pythia_event[i].acol()

                hepmc_particles[i].attributes["flow1"] = flow1
                hepmc_particles[i].attributes["flow2"] = flow2

    def store_event_info(self, pythia, hepmc_event):
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
        if self.m_store_weights:
            hepmc_event.weights.clear()
            for i in range(pyinfo.nWeights()):
                hepmc_event.weights.append(pyinfo.weight(i))

    def _check_if_free_particle(self, hepevt_particle):
        end_vertex = getattr(hepevt_particle, "end_vertex", None)
        if end_vertex is None:
            return True

        particles_out = getattr(end_vertex, "particles_out", None)
        if particles_out is None:
            return True
        if len(particles_out) == 0:
            return True

        return False
