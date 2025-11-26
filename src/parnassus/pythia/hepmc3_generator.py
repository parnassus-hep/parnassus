import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from joblib import Parallel, delayed

from parnassus.utils.logger import setup_logger

LOG = setup_logger()


class HepMC3Generator:
    """Generate HepMC3 events using Pythia8 in parallel and merge outputs.

    This class manages launching a single-core Pythia8 job (via a small helper
    script) multiple times in parallel using joblib, collects and merges the
    outputs into a single HepMC3 file, and writes per-job logs for debugging.

    Attributes
    ----------
    cmnd_file : str
        Path to the Pythia .cmnd configuration file.
    output_dir : Path
        Directory where generated HepMC files are written.
    log_dir : Path
        Directory where per-job logs are written.
    hadronization_on : bool
        Whether hadronization is enabled in the Pythia configuration.

    Methods
    -------
    generate(n_events, max_workers, debug=False)
        Generate n_events using up to max_workers parallel processes
        and merge results into one HepMC3 file.
    """

    def __init__(self, cmnd_file: str, output_dir: str, log_dir: str | None = None):
        self.cmnd_file = cmnd_file
        self.output_dir = Path(output_dir)
        if log_dir is None:
            self.log_dir = Path(self.output_dir) / "logs"
        else:
            self.log_dir = Path(log_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.hadronization_on = self._is_hadronization_on()

    def generate(self, n_events: int, max_workers: int, debug: bool = False) -> Path:
        tic = time.time()

        n_events_per_job = n_events // max_workers
        n_remainder = n_events % max_workers
        # Create list of event counts: first n_remainder workers get +1 event
        n_events_list = [
            n_events_per_job + (1 if i < n_remainder else 0) for i in range(max_workers)
        ]
        seeds = list(range(1, max_workers + 1))
        fpaths_output = [Path(self.output_dir) / f"events_part_{i}.hepmc" for i in seeds]
        fpaths_log = [Path(self.log_dir) / f"job_{i}.log" for i in seeds]

        # Generate events in parallel
        self._write_single_job()
        tic_gen = time.time()
        if debug:
            LOG.info("HepMC3Generator: DEBUG MODE: Generating 5 events using single core...")
            self._gen_hepmc_single_job(
                cmnd_file=self.cmnd_file,
                n_events=5,
                seed=42,
                fpath_output=Path(self.output_dir) / "events_DEBUG.hepmc",
                fpath_log=Path(self.log_dir) / "job_DEBUG.log",
            )
            n_events = 5
            fpath_merged = Path(self.output_dir) / "events_DEBUG.hepmc"

        else:
            LOG.info(f"HepMC3Generator: Generating {n_events} events using {max_workers} cores...")
            Parallel(n_jobs=max_workers, backend="multiprocessing", verbose=100)(
                delayed(self._gen_hepmc_single_job)(
                    cmnd_file=self.cmnd_file,
                    n_events=n_events_list[i],
                    seed=seeds[i],
                    fpath_output=fpaths_output[i],
                    fpath_log=fpaths_log[i],
                )
                for i in range(max_workers)
            )
            fpath_merged = Path(self.output_dir) / "events.hepmc"

        toc_gen = time.time()
        dur_gen = toc_gen - tic_gen
        LOG.info(
            f"HepMC3Generator: Generated {n_events} events in {dur_gen // 60}m {dur_gen % 60:.1f}s"
        )

        # Merge output files
        if not debug:
            LOG.info("HepMC3Generator: \nMerging files...")
            tic_merge = time.time()
            self._merge_hepmc_files(fpaths_output, fpath_merged, max_workers=max_workers)
            toc_merge = time.time()
            dur_merge = toc_merge - tic_merge
            LOG.info(
                f"HepMC3Generator: "
                f"Merged {max_workers} files in {dur_merge // 60}m {dur_merge % 60:.1f}s"
            )

        toc = time.time()
        dur = toc - tic
        LOG.info(f"HepMC3Generator: Total duration: {dur // 60} min {dur % 60:.1f} sec")

        return fpath_merged

    def _write_single_job(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
            self.fpath_script = f.name
            LOG.info(f"HepMC3Generator: self.fpath_script = {self.fpath_script}")
            f.write(SINGLE_JOB_SCRIPT.encode())

    def _gen_hepmc_single_job(
        self,
        cmnd_file: str,
        n_events: int,
        seed: int,
        fpath_output: Path,
        fpath_log: Path,
    ):
        cmd = [
            "python",
            self.fpath_script,
            cmnd_file,
            "--n-events",
            str(n_events),
            "--seed",
            str(seed),
            "--output",
            fpath_output.as_posix(),
        ]
        if self.hadronization_on:
            cmd.append("--hadronization-on")

        with open(fpath_log, "w") as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

        if result.returncode != 0:
            LOG.error(f"HepMC3Generator: Job #{seed} failed")

    def _merge_hepmc_files(self, input_files: list[Path], output_file: Path, max_workers: int = 8):
        """Reads events from multiple input HepMC files and writes them to a single output file.

        Args:
            input_files (list): A list of paths to the input HepMC files.
            output_file (str): The path for the output merged HepMC file.
        """
        all_files: list[Path] = input_files.copy()

        # Merge files in pairs in parallel until only 1 file remains
        while len(all_files) > 1:
            LOG.info(f"HepMC3Generator: len(files) = {len(all_files)}")
            if len(all_files) % 2 != 0:
                all_files = [self._append_hepmc_file(all_files[0], all_files[1])] + all_files[2:]
            assert len(all_files) % 2 == 0

            all_files = Parallel(min(max_workers, len(all_files) // 2), backend="multiprocessing")(
                delayed(self._append_hepmc_file)(all_files[i], all_files[i + 1])
                for i in range(0, len(all_files), 2)
            )  # pyright: ignore[reportAssignmentType]

        # Rename the final file to the desired output file name
        shutil.move(all_files[0], output_file)
        LOG.info(f"HepMC3Generator: len(files) = {len(all_files)}")

    def _append_hepmc_file(self, fpath1: Path, fpath2: Path) -> Path:
        # Read files to memory
        with open(fpath1, "rb") as f:
            file1 = f.readlines()
        with open(fpath2, "rb") as f:
            file2 = f.readlines()

        # Update event numbers in file 2
        file1_event_lines = [i for i in file1 if i.startswith(b"E ")]
        idx_event = int(file1_event_lines[-1].split(b" ")[1])
        for line_idx, line in enumerate(file2):
            if line.startswith(b"E "):
                idx_event += 1
                new_line = line.split(b" ")
                new_line[1] = str(idx_event).encode()
                file2[line_idx] = b" ".join(new_line)

        # Drop tail of file 1
        file1 = file1[:-1]

        # Drop header of file 2
        file2_event_line_indeces = [
            line_idx for line_idx, line in enumerate(file2) if line.startswith(b"E ")
        ]
        file2 = file2[file2_event_line_indeces[0] :]

        # Merge files
        merged_file = file1 + file2
        with open(fpath1, "wb") as f:
            f.writelines(merged_file)

        fpath2.unlink()

        return fpath1

    def _is_hadronization_on(self) -> bool:
        with open(self.cmnd_file) as f:
            lines = f.readlines()
        for line in lines:
            if not line.strip().startswith("!") and (
                "HadronLevel:Hadronize = off" in line
                or "HadronLevel:Hadronize=off" in line
                or "HadronLevel:all = off" in line
                or "HadronLevel:all=off" in line
            ):
                return False
        return True


SINGLE_JOB_SCRIPT = """
#!/usr/bin/env python3
import pythia8mc
import pyhepmc
from parnassus.pythia import Pythia8ToHepMC3
from parnassus.utils.logger import setup_logger
import argparse

LOG = setup_logger()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate events with Pythia8 on a single core and save to HepMC3 format.")
    parser.add_argument("cmnd", help=".cmnd file with Pythia8 settings")
    parser.add_argument("--n-events", type=int, default=1000, help="Number of events to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Pythia8 (default: None)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for HepMC3 file (default: data_out/HZZ4l)")
    parser.add_argument("--hadronization-on", action="store_true", help="Flag to indicate if hadronization is on (default: True)")
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    pythia = pythia8mc.Pythia()

    # Random seed
    pythia.readString("Random:setSeed = on")
    pythia.readString(f"Random:seed = {args.seed}")

    # Read settings from .cmnd file
    pythia.readFile(args.cmnd)

    if not pythia.init():
        LOG.info("HepMC3Generator: Pythia initialization failed!")
        return 1

    # HepMC3 writer
    converter = Pythia8ToHepMC3(m_hadronization_on=args.hadronization_on)
    writer = pyhepmc.io.WriterAscii(f"{args.output}")

    # Generate exactly args.nEvents successful events (retry on failed ones).
    n_written = 0
    idx_event = 0
    while n_written < args.n_events:
        if not pythia.next():
            continue  # event failed, try again

        hepmcEvent = converter.fill_next_event(pythia, idx_event+1)
        writer.write_event(hepmcEvent)
        n_written += 1
        idx_event += 1

        if n_written % 10000 == 0:
            LOG.info(f"HepMC3Generator: Generated {n_written} events...")

    pythia.stat()
    writer.close()
    return 0

if __name__ == "__main__":
    main()
"""  # noqa: E501
