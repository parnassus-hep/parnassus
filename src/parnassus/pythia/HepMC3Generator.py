from pathlib import Path
import shutil
import subprocess
import time

from joblib import Parallel, delayed


class HepMC3Generator:
    def __init__(self, cmnd_file: str, output_dir: str, log_dir: str | None = None):
        self.cmnd_file = cmnd_file
        self.output_dir = output_dir
        if log_dir is None:
            self.log_dir = Path(self.output_dir) / "logs"
        else:
            self.log_dir = log_dir
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        Path(self.log_dir).mkdir(exist_ok=True, parents=True)

        self.hadronization_on = self._is_hadronization_on()

    def generate(self, n_events, max_workers, debug=False):
        tic = time.time()

        n_events_per_job = n_events // max_workers
        seeds = list(range(1, max_workers + 1))
        fpaths_output = [Path(self.output_dir) / f"events_part_{i}.hepmc" for i in seeds]
        fpaths_log = [Path(self.log_dir) / f"job_{i}.log" for i in seeds]

        # Generate events in parallel
        self._write_single_job()
        tic_gen = time.time()
        if debug:
            print("DEBUG MODE: Generating 5 events using single core...")
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
            print(f"Generating {n_events} events using {max_workers} cores...")
            Parallel(n_jobs=max_workers, backend="multiprocessing", verbose=100)(
                delayed(self._gen_hepmc_single_job)(
                    cmnd_file=self.cmnd_file,
                    n_events=n_events_per_job,
                    seed=seeds[i],
                    fpath_output=fpaths_output[i],
                    fpath_log=fpaths_log[i],
                )
                for i in range(max_workers)
            )

        toc_gen = time.time()
        dur_gen = toc_gen - tic_gen
        print(f"\nGenerated {n_events} events in {dur_gen // 60}m {dur_gen % 60:.1f}s")

        # Merge output files
        if not debug:
            print("\nMerging files...")
            tic_merge = time.time()
            fpath_merged = Path(self.output_dir) / "events.hepmc"
            self._merge_hepmc_files(fpaths_output, fpath_merged, max_workers=max_workers)
            toc_merge = time.time()
            dur_merge = toc_merge - tic_merge
            print(f"Merged {max_workers} files in {dur_merge // 60}m {dur_merge % 60:.1f}s")

        toc = time.time()
        dur = toc - tic
        print(f"Total duration: {dur // 60} min {dur % 60:.1f} sec")

        return fpath_merged

    def _write_single_job(self):
        with open("single_job.py", "w") as f:
            f.write(SINGLE_JOB_SCRIPT)

    def _gen_hepmc_single_job(
        self, cmnd_file: str, n_events: int, seed: int, fpath_output: str, fpath_log: str
    ):
        cmd = [
            "python",
            "single_job.py",
            cmnd_file,
            "--n-events",
            str(n_events),
            "--seed",
            str(seed),
            "--output",
            fpath_output,
        ]
        if self.hadronization_on:
            cmd.append("--hadronization-on")

        with open(fpath_log, "w") as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

        if result.returncode != 0:
            print(f"Error running job #{seed}")

    def _merge_hepmc_files(self, input_files: list, output_file: str, max_workers: int = 8):
        """Reads events from multiple input HepMC files and writes them to a single output file.

        Args:
            input_files (list): A list of paths to the input HepMC files.
            output_file (str): The path for the output merged HepMC file.
        """
        all_files = input_files.copy()

        # Merge files in pairs in parallel until only 1 file remains
        while len(all_files) > 1:
            print(f"len(files) = {len(all_files)}")
            if len(all_files) % 2 != 0:
                all_files = [self._append_hepmc_file(all_files[0], all_files[1])] + all_files[2:]
            assert len(all_files) % 2 == 0

            all_files = Parallel(min(max_workers, len(all_files) // 2), backend="multiprocessing")(
                delayed(self._append_hepmc_file)(all_files[i], all_files[i + 1])
                for i in range(0, len(all_files), 2)
            )

        # Rename the final file to the desired output file name
        shutil.move(all_files[0], output_file)
        print(f"len(files) = {len(all_files)}")

    def _append_hepmc_file(self, fpath1: str, fpath2: str):
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
                new_line = b" ".join(new_line)
                file2[line_idx] = new_line

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

        Path(fpath2).unlink()

        return fpath1

    def _is_hadronization_on(self):
        with open(self.cmnd_file) as f:
            lines = f.readlines()
        for line in lines:
            if not line.strip().startswith("!"):
                if (
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
from parnassus.pythia.Pythia8ToHepMC3 import Pythia8ToHepMC3
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate events with Pythia8 on a single core and save to HepMC3 format.")
    parser.add_argument("cmnd", help=".cmnd file with Pythia8 settings")
    parser.add_argument("--n-events", type=int, default=1000, help="Number of events to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Pythia8 (default: None)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for HepMC3 file (default: data_out/HZZ4l)")
    parser.add_argument("--hadronization-on", action="store_true", help="Flag to indicate if hadronization is on (default: True)")
    return parser.parse_args()

def main():
    args = parse_args()

    pythia = pythia8mc.Pythia()

    # Random seed
    pythia.readString("Random:setSeed = on")
    pythia.readString(f"Random:seed = {args.seed}")

    # Read settings from .cmnd file
    pythia.readFile(args.cmnd)

    if not pythia.init():
        print("Pythia initialization failed!")
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
            print(f"Generated {n_written} events...")

    pythia.stat()
    writer.close()
    return 0

if __name__ == "__main__":
    main()
"""
