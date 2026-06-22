import argparse

import pyhepmc

from parnassus.pythia import HepMC3Generator

N_EVENTS = 1000
N_JOBS = 10

parser = argparse.ArgumentParser(description="Generate HepMC3 events using Pythia8.")
parser.add_argument(
    "-i", "--input", type=str, required=True, help="Input Pythia8 command file (.cmnd)."
)
parser.add_argument(
    "-o", "--output", type=str, required=True, help="Output directory for HepMC3 files."
)
parser.add_argument(
    "-n",
    "--n-events",
    type=int,
    default=N_EVENTS,
    help="Number of events to generate (default: 1000).",
)
parser.add_argument(
    "-j",
    "--n-jobs",
    type=int,
    default=N_JOBS,
    help="Number of parallel jobs to run (default: 10).",
)


def main():
    """Test the HepMC3Generator by generating a sample dataset
    and verifying the number of events in the merged output file.
    """
    args = parser.parse_args()
    generator = HepMC3Generator(
        cmnd_file=args.input,
        output_dir=args.output,
        log_dir="logs/Parnassus_Pythia",
    )
    fpath_merged = generator.generate(n_events=args.n_events, max_workers=args.n_jobs, debug=False)
    print(f"Wrote to file {fpath_merged}")
    with pyhepmc.open(fpath_merged, "r") as f_merged:
        n_events_merged = sum(1 for _ in f_merged)
        assert n_events_merged == args.n_events, (
            f"HepMC3Generator: Merged file has {n_events_merged} events, expected {args.n_events}"
        )


if __name__ == "__main__":
    main()
