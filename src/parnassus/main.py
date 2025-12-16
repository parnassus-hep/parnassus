import argparse
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from parnassus.configs import Config
from parnassus.configs.generators import NeuralGeneratorConfig, ParametricGeneratorConfig
from parnassus.configs.pipeline import IsolationConfig, JetClusteringConfig
from parnassus.pipelines import GenerationPipeline, IsolationPipeline, JetClusteringPipeline
from parnassus.utils.logger import setup_logger
from parnassus.writers import RootWriter

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("configs/default_config.yaml")


def parse_args(args: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for Parnassus.

    Parameters
    ----------
    args : Sequence[str] | None
        List of command-line arguments. If None, defaults to sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.

    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="cli",
    )
    parser_init = subparsers.add_parser(
        "init",
        help="Initialize a default configuration file in the provided path (or current directory)",
    )
    _ = parser_init.add_argument(
        "dir",
        nargs="?",
        type=str,
        default=".",
        help="Path to directory to save the default configuration file",
    )
    parser_run = subparsers.add_parser("run", help="Execute the Parnassus pipeline")
    _ = parser_run.add_argument(
        "-c", "--config", type=str, help="Path to the configuration YAML file"
    )
    _ = parser_run.add_argument(
        "-i",
        "--input_path",
        type=str,
        default=None,
        help="Path to input data file (overrides config)",
    )
    _ = parser_run.add_argument(
        "-o", "--output_path", type=str, default=None, help="Path to output file (overrides config)"
    )
    _ = parser_run.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=None,
        help="Number of sampler steps for neural models (overrides config)",
    )
    _ = parser_run.add_argument(
        "-ne",
        "--num_events",
        type=int,
        default=None,
        help="Number of events to process (overrides config)",
    )
    _ = parser_run.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for processing (overrides config)",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    """Main function to run Parnassus CLI.

    Parameters
    ----------
    args : Sequence[str] | None
        List of command-line arguments. If None, defaults to sys.argv.
    """
    parsed_args = parse_args(args)
    if parsed_args.cli == "init":
        shutil.copy(DEFAULT_CONFIG_PATH, Path(parsed_args.dir).absolute())
    elif parsed_args.cli == "run":
        log = setup_logger()
        title = " Starting Parnassus "
        log.info(f"[bold green]{title:-^100}")
        start = datetime.now()
        log.info(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        config = Config.from_yaml(parsed_args.config)
        accessor_store = config.writer_config.accessor_store

        if parsed_args.input_path:
            config.dataset_config.file_path = Path(parsed_args.input_path).absolute()
        if parsed_args.output_path:
            config.writer_config.file_path = Path(parsed_args.output_path).absolute()
        if parsed_args.num_events:
            config.dataset_config.num_events = parsed_args.num_events
        if parsed_args.batch_size:
            config.batch_size = parsed_args.batch_size
        if parsed_args.num_steps:
            # Override num_steps for neural generator configs
            if isinstance(config.generator_config, NeuralGeneratorConfig):
                config.generator_config.set_num_steps(parsed_args.num_steps)
            elif isinstance(config.generator_config, ParametricGeneratorConfig):
                log.warning(
                    f"--num_steps is ignored for parametric generator "
                    f"'{config.generator_config.name}'. "
                    "This parameter only applies to neural generators.",
                )

        generation_pipeline = GenerationPipeline(config)
        gen_events, accessors_dict = generation_pipeline.run()
        accessor_store.update_from_dict(accessors_dict)
        log.info("[green]Starting postprocessing.")
        pipeline: JetClusteringPipeline | IsolationPipeline
        for pipeline_config in config.pipeline_configs:
            if isinstance(pipeline_config, JetClusteringConfig):
                pipeline = JetClusteringPipeline(pipeline_config)
                pipeline.process(gen_events)
                accessor_store.update_from_dict(pipeline.get_accessors())
            elif isinstance(pipeline_config, IsolationConfig):
                pipeline = IsolationPipeline(pipeline_config)
                pipeline.process(gen_events)
                accessor_store.update_from_dict(pipeline.get_accessors())
        log.info("[green]Postprocessing completed.")
        log.info("[green]Using following accessors for writer:")
        print(accessor_store)
        writer = RootWriter(config.writer_config)
        writer.write(gen_events)
        end = datetime.now()
        title = " Completed! "
        log.info(f"[bold green]{title:-^100}")
        log.info(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"Elapsed time: {str(end - start).split('.')[0]}")


if __name__ == "__main__":
    main()
