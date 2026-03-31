"""Variable requirements configuration for models and datasets."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from parnassus.utils.typing import VarNameTuple

if TYPE_CHECKING:
    from parnassus.configs.generators import GeneratorConfig


@dataclass(frozen=True, slots=True)
class VariableRequirements:
    """Shared variable configuration used by both models and datasets.

    This class encapsulates the variable names required for data processing,
    eliminating circular dependencies between DatasetConfig and ModelConfig.
    """

    truth_vars_to_load: VarNameTuple
    ctxt_vars: VarNameTuple
    ctxt_global_vars: VarNameTuple

    @classmethod
    def from_dict(cls, config: dict[str, Sequence[str]]) -> "VariableRequirements":
        """Create VariableRequirements from a dictionary.

        Parameters
        ----------
        config : dict[str, list[str]]
            Dictionary containing variable lists with keys 'truth_vars_to_load',
            'ctxt_vars', and 'ctxt_global_vars'.

        Returns
        -------
        VariableRequirements
            A new VariableRequirements instance populated from the dictionary.
        """
        return cls(
            truth_vars_to_load=tuple(config.get("truth_vars_to_load", [])),
            ctxt_vars=tuple(config.get("ctxt_vars", [])),
            ctxt_global_vars=tuple(config.get("ctxt_global_vars", [])),
        )

    @classmethod
    def from_model_config(cls, generator_config: "GeneratorConfig") -> "VariableRequirements":
        """Create VariableRequirements from a GeneratorConfig.

        Parameters
        ----------
        generator_config : GeneratorConfig
            The generator configuration to extract variables from.

        Returns
        -------
        VariableRequirements
            A new VariableRequirements instance with variables from the generator.

        """
        # For neural generators, extract from model configs
        from parnassus.configs.generators import NeuralGeneratorConfig  # noqa: PLC0415

        if isinstance(generator_config, NeuralGeneratorConfig):
            variable_requirements = generator_config.variable_requirements
            return cls(
                truth_vars_to_load=variable_requirements.truth_vars_to_load,
                ctxt_vars=variable_requirements.ctxt_vars,
                ctxt_global_vars=variable_requirements.ctxt_global_vars,
            )

        # For non-neural generators, we don't have variable requirements, so return empty.
        return cls(
            truth_vars_to_load=(),
            ctxt_vars=(),
            ctxt_global_vars=(),
        )

    @property
    def ctxt_vars_stripped(self) -> list[str]:
        """Context variables with 'truth_' prefix removed.

        Returns
        -------
        list[str]
            List of context variable names without 'truth_' prefix.
        """
        return [var.removeprefix("truth_") for var in self.ctxt_vars]

    @property
    def ctxt_global_vars_stripped(self) -> list[str]:
        """Global context variables with 'truth_' prefix removed.

        Returns
        -------
        list[str]
            List of global context variable names without 'truth_' prefix.
        """
        return [var.removeprefix("truth_") for var in self.ctxt_global_vars if "pflow" not in var]
