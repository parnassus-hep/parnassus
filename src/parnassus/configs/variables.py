"""Variable requirements configuration for models and datasets."""

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

        Raises
        ------
        TypeError
            If generator type doesn't support variable extraction.
        """
        # For neural generators, extract from model configs
        from parnassus.configs.generators import NeuralGeneratorConfig  # noqa: PLC0415

        if isinstance(generator_config, NeuralGeneratorConfig):
            return cls(
                truth_vars_to_load=generator_config.truth_vars_to_load,
                ctxt_vars=generator_config.event_model_config.variables_config.ctxt_vars,
                ctxt_global_vars=generator_config.event_model_config.variables_config.ctxt_global_vars,
            )
        # For future generator types, provide defaults or raise error
        raise TypeError(
            "Variable extraction not implemented for generator type: "
            f"{type(generator_config).__name__}"
        )

    @property
    def ctxt_vars_stripped(self) -> list[str]:
        """Context variables with 'truth_' prefix removed.

        Returns
        -------
        list[str]
            List of context variable names without 'truth_' prefix.
        """
        return [var.replace("truth_", "") for var in self.ctxt_vars]

    @property
    def ctxt_global_vars_stripped(self) -> list[str]:
        """Global context variables with 'truth_' prefix removed.

        Returns
        -------
        list[str]
            List of global context variable names without 'truth_' prefix.
        """
        return [var.replace("truth_", "") for var in self.ctxt_global_vars]
