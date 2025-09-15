from pydantic import BaseModel

from src.typing import AnyDict, StringOrPath

from ._utils import load_yaml


class BaseConfig(BaseModel):
    @classmethod
    def from_dict(cls, data: AnyDict) -> "BaseConfig":
        """
        Create an instance of the configuration class from a dictionary.

        Args:
            data (dict[str, Any]):
                The dictionary containing the configuration data.

        Returns:
            BaseConfig:
                An instance of the configuration class.
        """
        return cls(**data)

    @classmethod
    def from_yaml(cls, filepath: StringOrPath) -> "BaseConfig":
        """
        Create an instance of the configuration class from a YAML file.

        Args:
            filepath (Path):
                The path to the YAML file.

        Returns:
            BaseConfig:
                An instance of the configuration class.
        """
        data = load_yaml(filepath)
        return cls.from_dict(data)
