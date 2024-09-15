from synth_fl.generators import GeneratorConfig
from synth_fl.utils import Factory

from .aim import AIM
from .random_aim import RandomAIM
from .base_aim import BaseAIM
from .baseline import ZeroGenerator, RealDataGenerator
from .mwem import MWEM
from .ctgan import CTGAN

FACTORY_MAP = {
    "aim": AIM,
    "random_aim": RandomAIM,
    "mwem": MWEM,
    "zero_gen": ZeroGenerator,
    "real_data": RealDataGenerator,
    "base_aim": BaseAIM,
    "ctgan": CTGAN,
}


class CentralGeneratorFactory(Factory):
    def __init__(self) -> None:
        self.factory_map = FACTORY_MAP

    @staticmethod
    def get_factory_map():
        return FACTORY_MAP

    def _sanitize_gen_name(self, generator_name: str):
        sanitised_name = generator_name.lower()
        if sanitised_name not in self.factory_map:
            raise NotImplementedError(
                f"generator_name '{sanitised_name}' is not valid. Choose from {self.factory_map.keys()}"
            )
        return sanitised_name

    def create_obj(self, generator_name: str, generator_cfg: GeneratorConfig):
        return self.create_cls(generator_name)(generator_cfg)

    def create_cls(self, generator_name: str):
        return self.factory_map[self._sanitize_gen_name(generator_name)]


__all__ = ["PrivBayes", "AIM", "PGM", "RAP", "CentralGeneratorFactory"]
