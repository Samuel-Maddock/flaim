from synth_fl.generators import FederatedGeneratorConfig
from synth_fl.generators.central import CentralGeneratorFactory
from synth_fl.generators.federated.distributed_aim import DistributedAIM
from synth_fl.generators.federated.distributed_aim_mpc import DistributedAIMMPC

from synth_fl.generators.federated.naive_hfl import NaiveHFLGenerator
from synth_fl.generators.federated.flaim import FLAIM
from synth_fl.generators.federated.flaim_init import FLAIMInit
from synth_fl.generators.federated.fed_ctgan import FedCTGAN

FACTORY_MAP = {
    "naivehfl": NaiveHFLGenerator,
    "flaim": FLAIM,
    "flaim_init": FLAIMInit,
    "distributedaim": DistributedAIM,
    "distributedaimmpc": DistributedAIMMPC,
    "fedctgan": FedCTGAN,
}


# Central + FL Generator factory
class FLGeneratorFactory(CentralGeneratorFactory):
    def __init__(self) -> None:
        super().__init__()
        self.fl_factory_map = FACTORY_MAP

    @staticmethod
    def get_factory_map():
        return FACTORY_MAP

    @staticmethod
    def get_central_factory_map():
        return CentralGeneratorFactory.get_factory_map()

    def _process_gen_name(self, generator_name: str):
        fl_gen_name = generator_name.lower()
        central_gen_name = None

        # If gen name invalid, check to see if it is a FL-central hybrid
        if fl_gen_name not in self.fl_factory_map:
            # Format should be NaiveHFL_PGM or NaiveHFL_PrivBayes etc
            gen_name_split = generator_name.split("_")
            assert len(gen_name_split) == 2
            fl_gen_name, central_gen_name = gen_name_split

        return fl_gen_name, central_gen_name

    def create_obj(self, generator_name: str, generator_cfg: FederatedGeneratorConfig):
        if generator_name in self.factory_map:
            return super().create_cls(generator_name)(generator_cfg)

        fl_gen_name, central_gen_name = self._process_gen_name(generator_name)
        if central_gen_name:
            return self.create_cls(fl_gen_name)(central_gen_name, generator_cfg)
        else:
            return self.create_cls(fl_gen_name)(generator_cfg)

    def create_cls(self, generator_name: str):
        if generator_name in self.factory_map:
            return super().create_cls(generator_name)
        fl_gen_name, _ = self._process_gen_name(generator_name)
        return self.fl_factory_map[fl_gen_name]
