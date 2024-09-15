import numpy as np
from synth_fl.generators import FederatedGeneratorConfig
from synth_fl.generators.federated import DistributedAIM

from synth_fl.utils import logger


class DistributedAIMMPC(DistributedAIM):
    def __init__(self, config: FederatedGeneratorConfig) -> None:
        super().__init__(config)
        self.shared_answers = None
        self.shared_ids = set()

    def _aim_step(
        self,
        engine,
        model,
        measurements,
        client_answers,
        viable_candidates,
        exp_epsilon,
        gauss_sigma,
        t=1,
    ):
        subset_answers = self._subset_client_answers(
            client_answers, q=self.clients_per_round / len(client_answers)
        )
        self._commit_client_participation(subset_answers)
        client_ids, round_answers = zip(*subset_answers)  # (client ids, client answers)

        # 'Secret-share' client answer if not done before, and aggregate
        new_client_ids = []
        if self.shared_answers is None:
            self.shared_answers = list(round_answers)
            self.shared_ids.update(client_ids)
            new_client_ids = client_ids
        else:
            for i, client_id in enumerate(client_ids):
                if client_id not in self.shared_ids:
                    self.shared_answers.append(round_answers[i])
                    self.shared_ids.add(client_id)
                    new_client_ids.append(client_id)
        round_answers = self.shared_answers
        client_ids = new_client_ids

        batch_query_ans = None
        logger.debug(
            f"Current shared id list - {self.shared_ids}, length of shared answers {len(self.shared_answers)}"
        )
        logger.debug(f"Clients communicating - {len(client_ids)} - {client_ids}")

        # Find worst-query via exp mech
        # if self.backend == "secagg":
        #     batch_query_ans = self._mock_sec_agg(
        #         client_ids, round_answers, viable_candidates, t=t, log_msg="exp"
        #     )
        if self.selection_method == "server_random":
            cl = np.random.choice(list(viable_candidates.keys()))
        else:
            cl = self._exp_query(
                round_answers,
                model,
                viable_candidates,
                exp_epsilon,
                gauss_sigma,
                t=t,
                batch_query_ans=batch_query_ans,
            )
        round_sample_size = sum([answer[cl] for answer in round_answers]).sum()
        logger.debug(f"Selected query: {cl}, sample_size={round_sample_size}")

        # Measure chosen query and add to measurements
        # if self.backend == "secagg":
        #     batch_query_ans = self._mock_sec_agg(
        #         client_ids, round_answers, [cl], t=t, log_msg="gauss"
        #     )
        measurements += self._gauss_query(
            round_answers, [cl], gauss_sigma, t=t, batch_query_ans=batch_query_ans
        )

        if "local_decisions" in self.log_internal_metrics:
            _, all_client_answers = zip(*client_answers)
            tau_q = self._average_hetero(
                cl, round_answers, all_client_answers, gauss_sigma
            )
            self.internal_metrics["local_decisions"].append(tau_q)

        # Log communication sent/receieved for this round
        self._log_communication(client_ids, cl, viable_candidates, model, t)

        # Calculate query improvement
        prev_model_est = model.project(cl).datavector()
        model = engine.estimate(measurements, total=self.n)
        new_model_est = model.project(cl).datavector()

        return model, prev_model_est, new_model_est, cl, measurements, round_sample_size
