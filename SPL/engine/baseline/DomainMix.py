import torch
from torch.nn import functional as F
from SPL.engine import TRAINER_REGISTRY, GenericTrainer
from SPL.utils import compute_top_k_accuracy, compute_gradients_length


@TRAINER_REGISTRY.register()
class DomainMix(GenericTrainer):
    """DomainMix.

    Dynamic Domain Generalization.

    https://github.com/MetaVisionLab/DDG
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mix_type = cfg.TRAINER.DOMAINMIX.TYPE
        self.alpha = cfg.TRAINER.DOMAINMIX.ALPHA
        self.beta = cfg.TRAINER.DOMAINMIX.BETA
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)

    def forward_backward(self, batch_data):
        input_data, label_a, label_b, lam = self.parse_batch_train(batch_data)
        output = self.model(input_data)
        loss = lam * F.cross_entropy(output, label_a) + (1 - lam) * F.cross_entropy(output, label_b)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_top_k_accuracy(output, label_a)[0].item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        examples_difficulty = []

        return loss_summary, examples_difficulty

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)

        input_data, label_a, label_b, lam = self.domain_mix(input_data, class_label, domain_label)

        return input_data, label_a, label_b, lam

    def domain_mix(self, input_data, class_label, domain_label):
        if self.alpha > 0:
            lam = self.dist_beta.rsample((1, )).to(input_data.device)
        else:
            lam = torch.tensor(1).to(input_data.device)

        perm = torch.randperm(input_data.size(0), dtype=torch.int64, device=input_data.device)

        if self.mix_type == "crossdomain":
            domain_list = torch.unique(domain_label)
            if len(domain_list) > 1:
                for current_domain_index in domain_list:
                    # Count the number of examples in the current domain.
                    count_current_domain = torch.sum(domain_label == current_domain_index)
                    # Retrieve the index of examples other than the current domain.
                    other_domain_index = (domain_label != current_domain_index).nonzero().squeeze(-1)
                    # Count the number of examples in the other domains.
                    count_other_domain = other_domain_index.shape[0]
                    # Generate the Perm index of examples in the other domains that are going to be mixed.
                    perm_other_domain = torch.ones(count_other_domain).multinomial(
                        num_samples=count_current_domain, replacement=bool(count_current_domain > count_other_domain))
                    # Replace current domain label with another domain label.
                    perm[domain_label == current_domain_index] = other_domain_index[perm_other_domain]
        elif self.mix_type != "random":
            raise NotImplementedError(f"Mix Type should be within {'random', 'crossdomain'}, but got {self.mix_type}")

        mixed_input_data = lam * input_data + (1 - lam) * input_data[perm, :]
        label_a, label_b = class_label, class_label[perm]
        return mixed_input_data, label_a, label_b, lam
