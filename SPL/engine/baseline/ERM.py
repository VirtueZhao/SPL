from torch.nn import functional as F
from SPL.engine import TRAINER_REGISTRY, GenericTrainer
from SPL.utils import compute_top_k_accuracy


@TRAINER_REGISTRY.register()
class ERM(GenericTrainer):
    """
    ERM (Empirical Risk Minimization)

    """

    def forward_backward(self, batch_data):
        img_id, input_data, class_label = self.parse_batch_train(batch_data)
        output = self.model(input_data)
        loss = F.cross_entropy(output, class_label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_top_k_accuracy(output, class_label)[0].item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        print(img_id)

        return loss_summary

    def parse_batch_train(self, batch_data):
        img_id = batch_data["img_id"]
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        return img_id, input_data, class_label
