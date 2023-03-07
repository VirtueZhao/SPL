from torch.nn import functional as F
from SPL.engine import TRAINER_REGISTRY, GenericTrainer
from SPL.utils import compute_top_k_accuracy, compute_gradients_length


@TRAINER_REGISTRY.register()
class ERM(GenericTrainer):
    """
    ERM (Empirical Risk Minimization)

    """

    def forward_backward(self, batch_data):
        img_id, input_data, class_label = self.parse_batch_train(batch_data)

        output = self.model(input_data)
        loss = F.cross_entropy(output, class_label)
        loss = loss * self.current_batch_loss_weight
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_top_k_accuracy(output, class_label)[0].item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        examples_difficulty = self.compute_difficulty(img_id=img_id, class_label=class_label, output=output, input_data_grad=input_data.grad)

        return loss_summary, examples_difficulty

    def parse_batch_train(self, batch_data):
        img_id = batch_data["img_id"]
        input_data = batch_data["img"].to(self.device)
        input_data.requires_grad = True
        class_label = batch_data["class_label"].to(self.device)
        return img_id, input_data, class_label

    def compute_difficulty(self, img_id, class_label, output, input_data_grad):
        examples_difficulty = []

        for i in range(len(img_id)):
            current_img_id = img_id[i].item()
            current_class_label = class_label[i].item()
            current_loss = F.cross_entropy(output[i], class_label[i]).cpu().item()
            current_prediction_confidence = F.softmax(output[i], dim=0).cpu().detach().numpy()[current_class_label]
            current_gradients_length = compute_gradients_length(input_data_grad[i].cpu().numpy())

            if self.cfg.SPL.CURRICULUM == "GCDM":
                current_img_difficulty = current_gradients_length / current_prediction_confidence
            elif self.cfg.SPL.CURRICULUM == "loss":
                current_img_difficulty = current_loss
            elif self.cfg.SPL.CURRICULUM == "confidence":
                current_img_difficulty = current_prediction_confidence
            elif self.cfg.SPL.CURRICULUM == "gradients":
                current_img_difficulty = current_gradients_length
            else:
                raise NotImplementedError("Curriculum: {} Not Implemented.".format(self.cfg.SPL.CURRICULUM))

            examples_difficulty.append((current_img_id, current_img_difficulty))

        return examples_difficulty
