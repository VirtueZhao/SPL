import torch
from tabulate import tabulate
from torch.nn import functional as F
from SPL.engine.trainer import GenericNet
from SPL.engine import TRAINER_REGISTRY, GenericTrainer
from SPL.optim import build_optimizer, build_lr_scheduler
from SPL.utils import count_num_parameters, compute_gradients_length

@TRAINER_REGISTRY.register()
class CrossGrad(GenericTrainer):
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eps_l = cfg.TRAINER.CROSSGRAD.EPS_L
        self.eps_d = cfg.TRAINER.CROSSGRAD.EPS_D
        self.alpha_l = cfg.TRAINER.CROSSGRAD.ALPHA_L
        self.alpha_d = cfg.TRAINER.CROSSGRAD.ALPHA_D

    def build_model(self):
        print("Building Label Classifier")
        self.label_classifier = GenericNet(self.cfg, self.num_classes)
        self.label_classifier.to(self.device)
        self.optimizer_label = build_optimizer(self.label_classifier, self.cfg.OPTIM)
        self.scheduler_label = build_lr_scheduler(self.optimizer_label, self.cfg.OPTIM)
        self.model_registration("label_classifier", self.label_classifier, self.optimizer_label, self.scheduler_label)

        print("Building Domain Classifier")
        self.domain_classifier = GenericNet(self.cfg, self.num_source_domains)
        self.domain_classifier.to(self.device)
        self.optimizer_domain = build_optimizer(self.domain_classifier, self.cfg.OPTIM)
        self.scheduler_domain = build_lr_scheduler(self.optimizer_domain, self.cfg.OPTIM)
        self.model_registration("domain_classifier", self.domain_classifier, self.optimizer_domain, self.scheduler_domain)

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Label Classifier", f"{count_num_parameters(self.label_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"]
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        img_id, input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        # Compute Domain Perturbation
        loss_domain = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        loss_domain.backward()
        grad_domain = torch.clamp(input_data.grad.data, min=-0.1, max=0.1)
        input_data_domain_perturb = input_data.data + self.eps_l * grad_domain


        # Compute Label Perturbation
        input_data.grad.data.zero_()
        loss_label = F.cross_entropy(self.label_classifier(input_data), class_label)
        loss_label.backward()
        grad_label = torch.clamp(input_data.grad.data, min=-0.1, max=0.1)
        input_data_label_perturb = input_data.data + self.eps_d * grad_label

        input_data = input_data.detach()
        input_data.requires_grad = True
        input_data_domain_perturb.requires_grad = True

        # Update Label Classifier
        pred_l1 = self.label_classifier(input_data)
        loss_l1 = F.cross_entropy(pred_l1, class_label)
        pred_l2 = self.label_classifier(input_data_domain_perturb)
        loss_l2 = F.cross_entropy(pred_l2, class_label)
        loss_l = (1 - self.alpha_l) * loss_l1 + self.alpha_l * loss_l2
        loss_l = loss_l * self.current_batch_loss_weight
        self.model_backward_and_update(loss_l, "label_classifier")

        # Update Domain Classifier
        loss_d1 = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        loss_d2 = F.cross_entropy(self.domain_classifier(input_data_label_perturb), domain_label)
        loss_d = (1 - self.alpha_d) * loss_d1 + self.alpha_d * loss_d2
        self.model_backward_and_update(loss_d, "domain_classifier")

        loss_summary = {
            "loss_l": loss_l.item(),
            "loss_d": loss_d.item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        examples_difficulty = self.compute_difficulty(
            img_id=img_id, class_label=class_label, pred_l1=pred_l1, pred_l2=pred_l2, input_data_grad=input_data.grad,
            input_data_domain_perturb_grad=input_data_domain_perturb.grad
        )

        return loss_summary, examples_difficulty

    def parse_batch_train(self, batch_data):
        img_id = batch_data["img_id"]
        input_data = batch_data["img"].to(self.device)
        input_data.requires_grad = True
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return img_id, input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.label_classifier(input_data)

    def compute_difficulty(self, img_id, class_label, pred_l1, pred_l2, input_data_grad, input_data_domain_perturb_grad):
        alpha = 0.5
        examples_difficulty = []

        for i in range(len(img_id)):
            current_img_id = img_id[i].item()
            current_class_label = class_label[i].item()
            current_loss_l1 = F.cross_entropy(pred_l1[i], current_class_label)
            current_loss_l2 = F.cross_entropy(pred_l2[i], current_class_label)



            current_prediction_confidence = F.softmax(pred_l1[i], dim=0).cpu().detach().numpy()[current_class_label]
            current_prediction_confidence_perturb = F.softmax(pred_l2[i], dim=0).cpu().detach().numpy()[current_class_label]
            current_gradients_length = compute_gradients_length(input_data_grad[i].cpu().numpy())
            current_gradients_length_perturb = compute_gradients_length(input_data_domain_perturb_grad[i].cpu().numpy())
            current_img_difficulty = (1 - alpha) * (current_gradients_length / current_prediction_confidence) + \
                                     alpha * (current_gradients_length_perturb / current_prediction_confidence_perturb)
            examples_difficulty.append((current_img_id, current_img_difficulty))

        return examples_difficulty
