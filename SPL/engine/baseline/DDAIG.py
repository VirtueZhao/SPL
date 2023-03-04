import torch
from tabulate import tabulate
from torch.nn import functional as F
from SPL.model import build_network
from SPL.engine.trainer import GenericNet
from SPL.engine import TRAINER_REGISTRY, GenericTrainer
from SPL.optim import build_optimizer, build_lr_scheduler
from SPL.utils import count_num_parameters, compute_gradients_length


@TRAINER_REGISTRY.register()
class DDAIG(GenericTrainer):
    """Deep Domain-Adversarial Image Generation.

    https://arxiv.org/abs/2003.06054.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.DDAIG.LMDA
        self.clamp = cfg.TRAINER.DDAIG.CLAMP
        self.clamp_min = cfg.TRAINER.DDAIG.CLAMP_MIN
        self.clamp_max = cfg.TRAINER.DDAIG.CLAMP_MAX
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        self.alpha = cfg.TRAINER.DDAIG.ALPHA

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

        print("Building Domain Transformation Net")
        self.domain_transformation_net = build_network(self.cfg.TRAINER.DDAIG.G_ARCH)
        self.domain_transformation_net.to(self.device)
        self.optimizer_domain_transformation = build_optimizer(self.domain_transformation_net, self.cfg.OPTIM)
        self.scheduler_domain_transformation = build_lr_scheduler(self.optimizer_domain_transformation, self.cfg.OPTIM)
        self.model_registration("domain_transformation_net", self.domain_transformation_net, self.optimizer_domain_transformation, self.scheduler_domain_transformation)

        model_parameters_table = [
            ["Model", "# Parameters"],
            ["Label Classifier", f"{count_num_parameters(self.label_classifier):,}"],
            ["Domain Classifier", f"{count_num_parameters(self.domain_classifier):,}"],
            ["Domain Transformation Net", f"{count_num_parameters(self.domain_transformation_net):,}"]
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        img_id, input_data, class_label, domain_label = self.parse_batch_train(batch_data)

        #############
        # Update Domain Transformation Net
        #############
        input_data_p = self.domain_transformation_net(input_data, lmda=self.lmda)
        #
        loss_dt = 0
        # Minimize Label Classifier Loss
        loss_dt += F.cross_entropy(self.label_classifier(input_data_p), class_label)
        # # Maximize Domain Classifier Loss
        loss_dt -= F.cross_entropy(self.domain_classifier(input_data_p), domain_label)
        self.model_backward_and_update(loss_dt, "domain_transformation_net")

        # Perturb Data with Updated Domain Transformation Net
        with torch.no_grad():
            input_data_p = self.domain_transformation_net(input_data, lmda=self.lmda)

        input_data.requires_grad = True
        input_data_p.requires_grad = True

        #############
        # Update Label Classifier
        #############
        pred_l = self.label_classifier(input_data)
        loss_l = F.cross_entropy(pred_l, class_label)
        pred_lp = self.label_classifier(input_data_p)
        loss_lp = F.cross_entropy(pred_lp, class_label)
        loss_l = (1.0 - self.alpha) * loss_l + self.alpha * loss_lp
        loss_l = loss_l * self.current_batch_loss_weight
        self.model_backward_and_update(loss_l, "label_classifier")

        examples_difficulty = self.compute_difficulty(img_id=img_id, class_label=class_label, pred_l=pred_l,
                                                      pred_lp=pred_lp, input_data_grad=input_data.grad,
                                                      input_data_p_grad=input_data_p.grad)

        #############
        # Update Domain Classifier
        #############
        loss_d = F.cross_entropy(self.domain_classifier(input_data), domain_label)
        self.model_backward_and_update(loss_d, "domain_classifier")

        loss_summary = {
            # "loss_dt": loss_dt.item(),
            "loss_l": loss_l.item(),
            "loss_d": loss_d.item()
        }

        if self.batch_index + 1 == self.num_batches:
            self.update_lr()

        return loss_summary, examples_difficulty

    def parse_batch_train(self, batch_data):
        img_id = batch_data["img_id"]
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return img_id, input_data, class_label, domain_label

    def model_inference(self, input_data):
        return self.label_classifier(input_data)

    def compute_difficulty(self, img_id, class_label, pred_l, pred_lp, input_data_grad, input_data_p_grad):
        examples_difficulty = []
        alpha = 0.5

        for i in range(len(img_id)):
            current_img_id = img_id[i].item()
            current_class_label = class_label[i].item()
            current_prediction_confidence = F.softmax(pred_l[i], dim=0).cpu().detach().numpy()[current_class_label]
            current_prediction_confidence_p = F.softmax(pred_lp[i], dim=0).cpu().detach().numpy()[current_class_label]
            current_gradients_length = compute_gradients_length(input_data_grad[i].cpu().numpy())
            current_gradients_length_p = compute_gradients_length(input_data_p_grad[i].cpu().numpy())
            current_img_difficulty = (1 - alpha) * (current_gradients_length / current_prediction_confidence) + \
                                     alpha * (current_gradients_length_p / current_prediction_confidence_p)
            examples_difficulty.append((current_img_id, current_img_difficulty))

        return examples_difficulty
