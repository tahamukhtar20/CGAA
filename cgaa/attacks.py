import torch
import torch.nn.functional as F


class BIM:
    """Basic Iterative Method (Kurakin et al. 2016), also known as PGD without random start."""

    def __init__(self, model, eps=16 / 255, alpha=2 / 255, steps=50):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = next(model.parameters()).device

    def get_loss(self, logits, labels, acts=None):
        return F.cross_entropy(logits, labels)

    def forward(self, images, labels, callback=None):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = images.clone().detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            logits, acts = outputs if isinstance(outputs, tuple) else (outputs, None)

            loss = self.get_loss(logits, labels, acts)
            grad = torch.autograd.grad(loss, adv_images)[0]

            if callback:
                callback(i, adv_images, acts, grad)

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images


class CGAA(BIM):
    """Concept-Guided Adversarial Attack.

    Extends BIM with a concept-alignment term in the loss:
    loss = cross_entropy + lambda * direction * dot(mean_pool(activations), cav)

    A positive direction pushes activations toward the concept; negative pushes away.
    The model argument is expected to be Sequential(Normalize, resnet).
    """

    def __init__(self, model, cav, layer_name, direction, lambda_val=100.0, **kwargs):
        super().__init__(model, **kwargs)
        self.cav = cav.detach().to(self.device)
        self.lambda_val = lambda_val
        self.direction = direction
        self.layer_name = layer_name
        self.activations = None
        self._layer = dict(self.model[1].named_modules())[layer_name]

    def _hook(self, _mod, _inp, out):
        self.activations = out.mean(dim=[2, 3])

    def forward(self, images, labels, callback=None):
        handle = self._layer.register_forward_hook(self._hook)
        original_forward = self.model.forward

        def forward_wrapper(x):
            logits = original_forward(x)
            return logits, self.activations

        self.model.forward = forward_wrapper
        try:
            adv_images = super().forward(images, labels, callback)
        finally:
            self.model.forward = original_forward
            handle.remove()
        return adv_images

    def get_loss(self, logits, labels, acts):
        loss_ce = F.cross_entropy(logits, labels)
        concept_score = torch.matmul(acts, self.cav).mean()
        return loss_ce + (self.lambda_val * self.direction * concept_score)
