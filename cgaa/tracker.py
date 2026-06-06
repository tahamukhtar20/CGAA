class AttackTracker:
    """Records per-step activations and input-space gradients during an attack."""

    def __init__(self, cav):
        self.acts_history = []
        self.grads_history = []
        self.cav = cav.detach().cpu()

    def callback(self, _step, _x_adv, acts, grad):
        if acts is not None:
            self.acts_history.append(acts.detach().cpu().numpy())
        self.grads_history.append(grad.detach().cpu().view(grad.size(0), -1).numpy())
