import torch


def evaluate(model, x, x_adv, cav, layer_name):
    """Attack success rate, concept-score delta, and mean L2 perturbation."""
    layer_mod = dict(model[1].named_modules())[layer_name]

    acts = {}

    def hook(_m, _i, o):
        acts["v"] = o.mean(dim=[2, 3])

    handle = layer_mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits_orig = model(x)
            a_orig = acts["v"]
            logits_adv = model(x_adv)
            a_adv = acts["v"]
    finally:
        handle.remove()

    score_orig = torch.matmul(a_orig, cav).mean().item()
    score_adv = torch.matmul(a_adv, cav).mean().item()
    asr = (logits_orig.argmax(1) != logits_adv.argmax(1)).float().mean().item()
    l2 = (torch.norm(x_adv - x, p=2) / x.size(0)).item()

    return {"ASR": asr, "Delta": score_adv - score_orig, "L2": l2}
