import torch

from torchhd.tensors.mcr import BaseMCRTensor

class CGRTensor(BaseMCRTensor):
    """Cyclic Group Representation

    Proposed in `Understanding hyperdimensional computing for parallel single-pass learning`. The model is a integer quantized alternative to FHRR, similar to Modular Composite Representations (MCR), but with bundle by majority.
    """

    def bundle(self, other: "BaseMCRTensor", mod) -> "BaseMCRTensor":
        pass

    def multibundle(self, mod) -> "BaseMCRTensor":
        """Bundle multiple hypervectors"""
        val, _ = torch.mode(self, dim=-2)
        return val
