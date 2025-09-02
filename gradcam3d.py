import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from segmentation_model import UNet3D
from brats_dataset import BraTSDataset

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks. Use full_backward_hook if available, otherwise backward_hook.
        self.fh = target_layer.register_forward_hook(self._forward_hook)
        try:
            self.bh = target_layer.register_full_backward_hook(self._backward_hook)
        except Exception:
            # older PyTorch fallback
            self.bh = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # store the raw activations (keep graph)
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out[0] is the gradient w.r.t. the module's output
        self.gradients = grad_out[0]

    def remove_hooks(self):
        self.fh.remove()
        self.bh.remove()

    def generate(self, input_tensor, class_idx:int, upsample_to=None):
        """
        input_tensor: (B, C, D, H, W) tensor on same device as model
        class_idx: int channel index to explain (e.g. 1,2,3)
        upsample_to: tuple (D,H,W) to upsample CAM to input resolution
        returns: cam: (B, D, H, W) numpy array (values 0..1), logits: raw model output
        """
        device = next(self.model.parameters()).device
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # ensure model in eval for deterministic behavior (dropout off)
        self.model.eval()
        # forward (keep grad enabled so backward works)
        with torch.enable_grad():
            logits = self.model(input_tensor)  # (B, C, D, H, W)

            # score: mean over spatial dims for the chosen class
            if not isinstance(class_idx, int):
                raise ValueError("class_idx must be int (e.g. 1,2,3)")
            score = logits[:, class_idx].mean()

            # backward
            score.backward(retain_graph=False)

        # basic checks
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Backward hook didn't run. "
                               "Check that target_layer is correct and gradients are enabled.")

        # shapes: activations (B, K, d, h, w); gradients same
        A = self.activations  # (B, K, d, h, w)
        G = self.gradients    # (B, K, d, h, w)

        # compute weights: global average pooling of gradients over spatial dims
        weights = G.mean(dim=(2,3,4), keepdim=True)  # (B, K, 1, 1, 1)

        # weighted combination
        cam = F.relu((weights * A).sum(dim=1, keepdim=False))  # (B, d, h, w)

        # normalize per-sample
        cam_min = cam.amin(dim=(1,2,3), keepdim=True)
        cam_max = cam.amax(dim=(1,2,3), keepdim=True)
        denom = cam_max - cam_min
        # avoid divide by zero
        denom[denom == 0] = 1.0
        cam = (cam - cam_min) / denom

        # upsample to input size if requested
        if upsample_to is not None:
            cam = cam.unsqueeze(1)  # (B,1,d,h,w)
            cam = F.interpolate(cam, size=upsample_to, mode='trilinear', align_corners=False)
            cam = cam.squeeze(1)    # (B, D, H, W)

        return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()

if __name__ == '__main__':
    best_model = "best_unet3d.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet3D(n_channels=4, n_classes=4).to(device)
    model.load_state_dict(torch.load(best_model))
    target_layer = model.bottleneck  # pick the bottleneck
    
    gradcam = GradCAM3D(model, target_layer)

    val_ds   = BraTSDataset(root="data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData", split='val')    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    for img, seg, _ in val_loader:
        img, seg = img.to(device), seg.to(device)
        break

    cam_np, logits_np = gradcam.generate(img, class_idx=1, upsample_to=img.shape[2:])
    
    # Visualize a middle slice
    slice_idx = 155 // 2

    # Original image (choose a modality, e.g. FLAIR = channel 0)
    original_slice = img[0, 0, :, :, slice_idx].cpu().numpy()
    # GradCAM heatmap
    cam_slice = cam_np[0, :, :, slice_idx]
    

    # Plot side by side
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    # Original
    axes[0].imshow(original_slice, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # GradCAM overlay
    axes[1].imshow(original_slice, cmap="gray")
    axes[1].imshow(cam_slice, cmap="jet", alpha=0.5)
    axes[1].set_title("GradCAM class 1")
    axes[1].axis("off")

    cam_np, logits_np = gradcam.generate(img, class_idx=2, upsample_to=img.shape[2:])
    cam_slice = cam_np[0, :, :, slice_idx]

    axes[2].imshow(original_slice, cmap="gray")
    axes[2].imshow(cam_slice, cmap="jet", alpha=0.5)
    axes[2].set_title("GradCAM class 2")
    axes[2].axis("off")

    cam_np, logits_np = gradcam.generate(img, class_idx=3, upsample_to=img.shape[2:])
    cam_slice = cam_np[0, :, :, slice_idx]

    axes[3].imshow(original_slice, cmap="gray")
    axes[3].imshow(cam_slice, cmap="jet", alpha=0.5)
    axes[3].set_title("GradCAM class 3")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()