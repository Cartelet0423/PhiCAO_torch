import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

from utils import load_dv_as_numpy, resize_stack
from phicao import run_phicao_subregion
from viewer import OrthogonalViewer

def main():
    # 1. Loading Data
    # NOTE: Replace 'path' with your actual .dv file path.
    path = "TestImageSet/Fig1d_beads_original.dv"
    imgs = load_dv_as_numpy(path, width=256, height=256, header_size=512)
    imgs = imgs / imgs.max()

    params = {
        'dz': 0.125, 'dy': 0.08, 'dx': 0.08,
        'NA': 1.42, 'lambda_em': 0.528, 'n_imm': 1.518
    }

    # =========================================================
    # Option B: Subregion Optimization
    # =========================================================
    print("Running Subregion PhiCAO...")
    imgs_resized = resize_stack(imgs, size=(512, 512))
    params.update({'dy': 0.04, 'dx': 0.04})
    
    grid_y, grid_x = 3, 3
    corrected_img_sr, patch_data = run_phicao_subregion(
        image_stack=imgs_resized, grid_y=grid_y, grid_x=grid_x,
        target_modes=np.r_[5:16], epochs=50, **params
    )
    
    print("Estimated Fringe Zernike Coefficients:")
    for patch_idx, pd in patch_data.items():
        print(f"Patch {patch_idx}:")
        for mode, coeff in pd['alphas'].items():
            print(f"  Mode {mode}: {coeff/params['lambda_em']:.4f} waves")

    # Plot Wavefronts (Spatially Variant)
    fig, axs = plt.subplots(grid_y, grid_x, figsize=(6, 6))
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    axs = np.atleast_2d(axs)
    wavefronts = [pd['wavefront'] for pd in patch_data.values()]
    for ax, p in zip(axs.flatten(), wavefronts):
        ax.imshow(np.fft.fftshift(p/params['lambda_em']), cmap='nipy_spectral_r', vmin=-0.2, vmax=0.2)
        ax.axis('off')
    plt.colorbar(axs[0, 0].images[0], ax=axs, orientation='vertical', label='Phase (radians)')
    fig.suptitle("Subregion Wavefronts")
    plt.show(block=False)

    # Display result using Orthogonal Viewer
    app_sv = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer_sv = OrthogonalViewer(imgs_resized, corrected_img_sr)
    viewer_sv.setWindowTitle("Subregion Correction Result")
    viewer_sv.show()
    app_sv.exec_()

if __name__ == '__main__':
    main()