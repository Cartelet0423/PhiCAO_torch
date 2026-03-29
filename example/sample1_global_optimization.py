import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

from utils import load_dv_as_numpy
from phicao import run_phicao
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
    target_modes = np.r_[5:51]

    # =========================================================
    # Option A: Global Optimization
    # =========================================================
    print("Running Global PhiCAO...")
    alphas, psf_final, corrected_imgs, wavefront = run_phicao(
        image_stack=imgs.copy(), target_modes=target_modes, epochs=150, **params
    )

    # Print estimated Zernike coefficients
    print("Estimated Fringe Zernike Coefficients:")
    for mode, coeff in alphas.items():
        print(f"Mode {mode}: {coeff/params['lambda_em']:.4f} waves")

    # Plot Wavefront (Global)
    plt.figure(figsize=(4, 4))
    plt.imshow(np.fft.fftshift(wavefront/params['lambda_em']), cmap='nipy_spectral_r')
    plt.colorbar(label='Phase ($\lambda$)')
    plt.title("Estimated Wavefront")
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)

    # Display result using Orthogonal Viewer
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viewer_global = OrthogonalViewer(imgs, corrected_imgs)
    viewer_global.setWindowTitle("Global Correction Result")
    viewer_global.show()
    app.exec_()

if __name__ == '__main__':
    main()