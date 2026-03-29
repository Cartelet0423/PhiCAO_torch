import numpy as np
import pyqtgraph as qt
from PyQt5 import QtWidgets, QtCore, QtGui

class OrthogonalViewer(QtWidgets.QWidget):
    def __init__(self, data1, data2):
        super().__init__()
        self.data1 = data1
        self.data2 = data2
        self.data = data2
        
        self.z_max, self.y_max, self.x_max = self.data.shape
        self.pos = [self.z_max // 2, self.y_max // 2, self.x_max // 2]

        font = QtGui.QFont("Arial", 10)
        self.setFont(font)
        
        self.init_ui()

    def init_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        top_panel = QtWidgets.QHBoxLayout()
        top_panel.setContentsMargins(10, 0, 10, 0)
        
        self.btn_data1 = QtWidgets.QPushButton("Original")
        self.btn_data1.setCheckable(True)
        
        self.btn_data2 = QtWidgets.QPushButton("Corrected")
        self.btn_data2.setCheckable(True)
        self.btn_data2.setChecked(True)
        
        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.btn_group.addButton(self.btn_data1)
        self.btn_group.addButton(self.btn_data2)
        self.btn_group.buttonClicked.connect(self.on_dataset_changed)
        
        top_panel.addWidget(self.btn_data1)
        top_panel.addWidget(self.btn_data2)
        top_panel.addSpacing(20)

        # 2. ガンマ調整パネル
        gamma_label_title = QtWidgets.QLabel("Gamma:")
        
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)
        self.gamma_slider.setValue(100)
        self.gamma_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.gamma_slider.setTickInterval(50)
        
        self.gamma_value_label = QtWidgets.QLabel("1.00")
        self.gamma_value_label.setMinimumWidth(40)
        
        top_panel.addWidget(gamma_label_title)
        top_panel.addWidget(self.gamma_slider)
        top_panel.addWidget(self.gamma_value_label)
        
        self.main_layout.addLayout(top_panel)

        self.v_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.v_splitter.setStyleSheet("QSplitter::handle { background-color: #555; }")
        
        self.top_h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        self.bottom_h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.win_xy = qt.GraphicsLayoutWidget()
        self.win_zy = qt.GraphicsLayoutWidget()
        self.win_xz = qt.GraphicsLayoutWidget()
        self.win_hist = qt.GraphicsLayoutWidget()

        self.top_h_splitter.addWidget(self.win_xy)
        self.top_h_splitter.addWidget(self.win_zy)
        self.bottom_h_splitter.addWidget(self.win_xz)
        self.bottom_h_splitter.addWidget(self.win_hist)

        self.v_splitter.addWidget(self.top_h_splitter)
        self.v_splitter.addWidget(self.bottom_h_splitter)
        
        self.main_layout.addWidget(self.v_splitter)

        self.top_h_splitter.setStretchFactor(0, 3)
        self.top_h_splitter.setStretchFactor(1, 1)
        
        self.bottom_h_splitter.setStretchFactor(0, 3)
        self.bottom_h_splitter.setStretchFactor(1, 1)
        
        self.v_splitter.setStretchFactor(0, 3)
        self.v_splitter.setStretchFactor(1, 1)

        self.v_xy = self.win_xy.addPlot(title="XY Plane (Axial)")
        self.v_zy = self.win_zy.addPlot(title="ZY Plane (Sagittal)")
        self.v_xz = self.win_xz.addPlot(title="XZ Plane (Coronal)")
        
        self.img_xy = qt.ImageItem()
        self.img_zy = qt.ImageItem()
        self.img_xz = qt.ImageItem()
    
        for v, img in zip([self.v_xy, self.v_zy, self.v_xz], 
                          [self.img_xy, self.img_zy, self.img_xz]):
            v.addItem(img)
            v.setAspectLocked(True)

        self.hist = qt.HistogramLUTItem()
        self.hist.setImageItem(self.img_xy)
        self.win_hist.addItem(self.hist)

        self.line_xy_x = qt.InfiniteLine(pos=self.pos[2], angle=90, movable=True, pen=qt.mkPen('r', width=.5))
        self.line_xy_y = qt.InfiniteLine(pos=self.pos[1], angle=0,  movable=True, pen=qt.mkPen('g', width=.5))
        
        self.line_zy_z = qt.InfiniteLine(pos=self.pos[0], angle=90, movable=True, pen=qt.mkPen('b', width=.5))
        self.line_zy_y = qt.InfiniteLine(pos=self.pos[1], angle=0,  movable=True, pen=qt.mkPen('g', width=.5))
        
        self.line_xz_x = qt.InfiniteLine(pos=self.pos[2], angle=90, movable=True, pen=qt.mkPen('r', width=.5))
        self.line_xz_z = qt.InfiniteLine(pos=self.pos[0], angle=0,  movable=True, pen=qt.mkPen('b', width=.5))
        
        self.target_xy = qt.TargetItem(pos=(self.pos[2], self.pos[1]), size=15, movable=True, pen=qt.mkPen('w', width=.5))
        self.target_zy = qt.TargetItem(pos=(self.pos[0], self.pos[1]), size=15, movable=True, pen=qt.mkPen('w', width=.5))
        self.target_xz = qt.TargetItem(pos=(self.pos[2], self.pos[0]), size=15, movable=True, pen=qt.mkPen('w', width=.5))

        self.line_xy_x.setBounds([0, self.x_max - 1])
        self.line_xy_y.setBounds([0, self.y_max - 1])
        
        self.line_zy_z.setBounds([0, self.z_max - 1])
        self.line_zy_y.setBounds([0, self.y_max - 1])
        
        self.line_xz_x.setBounds([0, self.x_max - 1])
        self.line_xz_z.setBounds([0, self.z_max - 1])

        items_to_front = [
            self.line_xy_x, self.line_xy_y, self.target_xy,
            self.line_zy_z, self.line_zy_y, self.target_zy,
            self.line_xz_x, self.line_xz_z, self.target_xz
        ]
        for item in items_to_front:
            item.setZValue(10)

        self.v_xy.addItem(self.line_xy_x); self.v_xy.addItem(self.line_xy_y); self.v_xy.addItem(self.target_xy)
        self.v_zy.addItem(self.line_zy_z); self.v_zy.addItem(self.line_zy_y); self.v_zy.addItem(self.target_zy)
        self.v_xz.addItem(self.line_xz_x); self.v_xz.addItem(self.line_xz_z); self.v_xz.addItem(self.target_xz)

        self.line_xy_x.sigPositionChanged.connect(lambda l: self.update_from_line(2, l.value()))
        self.line_xz_x.sigPositionChanged.connect(lambda l: self.update_from_line(2, l.value()))
        self.line_xy_y.sigPositionChanged.connect(lambda l: self.update_from_line(1, l.value()))
        self.line_zy_y.sigPositionChanged.connect(lambda l: self.update_from_line(1, l.value()))
        self.line_zy_z.sigPositionChanged.connect(lambda l: self.update_from_line(0, l.value()))
        self.line_xz_z.sigPositionChanged.connect(lambda l: self.update_from_line(0, l.value()))

        self.target_xy.sigPositionChanged.connect(self.update_from_target_xy)
        self.target_zy.sigPositionChanged.connect(self.update_from_target_zy)
        self.target_xz.sigPositionChanged.connect(self.update_from_target_xz)

        self.hist.sigLevelsChanged.connect(self.sync_levels)
        self.hist.gradient.sigGradientChanged.connect(self.sync_lut)
        
        self.gamma_slider.valueChanged.connect(self.update_gamma)

        self.update_images(auto_levels=True)
        self.sync_lut()
        self.sync_levels()

    def on_dataset_changed(self, button):
        """画像切り替えボタンがクリックされた時の処理"""
        if button == self.btn_data1:
            self.data = self.data1
        else:
            self.data = self.data2
            
        # データを切り替えたので、コントラストレベルを自動調整して再描画
        self.update_images(auto_levels=True)

    def update_from_target_xy(self, item):
        x, y = item.pos()
        new_x = int(np.clip(x, 0, self.x_max - 1))
        new_y = int(np.clip(y, 0, self.y_max - 1))
        
        if x != new_x or y != new_y:
            item.blockSignals(True)
            item.setPos(new_x, new_y)
            item.blockSignals(False)

        if self.pos[2] == new_x and self.pos[1] == new_y:
            return
            
        self.pos[2] = new_x
        self.pos[1] = new_y
        self.sync_ui_elements()
        self.update_images(auto_levels=False)

    def update_from_target_zy(self, item):
        z, y = item.pos()
        new_z = int(np.clip(z, 0, self.z_max - 1))
        new_y = int(np.clip(y, 0, self.y_max - 1))
        
        if z != new_z or y != new_y:
            item.blockSignals(True)
            item.setPos(new_z, new_y)
            item.blockSignals(False)

        if self.pos[0] == new_z and self.pos[1] == new_y:
            return
            
        self.pos[0] = new_z
        self.pos[1] = new_y
        self.sync_ui_elements()
        self.update_images(auto_levels=False)

    def update_from_target_xz(self, item):
        x, z = item.pos()
        new_x = int(np.clip(x, 0, self.x_max - 1))
        new_z = int(np.clip(z, 0, self.z_max - 1))
        
        if x != new_x or z != new_z:
            item.blockSignals(True)
            item.setPos(new_x, new_z)
            item.blockSignals(False)

        if self.pos[2] == new_x and self.pos[0] == new_z:
            return
            
        self.pos[2] = new_x
        self.pos[0] = new_z
        self.sync_ui_elements()
        self.update_images(auto_levels=False)

    def update_from_line(self, axis, value):
        max_val = [self.z_max, self.y_max, self.x_max][axis]
        val = int(np.clip(value, 0, max_val - 1))
        
        if self.pos[axis] == val:
            return
        
        self.pos[axis] = val
        self.sync_ui_elements()
        self.update_images(auto_levels=False)


    def sync_ui_elements(self):
        z, y, x = self.pos
        
        elements = [
            self.line_xy_x, self.line_xy_y, self.target_xy,
            self.line_zy_z, self.line_zy_y, self.target_zy,
            self.line_xz_x, self.line_xz_z, self.target_xz
        ]

        for el in elements:
            el.blockSignals(True)
            
        self.line_xy_x.setValue(x)
        self.line_xy_y.setValue(y)
        self.target_xy.setPos(x, y)
        
        self.line_zy_z.setValue(z)
        self.line_zy_y.setValue(y)
        self.target_zy.setPos(z, y)
        
        self.line_xz_x.setValue(x)
        self.line_xz_z.setValue(z)
        self.target_xz.setPos(x, z)
            
        for el in elements:
            el.blockSignals(False)

    def update_gamma(self):
        gamma = self.gamma_slider.value() / 100.0
        self.gamma_value_label.setText(f"{gamma:.2f}")
        self.update_images(auto_levels=False)

    def apply_gamma(self, img_data):
        gamma = self.gamma_slider.value() / 100.0
        if gamma == 1.0:
            return img_data
            
        d_min, d_max = self.data.min(), self.data.max()
        if d_max == d_min:
            return img_data
            
        norm_data = (img_data - d_min) / (d_max - d_min)
        gamma_data = np.power(norm_data, 1.0 / gamma)
        return gamma_data * (d_max - d_min) + d_min

    def update_images(self, auto_levels=False):
        slice_xy = self.apply_gamma(self.data[self.pos[0], :, :].T)
        slice_zy = self.apply_gamma(self.data[:, :, self.pos[2]])
        slice_xz = self.apply_gamma(self.data[:, self.pos[1], :].T)

        self.img_xy.setImage(slice_xy, autoLevels=auto_levels)
        self.img_zy.setImage(slice_zy, autoLevels=auto_levels)
        self.img_xz.setImage(slice_xz, autoLevels=auto_levels)

    def sync_levels(self):
        levels = self.hist.getLevels()
        self.img_zy.setLevels(levels)
        self.img_xz.setLevels(levels)

    def sync_lut(self):
        lut = self.hist.gradient.getLookupTable(512)
        self.img_zy.setLookupTable(lut)
        self.img_xz.setLookupTable(lut)