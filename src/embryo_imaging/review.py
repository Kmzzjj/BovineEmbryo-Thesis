"""Review developmental stage labels with their centre-crop images."""

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets


STATE_JSON = Path.home() / ".embryo_label_grid_state.json"
DEFAULT_PAGE_SIZE = 9


def load_state():
    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict):
    try:
        STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def find_center_img_path(row, csv_dir: Path, center_root: Path):
    candidates = (
        row.get("center_img_path"),
        center_root / f"{row['seq_id']}_center_320.jpg",
        row.get("filename"),
    )
    for value in candidates:
        if not isinstance(value, (str, Path)) or not value:
            continue
        path = Path(value).expanduser()
        for candidate in (csv_dir / path, path):
            if candidate.is_file():
                return str(candidate)
    return None


def csv_version(path: Path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def save_label_csv(frame, path: Path, expected_version):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")

    with TemporaryDirectory(prefix=".labels-", dir=path.parent) as staging:
        updated = Path(staging) / path.name
        frame.to_csv(updated, index=False, encoding="utf-8-sig")
        lock = QtCore.QLockFile(str(path.with_name(f"{path.name}.lock")))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            raise OSError("Could not lock the label CSV for saving.")
        try:
            if csv_version(path) != expected_version:
                raise ValueError("The CSV has changed on disk. Reload it before saving.")
            shutil.copymode(path, updated)
            shutil.copy2(path, backup)
            updated.replace(path)
        finally:
            lock.unlock()
    return backup


class EmbryoItemWidget(QtWidgets.QWidget):
    """显示一个胚胎：图片 + seq_id + 来源 + 标签下拉框"""

    label_changed = QtCore.pyqtSignal(int, str)  # row_idx, new_label

    def __init__(self, row_idx, row, class_list, image_path, parent=None):
        super().__init__(parent)
        self.row_idx = row_idx
        self.row = row
        self.class_list = class_list

        self.init_ui(image_path)

    def init_ui(self, image_path):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(4)

        
        self.img_label = QtWidgets.QLabel()
        self.img_label.setAlignment(QtCore.Qt.AlignCenter)
        self.img_label.setMinimumSize(160, 160)
        self.img_label.setMaximumSize(220, 220)
        self.img_label.setFrameShape(QtWidgets.QFrame.Box)

        if image_path is not None:
            pix = QtGui.QPixmap(image_path)
            if not pix.isNull():
                self.img_label.setPixmap(
                    pix.scaled(
                        200,
                        200,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )
            else:
                self.img_label.setText("图像读取失败")
        else:
            self.img_label.setText("未找到图像")

        layout.addWidget(self.img_label)

        
        info = QtWidgets.QLabel()
        seq_id = str(self.row.get("seq_id", ""))
        src = str(self.row.get("source", "?"))
        info.setText(f"{seq_id}\n[{src}]")
        info.setWordWrap(True)
        info.setAlignment(QtCore.Qt.AlignCenter)
        font = info.font()
        font.setPointSize(8)
        info.setFont(font)
        layout.addWidget(info)

        
        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)

        lab_label = QtWidgets.QLabel("Label:")
        font2 = lab_label.font()
        font2.setPointSize(8)
        lab_label.setFont(font2)
        h.addWidget(lab_label)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(self.class_list)
        cur_label = str(self.row.get("label", ""))
        if cur_label in self.class_list:
            self.combo.setCurrentIndex(self.class_list.index(cur_label))
        else:
            
            self.combo.addItem(cur_label)
            self.class_list.append(cur_label)
            self.combo.setCurrentIndex(len(self.class_list) - 1)
        self.combo.currentTextChanged.connect(self._on_label_changed)

        h.addWidget(self.combo)
        layout.addLayout(h)

    def _on_label_changed(self, text):
        self.label_changed.emit(self.row_idx, text)



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, label_csv: Path, center_root: Path):
        super().__init__()
        self.label_csv = label_csv
        self.center_root = center_root
        self.setWindowTitle("Embryo Label Grid Viewer (GUI + pseudo_600)")
        self.resize(1400, 900)

        self.df = None             
        self.filtered_indices = [] 
        self.class_list = []       
        self.page_size = DEFAULT_PAGE_SIZE
        self.cur_page = 0
        self.total_pages = 0
        self.dirty = False         

        self.state = load_state()

        self._init_ui()
        self.load_data()
        self.apply_filters(update_page_from_state=True)
        self.update_grid()

    
    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(5, 5, 5, 5)
        vbox.setSpacing(5)

        
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(10)

        
        ctrl.addWidget(QtWidgets.QLabel("Label 过滤:"))
        self.combo_label_filter = QtWidgets.QComboBox()
        self.combo_label_filter.addItem("All")
        self.combo_label_filter.currentTextChanged.connect(self._on_filter_changed)
        ctrl.addWidget(self.combo_label_filter)

        
        ctrl.addWidget(QtWidgets.QLabel("Source 过滤:"))
        self.combo_src_filter = QtWidgets.QComboBox()
        self.combo_src_filter.addItem("All")
        self.combo_src_filter.addItem("GUI")
        self.combo_src_filter.addItem("pseudo_600")
        self.combo_src_filter.currentTextChanged.connect(self._on_filter_changed)
        ctrl.addWidget(self.combo_src_filter)

        
        ctrl.addWidget(QtWidgets.QLabel("每页条数:"))
        self.spin_page_size = QtWidgets.QSpinBox()
        self.spin_page_size.setRange(1, 20)
        self.spin_page_size.setValue(
            int(self.state.get("page_size", DEFAULT_PAGE_SIZE))
        )
        self.spin_page_size.valueChanged.connect(self._on_page_size_changed)
        ctrl.addWidget(self.spin_page_size)

        
        self.chk_shuffle = QtWidgets.QCheckBox("随机顺序")
        self.chk_shuffle.setChecked(bool(self.state.get("shuffle", False)))
        self.chk_shuffle.stateChanged.connect(self._on_filter_changed)
        ctrl.addWidget(self.chk_shuffle)

        ctrl.addStretch()

        
        self.btn_save = QtWidgets.QPushButton("保存标签 (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_labels)
        ctrl.addWidget(self.btn_save)

        
        self.btn_reload = QtWidgets.QPushButton("重新加载 CSV")
        self.btn_reload.clicked.connect(self._on_reload_clicked)
        ctrl.addWidget(self.btn_reload)

        vbox.addLayout(ctrl)

        
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(5, 5, 5, 5)
        self.grid_layout.setSpacing(8)

        self.scroll_area.setWidget(self.grid_container)
        vbox.addWidget(self.scroll_area, 1)

        
        bottom = QtWidgets.QHBoxLayout()
        bottom.setSpacing(10)

        self.btn_prev = QtWidgets.QPushButton("上一页 (A / ←)")
        self.btn_prev.clicked.connect(self.prev_page)
        bottom.addWidget(self.btn_prev)

        self.btn_next = QtWidgets.QPushButton("下一页 (D / →)")
        self.btn_next.clicked.connect(self.next_page)
        bottom.addWidget(self.btn_next)

        bottom.addStretch()

        self.lbl_page_info = QtWidgets.QLabel("页码: 0 / 0")
        bottom.addWidget(self.lbl_page_info)

        vbox.addLayout(bottom)

        
        save_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_labels)

        left_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        left_shortcut.activated.connect(self.prev_page)
        a_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("A"), self)
        a_shortcut.activated.connect(self.prev_page)

        right_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        right_shortcut.activated.connect(self.next_page)
        d_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("D"), self)
        d_shortcut.activated.connect(self.next_page)

    
    def load_data(self):
        if not self.label_csv.is_file():
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"找不到标签 CSV：\n{self.label_csv}\n\n请先确认路径。",
            )
            sys.exit(1)

        self.csv_version = csv_version(self.label_csv)
        df = pd.read_csv(self.label_csv, encoding="utf-8-sig")
        if "seq_id" not in df.columns or "label" not in df.columns:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                "CSV 中必须包含 'seq_id' 和 'label' 列。",
            )
            sys.exit(1)

        
        if "source" not in df.columns:
            df["source"] = "GUI"

        self.df = df
        self.class_list = sorted(df["label"].astype(str).unique().tolist())

        
        self.combo_label_filter.blockSignals(True)
        self.combo_label_filter.clear()
        self.combo_label_filter.addItem("All")
        for c in self.class_list:
            self.combo_label_filter.addItem(c)
        self.combo_label_filter.blockSignals(False)

        
        prev_label = self.state.get("label_filter", "All")
        if prev_label in ["All"] + self.class_list:
            idx = self.combo_label_filter.findText(prev_label)
            if idx >= 0:
                self.combo_label_filter.setCurrentIndex(idx)

        prev_src = self.state.get("source_filter", "All")
        idx2 = self.combo_src_filter.findText(prev_src)
        if idx2 >= 0:
            self.combo_src_filter.setCurrentIndex(idx2)

        self.page_size = int(self.state.get("page_size", DEFAULT_PAGE_SIZE))
        self.spin_page_size.setValue(self.page_size)

    
    def apply_filters(self, update_page_from_state=False):
        """根据 label_filter / source_filter / shuffle 计算 filtered_indices"""
        if self.df is None:
            return

        label_f = self.combo_label_filter.currentText()
        src_f = self.combo_src_filter.currentText()

        if label_f != "All":
            mask = (self.df["label"].astype(str) == label_f)
        else:
            mask = pd.Series([True] * len(self.df), index=self.df.index)

        if src_f != "All":
            mask = mask & (self.df["source"].astype(str) == src_f)

        idxs = self.df[mask].index.tolist()

        if self.chk_shuffle.isChecked():
            import random

            random.shuffle(idxs)
        else:
            
            df_sub = self.df.loc[idxs].copy()
            if "dish" in df_sub.columns and "center_min" in df_sub.columns:
                df_sub = df_sub.sort_values(
                    by=["label", "dish", "center_min"], ascending=[True, True, True]
                )
            else:
                df_sub = df_sub.sort_values(by=["label"], ascending=True)
            idxs = df_sub.index.tolist()

        self.filtered_indices = idxs
        self.total_pages = math.ceil(len(self.filtered_indices) / self.page_size) if self.page_size > 0 else 0

        if update_page_from_state:
            self.cur_page = int(self.state.get("page", 0))
            if self.cur_page >= self.total_pages:
                self.cur_page = max(0, self.total_pages - 1)
        else:
            self.cur_page = 0

        self.update_page_label()

    def update_page_label(self):
        if self.total_pages == 0:
            self.lbl_page_info.setText("页码: 0 / 0 | 共 0 条")
        else:
            start = self.cur_page * self.page_size
            end = min(start + self.page_size, len(self.filtered_indices))
            self.lbl_page_info.setText(
                f"页码: {self.cur_page+1} / {self.total_pages} | "
                f"当前显示 {end-start} 条 / 过滤后 {len(self.filtered_indices)} 条 / 总 {len(self.df)} 条"
            )

    
    def clear_grid(self):
        
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def update_grid(self):
        self.clear_grid()

        if self.df is None or self.total_pages == 0:
            return

        start = self.cur_page * self.page_size
        end = min(start + self.page_size, len(self.filtered_indices))
        cur_indices = self.filtered_indices[start:end]

        
        cols = 3

        for i, idx in enumerate(cur_indices):
            row = self.df.loc[idx]
            r = i // cols
            c = i % cols

            image_path = find_center_img_path(row, self.label_csv.parent, self.center_root)
            widget = EmbryoItemWidget(idx, row, self.class_list, image_path, self)
            widget.label_changed.connect(self.on_label_changed)
            self.grid_layout.addWidget(widget, r, c)

        self.update_page_label()

        
        self.state["page"] = self.cur_page
        self.state["label_filter"] = self.combo_label_filter.currentText()
        self.state["source_filter"] = self.combo_src_filter.currentText()
        self.state["page_size"] = self.page_size
        self.state["shuffle"] = self.chk_shuffle.isChecked()
        save_state(self.state)

    
    def on_label_changed(self, row_idx, new_label):
        """单个胚胎标签变更"""
        if self.df is None:
            return
        self.df.loc[row_idx, "label"] = new_label
        self.dirty = True

    def _on_filter_changed(self, *args):
        self.apply_filters(update_page_from_state=False)
        self.update_grid()

    def _on_page_size_changed(self, val):
        self.page_size = int(val)
        self.apply_filters(update_page_from_state=False)
        self.update_grid()

    def prev_page(self):
        if self.cur_page > 0:
            self.cur_page -= 1
            self.update_grid()

    def next_page(self):
        if self.cur_page + 1 < self.total_pages:
            self.cur_page += 1
            self.update_grid()

    def _on_reload_clicked(self):
        if self.dirty:
            ret = QtWidgets.QMessageBox.question(
                self,
                "确认",
                "当前有尚未保存的标签修改，重新加载会丢失这些改动。\n确定要重新加载吗？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                return

        self.load_data()
        self.apply_filters(update_page_from_state=True)
        self.update_grid()
        self.dirty = False

    def save_labels(self):
        if self.df is None:
            return
        
        try:
            backup = save_label_csv(self.df, self.label_csv, self.csv_version)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"保存失败：\n{self.label_csv}\n\n{error}",
            )
            return

        self.csv_version = csv_version(self.label_csv)
        self.dirty = False
        QtWidgets.QMessageBox.information(
            self,
            "保存成功",
            f"已保存到：\n{self.label_csv}\n\n备份：\n{backup}",
        )

    def closeEvent(self, event):
        if self.dirty:
            ret = QtWidgets.QMessageBox.question(
                self,
                "确认退出",
                "当前有尚未保存的标签修改，确定要退出吗？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        event.accept()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Label CSV with seq_id and label columns")
    parser.add_argument("--image-root", type=Path, help="Centre crop directory (defaults to the CSV directory)")
    args = parser.parse_args()
    label_csv = args.labels.expanduser().resolve()
    if not label_csv.is_file():
        parser.error(f"label CSV does not exist: {label_csv}")
    center_root = args.image_root.expanduser().resolve() if args.image_root else label_csv.parent
    if not center_root.is_dir():
        parser.error(f"image directory does not exist: {center_root}")

    app = QtWidgets.QApplication([sys.argv[0]])
    win = MainWindow(label_csv, center_root)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
