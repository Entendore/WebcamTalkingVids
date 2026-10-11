from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider, QPushButton

def dark_stylesheet():
    return """
        QMainWindow, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Segoe UI, Arial; font-size: 13px; }
        QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 12px; padding: 14px 10px 10px 10px; font-weight: bold; color: #ff4444; background-color: #252525; }
        QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; background-color: #1e1e1e; }
        QPushButton { background-color: #3a3a3a; color: #ffffff; border: 1px solid #4a4a4a; padding: 7px 14px; border-radius: 4px; }
        QPushButton:hover { background-color: #4a4a4a; border-color: #5a5a5a; }
        QPushButton:pressed { background-color: #2d2d2d; }
        QPushButton:disabled { background-color: #2d2d2d; color: #666666; border-color: #333333; }
        QComboBox, QSpinBox, QLineEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #3a3a3a; border-radius: 4px; padding: 5px 8px; selection-background-color: #ff4444; }
        QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #5a5a5a; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView { background-color: #2d2d2d; color: #ffffff; selection-background-color: #ff4444; border: 1px solid #3a3a3a; outline: none; }
        QSlider::groove:horizontal { border: none; height: 6px; background: #2d2d2d; border-radius: 3px; }
        QSlider::sub-page:horizontal { background: #ff4444; border-radius: 3px; }
        QSlider::add-page:horizontal { background: #2d2d2d; border-radius: 3px; }
        QSlider::handle:horizontal { background: #e0e0e0; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; border: 2px solid #ff4444; }
        QSlider::handle:horizontal:hover { background: #ffffff; border-color: #ff6666; width: 18px; height: 18px; }
        QProgressBar { background-color: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 4px; text-align: center; color: #ffffff; }
        QProgressBar::chunk { background-color: #ff4444; border-radius: 3px; }
        QTextEdit { background-color: #2d2d2d; color: #ffffff; border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px; }
        QCheckBox { spacing: 8px; padding: 2px 0; }
        QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; }
        QCheckBox::indicator:unchecked { background-color: #2d2d2d; border: 1px solid #555555; border-radius: 3px; }
        QCheckBox::indicator:checked { background-color: #ff4444; border: 1px solid #ff4444; border-radius: 3px; }
        QCheckBox::indicator:hover { border-color: #ff6666; }
        QLabel { color: #d4d4d4; }
        QTabWidget::pane { border: 1px solid #3a3a3a; border-radius: 4px; background: #1e1e1e; }
        QTabBar::tab { background: #2d2d2d; color: #aaaaaa; padding: 8px 16px; border: 1px solid #3a3a3a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
        QTabBar::tab:selected { background: #252525; color: #ff4444; border-bottom: 2px solid #ff4444; }
        QTabBar::tab:hover { background: #353535; color: #ffffff; }
        QScrollBar:vertical { background: #2d2d2d; width: 10px; border: none; }
        QScrollBar::handle:vertical { background: #4a4a4a; min-height: 30px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #5a5a5a; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0; }
        QScrollBar:horizontal { background: #2d2d2d; height: 10px; border: none; }
        QScrollBar::handle:horizontal { background: #4a4a4a; min-width: 30px; border-radius: 5px; }
        QScrollBar::handle:horizontal:hover { background: #5a5a5a; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0; }
        QSplitter::handle { background-color: #3a3a3a; }
        QSplitter::handle:horizontal { width: 4px; }
        QSplitter::handle:vertical { height: 4px; }
        QSplitter::handle:hover { background-color: #ff4444; }
        QFrame[frameShape="4"] { background-color: #333333; max-height: 1px; margin: 4px 0; }
    """

def create_section_label(text):
    label = QLabel(text)
    label.setStyleSheet("QLabel { color: #888888; font-size: 11px; font-weight: bold; padding: 10px 0 2px 0; border-bottom: 1px solid #333333; }")
    return label

def create_slider_row(label_text, min_val, max_val, default_val, scale=1.0, suffix="", format_str=None, label_width=130, value_width=70, custom_formatter=None, show_reset=True, tooltip=""):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 1, 0, 1)
    layout.setSpacing(8)
    label = QLabel(label_text); label.setMinimumWidth(label_width); label.setToolTip(tooltip)
    layout.addWidget(label)
    slider = QSlider(Qt.Horizontal); slider.setRange(min_val, max_val); slider.setValue(int(default_val / scale)); slider.setToolTip(tooltip)
    layout.addWidget(slider, stretch=1)
    value_label = QLabel(); value_label.setMinimumWidth(value_width); value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    value_label.setStyleSheet("color: #ff6666; font-weight: bold; font-family: Consolas, 'Courier New', monospace;")
    def update_label(v):
        real_val = v / scale
        if custom_formatter: value_label.setText(custom_formatter(v, real_val))
        elif format_str: value_label.setText(format_str.format(real_val))
        else: value_label.setText(f"{real_val:.2f}{suffix}")
    slider.valueChanged.connect(update_label); update_label(slider.value())
    layout.addWidget(value_label)
    if show_reset:
        reset_btn = QPushButton("⟲"); reset_btn.setFixedSize(22, 22); reset_btn.setStyleSheet("padding: 0; margin: 0; font-size: 11px;")
        reset_btn.setToolTip("Reset to default"); reset_btn.clicked.connect(lambda: slider.setValue(int(default_val / scale)))
        layout.addWidget(reset_btn)
    return widget, slider, value_label
