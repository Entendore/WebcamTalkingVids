from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QComboBox, QSlider, QSpinBox, 
                               QGroupBox, QTextEdit, QProgressBar)
from PySide6.QtGui import QTextCursor

class TeleprompterWidget(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.is_recording = False
        self.current_bullet_idx = 0
        self.bullets = []
        self.auto_scroll_timer = None
        self.setup_ui()
        self.load_settings(settings)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        tele_group = QGroupBox(" 📝 Teleprompter ")
        tele_layout = QVBoxLayout(tele_group)
        tele_layout.setSpacing(8)
        tele_layout.setContentsMargins(12, 18, 12, 12)
        
        self.tele_text = QTextEdit()
        self.tele_text.setStyleSheet(f"font-size: {self.settings.get('tele_font', 18)}pt; line-height: 1.6; padding: 16px;")
        self.tele_text.setMinimumHeight(200)
        tele_layout.addWidget(self.tele_text, stretch=1)
        
        self.tele_progress = QProgressBar()
        self.tele_progress.setMaximumHeight(8)
        self.tele_progress.setTextVisible(False)
        self.tele_progress.setStyleSheet("QProgressBar { background-color: #2d2d2d; border: none; border-radius: 4px; } QProgressBar::chunk { background-color: #ff4444; border-radius: 4px; }")
        tele_layout.addWidget(self.tele_progress)
        
        self.tele_counter_label = QLabel("📊 Bullet: 0 / 0")
        self.tele_counter_label.setStyleSheet("color: #888; font-size: 12px; font-weight: bold; padding: 2px 0;")
        tele_layout.addWidget(self.tele_counter_label)
        
        tele_ctrl = QHBoxLayout()
        tele_ctrl.setSpacing(8)
        lbl = QLabel("Mode:"); lbl.setMinimumWidth(50); tele_ctrl.addWidget(lbl)
        self.tele_mode_combo = QComboBox()
        self.tele_mode_combo.addItems(["Manual Scroll", "Bullet Points", "Auto Scroll"])
        self.tele_mode_combo.currentTextChanged.connect(self.on_tele_mode_changed)
        tele_ctrl.addWidget(self.tele_mode_combo)
        
        tele_ctrl.addSpacing(12)
        lbl = QLabel("Font:"); lbl.setMinimumWidth(40); tele_ctrl.addWidget(lbl)
        self.tele_font_spin = QSpinBox()
        self.tele_font_spin.setRange(8, 72)
        self.tele_font_spin.valueChanged.connect(self.update_tele_font)
        tele_ctrl.addWidget(self.tele_font_spin)
        
        tele_ctrl.addSpacing(12)
        lbl = QLabel("Visible:"); lbl.setMinimumWidth(50); tele_ctrl.addWidget(lbl)
        self.tele_max_visible_spin = QSpinBox()
        self.tele_max_visible_spin.setRange(3, 15)
        self.tele_max_visible_spin.setToolTip("Number of bullets visible in preview")
        self.tele_max_visible_spin.valueChanged.connect(self.update_bullet_display)
        tele_ctrl.addWidget(self.tele_max_visible_spin)
        tele_ctrl.addStretch()
        
        self.tele_prev_btn = QPushButton("◀ Prev"); self.tele_prev_btn.setFixedSize(70, 28)
        self.tele_prev_btn.clicked.connect(self.prev_bullet); tele_ctrl.addWidget(self.tele_prev_btn)
        self.tele_next_btn = QPushButton("Next ▶"); self.tele_next_btn.setFixedSize(70, 28)
        self.tele_next_btn.clicked.connect(self.next_bullet); tele_ctrl.addWidget(self.tele_next_btn)
        tele_layout.addLayout(tele_ctrl)
        
        auto_scroll_row = QHBoxLayout()
        auto_scroll_row.setSpacing(8)
        lbl = QLabel("Auto-Scroll Speed:"); lbl.setMinimumWidth(130); auto_scroll_row.addWidget(lbl)
        self.tele_scroll_slider = QSlider(Qt.Horizontal)
        self.tele_scroll_slider.setRange(0, 100)
        self.tele_scroll_slider.valueChanged.connect(self.update_auto_scroll_speed)
        auto_scroll_row.addWidget(self.tele_scroll_slider, stretch=1)
        self.tele_scroll_label = QLabel("Off")
        self.tele_scroll_label.setMinimumWidth(50)
        self.tele_scroll_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.tele_scroll_label.setStyleSheet("color: #ff6666; font-weight: bold;")
        auto_scroll_row.addWidget(self.tele_scroll_label)
        tele_layout.addLayout(auto_scroll_row)
        
        self.tele_hint_label = QLabel("▶  [Space / →] Next    ◀  [←] Previous    [Ctrl + / −] Font Size    [Ctrl+Enter] Jump to Start    [Ctrl+End] Jump to End")
        self.tele_hint_label.setStyleSheet("color: #888888; font-size: 11px; padding: 4px 0;")
        self.tele_hint_label.setWordWrap(True)
        tele_layout.addWidget(self.tele_hint_label)
        
        help_label = QLabel("💡 <b>Manual Scroll:</b> Paste your script and use arrow keys to scroll.<br>💡 <b>Bullet Points:</b> Each line becomes a bullet. Current is highlighted, next/prev shown for context.<br>💡 <b>Auto Scroll:</b> Script scrolls automatically at the set speed.")
        help_label.setStyleSheet("color: #666; font-size: 11px; padding: 8px; background: #2d2d2d; border-radius: 4px;")
        help_label.setWordWrap(True)
        tele_layout.addWidget(help_label)
        
        layout.addWidget(tele_group)

    def load_settings(self, settings):
        self.tele_font_spin.setValue(settings.get("tele_font", 18))
        self.tele_mode_combo.setCurrentText(settings.get("tele_mode", "Manual Scroll"))
        self.tele_scroll_slider.setValue(settings.get("tele_auto_scroll_speed", 0))
        self.tele_max_visible_spin.setValue(settings.get("tele_max_visible", 5))

    def save_settings(self):
        # Explicitly capturing everything originally listed in DEFAULT_SETTINGS
        return {
            "tele_font": self.tele_font_spin.value(),
            "tele_mode": self.tele_mode_combo.currentText(),
            "tele_auto_scroll_speed": self.tele_scroll_slider.value(),
            "tele_max_visible": self.tele_max_visible_spin.value(),
            "tele_text_color": self.settings.get("tele_text_color", "#ffffff"),
            "tele_highlight_color": self.settings.get("tele_highlight_color", "#ff4444")
        }

    def on_tele_mode_changed(self):
        mode = self.tele_mode_combo.currentText()
        if mode == "Auto Scroll": self.tele_text.setReadOnly(True)
        else: self.tele_text.setReadOnly(False)
        if mode == "Bullet Points": self.tele_scroll_slider.setEnabled(False)
        else: self.tele_scroll_slider.setEnabled(True)

    def update_auto_scroll_speed(self, val):
        if val == 0:
            self.tele_scroll_label.setText("Off")
            if self.auto_scroll_timer: self.auto_scroll_timer.stop(); self.auto_scroll_timer = None
        else:
            self.tele_scroll_label.setText(f"{val}%")
            if self.is_recording and self.tele_mode_combo.currentText() in ["Auto Scroll", "Manual Scroll"]: self.start_auto_scroll(val)

    def start_auto_scroll(self, speed):
        if self.auto_scroll_timer: self.auto_scroll_timer.stop()
        interval = max(10, int(200 - speed * 1.9))
        self.auto_scroll_timer = QTimer(self)
        self.auto_scroll_timer.timeout.connect(lambda: self.scroll_manual(1))
        self.auto_scroll_timer.start(interval)

    def handle_keypress(self, event):
        mode = self.tele_mode_combo.currentText()
        if mode == "Bullet Points":
            if event.key() in [Qt.Key_Space, Qt.Key_Right, Qt.Key_Down]: self.next_bullet(); event.accept(); return
            elif event.key() in [Qt.Key_Left, Qt.Key_Up]: self.prev_bullet(); event.accept(); return
        elif mode in ["Manual Scroll", "Auto Scroll"]:
            if event.key() in [Qt.Key_Space, Qt.Key_Right, Qt.Key_Down]: self.scroll_manual(1); event.accept(); return
            elif event.key() in [Qt.Key_Left, Qt.Key_Up]: self.scroll_manual(-1); event.accept(); return
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal: self.increase_tele_font(); event.accept(); return
            elif event.key() == Qt.Key_Minus: self.decrease_tele_font(); event.accept(); return
            elif event.key() == Qt.Key_Return: self.current_bullet_idx = 0; self.update_bullet_display(); event.accept(); return

    def increase_tele_font(self):
        self.tele_font_spin.setValue(self.tele_font_spin.value() + 1)

    def decrease_tele_font(self):
        self.tele_font_spin.setValue(self.tele_font_spin.value() - 1)

    def update_tele_font(self):
        if self.tele_mode_combo.currentText() == "Bullet Points" and self.is_recording: self.update_bullet_display()
        else: self.tele_text.setStyleSheet(f"font-size: {self.tele_font_spin.value()}pt; line-height: 1.6; padding: 16px;")

    def scroll_manual(self, direction):
        scrollbar = self.tele_text.verticalScrollBar()
        step = self.tele_text.fontMetrics().height() * 2
        scrollbar.setValue(scrollbar.value() + (step * direction))

    def next_bullet(self):
        if self.current_bullet_idx < len(self.bullets) - 1:
            self.current_bullet_idx += 1; self.update_bullet_display()

    def prev_bullet(self):
        if self.current_bullet_idx > 0:
            self.current_bullet_idx -= 1; self.update_bullet_display()

    def update_bullet_display(self):
        if not self.bullets: return
        base_font = self.tele_font_spin.value(); max_visible = self.tele_max_visible_spin.value()
        half = max_visible // 2
        start_idx = max(0, self.current_bullet_idx - half)
        end_idx = min(len(self.bullets), start_idx + max_visible)
        if end_idx - start_idx < max_visible: start_idx = max(0, end_idx - max_visible)
        html = ""
        if start_idx > 0: html += f"<p style='color:#444445; font-size:{base_font - 4}pt; text-align:center; margin:4px 0;'>··· ({start_idx} more above) ···</p>"
        for i in range(start_idx, end_idx):
            bullet = self.bullets[i]; distance = abs(i - self.current_bullet_idx)
            if i == self.current_bullet_idx:
                html += f"<div style='background-color: #2a1a1a; border-left: 4px solid #ff4444; padding: 16px 12px; margin: 12px 0; border-radius: 4px;'><p style='color:#ff4444; font-size:{base_font + 8}pt; font-weight:bold; margin:0; line-height:1.4;'>▶ {bullet}</p></div>"
            elif distance == 1: html += f"<p style='color:#aaaabb; font-size:{base_font}pt; margin:8px 0; padding-left:20px;'>{bullet}</p>"
            elif distance == 2: html += f"<p style='color:#777788; font-size:{base_font - 2}pt; margin:6px 0; padding-left:20px;'>{bullet}</p>"
            else: html += f"<p style='color:#555566; font-size:{base_font - 4}pt; margin:4px 0; padding-left:20px;'>{bullet}</p>"
        if end_idx < len(self.bullets):
            remaining = len(self.bullets) - end_idx
            html += f"<p style='color:#444455; font-size:{base_font - 4}pt; text-align:center; margin:4px 0;'>··· ({remaining} more below) ···</p>"
        self.tele_text.setHtml(html)
        if len(self.bullets) > 0:
            progress_val = int((self.current_bullet_idx + 1) / len(self.bullets) * 100)
            self.tele_progress.setValue(progress_val)
            self.tele_counter_label.setText(f"📊 Bullet: {self.current_bullet_idx + 1} / {len(self.bullets)}  ({progress_val}%)")
        cursor = self.tele_text.textCursor()
        block = self.tele_text.document().findBlockByNumber(self.current_bullet_idx - start_idx)
        if block.isValid(): cursor.setPosition(block.position()); self.tele_text.setTextCursor(cursor); self.tele_text.ensureCursorVisible()

    def start_recording(self):
        self.is_recording = True
        self.tele_text.setReadOnly(True)
        self.tele_text.setFocusPolicy(Qt.NoFocus)
        mode = self.tele_mode_combo.currentText()
        if mode == "Bullet Points":
            self.bullets = [line.strip() for line in self.tele_text.toPlainText().split('\n') if line.strip()]
            self.current_bullet_idx = 0
            if self.bullets: self.update_bullet_display()
        elif mode in ["Manual Scroll", "Auto Scroll"]:
            cursor = self.tele_text.textCursor(); cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.tele_text.setTextCursor(cursor); self.tele_text.verticalScrollBar().setValue(0)
            if mode == "Auto Scroll" and self.tele_scroll_slider.value() > 0:
                self.start_auto_scroll(self.tele_scroll_slider.value())

    def stop_recording(self):
        self.is_recording = False
        if self.auto_scroll_timer: self.auto_scroll_timer.stop(); self.auto_scroll_timer = None
        self.tele_text.setReadOnly(False)
        self.tele_text.setFocusPolicy(Qt.StrongFocus)
        self.tele_text.setHtml("")
        self.tele_progress.setValue(0)
        self.tele_counter_label.setText("📊 Bullet: 0 / 0")