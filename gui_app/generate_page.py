# -*- coding: utf-8 -*-
import os
import tempfile
import cv2
import torch
from PyQt5.QtCore import Qt, QSize, QUrl
from PyQt5.QtGui import QDesktopServices, QIcon, QPixmap, QImage, QPalette, QColor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, 
                             QButtonGroup, QStyle, QLayout)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from qfluentwidgets import (PushButton, PrimaryPushButton, TextEdit, LineEdit, 
                            ComboBox, Slider, SwitchButton, ProgressBar, 
                            BodyLabel, SubtitleLabel, CardWidget, FluentIcon, 
                            InfoBar, InfoBarPosition, RadioButton, ScrollArea)
from gui_app import config as cfg

class DragDropZone(CardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.path = ""
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)
        
        # Preview Box
        self.lbl_preview = BodyLabel(self)
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setFixedSize(140, 140)
        self.lbl_preview.setStyleSheet("background-color: #f3f3f3; border: 2px dashed #ccc; border-radius: 8px;")
        self.lbl_preview.setText("📺 Chưa có tệp")
        self.layout.addWidget(self.lbl_preview, 0, Qt.AlignCenter)
        
        self.lbl_info = BodyLabel("Kéo thả ảnh hoặc video vào đây", self)
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color: #666; font-size: 11px;")
        self.layout.addWidget(self.lbl_info)
        
        self.btn = PushButton("Chọn tệp", self)
        self.btn.clicked.connect(self._select_file)
        self.layout.addWidget(self.btn, 0, Qt.AlignCenter)

    def _select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh hoặc video chân dung", "",
            "Tệp hình ảnh/video (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.set_path(file_path)

    def set_path(self, path):
        self.path = path
        if not path or not os.path.exists(path):
            self.lbl_preview.setPixmap(QPixmap())
            self.lbl_preview.setText("📺 Chưa có tệp")
            self.lbl_info.setText("Kéo thả ảnh hoặc video vào đây")
            self.lbl_info.setStyleSheet("color: #666; font-size: 11px;")
            return

        self.lbl_info.setText(os.path.basename(path))
        self.lbl_info.setStyleSheet("color: #0078D4; font-weight: bold; font-size: 12px;")
        
        # Load preview
        ext = path.lower()
        if ext.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(QSize(136, 136), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_preview.setPixmap(scaled)
            else:
                self.lbl_preview.setText("🖼️ Lỗi nạp ảnh")
        elif ext.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(QSize(136, 136), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_preview.setPixmap(scaled)
            else:
                self.lbl_preview.setText("🎬 (Video)")
        else:
            self.lbl_preview.setText("📁 Tệp khác")

    def get_path(self):
        return self.path

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.set_path(path)

class GeneratePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._output_path = ""
        self._init_ui()

    def _init_ui(self):
        # Create main layout for the Page widget
        self.main_page_layout = QVBoxLayout(self)
        self.main_page_layout.setContentsMargins(0, 0, 0, 0)
        self.main_page_layout.setSpacing(0)

        # Create the ScrollArea as a child
        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("ScrollArea { border: none; background-color: transparent; }")
        
        self.main_page_layout.addWidget(self.scroll_area)

        # Container widget for scroll area content
        self.scroll_widget = QWidget()
        self.scroll_widget.setObjectName("scroll_widget")
        self.scroll_widget.setStyleSheet("QWidget#scroll_widget { background-color: transparent; }")
        self.scroll_widget.setMinimumSize(1000, 1050)
        self.scroll_area.setWidget(self.scroll_widget)

        # Main layout inside scroll container
        main_layout = QHBoxLayout(self.scroll_widget)
        main_layout.setContentsMargins(60, 50, 15, 15)
        main_layout.setSpacing(20)
        main_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        # ── CỘT TRÁI: NHẬP LIỆU & XỬ LÝ ────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        # 1. Ảnh/Video nguồn
        left_layout.addWidget(SubtitleLabel("1. Hình ảnh hoặc Video chân dung"))
        self.drop_zone = DragDropZone(self)
        self.drop_zone.setFixedHeight(230)
        left_layout.addWidget(self.drop_zone)

        # 2. Âm thanh / Kịch bản
        left_layout.addWidget(SubtitleLabel("2. Âm thanh / Kịch bản"))
        audio_card = CardWidget()
        audio_layout = QVBoxLayout(audio_card)
        audio_layout.setSpacing(10)

        self._btn_group = QButtonGroup(self)
        self._rb_script = RadioButton()
        self._rb_script.setText("Nhập kịch bản (TTS)")
        self._rb_script.setChecked(True)
        self._rb_script.toggled.connect(self._toggle_audio_mode)
        
        self._rb_audio = RadioButton()
        self._rb_audio.setText("Sử dụng file âm thanh")
        self._rb_audio.toggled.connect(self._toggle_audio_mode)

        self._btn_group.addButton(self._rb_script)
        self._btn_group.addButton(self._rb_audio)

        audio_layout.addWidget(self._rb_script)
        
        # Nhập kịch bản widget
        self.script_widget = QWidget()
        script_v = QVBoxLayout(self.script_widget)
        script_v.setContentsMargins(0, 0, 0, 0)
        self.script_edit = TextEdit()
        self.script_edit.setPlaceholderText("Nhập đoạn văn bản tiếng Việt bạn muốn avatar đọc tại đây...")
        self.script_edit.setFixedHeight(95)
        script_v.addWidget(self.script_edit)

        # Tốc độ đọc & Giọng nói
        voice_layout = QHBoxLayout()
        self.voice_combo = ComboBox()
        voices = [
            ("Microsoft Hoài My (Nữ)", "vi-VN-HoaiMyNeural"),
            ("Microsoft Nam Minh (Nam)", "vi-VN-NamMinhNeural"),
            ("Microsoft An (Nam - En)", "en-US-AnaNeural"),
            ("Microsoft Brian (Nam - En)", "en-US-BrianNeural"),
        ]
        for name, val in voices:
            self.voice_combo.addItem(name, val)
        
        voice_layout.addWidget(BodyLabel("Giọng:"))
        voice_layout.addWidget(self.voice_combo, 1)

        self.rate_slider = Slider(Qt.Horizontal)
        self.rate_slider.setRange(-50, 50)
        self.rate_slider.setValue(cfg.tts_rate())
        self.rate_lbl = BodyLabel(f"{cfg.tts_rate()}%")
        self.rate_slider.valueChanged.connect(lambda v: self.rate_lbl.setText(f"{v:+}%"))
        
        voice_layout.addWidget(BodyLabel("Tốc độ:"))
        voice_layout.addWidget(self.rate_slider, 1)
        voice_layout.addWidget(self.rate_lbl)

        script_v.addLayout(voice_layout)
        audio_layout.addWidget(self.script_widget)

        audio_layout.addWidget(self._rb_audio)
        
        # Chọn file âm thanh widget
        self.audio_file_widget = QWidget()
        audio_file_h = QHBoxLayout(self.audio_file_widget)
        audio_file_h.setContentsMargins(0, 0, 0, 0)
        self.audio_path_edit = LineEdit()
        self.audio_path_edit.setPlaceholderText("Đường dẫn tệp âm thanh (.mp3, .wav)...")
        self.btn_browse_audio = PushButton("Duyệt")
        self.btn_browse_audio.clicked.connect(self._browse_audio)
        audio_file_h.addWidget(self.audio_path_edit, 1)
        audio_file_h.addWidget(self.btn_browse_audio)
        
        audio_layout.addWidget(self.audio_file_widget)
        left_layout.addWidget(audio_card)

        # 3. Thư mục đầu ra & Điều khiển xử lý
        left_layout.addWidget(SubtitleLabel("3. Thư mục đầu ra & Xử lý"))
        proc_card = CardWidget()
        proc_layout = QVBoxLayout(proc_card)
        proc_layout.setSpacing(10)

        # Hàng đầu ra
        out_layout = QHBoxLayout()
        self.output_dir_edit = LineEdit()
        self.output_dir_edit.setText(cfg.output_dir())
        btn_out = PushButton()
        btn_out.setIcon(QIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon)))
        btn_out.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(self.output_dir_edit, 1)
        out_layout.addWidget(btn_out)
        proc_layout.addLayout(out_layout)

        # GPU / CPU Status Indicator
        gpu_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else "CPU Only"
        self.device_lbl = BodyLabel(f"💻 Thiết bị xử lý: {'GPU' if gpu_avail else 'CPU'} ({gpu_name})")
        self.device_lbl.setStyleSheet("color: #0078D4; font-weight: bold;" if gpu_avail else "color: #d83b01; font-weight: bold;")
        proc_layout.addWidget(self.device_lbl)

        # Nút điều khiển
        proc_btn_layout = QHBoxLayout()
        self.generate_btn = PrimaryPushButton("Bắt đầu chạy")
        self.generate_btn.clicked.connect(self._start_generate)
        self.stop_btn = PushButton("Dừng")
        self.stop_btn.clicked.connect(self._stop_generate)
        self.stop_btn.setEnabled(False)
        self.open_btn = PushButton("Mở thư mục")
        self.open_btn.clicked.connect(self._open_video)
        self.open_btn.setEnabled(False)

        proc_btn_layout.addWidget(self.generate_btn, 1)
        proc_btn_layout.addWidget(self.stop_btn)
        proc_btn_layout.addWidget(self.open_btn)
        proc_layout.addLayout(proc_btn_layout)

        # Progress bar
        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        self.status_lbl = BodyLabel("Sẵn sàng.")
        self.status_lbl.setStyleSheet("color: #666;")
        proc_layout.addWidget(self.progress_bar)
        proc_layout.addWidget(self.status_lbl)

        left_layout.addWidget(proc_card)

        # ── CỘT PHẢI: KẾT QUẢ & CẤU HÌNH ──────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)

        # 4. Trình phát video đầu ra
        right_layout.addWidget(SubtitleLabel("4. Xem video kết quả trực tiếp"))
        self.video_card = CardWidget()
        video_v = QVBoxLayout(self.video_card)
        video_v.setContentsMargins(10, 10, 10, 10)
        video_v.setSpacing(8)

        self.video_widget = QVideoWidget()
        self.video_widget.setFixedHeight(230)
        # Set palette background to black to ensure native compatibility
        palette = self.video_widget.palette()
        palette.setColor(QPalette.Background, QColor(0, 0, 0))
        palette.setColor(QPalette.Window, QColor(0, 0, 0))
        self.video_widget.setPalette(palette)
        self.video_widget.setAutoFillBackground(True)
        video_v.addWidget(self.video_widget)

        # Controls của Video Player
        player_ctrl = QHBoxLayout()
        self.play_btn = PushButton("Phát")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_playback)
        player_ctrl.addWidget(self.play_btn)

        self.position_slider = Slider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._set_position)
        player_ctrl.addWidget(self.position_slider, 1)

        video_v.addLayout(player_ctrl)
        right_layout.addWidget(self.video_card)

        # Media Player Object
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.positionChanged.connect(self._position_changed)
        self.media_player.durationChanged.connect(self._duration_changed)
        # Auto-play only after media is fully loaded (avoids silent failure when
        # play() is called before the media service has buffered the file)
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.media_player.error.connect(self._on_player_error)
        self._auto_play_pending = False

        # 5. Cấu hình nâng cao MuseTalk
        right_layout.addWidget(SubtitleLabel("5. Tùy chọn xử lý nâng cao"))
        opts_card = CardWidget()
        opts_l = QVBoxLayout(opts_card)
        opts_l.setSpacing(10)

        def _row(label_text, widget):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            lbl = BodyLabel(label_text)
            lbl.setFixedWidth(180)
            row_l.addWidget(lbl)
            row_l.addWidget(widget, 1)
            return row_w

        # BBox shift
        bbox_w = QWidget()
        bbox_l = QHBoxLayout(bbox_w)
        bbox_l.setContentsMargins(0, 0, 0, 0)
        bbox_l.setSpacing(8)
        self.bbox_slider = Slider(Qt.Horizontal)
        self.bbox_slider.setRange(-50, 50)
        self.bbox_slider.setValue(cfg.bbox_shift())
        self.bbox_lbl = BodyLabel(str(self.bbox_slider.value()))
        self.bbox_lbl.setStyleSheet("color:#0078D4;font-weight:600;min-width:30px;")
        self.bbox_slider.valueChanged.connect(lambda v: self.bbox_lbl.setText(str(v)))
        bbox_l.addWidget(self.bbox_slider)
        bbox_l.addWidget(self.bbox_lbl)
        opts_l.addWidget(_row("Dịch khung mặt (BBox shift):", bbox_w))

        # Extra margin
        margin_w = QWidget()
        margin_l = QHBoxLayout(margin_w)
        margin_l.setContentsMargins(0, 0, 0, 0)
        margin_l.setSpacing(8)
        self.margin_slider = Slider(Qt.Horizontal)
        self.margin_slider.setRange(0, 40)
        self.margin_slider.setValue(cfg.extra_margin())
        self.margin_lbl = BodyLabel(str(self.margin_slider.value()))
        self.margin_lbl.setStyleSheet("color:#0078D4;font-weight:600;min-width:30px;")
        self.margin_slider.valueChanged.connect(lambda v: self.margin_lbl.setText(str(v)))
        margin_l.addWidget(self.margin_slider)
        margin_l.addWidget(self.margin_lbl)
        opts_l.addWidget(_row("Lề mở rộng cằm (Margin):", margin_w))

        # Parsing Mode
        self.parsing_combo = ComboBox()
        self.parsing_combo.addItem("Hàm cằm tự nhiên (jaw)", "jaw")
        self.parsing_combo.addItem("Toàn bộ khuôn mặt (raw)", "raw")
        self.parsing_combo.setFixedWidth(200)
        self.parsing_combo.setCurrentIndex(0 if cfg.parsing_mode() == "jaw" else 1)
        opts_l.addWidget(_row("Chế độ cắt mặt (Parsing):", self.parsing_combo))

        # Cheek Width Left
        left_cheek_w = QWidget()
        lc_l = QHBoxLayout(left_cheek_w)
        lc_l.setContentsMargins(0, 0, 0, 0)
        lc_l.setSpacing(8)
        self.left_cheek_slider = Slider(Qt.Horizontal)
        self.left_cheek_slider.setRange(20, 160)
        self.left_cheek_slider.setValue(cfg.left_cheek_width())
        self.left_cheek_lbl = BodyLabel(str(self.left_cheek_slider.value()))
        self.left_cheek_lbl.setStyleSheet("color:#0078D4;font-weight:600;min-width:30px;")
        self.left_cheek_slider.valueChanged.connect(lambda v: self.left_cheek_lbl.setText(str(v)))
        lc_l.addWidget(self.left_cheek_slider)
        lc_l.addWidget(self.left_cheek_lbl)
        opts_l.addWidget(_row("Độ rộng má trái:", left_cheek_w))

        # Cheek Width Right
        right_cheek_w = QWidget()
        rc_l = QHBoxLayout(right_cheek_w)
        rc_l.setContentsMargins(0, 0, 0, 0)
        rc_l.setSpacing(8)
        self.right_cheek_slider = Slider(Qt.Horizontal)
        self.right_cheek_slider.setRange(20, 160)
        self.right_cheek_slider.setValue(cfg.right_cheek_width())
        self.right_cheek_lbl = BodyLabel(str(self.right_cheek_slider.value()))
        self.right_cheek_lbl.setStyleSheet("color:#0078D4;font-weight:600;min-width:30px;")
        self.right_cheek_slider.valueChanged.connect(lambda v: self.right_cheek_lbl.setText(str(v)))
        rc_l.addWidget(self.right_cheek_slider)
        rc_l.addWidget(self.right_cheek_lbl)
        opts_l.addWidget(_row("Độ rộng má phải:", right_cheek_w))

        # Use Enhancer
        self.enhancer_sw = SwitchButton()
        self.enhancer_sw.setChecked(cfg.use_enhancer())
        opts_l.addWidget(_row("Làm nét mặt (GFPGAN):", self.enhancer_sw))

        # Batch Size
        batch_w = QWidget()
        batch_l = QHBoxLayout(batch_w)
        batch_l.setContentsMargins(0, 0, 0, 0)
        batch_l.setSpacing(8)
        self.batch_slider = Slider(Qt.Horizontal)
        self.batch_slider.setRange(1, 16)
        self.batch_slider.setValue(cfg.batch_size())
        self.batch_lbl = BodyLabel(str(self.batch_slider.value()))
        self.batch_lbl.setStyleSheet("color:#0078D4;font-weight:600;min-width:30px;")
        self.batch_slider.valueChanged.connect(lambda v: self.batch_lbl.setText(str(v)))
        batch_l.addWidget(self.batch_slider)
        batch_l.addWidget(self.batch_lbl)
        opts_l.addWidget(_row("Batch size:", batch_w))

        right_layout.addWidget(opts_card)

        # 6. Log console
        right_layout.addWidget(SubtitleLabel("6. Nhật ký xử lý"))
        self.log_edit = TextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("Nhật ký xử lý sẽ hiển thị ở đây...")
        self.log_edit.setFixedHeight(115)
        right_layout.addWidget(self.log_edit)

        # Thêm các cột vào layout chính
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 1)

        # Khởi tạo trạng thái
        self._toggle_audio_mode()

    def _toggle_audio_mode(self):
        is_script = self._rb_script.isChecked()
        self.script_widget.setVisible(is_script)
        self.audio_file_widget.setVisible(not is_script)

    def _browse_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file âm thanh", "", "Tệp âm thanh (*.wav *.mp3 *.m4a)"
        )
        if file_path:
            self.audio_path_edit.setText(file_path)

    def _browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục đầu ra", self.output_dir_edit.text())
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _open_video(self):
        if self._output_path and os.path.exists(self._output_path):
            os.startfile(os.path.dirname(self._output_path))
        else:
            InfoBar.error(
                title="Lỗi",
                content="Không tìm thấy tệp video đầu ra!",
                orient=Qt.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )

    def _toggle_playback(self):
        state = self.media_player.state()
        if state == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_btn.setText("▶ Phát")
        elif state == QMediaPlayer.PausedState:
            self.media_player.play()
            self.play_btn.setText("⏸ Tạm dừng")
        else:
            # Stopped or invalid — reload and play
            if self._output_path and os.path.exists(self._output_path):
                self._auto_play_pending = True
                self.media_player.setMedia(
                    QMediaContent(QUrl.fromLocalFile(self._output_path))
                )
            self.play_btn.setText("⏸ Tạm dừng")

    def _position_changed(self, position):
        self.position_slider.setValue(position)

    def _duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def _set_position(self, position):
        self.media_player.setPosition(position)

    def _validate(self):
        if not self.drop_zone.get_path():
            self._show_info("Lỗi", "Vui lòng chọn hình ảnh hoặc video chân dung nguồn!", is_error=True)
            return False
        if self._rb_script.isChecked() and not self.script_edit.toPlainText().strip():
            self._show_info("Lỗi", "Vui lòng nhập kịch bản văn bản!", is_error=True)
            return False
        if not self._rb_script.isChecked() and not self.audio_path_edit.text().strip():
            self._show_info("Lỗi", "Vui lòng chọn tệp âm thanh nguồn!", is_error=True)
            return False
        return True

    def _show_info(self, title, content, is_error=False):
        if is_error:
            InfoBar.error(title=title, content=content, parent=self, position=InfoBarPosition.TOP)
        else:
            InfoBar.success(title=title, content=content, parent=self, position=InfoBarPosition.TOP)

    def _set_proc(self, is_processing):
        self.generate_btn.setEnabled(not is_processing)
        self.stop_btn.setEnabled(is_processing)
        self.drop_zone.setEnabled(not is_processing)
        self.script_edit.setEnabled(not is_processing)
        self.audio_path_edit.setEnabled(not is_processing)
        self.btn_browse_audio.setEnabled(not is_processing)
        self.output_dir_edit.setEnabled(not is_processing)

    def _start_generate(self):
        if not self._validate():
            return

        self.media_player.stop()
        self.play_btn.setEnabled(False)
        self.play_btn.setText("Phát")

        rate_v = self.rate_slider.value()
        voice  = self.voice_combo.currentData() or 'vi-VN-HoaiMyNeural'

        gpu_avail = torch.cuda.is_available()
        device_type = "cuda" if gpu_avail else "cpu"
        self._append_log(f"💻 Chạy thiết bị: {'GPU (CUDA)' if gpu_avail else 'CPU (Eager mode)'}")

        params = {
            'sadtalker_root'   : cfg.sadtalker_root(),
            'checkpoint_path'  : cfg.checkpoint_path(),
            'ffmpeg_path'      : cfg.ffmpeg_path(),
            'output_dir'       : self.output_dir_edit.text().strip() or cfg.output_dir(),
            'source_image'     : self.drop_zone.get_path(),
            'mode'             : 'script' if self._rb_script.isChecked() else 'audio',
            'script'           : self.script_edit.toPlainText().strip(),
            'voice'            : voice,
            'rate'             : f"+{rate_v}%" if rate_v >= 0 else f"{rate_v}%",
            'audio_file'       : self.audio_path_edit.text().strip(),
            'bbox_shift'       : self.bbox_slider.value(),
            'extra_margin'     : self.margin_slider.value(),
            'parsing_mode'     : self.parsing_combo.currentData(),
            'left_cheek_width' : self.left_cheek_slider.value(),
            'right_cheek_width': self.right_cheek_slider.value(),
            'use_enhancer'     : self.enhancer_sw.isChecked(),
            'batch_size'       : self.batch_slider.value(),
            'device'           : device_type
        }

        # Lưu cấu hình nâng cao
        cfg.set('bbox_shift', params['bbox_shift'])
        cfg.set('extra_margin', params['extra_margin'])
        cfg.set('parsing_mode', params['parsing_mode'])
        cfg.set('left_cheek_width', params['left_cheek_width'])
        cfg.set('right_cheek_width', params['right_cheek_width'])
        cfg.set('use_enhancer', params['use_enhancer'])
        cfg.set('batch_size', params['batch_size'])

        self.log_edit.clear()
        self._set_proc(True)
        self.progress_bar.setValue(0)
        self.status_lbl.setText(f"⏳ Đang xử lý... (Thiết bị: {'GPU' if gpu_avail else 'CPU'})")
        self._output_path = ''
        self.open_btn.setEnabled(False)

        # Import VideoWorker here to avoid early dependency errors
        from gui_app.worker import VideoWorker
        self._worker = VideoWorker(params, parent=self)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(lambda _: self._set_proc(False))
        self._worker.error.connect(lambda _: self._set_proc(False))
        self._worker.start()

    def _stop_generate(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.status_lbl.setText("⛔ Đang dừng...")

    def _append_log(self, text):
        self.log_edit.append(text)
        self.log_edit.ensureCursorVisible()

    def _on_finished(self, out_path):
        self._output_path = out_path
        self.status_lbl.setText("✅ Hoàn thành xuất sắc!")
        self._show_info("Thành công", f"Video đã lưu tại: {os.path.basename(out_path)}")
        self.open_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.play_btn.setText("⏸ Tạm dừng")

        # Mark that we want to auto-play once media is loaded
        self._auto_play_pending = True
        self.media_player.stop()  # ensure clean state
        self.media_player.setMedia(
            QMediaContent(QUrl.fromLocalFile(os.path.abspath(out_path)))
        )
        # Actual play() is triggered by _on_media_status_changed when LoadedMedia

    def _on_media_status_changed(self, status):
        """Auto-play once the media service has finished buffering the file."""
        if self._auto_play_pending and status == QMediaPlayer.LoadedMedia:
            self._auto_play_pending = False
            self.media_player.play()

    def _on_player_error(self, error):
        if error != QMediaPlayer.NoError:
            err_str = self.media_player.errorString()
            self._append_log(f"⚠️ Trình phát video lỗi: {err_str} (code={error})")
            self.play_btn.setText("▶ Phát")

    def _on_error(self, err_msg):
        self.status_lbl.setText("❌ Lỗi xử lý.")
        self._append_log(f"❌ LỖI HỆ THỐNG: {err_msg}")
        self._show_info("Lỗi hệ thống", err_msg, is_error=True)
