# -*- coding: utf-8 -*-
import sys
import os

# Disable torch.compile to avoid MSVC compiler warm-up time and Triton errors
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Force QMediaPlayer to use Windows Media Foundation (WMF) instead of DirectShow.
# DirectShow gives error 0x80040266 when playing h264/mp4 without extra codec packs.
os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "windowsmediafoundation"

# ── Monkeypatch basicsr/torchvision functional_tensor ────────────────
try:
    import torchvision
    import torchvision.transforms.functional as tv_F
    from types import ModuleType
    m = ModuleType("torchvision.transforms.functional_tensor")
    m.rgb_to_grayscale = tv_F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = m
except Exception as e:
    print(f"Basicsr monkeypatch failed: {e}")

# Resolve asynchronous IO issues on Windows
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from PyQt5.QtCore import Qt, QSize, QLockFile, QTimer
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import QApplication, QMessageBox
from qfluentwidgets import (FluentWindow, NavigationItemPosition, FluentIcon,
                            setTheme, Theme, setThemeColor)

from gui_app.generate_page import GeneratePage

class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self._init_window()
        self._init_navigation()

    def closeEvent(self, event):
        # Stop worker thread if running to ensure clean process termination
        if hasattr(self, 'generate_page') and hasattr(self.generate_page, '_worker') and self.generate_page._worker and self.generate_page._worker.isRunning():
            self.generate_page._worker.stop()
            self.generate_page._worker.wait()
        event.accept()

    def _init_window(self):
        self.setWindowTitle("NhepMieng - Video Nhép Miệng AI SOTA (MuseTalk)")
        self.resize(1200, 800)
        self.setMinimumSize(1100, 750)

        # Set modern window icon
        icon = self.style().standardIcon(self.style().StandardPixmap(18))
        self.setWindowIcon(icon)

        # Set harmonious blue theme
        setTheme(Theme.LIGHT)
        setThemeColor("#0078D4")
        # NOTE: do NOT call move() here — centering is done AFTER show()
        # to prevent DWM from caching a ghost of the window at the pre-move position

    def center_on_screen(self):
        desktop = QApplication.desktop()
        screen_rect = desktop.screenGeometry()
        self.move((screen_rect.width() - self.width()) // 2,
                  (screen_rect.height() - self.height()) // 2)

    def _init_navigation(self):
        self.generate_page = GeneratePage(self)
        self.generate_page.setObjectName("generate_page")
        self.addSubInterface(
            self.generate_page,
            FluentIcon.VIDEO,
            "Tạo Video"
        )

if __name__ == '__main__':
    try:
        # High DPI scaling — must be set before QApplication is created
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        app = QApplication(sys.argv)

        # Use QLockFile to enforce a single instance lock (crash-safe)
        import tempfile
        lock_path = os.path.join(tempfile.gettempdir(), "nhepmieng_app.lock")
        lock_file = QLockFile(lock_path)
        if not lock_file.tryLock(100):
            QMessageBox.warning(None, "Cảnh báo", "Ứng dụng NhepMieng đang chạy ngầm hoặc đã được mở ở một cửa sổ khác!")
            sys.exit(0)

        # Use modern global font
        font = QFont("Segoe UI", 9)
        app.setFont(font)

        window = MainWindow()
        window.show()

        # Center AFTER show() so DWM only composites the window at its final position.
        # Calling move() inside __init__ (before show) causes DWM to cache a ghost
        # of the window at the pre-move coordinates, which stays visible until first click.
        window.center_on_screen()

        sys.exit(app.exec_())
    except Exception as e:
        import traceback
        with open("main_error.log", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        sys.exit(1)
