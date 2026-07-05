"""
esp32-app -- control panel CLIENT GUI for esp32-ssh.

The "open a computer and see its screen" experience for the ESP32, as far
as that's actually meaningful for a headless microcontroller: a file
browser, wifi info, installed/available apps, background job control
(bg/close/kill), and a raw terminal -- all built on top of the generic
`shell` bridge package (shell.run("<any NoorShell command>")).

Run it with:
    esp32-ssh --app path\to\apps\esp32-app
"""
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QListWidget, QMessageBox,
    QSplitter, QListWidgetItem, QMdiArea, QToolButton, QMdiSubWindow,
    QFrame, QSizePolicy, QGraphicsDropShadowEffect, QGraphicsOpacityEffect,
    QSizeGrip,
)
from PyQt5.QtCore import (
    QThread, pyqtSignal, Qt, QSize, QPoint, QPointF, QPropertyAnimation,
    QEasingCurve, QParallelAnimationGroup,
)
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QColor, QBrush, QPen, QFont, QLinearGradient,
    QPolygonF,
)

import shell


class WorkerMixin:
    """Keeps every in-flight Worker referenced in a list (self._workers)
    instead of a single attribute that a second async call could
    overwrite. Overwriting a QThread reference while it's still running
    lets Python garbage-collect it mid-execution -- destroying a running
    QThread is undefined behavior in Qt and was crashing this app on
    every single launch (AppsTab._refresh() fires two async calls back to
    back in its constructor, so the very first one was always at risk).
    Each worker removes itself from the list via its own `finished`
    signal, so the list only ever holds threads that are actually still
    running."""

    def _start_worker(self, fn, on_done, on_fail=None):
        if not hasattr(self, "_workers"):
            self._workers = []
        worker = Worker(fn)
        worker.succeeded.connect(on_done)
        if on_fail is not None:
            worker.failed.connect(on_fail)
        worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(worker)
        worker.start()
        return worker


class Worker(QThread):
    """Runs one shell.run(...) call on a background thread so the GUI
    never freezes while esp32-ssh's subprocess round-trips over the
    network. `fn` is a zero-arg callable; wrap the real call in a lambda."""
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            result = self.fn()
        except Exception as e:
            self.failed.emit(str(e))
        else:
            self.succeeded.emit(result)


def _join(cwd: str, name: str) -> str:
    """Joins a virtual path segment onto cwd. Needed because each
    shell.run() call is a BRAND NEW esp32-ssh connection/login -- NoorShell's
    server-side `cd` only lives for one TCP session and is gone the instant
    that one-shot --command call finishes, so navigation state has to be
    tracked here client-side and passed as a full path on every call,
    never relied on to persist between clicks."""
    if cwd == "/":
        return "/" + name
    return cwd.rstrip("/") + "/" + name


def _parent(cwd: str) -> str:
    if cwd in ("/", ""):
        return "/"
    parts = cwd.rstrip("/").rsplit("/", 1)
    return parts[0] if parts[0] else "/"


def _flash_in(widget, duration: int = 260):
    """Fades a widget's contents in from transparent -- used every time a
    panel repopulates after an async refresh, so new data visibly
    *arrives* instead of snapping into place like a static text dump."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    widget._flash_anim = anim  # keep a live reference tied to the widget
    anim.start()


class StatusBanner(QFrame):
    """A prominent, always-visible connection-state strip for panels that
    talk to the ESP32 (Files/Wifi/Apps). Before this, a failed shell.run()
    only changed a small gray status label while the panel above it
    stayed blank -- easy to miss entirely, which is exactly what made
    those tabs look like they were doing nothing while Terminal (which
    prints every error in big colored text) looked alive. This surfaces
    'loading' and 'can't reach the ESP32' the way a real OS would: a
    colored banner with an icon, a message, and (on error) a Retry
    button, impossible to overlook."""

    def __init__(self, on_retry=None):
        super().__init__()
        self._on_retry = on_retry
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(8)

        self.dot = QLabel()
        self.dot.setFixedSize(9, 9)
        layout.addWidget(self.dot)

        self.text = QLabel("")
        self.text.setWordWrap(True)
        layout.addWidget(self.text, 1)

        self.retry_btn = QPushButton("Retry")
        self.retry_btn.setFixedHeight(26)
        self.retry_btn.setVisible(False)
        self.retry_btn.setCursor(Qt.PointingHandCursor)
        self.retry_btn.clicked.connect(lambda: self._on_retry and self._on_retry())
        layout.addWidget(self.retry_btn)

        self.setVisible(False)

    def show_loading(self, message: str = "Connecting to ESP32..."):
        self.setStyleSheet(
            "QFrame { background: #1b2a3f; border: 1px solid #2c4568; border-radius: 8px; }"
        )
        self.dot.setStyleSheet("background: #4facfe; border-radius: 4px;")
        self.text.setStyleSheet("color: #bcd6ff; font-weight: 600; background: transparent;")
        self.text.setText(message)
        self.retry_btn.setVisible(False)
        self.setVisible(True)
        _flash_in(self)

    def show_error(self, message: str):
        self.setStyleSheet(
            "QFrame { background: #3a1f24; border: 1px solid #6a2f38; border-radius: 8px; }"
        )
        self.dot.setStyleSheet("background: #e0555f; border-radius: 4px;")
        self.text.setStyleSheet("color: #ffd0d5; font-weight: 600; background: transparent;")
        self.text.setText(message)
        self.retry_btn.setVisible(self._on_retry is not None)
        self.setVisible(True)
        _flash_in(self)

    def hide_banner(self):
        self.setVisible(False)


def _glyph_icon(kind: str, color: str = "#eef2f7", size: int = 18) -> QIcon:
    """Small line-art action icons (up, refresh, trash, play, ...) used
    on every toolbar button in place of plain words."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(max(1.6, size * 0.11))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    s = size

    if kind == "up":
        p.drawLine(QPointF(s * 0.5, s * 0.8), QPointF(s * 0.5, s * 0.24))
        p.drawLine(QPointF(s * 0.5, s * 0.24), QPointF(s * 0.28, s * 0.48))
        p.drawLine(QPointF(s * 0.5, s * 0.24), QPointF(s * 0.72, s * 0.48))
    elif kind == "refresh":
        r = s * 0.6
        p.drawArc(int(s / 2 - r / 2), int(s / 2 - r / 2), int(r), int(r), 20 * 16, 300 * 16)
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        ax, ay = s * 0.78, s * 0.3
        p.drawPolygon(QPolygonF([QPointF(ax, ay - 4), QPointF(ax + 5, ay + 2), QPointF(ax - 3, ay + 5)]))
    elif kind == "folder-plus":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(int(s * 0.08), int(s * 0.34), int(s * 0.5), int(s * 0.4), 2, 2)
        p.drawRoundedRect(int(s * 0.08), int(s * 0.24), int(s * 0.24), int(s * 0.14), 2, 2)
        pen2 = QPen(QColor("#10131a"))
        pen2.setWidthF(s * 0.1)
        pen2.setCapStyle(Qt.RoundCap)
        p.setPen(pen2)
        cx, cy = s * 0.72, s * 0.62
        p.drawLine(QPointF(cx - s * 0.09, cy), QPointF(cx + s * 0.09, cy))
        p.drawLine(QPointF(cx, cy - s * 0.09), QPointF(cx, cy + s * 0.09))
    elif kind == "trash":
        p.drawLine(QPointF(s * 0.2, s * 0.32), QPointF(s * 0.8, s * 0.32))
        p.drawLine(QPointF(s * 0.38, s * 0.32), QPointF(s * 0.42, s * 0.2))
        p.drawLine(QPointF(s * 0.42, s * 0.2), QPointF(s * 0.58, s * 0.2))
        p.drawLine(QPointF(s * 0.58, s * 0.2), QPointF(s * 0.62, s * 0.32))
        p.drawLine(QPointF(s * 0.26, s * 0.32), QPointF(s * 0.3, s * 0.84))
        p.drawLine(QPointF(s * 0.3, s * 0.84), QPointF(s * 0.7, s * 0.84))
        p.drawLine(QPointF(s * 0.7, s * 0.84), QPointF(s * 0.74, s * 0.32))
    elif kind == "play":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(s * 0.3, s * 0.2), QPointF(s * 0.3, s * 0.8), QPointF(s * 0.82, s * 0.5)]))
    elif kind == "cloud-play":
        p.drawEllipse(QPointF(s * 0.36, s * 0.5), s * 0.16, s * 0.16)
        p.drawEllipse(QPointF(s * 0.58, s * 0.42), s * 0.2, s * 0.2)
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(s * 0.44, s * 0.56), QPointF(s * 0.44, s * 0.74), QPointF(s * 0.62, s * 0.65)]))
    elif kind == "stop":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(int(s * 0.28), int(s * 0.28), int(s * 0.44), int(s * 0.44), 3, 3)
    elif kind == "bolt":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([
            QPointF(s * 0.56, s * 0.14), QPointF(s * 0.28, s * 0.56), QPointF(s * 0.46, s * 0.56),
            QPointF(s * 0.4, s * 0.86), QPointF(s * 0.72, s * 0.42), QPointF(s * 0.52, s * 0.42),
        ]))
    elif kind == "eye":
        p.drawArc(int(s * 0.08), int(s * 0.28), int(s * 0.84), int(s * 0.44), 0, 360 * 16)
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.11, s * 0.11)
    elif kind == "send":
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([
            QPointF(s * 0.14, s * 0.5), QPointF(s * 0.86, s * 0.16),
            QPointF(s * 0.58, s * 0.5), QPointF(s * 0.86, s * 0.84),
        ]))
    elif kind == "download":
        p.drawLine(QPointF(s * 0.5, s * 0.16), QPointF(s * 0.5, s * 0.58))
        p.drawLine(QPointF(s * 0.32, s * 0.4), QPointF(s * 0.5, s * 0.6))
        p.drawLine(QPointF(s * 0.68, s * 0.4), QPointF(s * 0.5, s * 0.6))
        p.drawLine(QPointF(s * 0.2, s * 0.8), QPointF(s * 0.8, s * 0.8))
    elif kind == "wifi":
        for r in (0.38, 0.24, 0.1):
            rad = s * r
            p.drawArc(int(s / 2 - rad), int(s * 0.68 - rad), int(rad * 2), int(rad * 2), 35 * 16, 110 * 16)
        p.setBrush(QBrush(QColor(color)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(s * 0.5, s * 0.68), s * 0.05, s * 0.05)

    p.end()
    return QIcon(pix)


def _explorer_folder_icon(size: int = 60) -> QIcon:
    """A real folder glyph (not a '[DIR]' prefix) for the file grid."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    grad = QLinearGradient(0, 0, 0, size)
    grad.setColorAt(0, QColor("#ffd166"))
    grad.setColorAt(1, QColor("#f0a020"))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(int(size * 0.08), int(size * 0.22), int(size * 0.34), int(size * 0.16), 4, 4)
    p.drawRoundedRect(int(size * 0.08), int(size * 0.34), int(size * 0.84), int(size * 0.5), 6, 6)
    p.setBrush(QBrush(QColor(255, 255, 255, 40)))
    p.drawRoundedRect(int(size * 0.08), int(size * 0.34), int(size * 0.84), int(size * 0.14), 6, 6)
    p.end()
    return QIcon(pix)


def _explorer_file_icon(size: int = 60) -> QIcon:
    """A real document glyph (folded corner + text lines) for the file
    grid, replacing the old '[FILE] name' text prefix."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    page_w, page_h = size * 0.62, size * 0.8
    x, y = (size - page_w) / 2, (size - page_h) / 2
    corner = page_w * 0.32
    p.setPen(QPen(QColor("#b9c4d6"), 1.4))
    p.setBrush(QBrush(QColor("#eef3fb")))
    body = QPolygonF([
        QPointF(x, y), QPointF(x + page_w - corner, y), QPointF(x + page_w, y + corner),
        QPointF(x + page_w, y + page_h), QPointF(x, y + page_h),
    ])
    p.drawPolygon(body)
    p.setBrush(QBrush(QColor("#d7e0ee")))
    p.setPen(Qt.NoPen)
    fold = QPolygonF([
        QPointF(x + page_w - corner, y), QPointF(x + page_w, y + corner),
        QPointF(x + page_w - corner, y + corner),
    ])
    p.drawPolygon(fold)
    p.setPen(QPen(QColor("#9fb0c8"), 1.6))
    for i in range(3):
        ly = y + page_h * 0.44 + i * page_h * 0.14
        p.drawLine(QPointF(x + page_w * 0.18, ly), QPointF(x + page_w * 0.82, ly))
    p.end()
    return QIcon(pix)


_APP_TILE_PALETTE = (
    ("#4facfe", "#2266d8"), ("#a06bff", "#6f3ef0"), ("#3ddc84", "#1a9b56"),
    ("#ffd166", "#f7a531"), ("#ff6b6b", "#c23b3b"), ("#22d3ee", "#0e7f96"),
)


def _app_icon(name: str, size: int = 56) -> QIcon:
    """A colorful 'package' tile for an installed/available app entry --
    a stable color per app name (via hash) so the same app always looks
    the same, like real app-store tiles instead of a plain text row."""
    color1, color2 = _APP_TILE_PALETTE[hash(name) % len(_APP_TILE_PALETTE)]
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    grad = QLinearGradient(0, 0, 0, size)
    grad.setColorAt(0, QColor(color1))
    grad.setColorAt(1, QColor(color2))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(3, 3, size - 6, size - 6, 14, 14)
    w, h = size * 0.5, size * 0.36
    cx, cy = size / 2, size / 2
    p.setBrush(QBrush(QColor(255, 255, 255, 235)))
    p.drawRoundedRect(int(cx - w / 2), int(cy - h / 2), int(w), int(h), 4, 4)
    p.setPen(QPen(QColor(color1), 2))
    p.drawLine(int(cx - w / 2), int(cy), int(cx + w / 2), int(cy))
    p.end()
    return QIcon(pix)


class ToolIconButton(QToolButton):
    """A compact icon-only action button (Up, Refresh, New Folder,
    Delete, ...) -- used everywhere a plain text button used to be."""

    def __init__(self, kind: str, tooltip: str, danger: bool = False, size: int = 34):
        super().__init__()
        self.setIcon(_glyph_icon(kind))
        self.setIconSize(QSize(18, 18))
        self.setFixedSize(size, size)
        self.setToolTip(tooltip)
        self.setCursor(Qt.PointingHandCursor)
        hover_border = "#e0555f" if danger else "#7c98ff"
        hover_bg = "#4a2b30" if danger else "#3a4260"
        self.setStyleSheet(f"""
            QToolButton {{
                background: #2c3348;
                border: 1px solid #454f6e;
                border-radius: 8px;
            }}
            QToolButton:hover {{ background: {hover_bg}; border: 1px solid {hover_border}; }}
            QToolButton:pressed {{ background: #1c2231; }}
        """)


class FilesTab(QWidget, WorkerMixin):
    """Virtual file browser over NoorShell's ls/cd/mkdir/rm/cat, navigated
    entirely client-side (see _join's docstring for why) -- shown as a
    real icon grid (folder/document tiles you double-click) with a
    breadcrumb path bar and a live preview pane, instead of a
    '[DIR]/[FILE] name' text list."""

    def __init__(self):
        super().__init__()
        self.cwd = "/"
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.banner = StatusBanner(on_retry=self._refresh)
        layout.addWidget(self.banner)

        path_row = QHBoxLayout()
        self.path_label = QLabel(self.cwd)
        self.path_label.setStyleSheet("""
            background: #1b2130;
            color: #9fb3e0;
            border: 1px solid #2a3040;
            border-radius: 14px;
            padding: 5px 14px;
            font-family: Consolas, monospace;
        """)
        up_btn = ToolIconButton("up", "Up one level")
        up_btn.clicked.connect(self._go_up)
        refresh_btn = ToolIconButton("refresh", "Refresh")
        refresh_btn.clicked.connect(self._refresh)
        path_row.addWidget(self.path_label, 1)
        path_row.addWidget(up_btn)
        path_row.addWidget(refresh_btn)
        layout.addLayout(path_row)

        splitter = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(52, 52))
        self.list.setGridSize(QSize(96, 92))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setMovement(QListWidget.Static)
        self.list.setSpacing(8)
        self.list.setWordWrap(True)
        self.list.itemClicked.connect(self._open_item)
        self.list.itemDoubleClicked.connect(self._open_item)
        splitter.addWidget(self.list)

        preview_frame = QFrame()
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        self.preview_title = QLabel("Preview")
        self.preview_title.setStyleSheet("color: #7fa8ff; font-weight: 600; padding: 2px 2px;")
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Double-click a file to preview it here.")
        self.preview.setStyleSheet("font-family: Consolas, monospace;")
        preview_layout.addWidget(self.preview_title)
        preview_layout.addWidget(self.preview, 1)
        splitter.addWidget(preview_frame)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        actions_row = QHBoxLayout()
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("New folder name")
        mkdir_btn = ToolIconButton("folder-plus", "Create folder")
        mkdir_btn.clicked.connect(self._mkdir)
        rm_btn = ToolIconButton("trash", "Delete selected", danger=True)
        rm_btn.clicked.connect(self._rm_selected)
        actions_row.addWidget(self.new_name, 1)
        actions_row.addWidget(mkdir_btn)
        actions_row.addWidget(rm_btn)
        layout.addLayout(actions_row)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #8891a5;")
        layout.addWidget(self.status)

        self._refresh()

    def _run_async(self, fn, on_done):
        self._start_worker(fn, on_done, lambda err: self.status.setText(f"Error: {err}"))

    def _refresh(self):
        self.path_label.setText(self.cwd)
        self.status.setText("Loading...")
        self.banner.show_loading("Loading files from the ESP32...")
        self._start_worker(lambda: shell.run(f"ls {self.cwd}"), self._on_listed, self._on_list_failed)

    def _on_list_failed(self, err):
        self.status.setText(f"Error: {err}")
        self.banner.show_error(f"Couldn't reach the ESP32 -- {err}")

    def _on_listed(self, raw):
        self.list.clear()
        if raw.startswith("error:"):
            self.status.setText(raw)
            self.banner.show_error(raw)
            placeholder = QListWidgetItem(_glyph_icon("refresh", "#8891a5", 40), "Tap to retry")
            placeholder.setTextAlignment(Qt.AlignHCenter)
            placeholder.setData(Qt.UserRole, None)
            self.list.addItem(placeholder)
            _flash_in(self.list)
            return
        if not raw.strip():
            self.banner.hide_banner()
            self.status.setText("This folder is empty.")
            placeholder = QListWidgetItem(_glyph_icon("refresh", "#8891a5", 40), "Tap to retry")
            placeholder.setTextAlignment(Qt.AlignHCenter)
            placeholder.setData(Qt.UserRole, None)
            self.list.addItem(placeholder)
            _flash_in(self.list)
            return
        self.banner.hide_banner()
        self.status.setText("Ready.")
        for line in raw.splitlines():
            line = line.strip()
            if not line or len(line) < 3:
                continue
            kind, name = line[0], line[2:]
            icon = _explorer_folder_icon() if kind == "d" else _explorer_file_icon()
            item = QListWidgetItem(icon, name)
            item.setTextAlignment(Qt.AlignHCenter)
            item.setData(Qt.UserRole, (kind, name))
            self.list.addItem(item)
        _flash_in(self.list)

    def _open_item(self, item):
        data = item.data(Qt.UserRole)
        if data is None:
            self._refresh()
            return
        kind, name = data
        if kind == "d":
            self.cwd = _join(self.cwd, name)
            self._refresh()
        else:
            self.status.setText("Loading file...")
            self.preview_title.setText(f"Preview — {name}")
            path = _join(self.cwd, name)
            self._run_async(lambda: shell.run(f"cat {path}"), self._on_file_loaded)

    def _on_file_loaded(self, text):
        self.preview.setPlainText(text)
        _flash_in(self.preview)

    def _go_up(self):
        self.cwd = _parent(self.cwd)
        self._refresh()

    def _mkdir(self):
        name = self.new_name.text().strip()
        if not name:
            return
        path = _join(self.cwd, name)
        self._run_async(lambda: shell.run(f"mkdir {path}"),
                         lambda res: (self.new_name.clear(), self._refresh()))

    def _rm_selected(self):
        item = self.list.currentItem()
        if item is None or item.data(Qt.UserRole) is None:
            self.status.setText("Select something to delete first.")
            return
        _, name = item.data(Qt.UserRole)
        path = _join(self.cwd, name)
        confirm = QMessageBox.question(self, "Delete", f"Delete {path}?")
        if confirm != QMessageBox.Yes:
            return
        self._run_async(lambda: shell.run(f"rm {path}"), lambda res: self._refresh())


class WifiTab(QWidget, WorkerMixin):
    """Read-only network info -- NoorShell's `wifi` command is a snapshot,
    not a settings screen (there's no remote wifi-change command by
    design; that's a hold-BOOT-and-power-cycle physical action). Shown as
    a status card with a live pulsing signal icon rather than a plain
    text dump."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.signal_icon = QLabel()
        self.signal_icon.setPixmap(_glyph_icon("wifi", "#4facfe", 26).pixmap(26, 26))
        header.addWidget(self.signal_icon)
        title = QLabel("Network Status")
        title.setStyleSheet("font-weight: 600; font-size: 13px; color: #eef2f7;")
        header.addWidget(title)
        header.addStretch(1)
        refresh_btn = ToolIconButton("refresh", "Refresh")
        refresh_btn.clicked.connect(self._refresh)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.banner = StatusBanner(on_retry=self._refresh)
        layout.addWidget(self.banner)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self.info, 1)

        # gentle infinite pulse on the signal icon -- a small sign of
        # life instead of a static glyph sitting next to static text.
        self._pulse_effect = QGraphicsOpacityEffect(self.signal_icon)
        self.signal_icon.setGraphicsEffect(self._pulse_effect)
        self._pulse = QPropertyAnimation(self._pulse_effect, b"opacity", self)
        self._pulse.setDuration(1100)
        self._pulse.setStartValue(0.45)
        self._pulse.setEndValue(1.0)
        self._pulse.setEasingCurve(QEasingCurve.InOutSine)
        self._pulse.setLoopCount(-1)
        self._pulse.start()

        self._refresh()

    def _refresh(self):
        self.info.setPlainText("Loading...")
        self.banner.show_loading("Checking network status...")
        self._start_worker(lambda: shell.run("wifi") + "\n\n" + shell.run("df"),
                            self._on_loaded,
                            self._on_load_failed)

    def _on_load_failed(self, err):
        self.info.setPlainText(f"Error: {err}")
        self.banner.show_error(f"Couldn't reach the ESP32 -- {err}")

    def _on_loaded(self, text):
        self.info.setPlainText(text)
        self.banner.hide_banner()
        _flash_in(self.info)


class AppsTab(QWidget, WorkerMixin):
    """Installed/available apps (app-installer) plus the background job
    panel (bg/jobs/close/kill/job-output) -- the ESP32 only has ONE
    background job slot (see task_manager.h), so this is a single status
    view + controls, not a multi-job queue. Apps are shown as colored
    package tiles (icon grid) rather than plain text rows."""

    def __init__(self):
        super().__init__()
        self._pending_refreshes = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.banner = StatusBanner(on_retry=self._refresh)
        root.addWidget(self.banner)

        outer = QHBoxLayout()
        root.addLayout(outer, 1)

        # -- left: installed / available apps --
        left = QVBoxLayout()
        left.addWidget(self._section_label("Installed apps"))
        self.installed_list = self._make_app_grid()
        left.addWidget(self.installed_list, 1)

        left.addWidget(self._section_label("Available apps (official repo)"))
        self.available_list = self._make_app_grid()
        left.addWidget(self.available_list, 1)

        btn_row = QHBoxLayout()
        refresh_btn = self._icon_button("refresh", "Refresh")
        refresh_btn.clicked.connect(self._refresh)
        install_btn = self._icon_button("download", "Install selected")
        install_btn.clicked.connect(self._install_selected)
        remove_btn = self._icon_button("trash", "Remove selected", danger=True)
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(install_btn)
        btn_row.addWidget(remove_btn)
        left.addLayout(btn_row)

        run_row = QHBoxLayout()
        run_btn = self._icon_button("play", "Run selected")
        run_btn.clicked.connect(lambda: self._run_selected(background=False))
        run_bg_btn = self._icon_button("cloud-play", "Run in background")
        run_bg_btn.clicked.connect(lambda: self._run_selected(background=True))
        run_row.addWidget(run_btn)
        run_row.addWidget(run_bg_btn)
        left.addLayout(run_row)

        outer.addLayout(left, 1)

        # -- right: background job panel (single job slot) --
        right = QVBoxLayout()
        right.addWidget(self._section_label("Background job (one slot)"))
        self.job_status = QTextEdit()
        self.job_status.setReadOnly(True)
        self.job_status.setStyleSheet("font-family: Consolas, monospace;")
        right.addWidget(self.job_status, 1)

        job_btn_row = QHBoxLayout()
        jobs_refresh_btn = self._icon_button("refresh", "Refresh jobs")
        jobs_refresh_btn.clicked.connect(self._refresh_jobs)
        output_btn = self._icon_button("eye", "View output")
        output_btn.clicked.connect(self._job_output)
        job_btn_row.addWidget(jobs_refresh_btn)
        job_btn_row.addWidget(output_btn)
        right.addLayout(job_btn_row)

        self.job_name = QLineEdit()
        self.job_name.setPlaceholderText("job name, e.g. run-45213 (see jobs)")
        right.addWidget(self.job_name)

        stop_row = QHBoxLayout()
        close_btn = self._icon_button("stop", "Close (graceful)")
        close_btn.clicked.connect(self._close_job)
        kill_btn = self._icon_button("bolt", "Kill (immediate)", danger=True)
        kill_btn.clicked.connect(self._kill_job)
        stop_row.addWidget(close_btn)
        stop_row.addWidget(kill_btn)
        right.addLayout(stop_row)

        outer.addLayout(right, 1)

        self._refresh()
        self._refresh_jobs()

    @staticmethod
    def _section_label(text):
        label = QLabel(text)
        label.setStyleSheet("color: #7fa8ff; font-weight: 600; padding-top: 4px;")
        return label

    @staticmethod
    def _make_app_grid():
        grid = QListWidget()
        grid.setViewMode(QListWidget.IconMode)
        grid.setIconSize(QSize(46, 46))
        grid.setGridSize(QSize(88, 82))
        grid.setResizeMode(QListWidget.Adjust)
        grid.setMovement(QListWidget.Static)
        grid.setSpacing(6)
        grid.setWordWrap(True)
        return grid

    @staticmethod
    def _icon_button(kind, tooltip, danger=False):
        btn = QPushButton(tooltip)
        btn.setIcon(_glyph_icon(kind))
        btn.setIconSize(QSize(15, 15))
        if danger:
            btn.setStyleSheet("""
                QPushButton { border: 1px solid #7a3a3f; }
                QPushButton:hover { background: #4a2b30; border: 1px solid #e0555f; }
            """)
        return btn

    def _run_async(self, fn, on_done, on_fail=None):
        self._start_worker(fn, on_done, on_fail or (lambda err: self.job_status.setPlainText(f"Error: {err}")))

    @staticmethod
    def _selected_name(list_widget):
        item = list_widget.currentItem()
        return item.text().strip() if item is not None else None

    def _refresh(self):
        self.banner.show_loading("Loading apps from the ESP32...")
        self._pending_refreshes = 2
        self._run_async(lambda: shell.run("app-installer list --installed"),
                         lambda raw: self._on_list_done(self.installed_list, raw),
                         self._on_list_failed)
        self._run_async(lambda: shell.run("app-installer list"),
                         lambda raw: self._on_list_done(self.available_list, raw),
                         self._on_list_failed)

    def _on_list_done(self, list_widget, raw):
        self._fill(list_widget, raw)
        self._pending_refreshes = max(0, self._pending_refreshes - 1)
        if self._pending_refreshes == 0:
            self.banner.hide_banner()

    def _on_list_failed(self, err):
        self.job_status.setPlainText(f"Error: {err}")
        self.banner.show_error(f"Couldn't reach the ESP32 -- {err}")

    @staticmethod
    def _fill(list_widget, raw):
        list_widget.clear()
        if raw.startswith("error:") or not raw.strip():
            msg = raw if raw.strip() else "None found."
            placeholder = QListWidgetItem(_glyph_icon("refresh", "#8891a5", 34), msg)
            placeholder.setTextAlignment(Qt.AlignHCenter)
            placeholder.setFlags(Qt.NoItemFlags)
            list_widget.addItem(placeholder)
            _flash_in(list_widget)
            return
        for line in raw.splitlines():
            name = line.strip()
            if not name:
                continue
            item = QListWidgetItem(_app_icon(name), name.split()[0])
            item.setTextAlignment(Qt.AlignHCenter)
            list_widget.addItem(item)
        _flash_in(list_widget)

    def _install_selected(self):
        name = self._selected_name(self.available_list)
        if not name:
            return
        self._run_async(lambda: shell.run(f"app-installer install {name}", timeout=60),
                         lambda res: self._refresh())

    def _remove_selected(self):
        name = self._selected_name(self.installed_list)
        if not name:
            return
        confirm = QMessageBox.question(self, "Remove", f"Remove app '{name}'?")
        if confirm != QMessageBox.Yes:
            return
        self._run_async(lambda: shell.run(f"app-installer remove {name}"),
                         lambda res: self._refresh())

    def _run_selected(self, background: bool):
        name = self._selected_name(self.installed_list)
        if not name:
            return
        cmd = f"bg run {name}" if background else f"run {name}"
        self._run_async(lambda: shell.run(cmd, timeout=60), self._on_job_text)
        if background:
            self._refresh_jobs()

    def _on_job_text(self, text):
        self.job_status.setPlainText(text)
        _flash_in(self.job_status)

    def _refresh_jobs(self):
        self._run_async(lambda: shell.run("jobs"), self._on_job_text)

    def _job_output(self):
        name = self.job_name.text().strip()
        if not name:
            self.job_status.setPlainText("Enter a job name first (see jobs).")
            return
        self._run_async(lambda: shell.run(f"job-output {name}"), self._on_job_text)

    def _close_job(self):
        name = self.job_name.text().strip()
        if not name:
            return
        self._run_async(lambda: shell.run(f"close {name}"), self._on_job_text)

    def _kill_job(self):
        name = self.job_name.text().strip()
        if not name:
            return
        self._run_async(lambda: shell.run(f"kill {name}"), self._on_job_text)


class TerminalTab(QWidget, WorkerMixin):
    """Raw NoorShell access -- every command this file's other tabs use
    under the hood (ls, run, bg, apt, os, curl, ...) plus anything else,
    typed directly. Styled like a real terminal (monospace, green-tinted
    prompt, glowing focus border) rather than a plain QTextEdit."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._history = []
        self._history_pos = 0

        header = QHBoxLayout()
        dot_colors = ("#e0555f", "#f5c451", "#3ddc84")  # a little traffic-light, like a real terminal
        for c in dot_colors:
            dot = QLabel()
            dot.setFixedSize(11, 11)
            dot.setStyleSheet(f"background: {c}; border-radius: 5px;")
            header.addWidget(dot)
        header.addSpacing(6)
        title = QLabel("NoorShell")
        title.setStyleSheet("color: #7fe89a; font-weight: 600; font-family: Consolas, monospace;")
        header.addWidget(title)
        header.addStretch(1)
        layout.addLayout(header)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Command output will appear here.")
        self.output.setStyleSheet("""
            QTextEdit {
                background: #0a0f0a;
                color: #7fe89a;
                font-family: Consolas, monospace;
                border: 1px solid #1f3a24;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        self.output.append(
            '<span style="color:#4d6b57;">NoorShell terminal -- type a command below '
            '(e.g. "help"), or use \u2191 / \u2193 for history.</span>'
        )
        layout.addWidget(self.output, 1)

        row = QHBoxLayout()
        prompt = QLabel("&gt;")
        prompt.setStyleSheet("color: #3ddc84; font-family: Consolas, monospace; font-weight: bold; padding: 0 2px;")
        self.input = QLineEdit()
        self.input.setPlaceholderText("type a NoorShell command, e.g. help")
        self.input.setStyleSheet("""
            QLineEdit {
                background: #0a0f0a;
                color: #7fe89a;
                font-family: Consolas, monospace;
                border: 1px solid #1f3a24;
                border-radius: 6px;
                padding: 5px 7px;
            }
            QLineEdit:focus { border: 1px solid #3ddc84; }
        """)
        self.input.installEventFilter(self)
        self.input.returnPressed.connect(self._send)
        send_btn = ToolIconButton("send", "Send command", size=36)
        send_btn.clicked.connect(self._send)
        row.addWidget(prompt)
        row.addWidget(self.input, 1)
        row.addWidget(send_btn)
        layout.addLayout(row)

    def eventFilter(self, obj, event):
        if obj is self.input and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Up:
                self._step_history(-1)
                return True
            if event.key() == Qt.Key_Down:
                self._step_history(1)
                return True
        return super().eventFilter(obj, event)

    def _step_history(self, direction: int):
        if not self._history:
            return
        self._history_pos = max(0, min(len(self._history), self._history_pos + direction))
        if self._history_pos == len(self._history):
            self.input.clear()
        else:
            self.input.setText(self._history[self._history_pos])

    def _append(self, html):
        self.output.append(html)
        self.output.moveCursor(self.output.textCursor().End)
        self.output.ensureCursorVisible()

    def _send(self):
        cmd = self.input.text().strip()
        if not cmd:
            return
        self.input.clear()
        self._history.append(cmd)
        self._history_pos = len(self._history)
        self._append(f'<span style="color:#3ddc84;">&gt; {cmd}</span>')
        self._start_worker(lambda: shell.run(cmd, timeout=60),
                            self._append,
                            lambda err: self._append(f'<span style="color:#e0555f;">Error: {err}</span>'))


APP_STYLESHEET = """
QWidget {
    background: #10131a;
    color: #e8edf5;
    font-family: "Segoe UI", sans-serif;
    font-size: 12px;
}
QLabel {
    background: transparent;
    color: #c9d3e0;
}
QMainWindow, QMdiArea {
    background: #0d1117;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #4c5770, stop:1 #394258);
    color: #f5f8ff;
    border: 1px solid #6a76a0;
    border-radius: 8px;
    padding: 7px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #5c688a, stop:1 #454f6e);
    border: 1px solid #8fa6ff;
}
QPushButton:pressed {
    background: #2c3348;
}
QPushButton:disabled {
    color: #6a7182;
    background: #232734;
    border: 1px solid #2c3140;
}
QToolButton {
    background: #2c3348;
    border: 1px solid #454f6e;
    border-radius: 8px;
    padding: 5px;
}
QToolButton:hover {
    background: #3a4260;
    border: 1px solid #7c98ff;
}
QToolButton:pressed {
    background: #20263a;
}
QLineEdit, QTextEdit {
    background: #0b0e14;
    color: #dce4f2;
    border: 1px solid #2a3040;
    border-radius: 6px;
    padding: 5px 7px;
    selection-background-color: #6f8cff;
    selection-color: #0b0e14;
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #6f8cff;
}
QListWidget {
    background: #0b0e14;
    color: #dce4f2;
    border: 1px solid #2a3040;
    border-radius: 6px;
    outline: none;
}
QListWidget::item {
    padding: 4px 6px;
    border-radius: 4px;
}
QListWidget::item:hover {
    background: #1b2131;
}
QListWidget::item:selected {
    background: #34406b;
    color: #ffffff;
}
QScrollBar:vertical {
    background: #0d1117;
    width: 11px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #33394a;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #495066;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QMessageBox {
    background: #171b24;
}
"""


def _shadow(blur=34, dy=10, alpha=185):
    """A reusable drop-shadow effect for elevation -- attached to app
    windows (deep) and dock/taskbar buttons (light, tinted) to make the
    UI feel layered instead of flat."""
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(blur)
    effect.setOffset(0, dy)
    effect.setColor(QColor(0, 0, 0, alpha))
    return effect


def _wallpaper_brush() -> QBrush:
    """A faint dot-grid tile for the workspace background, so the canvas
    behind the windows reads as an actual desktop rather than a blank
    QWidget."""
    tile = QPixmap(26, 26)
    tile.fill(QColor("#0d1117"))
    p = QPainter(tile)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(255, 255, 255, 16))
    p.drawEllipse(11, 11, 3, 3)
    p.end()
    return QBrush(tile)


def _icon_base(color1: str, color2: str, size: int = 72) -> (QPixmap, QPainter):
    """Draws the common rounded-square 'app tile' background every icon
    sits on (like iOS/Android/macOS app icons), leaving the QPainter open
    so the caller can draw its own glyph on top before calling p.end()."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    grad = QLinearGradient(0, 0, 0, size)
    grad.setColorAt(0, QColor(color1))
    grad.setColorAt(1, QColor(color2))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(3, 3, size - 6, size - 6, 16, 16)
    return pix, p


def _icon_files(size: int = 72) -> QIcon:
    pix, p = _icon_base("#ffd166", "#f7a531", size)
    p.setBrush(QBrush(QColor("#fffbe8")))
    w, h = size * 0.62, size * 0.42
    x, y = (size - w) / 2, size * 0.32
    p.drawRoundedRect(int(x), int(y), int(w), int(h), 4, 4)
    tab_w = w * 0.38
    p.drawRoundedRect(int(x + w * 0.08), int(y - h * 0.22), int(tab_w), int(h * 0.3), 3, 3)
    p.end()
    return QIcon(pix)


def _icon_wifi(size: int = 72) -> QIcon:
    pix, p = _icon_base("#4facfe", "#2266d8", size)
    pen = QPen(QColor("#ffffff"))
    pen.setWidth(max(2, size // 14))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    cx, cy = size / 2, size * 0.68
    for i, r in enumerate((0.42, 0.28, 0.14)):
        rad = size * r
        rect_ = (cx - rad, cy - rad, rad * 2, rad * 2)
        p.drawArc(int(rect_[0]), int(rect_[1]), int(rect_[2]), int(rect_[3]), 35 * 16, 110 * 16)
    p.setBrush(QBrush(QColor("#ffffff")))
    p.setPen(Qt.NoPen)
    dot_r = size * 0.045
    p.drawEllipse(int(cx - dot_r), int(cy - dot_r), int(dot_r * 2), int(dot_r * 2))
    p.end()
    return QIcon(pix)


def _icon_apps(size: int = 72) -> QIcon:
    pix, p = _icon_base("#a06bff", "#6f3ef0", size)
    p.setBrush(QBrush(QColor("#ffffff")))
    p.setPen(Qt.NoPen)
    pad, gap = size * 0.18, size * 0.08
    cell = (size - 2 * pad - gap) / 2
    for row in range(2):
        for col in range(2):
            x = pad + col * (cell + gap)
            y = pad + row * (cell + gap)
            p.drawRoundedRect(int(x), int(y), int(cell), int(cell), 5, 5)
    p.end()
    return QIcon(pix)


def _icon_terminal(size: int = 72) -> QIcon:
    pix, p = _icon_base("#2f3542", "#12151c", size)
    pen = QPen(QColor("#33ff77"))
    pen.setWidth(max(2, size // 16))
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    x0, y0 = size * 0.24, size * 0.38
    p.drawLine(int(x0), int(y0), int(x0 + size * 0.16), int(y0 + size * 0.14))
    p.drawLine(int(x0), int(y0 + size * 0.28), int(x0 + size * 0.16), int(y0 + size * 0.14))
    p.drawLine(int(x0 + size * 0.22), int(y0 + size * 0.28),
               int(x0 + size * 0.46), int(y0 + size * 0.28))
    p.end()
    return QIcon(pix)


class DesktopIcon(QToolButton):
    """One clickable 'app' tile in the dock -- a real drawn icon (see the
    _icon_* functions above) plus a label underneath. Hovering lifts the
    icon (it grows slightly) and blooms a soft glow tinted to the app's
    own color; clicking gives it a quick 'press' squash-back for tactile
    feedback -- the kind of micro-motion a real OS launcher has and a
    plain QToolButton doesn't."""

    BASE_SIZE = 56
    HOVER_SIZE = 64
    PRESS_SIZE = 48

    def __init__(self, icon: QIcon, label: str, glow_color: str):
        super().__init__()
        self.setIcon(icon)
        self.setIconSize(QSize(self.BASE_SIZE, self.BASE_SIZE))
        self.setText(label)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setFixedSize(96, 98)
        self.setCursor(Qt.PointingHandCursor)
        self.setAutoRaise(True)
        self.setStyleSheet("""
            QToolButton {
                color: #eef2f7;
                background: transparent;
                border: none;
                border-radius: 12px;
                font-size: 11px;
                padding-top: 4px;
            }
            QToolButton:hover { background: rgba(255, 255, 255, 26); }
            QToolButton:pressed { background: rgba(255, 255, 255, 46); }
        """)

        self._glow = _shadow(blur=0, dy=0, alpha=0)
        self._glow.setColor(QColor(glow_color))
        self.setGraphicsEffect(self._glow)

        self._size_anim = QPropertyAnimation(self, b"iconSize")
        self._size_anim.setDuration(160)
        self._size_anim.setEasingCurve(QEasingCurve.OutCubic)

        self._glow_anim = QPropertyAnimation(self._glow, b"blurRadius")
        self._glow_anim.setDuration(160)

    def _animate_icon(self, target: int, glow_blur: int):
        for anim, prop, end in (
            (self._size_anim, "size", QSize(target, target)),
            (self._glow_anim, "blur", glow_blur),
        ):
            anim.stop()
            anim.setEndValue(end)
            anim.start()

    def enterEvent(self, event):
        self._animate_icon(self.HOVER_SIZE, 24)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_icon(self.BASE_SIZE, 0)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._animate_icon(self.PRESS_SIZE, 6)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        target = self.HOVER_SIZE if self.underMouse() else self.BASE_SIZE
        self._animate_icon(target, 24 if self.underMouse() else 0)
        super().mouseReleaseEvent(event)


class RunningDot(QLabel):
    """A tiny colored dot under a dock icon that appears while that app
    has at least one window open -- the 'this is running' indicator every
    real OS dock/taskbar has."""

    def __init__(self, color: str):
        super().__init__()
        self.setFixedSize(7, 7)
        self.setStyleSheet(f"""
            background: {color};
            border-radius: 3px;
        """)
        self.setVisible(False)


class TaskbarButton(QToolButton):
    """A running app's entry in the bottom taskbar -- click to bring its
    window to the front or restore it if it was minimized. Fades in when
    the app opens instead of just popping into existence, and lights up
    with an accent underline while its window is the active one."""

    def __init__(self, icon: QIcon, label: str):
        super().__init__()
        self.setIcon(icon)
        self.setIconSize(QSize(18, 18))
        self.setText(" " + label)
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setCursor(Qt.PointingHandCursor)
        self.set_active(False)

        self._opacity_effect = QGraphicsOpacityEffect()
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._fade_in = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_in.setDuration(220)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_in.start()

    def set_active(self, active: bool):
        border = "#6f8cff" if active else "rgba(255, 255, 255, 40)"
        bg = "rgba(111, 140, 255, 45)" if active else "rgba(255, 255, 255, 20)"
        self.setStyleSheet(f"""
            QToolButton {{
                color: #eef2f7;
                background: {bg};
                border: 1px solid {border};
                border-bottom: 2px solid {"#6f8cff" if active else border};
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QToolButton:hover {{ background: rgba(255, 255, 255, 45); }}
        """)


class TitleBar(QWidget):
    """A fully custom, app-tinted title bar for each window, replacing
    the OS's own native chrome (which PyQt can't restyle -- its buttons
    and text are drawn by the platform theme engine, which is exactly
    why the screenshot showed a plain gray Windows title bar no matter
    what stylesheet was applied). Dragging this bar moves the window;
    the three buttons minimize (hide, restored from the taskbar),
    maximize/restore, and close (animated, via MainWindow)."""

    def __init__(self, subwindow, icon: QIcon, title: str,
                 color1: str, color2: str, on_close):
        super().__init__()
        self.subwindow = subwindow
        self._on_close = on_close
        self._drag_pos = None
        self._start_pos = None
        self._maximized = False

        self.setFixedHeight(34)
        self.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {color1}, stop:1 {color2});
                border-top-left-radius: 9px;
                border-top-right-radius: 9px;
            }}
            QLabel {{ color: #10131a; font-weight: 600; background: transparent; }}
            QToolButton {{
                color: #10131a;
                background: transparent;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 13px;
            }}
            QToolButton:hover {{ background: rgba(0, 0, 0, 55); color: #ffffff; }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 6, 0)
        layout.setSpacing(4)

        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(16, 16))
        layout.addWidget(icon_label)

        text = QLabel(title)
        layout.addWidget(text)
        layout.addStretch(1)

        min_btn = QToolButton()
        min_btn.setText("—")
        min_btn.setFixedSize(26, 24)
        min_btn.setCursor(Qt.PointingHandCursor)
        min_btn.clicked.connect(lambda: self.subwindow.hide())
        layout.addWidget(min_btn)

        self._max_btn = QToolButton()
        self._max_btn.setText("▢")
        self._max_btn.setFixedSize(26, 24)
        self._max_btn.setCursor(Qt.PointingHandCursor)
        self._max_btn.clicked.connect(self._toggle_maximize)
        layout.addWidget(self._max_btn)

        close_btn = QToolButton()
        close_btn.setText("✕")
        close_btn.setFixedSize(26, 24)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("QToolButton:hover { background: #e04b4b; color: white; }")
        close_btn.clicked.connect(self._on_close)
        layout.addWidget(close_btn)

    def _toggle_maximize(self):
        if self._maximized:
            self.subwindow.showNormal()
        else:
            self.subwindow.showMaximized()
        self._maximized = not self._maximized

    def mouseDoubleClickEvent(self, event):
        self._toggle_maximize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos()
            self._start_pos = self.subwindow.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and not self._maximized:
            delta = event.globalPos() - self._drag_pos
            self.subwindow.move(self._start_pos + delta)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None


class MainWindow(QMainWindow, WorkerMixin):
    """The 'open a computer and see its screen' control panel, styled as
    a tiny desktop OS: a dock of real, glowing app icons on the left
    instead of tabs; a workspace where each app opens in its own
    custom-chrome window with fade+scale open/close animations; and a
    taskbar along the bottom that tracks and highlights the active
    window."""

    # key, label, icon factory, tab class, title-bar gradient colors
    APPS = (
        ("files", "Files", _icon_files, FilesTab, ("#ffd166", "#f7a531")),
        ("wifi", "Wifi", _icon_wifi, WifiTab, ("#4facfe", "#2266d8")),
        ("apps", "Apps & Jobs", _icon_apps, AppsTab, ("#a06bff", "#6f3ef0")),
        ("terminal", "Terminal", _icon_terminal, TerminalTab, ("#3ddc84", "#1a9b56")),
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32 Control Panel")
        self.resize(1100, 700)
        self._open_windows = {}     # key -> QMdiSubWindow
        self._taskbar_buttons = {}  # key -> TaskbarButton
        self._dots = {}             # key -> RunningDot
        self._anims = []            # keeps in-flight animations alive

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        # -- left dock: the app icons, replacing the old tab bar --
        dock = QFrame()
        dock.setFixedWidth(128)
        dock.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1c2431, stop:1 #12161f);
                border-right: 1px solid #05070a;
            }
        """)
        dock_layout = QVBoxLayout(dock)
        dock_layout.setContentsMargins(12, 18, 12, 12)
        dock_layout.setSpacing(6)

        title = QLabel("ESP32 OS")
        title.setStyleSheet("color: #7fa8ff; font-weight: bold; font-size: 13px; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        dock_layout.addWidget(title)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addStretch(1)
        self.conn_dot = QLabel()
        self.conn_dot.setFixedSize(8, 8)
        self.conn_dot.setStyleSheet("background: #f5c451; border-radius: 4px;")
        status_row.addWidget(self.conn_dot)
        self.conn_label = QLabel("checking...")
        self.conn_label.setStyleSheet("color: #8891a5; font-size: 10px;")
        status_row.addWidget(self.conn_label)
        status_row.addStretch(1)
        dock_layout.addLayout(status_row)
        dock_layout.addSpacing(10)

        self._icon_factories = {}
        for key, label, icon_fn, tab_cls, colors in self.APPS:
            icon = icon_fn()
            self._icon_factories[key] = (icon, label, tab_cls, colors)

            btn = DesktopIcon(icon, label, colors[0])
            btn.clicked.connect(lambda _checked=False, k=key: self._open_app(k))
            dock_layout.addWidget(btn, 0, Qt.AlignHCenter)

            dot = RunningDot(colors[0])
            self._dots[key] = dot
            dock_layout.addWidget(dot, 0, Qt.AlignHCenter)
            dock_layout.addSpacing(4)

        dock_layout.addStretch(1)
        body.addWidget(dock)

        # -- right workspace: apps open as real windows here --
        self.mdi = QMdiArea()
        self.mdi.setViewMode(QMdiArea.SubWindowView)
        self.mdi.setBackground(_wallpaper_brush())
        self.mdi.setStyleSheet("QMdiArea { border: none; }")
        self.mdi.subWindowActivated.connect(self._on_active_changed)
        body.addWidget(self.mdi, 1)

        # -- bottom taskbar --
        self.taskbar = QFrame()
        self.taskbar.setFixedHeight(44)
        self.taskbar.setStyleSheet("""
            QFrame { background: #171b24; border-top: 1px solid #05070a; }
        """)
        self.taskbar_layout = QHBoxLayout(self.taskbar)
        self.taskbar_layout.setContentsMargins(10, 6, 10, 6)
        self.taskbar_layout.setSpacing(8)
        self.taskbar_layout.addStretch(1)
        outer.addWidget(self.taskbar)

        self.setCentralWidget(central)

        # Open the Files app by default, like a fresh desktop session.
        self._open_app("files")
        self._check_connection()

    def _check_connection(self):
        self.conn_label.setText("checking...")
        self.conn_dot.setStyleSheet("background: #f5c451; border-radius: 4px;")
        self._start_worker(
            lambda: shell.run("wifi", timeout=8),
            lambda res: self._set_connection(True),
            lambda err: self._set_connection(False),
        )

    def _set_connection(self, ok: bool):
        color = "#3ddc84" if ok else "#e0555f"
        self.conn_dot.setStyleSheet(f"background: {color}; border-radius: 4px;")
        self.conn_label.setText("connected" if ok else "not connected")

    # -- opening / restoring --------------------------------------------

    def _open_app(self, key: str):
        existing = self._open_windows.get(key)
        if existing is not None:
            existing.show()
            existing.showNormal()
            self.mdi.setActiveSubWindow(existing)
            existing.raise_()
            return

        icon, label, tab_cls, colors = self._icon_factories[key]

        container = QFrame()
        container.setObjectName("subwindowFrame")
        container.setStyleSheet("""
            #subwindowFrame {
                background: #151a23;
                border: 1px solid #2a3040;
                border-radius: 9px;
            }
        """)
        container.setGraphicsEffect(_shadow(blur=36, dy=12, alpha=190))

        frame_layout = QVBoxLayout(container)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        sub_ref = {}  # filled in after addSubWindow, used by the close callback
        title_bar = TitleBar(
            subwindow=None, icon=icon, title=label, color1=colors[0],
            color2=colors[1], on_close=lambda: self._close_app(key),
        )
        frame_layout.addWidget(title_bar)

        content = tab_cls()
        content.setContentsMargins(8, 8, 8, 8)
        frame_layout.addWidget(content, 1)

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 2, 2)
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(container))
        frame_layout.addLayout(grip_row)

        sub = self.mdi.addSubWindow(container, Qt.FramelessWindowHint)
        title_bar.subwindow = sub  # now that it exists, let the title bar drag it
        sub.resize(660, 480)
        sub.setWindowOpacity(0.0)
        sub.show()

        # -- fade + scale-in open animation --
        target_rect = sub.geometry()
        cx, cy = target_rect.center().x(), target_rect.center().y()
        shrink_w, shrink_h = int(target_rect.width() * 0.94), int(target_rect.height() * 0.94)
        start_rect = target_rect.__class__(0, 0, shrink_w, shrink_h)
        start_rect.moveCenter(target_rect.center())
        sub.setGeometry(start_rect)

        group = QParallelAnimationGroup(self)
        fade = QPropertyAnimation(sub, b"windowOpacity")
        fade.setDuration(220)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setEasingCurve(QEasingCurve.OutCubic)
        grow = QPropertyAnimation(sub, b"geometry")
        grow.setDuration(220)
        grow.setStartValue(start_rect)
        grow.setEndValue(target_rect)
        grow.setEasingCurve(QEasingCurve.OutCubic)
        group.addAnimation(fade)
        group.addAnimation(grow)
        self._anims.append(group)
        group.finished.connect(lambda g=group: g in self._anims and self._anims.remove(g))
        group.start()

        taskbar_btn = TaskbarButton(icon, label)
        taskbar_btn.clicked.connect(lambda _checked=False, k=key: self._open_app(k))
        self.taskbar_layout.insertWidget(self.taskbar_layout.count() - 1, taskbar_btn)

        self._open_windows[key] = sub
        self._taskbar_buttons[key] = taskbar_btn
        self._dots[key].setVisible(True)

    # -- closing (animated) ----------------------------------------------

    def _close_app(self, key: str):
        sub = self._open_windows.get(key)
        if sub is None:
            return

        start_rect = sub.geometry()
        end_rect = start_rect.__class__(0, 0, int(start_rect.width() * 0.94),
                                         int(start_rect.height() * 0.94))
        end_rect.moveCenter(start_rect.center())

        group = QParallelAnimationGroup(self)
        fade = QPropertyAnimation(sub, b"windowOpacity")
        fade.setDuration(180)
        fade.setStartValue(1.0)
        fade.setEndValue(0.0)
        fade.setEasingCurve(QEasingCurve.InCubic)
        shrink = QPropertyAnimation(sub, b"geometry")
        shrink.setDuration(180)
        shrink.setStartValue(start_rect)
        shrink.setEndValue(end_rect)
        shrink.setEasingCurve(QEasingCurve.InCubic)
        group.addAnimation(fade)
        group.addAnimation(shrink)
        self._anims.append(group)
        group.finished.connect(lambda g=group, k=key: self._finish_close(k, g))
        group.start()

        btn = self._taskbar_buttons.get(key)
        if btn is not None:
            btn.setEnabled(False)

    def _finish_close(self, key: str, group):
        if group in self._anims:
            self._anims.remove(group)

        sub = self._open_windows.pop(key, None)
        if sub is not None:
            self.mdi.removeSubWindow(sub)
            sub.deleteLater()

        btn = self._taskbar_buttons.pop(key, None)
        if btn is not None:
            btn.deleteLater()

        dot = self._dots.get(key)
        if dot is not None:
            dot.setVisible(False)

    # -- active-window tracking -------------------------------------------

    def _on_active_changed(self, sub):
        for key, window in self._open_windows.items():
            btn = self._taskbar_buttons.get(key)
            if btn is not None:
                btn.set_active(window is sub)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
