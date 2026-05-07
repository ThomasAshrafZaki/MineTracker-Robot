#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  Mine Tracker Robot — WebRTC Live Stream Setup
#  Target: NVIDIA Jetson Nano 4GB (JetPack 4.x / Ubuntu 18.04)
#
#  Run on your Jetson:
#    chmod +x setup_webrtc_jetson.sh && sudo ./setup_webrtc_jetson.sh
#
#  What this does:
#    1. Verifies JetPack + NVIDIA environment
#    2. Installs GStreamer NVENC plugins (hardware H.264 encoding)
#    3. Downloads and installs mediamtx (aarch64)
#    4. Writes optimized mediamtx.yml config
#    5. Creates a systemd service (auto-starts on boot)
# ════════════════════════════════════════════════════════════════

set -e

# ── Colors ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

INSTALL_DIR="/opt/mediamtx"
SERVICE_NAME="robot-stream"
MEDIAMTX_VERSION="v1.9.1"
MEDIAMTX_BINARY="mediamtx_${MEDIAMTX_VERSION}_linux_arm64v8.tar.gz"
MEDIAMTX_URL="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/${MEDIAMTX_BINARY}"

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   Mine Tracker — WebRTC Stream Setup             ║"
echo "  ║   NVIDIA Jetson Nano 4GB  |  NVENC H.264         ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Must be root ───────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}[ERROR] Run with sudo: sudo ./setup_webrtc_jetson.sh${NC}"
  exit 1
fi

# ── Verify this is a Jetson ────────────────────────────────────
echo -e "${GREEN}[1/7] Verifying Jetson Nano environment...${NC}"

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
  echo -e "${RED}[ERROR] Expected aarch64, got: ${ARCH}. Is this a Jetson Nano?${NC}"
  exit 1
fi

# Check for NVIDIA Jetson-specific file
if [ ! -f /etc/nv_tegra_release ] && [ ! -f /proc/device-tree/model ]; then
  echo -e "${YELLOW}[WARN] Cannot confirm Jetson hardware. Proceeding anyway...${NC}"
else
  JETSON_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Jetson")
  echo -e "${GREEN}    ✓ Detected: ${JETSON_MODEL}${NC}"
fi

# Check JetPack version
if [ -f /etc/nv_tegra_release ]; then
  JETPACK_VER=$(head -1 /etc/nv_tegra_release)
  echo -e "${GREEN}    ✓ JetPack: ${JETPACK_VER}${NC}"
fi

# Check CUDA
if command -v nvcc &>/dev/null; then
  CUDA_VER=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
  echo -e "${GREEN}    ✓ CUDA: ${CUDA_VER}${NC}"
else
  echo -e "${YELLOW}    ⚠ CUDA not in PATH (normal for JetPack 4.x, NVENC still works)${NC}"
fi

# ── System update + base dependencies ─────────────────────────
echo -e "${GREEN}[2/7] Updating system and installing base dependencies...${NC}"
apt-get update -qq
apt-get install -y -qq \
  wget curl tar \
  v4l-utils \
  python3-pip \
  libglib2.0-0

# ── GStreamer + NVIDIA NVENC plugins ───────────────────────────
echo -e "${GREEN}[3/7] Installing GStreamer with NVIDIA NVENC hardware encoder...${NC}"

# Core GStreamer
apt-get install -y -qq \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav

# NVIDIA-specific GStreamer plugins (JetPack ships these)
# nvv4l2h264enc  = hardware H.264 encoder (NVENC)
# nvvidconv      = GPU-accelerated color space conversion
# nvarguscamerasrc = CSI camera source
apt-get install -y -qq \
  gstreamer1.0-rtsp \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev || true

# NVIDIA multimedia packages (should already be on JetPack)
if dpkg -l | grep -q "nvidia-l4t-multimedia"; then
  echo -e "${GREEN}    ✓ NVIDIA multimedia packages already installed${NC}"
else
  echo -e "${YELLOW}    ⚠ nvidia-l4t-multimedia not found — NVENC may be limited${NC}"
  echo -e "${YELLOW}      Make sure you installed JetPack properly via SDK Manager${NC}"
fi

# ── Verify NVENC is working ────────────────────────────────────
echo -e "${GREEN}[4/7] Verifying NVENC hardware encoder...${NC}"

if gst-inspect-1.0 nvv4l2h264enc &>/dev/null; then
  echo -e "${GREEN}    ✓ nvv4l2h264enc (NVENC) is available — hardware encoding enabled${NC}"
  USE_NVENC=true
else
  echo -e "${YELLOW}    ⚠ nvv4l2h264enc not found — will fall back to software encoding${NC}"
  echo -e "${YELLOW}      This still works but adds ~100ms latency${NC}"
  USE_NVENC=false
fi

# Check for USB camera
echo ""
if v4l2-ctl --list-devices &>/dev/null; then
  echo -e "${CYAN}    Detected cameras:${NC}"
  v4l2-ctl --list-devices 2>/dev/null | grep -v "^$" | sed 's/^/      /'
fi

# Check for CSI camera
if [ -d /dev/nvhost-vi0 ] || ls /dev/video* &>/dev/null; then
  echo -e "${GREEN}    ✓ Camera device(s) found${NC}"
fi

# ── Download mediamtx ──────────────────────────────────────────
echo -e "${GREEN}[5/7] Downloading mediamtx ${MEDIAMTX_VERSION} (aarch64)...${NC}"
mkdir -p "$INSTALL_DIR"

wget -q --show-progress -O /tmp/mediamtx.tar.gz "$MEDIAMTX_URL"
tar -xzf /tmp/mediamtx.tar.gz -C "$INSTALL_DIR"
rm /tmp/mediamtx.tar.gz
chmod +x "$INSTALL_DIR/mediamtx"

echo -e "${GREEN}    ✓ mediamtx installed at ${INSTALL_DIR}${NC}"

# ── Write mediamtx config ──────────────────────────────────────
echo -e "${GREEN}[6/7] Writing optimized mediamtx.yml for Jetson Nano...${NC}"

cat > "$INSTALL_DIR/mediamtx.yml" << 'MEDIAMTX_CONFIG'
# ════════════════════════════════════════════════════════════════
#  mediamtx.yml — Mine Tracker Robot
#  Platform: NVIDIA Jetson Nano 4GB
#  Encoding: NVENC hardware H.264 (5–10ms vs Pi's 50–100ms)
#
#  Edit this file at: /opt/mediamtx/mediamtx.yml
#  Restart after changes: sudo systemctl restart robot-stream
# ════════════════════════════════════════════════════════════════

logLevel: info
logDestinations: [file]
logFile: /var/log/robot-stream.log

# ── Disable unused protocols ──────────────────────────────────
rtmpDisable: true
srtDisable: true

# ── RTSP (internal — GStreamer pushes H.264 here) ─────────────
rtsp:
  enable: true
  address: :8554

# ── HLS (browser fallback) ────────────────────────────────────
hls:
  enable: true
  address: :8888
  segmentCount: 3
  segmentDuration: 1s

# ── WebRTC (primary — lowest latency) ────────────────────────
webrtc:
  enable: true
  address: :8889
  # Add your Jetson's IP if browsers can't reach it:
  # additionalHosts: [192.168.1.50]
  iceServers:
    - urls: [stun:stun.l.google.com:19302]

# ── REST API ─────────────────────────────────────────────────
api:
  enable: true
  address: :9997

# ════════════════════════════════════════════════════════════════
#  CAMERA STREAM — path: "cam"
#  Browser connects to: http://<JETSON_IP>:8889/cam
#
#  GStreamer pipeline runs on demand and feeds H.264 into mediamtx
#  via RTSP on localhost. NVENC encodes on the GPU — not the CPU.
# ════════════════════════════════════════════════════════════════
paths:
  cam:

    # ── OPTION A: USB Camera + NVENC (DEFAULT) ────────────────
    #    Best for testing — works with any USB webcam
    #    Latency: ~80–150ms total end-to-end
    runOnReady: >-
      gst-launch-1.0
      v4l2src device=/dev/video0 do-timestamp=true
      ! video/x-raw,width=640,height=480,framerate=30/1,format=YUY2
      ! nvvidconv
      ! video/x-raw(memory:NVMM),format=I420
      ! nvv4l2h264enc
          bitrate=2000000
          iframeinterval=60
          preset-level=1
          control-rate=1
          vbv-size=33333
      ! h264parse
      ! rtspclientsink
          location=rtsp://127.0.0.1:$RTSP_PORT/$MTX_PATH
          protocols=tcp
    runOnReadyRestart: yes

    # ── OPTION B: USB Camera fallback (software libx264) ──────
    #    Use if NVENC is not available on your JetPack version
    #    Uncomment this block and comment out OPTION A
    #
    # runOnReady: >-
    #   gst-launch-1.0
    #   v4l2src device=/dev/video0 do-timestamp=true
    #   ! video/x-raw,width=640,height=480,framerate=30/1
    #   ! videoconvert
    #   ! x264enc
    #       tune=zerolatency
    #       speed-preset=ultrafast
    #       bitrate=2000
    #   ! h264parse
    #   ! rtspclientsink
    #       location=rtsp://127.0.0.1:$RTSP_PORT/$MTX_PATH
    #       protocols=tcp
    # runOnReadyRestart: yes

    # ── OPTION C: CSI Camera (IMX219 / Raspberry Pi Cam v2) ───
    #    Uncomment if using the CSI ribbon cable camera
    #    Latency: ~60–120ms — slightly faster than USB
    #
    # runOnReady: >-
    #   gst-launch-1.0
    #   nvarguscamerasrc sensor-id=0
    #   ! video/x-raw(memory:NVMM),
    #       width=640,height=480,
    #       framerate=30/1,
    #       format=NV12
    #   ! nvv4l2h264enc
    #       bitrate=2000000
    #       iframeinterval=60
    #       preset-level=1
    #       control-rate=1
    #   ! h264parse
    #   ! rtspclientsink
    #       location=rtsp://127.0.0.1:$RTSP_PORT/$MTX_PATH
    #       protocols=tcp
    # runOnReadyRestart: yes

    # ── OPTION D: High Quality (720p / LAN only) ──────────────
    #    Use on wired Ethernet for full quality
    #
    # runOnReady: >-
    #   gst-launch-1.0
    #   v4l2src device=/dev/video0 do-timestamp=true
    #   ! video/x-raw,width=1280,height=720,framerate=30/1
    #   ! nvvidconv
    #   ! video/x-raw(memory:NVMM),format=I420
    #   ! nvv4l2h264enc
    #       bitrate=4000000
    #       iframeinterval=60
    #       preset-level=1
    #       control-rate=1
    #   ! h264parse
    #   ! rtspclientsink
    #       location=rtsp://127.0.0.1:$RTSP_PORT/$MTX_PATH
    #       protocols=tcp
    # runOnReadyRestart: yes
MEDIAMTX_CONFIG

echo -e "${GREEN}    ✓ Config written to ${INSTALL_DIR}/mediamtx.yml${NC}"

# ── Create systemd service ─────────────────────────────────────
echo -e "${GREEN}[7/7] Creating systemd service '${SERVICE_NAME}'...${NC}"

cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Mine Tracker Robot — WebRTC Live Stream (Jetson Nano NVENC)
After=network.target nvargus-daemon.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
# Load Jetson NVENC environment
Environment="LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu"
Environment="GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0"
ExecStart=${INSTALL_DIR}/mediamtx ${INSTALL_DIR}/mediamtx.yml
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
# Allow enough time for NVENC to initialize
TimeoutStartSec=30

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

sleep 3

if systemctl is-active --quiet "$SERVICE_NAME"; then
  echo -e "${GREEN}    ✓ Service started and enabled on boot${NC}"
else
  echo -e "${YELLOW}    ⚠ Service may have issues. Check: journalctl -u ${SERVICE_NAME} -n 40${NC}"
fi

# ── Add user to video group (for camera access) ────────────────
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo '')}"
if [ -n "$REAL_USER" ]; then
  usermod -aG video "$REAL_USER" 2>/dev/null || true
  echo -e "${GREEN}    ✓ Added '${REAL_USER}' to video group${NC}"
fi

# ── Final summary ──────────────────────────────────────────────
JETSON_IP=$(hostname -I | awk '{print $1}')

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ✅  Jetson Nano Setup Complete!"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Jetson IP      : ${YELLOW}${JETSON_IP}${NC}"
echo -e "  NVENC Encoder  : ${USE_NVENC:+${GREEN}✓ Hardware (NVENC)${NC}}${USE_NVENC:-${YELLOW}⚠ Software fallback${NC}}"
echo ""
echo -e "  WebRTC Stream  : ${GREEN}http://${JETSON_IP}:8889/cam${NC}"
echo -e "  Built-in Player: ${GREEN}http://${JETSON_IP}:8889/cam/${NC}"
echo -e "  RTSP (internal): ${GREEN}rtsp://${JETSON_IP}:8554/cam${NC}"
echo -e "  API Status     : ${GREEN}http://${JETSON_IP}:9997/v3/paths/list${NC}"
echo ""
echo -e "  Commands:"
echo -e "    ${CYAN}sudo systemctl status ${SERVICE_NAME}${NC}"
echo -e "    ${CYAN}sudo journalctl -u ${SERVICE_NAME} -f${NC}"
echo -e "    ${CYAN}sudo systemctl restart ${SERVICE_NAME}${NC}"
echo ""
echo -e "  Config  : ${YELLOW}${INSTALL_DIR}/mediamtx.yml${NC}"
echo -e "  Logs    : ${YELLOW}/var/log/robot-stream.log${NC}"
echo ""
echo -e "  ${BOLD}Test GStreamer pipeline manually:${NC}"
echo -e "    ${CYAN}gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink${NC}"
echo ""
echo -e "${YELLOW}  📌 Update robot_viewer.html:${NC}"
echo -e "     const PI_IP = \"${JETSON_IP}\";"
echo ""
