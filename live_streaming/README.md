# WebRTC Live Stream — Mine Tracker Robot
## NVIDIA Jetson Nano 4GB Edition

---

## Why Jetson Nano is better than Pi 3 for this

| | Raspberry Pi 3 | Jetson Nano 4GB |
|--|----------------|-----------------|
| H.264 encoding | Software (libx264) | **Hardware NVENC** |
| Encode latency | 50–100ms | **5–10ms** |
| CPU during encode | ~70% one core | **<10%** |
| GPU | None | 128-core Maxwell |
| RAM | 1GB | **4GB** |
| Total stream latency | ~150–250ms | **~80–150ms** |

---

## Files

| File | Purpose |
|------|---------|
| `setup_webrtc_jetson.sh` | Run once on Jetson — installs everything |
| `mediamtx.yml` | Stream config (5 camera options pre-written) |
| `robot_viewer.html` | Web viewer — drop into your web app |
| `README.md` | This file |

---

## Requirements

- Jetson Nano 4GB with JetPack 4.5 or later
- USB webcam (for testing) or CSI camera (IMX219 / Pi Camera v2)
- Internet connection during setup (to download mediamtx)

---

## Setup

### 1. Flash JetPack (if not done)
Download from: https://developer.nvidia.com/embedded/jetpack
Use Balena Etcher to flash the SD card (64GB+ recommended).

### 2. Run the setup script on the Jetson

```bash
# From your laptop — copy script to Jetson
scp setup_webrtc_jetson.sh user@<JETSON_IP>:/home/user/

# SSH into Jetson
ssh user@<JETSON_IP>

# Run setup
chmod +x setup_webrtc_jetson.sh
sudo ./setup_webrtc_jetson.sh
```

### 3. Test the stream in a browser
Open: `http://<JETSON_IP>:8889/cam`
You should see the built-in mediamtx player with your camera feed.

### 4. Configure your web viewer
Edit `robot_viewer.html`:
```js
const JETSON_IP = "192.168.1.100";  // ← your Jetson's actual IP
```

---

## Service management

```bash
# Check status
sudo systemctl status robot-stream

# Live logs
sudo journalctl -u robot-stream -f

# Restart stream
sudo systemctl restart robot-stream

# Stop
sudo systemctl stop robot-stream
```

---

## Switching camera modes

Edit the config and restart:
```bash
sudo nano /opt/mediamtx/mediamtx.yml
sudo systemctl restart robot-stream
```

The config has 5 pre-written options — comment/uncomment as needed:

| Option | Source | Encoding | Use case |
|--------|--------|----------|----------|
| A | USB camera | NVENC (GPU) | **Default — testing** |
| B | USB camera | x264 (CPU) | Fallback if NVENC missing |
| C | CSI camera | NVENC (GPU) | Best quality + latency |
| D | USB camera | NVENC (GPU) | 720p on wired LAN |
| E | Dual cameras | NVENC (GPU) | Front + rear |

---

## Testing GStreamer pipeline manually

Before the service, test if your camera + NVENC works:

```bash
# Test USB camera preview (shows on Jetson display)
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink

# Test NVENC encoding (no display — just confirms it works)
gst-launch-1.0 \
  v4l2src device=/dev/video0 \
  ! video/x-raw,width=640,height=480,framerate=30/1 \
  ! nvvidconv \
  ! video/x-raw(memory:NVMM),format=I420 \
  ! nvv4l2h264enc bitrate=2000000 \
  ! fakesink

# Test CSI camera (IMX219)
gst-launch-1.0 \
  nvarguscamerasrc sensor-id=0 \
  ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 \
  ! nvvidconv \
  ! autovideosink
```

If the NVENC test runs without errors, the full stream will work.

---

## Troubleshooting

### "nvv4l2h264enc not found"
NVENC plugin is missing. Fix:
```bash
sudo apt-get install nvidia-l4t-multimedia
# or reflash with full JetPack via SDK Manager
```

### Black screen in browser
```bash
# Check camera is detected
v4l2-ctl --list-devices

# Check GStreamer can open it
gst-launch-1.0 v4l2src device=/dev/video0 ! fakesink

# Check service logs
sudo journalctl -u robot-stream -n 50
```

### "HTTP 404" when browser connects
- GStreamer pipeline may have failed to start
- Check: `sudo journalctl -u robot-stream -f`
- Try running the gst-launch command manually first

### High latency (>300ms)
- Switch to wired Ethernet (biggest improvement)
- Reduce resolution: `width=320,height=240`
- Lower FPS: `framerate=15/1`

### USB camera format error (YUY2 vs MJPEG)
Check what your camera supports:
```bash
v4l2-ctl --list-formats-ext -d /dev/video0
```
Then update the format in mediamtx.yml:
- Replace `format=YUY2` with `format=MJPG` if needed
- Add `-input_format mjpeg` before `-i` if using ffmpeg

---

## Port reference

| Port | Protocol | Purpose |
|------|----------|---------|
| 8554 | RTSP | Internal GStreamer → mediamtx |
| 8888 | HTTP | HLS fallback stream |
| 8889 | HTTP/WS | **WebRTC (main — browser connects here)** |
| 9997 | HTTP | REST API + health check |

---

## API health check

```bash
# From Jetson or any machine on the network
curl http://<JETSON_IP>:9997/v3/paths/list

# Expected response shows active path "cam" with status "ready"
```

---

## Remote access (outside LAN)

For field testing outside your local network:

```bash
# Install ngrok on Jetson
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt install ngrok

# Create tunnel
ngrok http 8889

# Use the HTTPS URL in robot_viewer.html
```

Note: WebRTC over ngrok may add 50–200ms depending on server location.
