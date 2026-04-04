# LCL Sovereign Audio Bridge — ASR Live Audio Integration

Real-time audio capture → WebSocket streaming → Whisper ASR + Speaker Diarization → Web Portal.

---

## Architecture

```
[ESP32-WROOM-32]  →  ws://launchcloud.duckdns.org:8765  →  [MacBook Hub]  →  [Web Portal]
 INMP441 I2S mic      raw 16kHz/16-bit PCM binary              Python ASR        Browser
 MAX9814 (unused)     WebSocket persistent connection           Whisper-tiny      Web Audio API
```

## Hardware (ESP32-WROOM-32 / ESP32-D0WD-V3 rev3.1)

| Component     | Pin        |
|---------------|------------|
| INMP441 VDD   | 3.3V       |
| INMP441 GND   | GND        |
| INMP441 SD    | GPIO 4     |
| INMP441 WS    | GPIO 15    |
| INMP441 SCK   | GPIO 14    |
| INMP441 L/R   | GND (left channel) |
| Onboard LED   | GPIO 2 (VU meter) |

**Critical:** L/R pin MUST be tied to GND. Floating = silence (RMS=0).

## Network

- **WebSocket Hub**: `ws://launchcloud.duckdns.org:8765` (port-forwarded → `192.168.1.160:8765`)
- **Web Portal**: `http://launchcloudlabs.com/demo/streaming/index.html`
- **SSH**: `launchcloud.duckdns.org:54321` → `192.168.1.160:22`

> ⚠️ Always use `ws://` (not `wss://`). The portal is served over HTTP.  
> ⚠️ Verizon router has hairpin NAT disabled — DuckDNS only works from outside the LAN.

## Files

| File | Purpose |
|------|---------|
| `LCL_Edge_Node.ino` | ESP32 firmware — I2S capture, LED VU meter, WebSocket streaming |
| `server_receiver.py` | Python hub — receives audio, 15x boost, Whisper ASR, MFCC diarization |
| `index.html` | Web portal — Web Audio API playback, transcript display, VU meter |
| `deploy_streaming.py` | FTP deployment to launchcloudlabs.com |
| `start_hub.sh` | Keep-alive wrapper — auto-restarts server on crash |

## How to Run

```bash
# Start the hub (auto-restarts on crash)
~/start_hub.sh

# Or directly:
export KMP_DUPLICATE_LIB_OK=TRUE
python3 -u server_receiver.py

# Monitor ESP32 serial
arduino-cli monitor -p /dev/cu.SLAB_USBtoUART -c baudrate=115200

# Deploy portal to FTP
python3 deploy_streaming.py
```

## Known Issues / Notes

- **ESP32 Core 3.3.7**: Do NOT use `analogRead()` — conflicts with `driver_ng`. I2S only.
- **i2s_pin_config_t**: Must include `.mck_io_num = I2S_PIN_NO_CHANGE` or I2S init fails silently.
- **faster_whisper / ctranslate2**: Must run in `loop.run_in_executor()` — calling from asyncio directly causes `SIGABRT`.
- **Audio gain**: Server applies 15x boost before broadcasting to browsers. Portal applies 10x. Total ~150x needed due to INMP441's low raw output (RMS ~0.001–0.003).

## Speaker Diarization

Pure numpy/scipy MFCC implementation (no pyannote, no resemblyzer — both fail on this machine).
- 13 MFCC coefficients per 2s window
- Cosine similarity threshold: 0.84
- Up to 10 speakers, each assigned a unique color

## Working State (as of 2026-04-04)

- ✅ ESP32 connects and streams audio to hub
- ✅ Hub receives, boosts, and broadcasts to browsers
- ✅ Whisper ASR transcribes in real-time
- ✅ Speaker diarization labels each transcript line
- ✅ Portal connects via `ws://launchcloud.duckdns.org:8765`
- ⚠️ Browser audio playback — packets reach browser, AudioContext scheduling active, volume may need tuning
