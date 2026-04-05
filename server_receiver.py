import asyncio
import websockets
import os
import json
import sys
import ssl
import audioop
import base64
import numpy as np
import urllib.parse
from collections import Counter

# LCL Sovereign Hub V15 — "Deepgram Edition"
# Telnyx phone bridge + ESP32 edge streaming + Deepgram Nova-3 ASR/Diarization

DEEPGRAM_API_KEY = "ea603f87124d18a21411f19c20368e27abdc696e"
GAIN_BOOST       = 15.0
SAMPLE_RATE      = 16000

CERT_DIR  = os.path.expanduser("~/certbot-duckdns/config/live/launchcloud.duckdns.org")
CERT_FILE = os.path.join(CERT_DIR, "fullchain.pem")
KEY_FILE  = os.path.join(CERT_DIR, "privkey.pem")

SPEAKER_COLORS = [
    "#38bdf8", "#f59e0b", "#a855f7", "#10b981",
    "#ef4444", "#f97316", "#00ffcc", "#ff007f",
    "#7fff00", "#6366f1"
]

# Shared state
browsers      = set()
edge_nodes    = set()
phone_streams = set()

# Phone bridge audio state
phone_ratecv_state = None   # audioop.ratecv state — persisted across packets
phone_pcmu_buf     = b""    # accumulator — send when >= 160 bytes (20ms at 8kHz)

# Deepgram feeder task
audio_queue: asyncio.Queue = None   # initialized in main()

# Helpers

def boost_bytes(raw: bytes, gain: float) -> bytes:
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return np.clip(arr * gain, -32768, 32767).astype(np.int16).tobytes()

def speaker_color(idx: int) -> str:
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]

def broadcast_transcript(text: str, speaker_idx: int):
    if not text.strip():
        return
    msg = json.dumps({
        "type":    "transcript",
        "text":    text.strip(),
        "speaker": f"Speaker {speaker_idx + 1}",
        "color":   speaker_color(speaker_idx)
    })
    print(f"[Speaker {speaker_idx + 1}] {text.strip()}")
    if browsers:
        websockets.broadcast(browsers, msg)

# Deepgram streaming engine (raw WebSocket — bypasses SDK for websockets v15 compat)

DG_WS_URL = (
    "wss://api.deepgram.com/v1/listen?"
    + urllib.parse.urlencode({
        "model":            "nova-3",
        "encoding":         "linear16",
        "sample_rate":      SAMPLE_RATE,
        "channels":         1,
        "diarize":          "true",
        "punctuate":        "true",
        "smart_format":     "true",
        "interim_results":  "true",   # required for utterance_end_ms
        "utterance_end_ms": 1000,
        "vad_events":       "true",
    })
)
DG_HEADERS = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

async def deepgram_engine():
    """
    Persistent Deepgram Nova-3 streaming loop.
    Feeds raw PCM from audio_queue and fires transcripts back to browsers.
    Reconnects automatically on any error.
    """
    global audio_queue

    while True:
        try:
            print("[DG] Connecting to Deepgram Nova-3 (diarize=True, smart_format=True)...")
            async with websockets.connect(
                DG_WS_URL,
                additional_headers=DG_HEADERS,
            ) as dg_ws:
                print("[DG] Nova-3 ACTIVE")

                async def feeder():
                    """Feed audio to Deepgram; send KeepAlive every 8s when idle."""
                    KEEPALIVE = json.dumps({"type": "KeepAlive"})
                    while True:
                        try:
                            chunk = await asyncio.wait_for(audio_queue.get(), timeout=8.0)
                        except asyncio.TimeoutError:
                            # No audio — send keepalive to prevent Deepgram timeout
                            try:
                                await dg_ws.send(KEEPALIVE)
                            except Exception:
                                break
                            continue
                        if chunk is None:   # sentinel — restart connection
                            break
                        try:
                            await dg_ws.send(chunk)
                        except Exception as e:
                            print(f"[DG SEND ERR] {e}")
                            break

                feed_task = asyncio.create_task(feeder())

                # Receive and broadcast transcripts
                async for raw in dg_ws:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    msg_type = data.get("type", "")
                    if msg_type != "Results":
                        continue
                    # Skip interim results — only process final utterances
                    if not data.get("is_final", False):
                        continue
                    channel = data.get("channel", {})
                    alts = channel.get("alternatives", [])
                    if not alts:
                        continue
                    alt = alts[0]
                    text = alt.get("transcript", "")
                    if not text.strip():
                        continue
                    # Dominant speaker from word-level diarization
                    words = alt.get("words", [])
                    speakers = [w.get("speaker") for w in words if w.get("speaker") is not None]
                    dominant = Counter(speakers).most_common(1)[0][0] if speakers else 0
                    broadcast_transcript(text, int(dominant))

                feed_task.cancel()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[DG ERROR] {e} — reconnecting in 3s...", file=sys.stderr)
            await asyncio.sleep(3)

# Telnyx phone bridge

async def telnyx_bridge_handler(websocket, stream_id):
    """Dedicated handler for Telnyx phone stream."""
    print(f"[PHONE] Bridge Active for Stream: {stream_id}")
    phone_streams.add((websocket, stream_id))
    try:
        async for msg in websocket:
            pass  # inbound phone  placeholder for future bidirectionalaudio 
    finally:
        phone_streams.discard((websocket, stream_id))
        print(f"[PHONE] Bridge Closed: {stream_id}")

# ── Main WebSocket handler ─────────────────────────────────────────────────────

async def handler(websocket):
    global audio_queue
    addr = websocket.remote_address
    pkt_count = 0

    # websockets v15: path is on websocket.request.path, not passed as argument
    try:
        path = websocket.request.path
    except Exception:
        path = "/"

    # Route Telnyx connections
    if path == "/telnyx":
        print(f"[PHONE] Incoming Telnyx Handshake from {addr}")
        async for message in websocket:
            try:
                data = json.loads(message)
                evt = data.get("event", "")
                print(f"[PHONE] Event: {evt} | keys: {list(data.keys())}")
                if evt == "connected":
                    print(f"[PHONE] Protocol connected: {data.get('version','?')}")
                    continue
                if evt == "start":
                    # Telnyx may use stream_id or stream_sid — try both
                    start_block = data.get("start", data)
                    sid = (start_block.get("stream_id")
                           or start_block.get("stream_sid")
                           or data.get("stream_id")
                           or "telnyx-stream")
                    print(f"[PHONE] Stream start: sid={sid} block={start_block}")
                    await telnyx_bridge_handler(websocket, sid)
                    break
            except Exception as e:
                print(f"[PHONE ERR] {e} | raw={message[:200]}")
        return

    print(f"[CONN] {addr}")
    try:
        first = await asyncio.wait_for(websocket.recv(), timeout=10.0)

        if isinstance(first, str):
            try:
                d = json.loads(first)
                if d.get("type") == "register_web":
                    print(f"[HUB] Browser verified ({addr})")
                    browsers.add(websocket)
                elif d.get("type") == "esp32_hello":
                    print(f"[HUB] Edge node verified ({addr})")
                    edge_nodes.add(websocket)
            except json.JSONDecodeError:
                pass

        elif isinstance(first, bytes):
            print(f"[HUB] Edge node binary-first ({addr})")
            edge_nodes.add(websocket)
            boosted = boost_bytes(first, GAIN_BOOST)
            if browsers: websockets.broadcast(browsers, boosted)
            await audio_queue.put(first)

        async for msg in websocket:
            if isinstance(msg, bytes):
                pkt_count += 1
                boosted = boost_bytes(msg, GAIN_BOOST)

# Browsers (boosted for audibility)
                if browsers:
                    websockets.broadcast(browsers, boosted)

# Deepgram (raw  better for ASR accuracy)unboosed 
                await audio_queue.put(msg)

# Telnyx phone bridge (G.711 PCMU transcode)
                # Per Telnyx docs: outgoing format is {"event":"media","media":{"payload":"base64"}}
                # Minimum chunk size is 20ms = 160 bytes at 8kHz PCMU
                # ratecv state is persisted across packets to avoid audio corruption
                if phone_streams:
                    global phone_ratecv_state, phone_pcmu_buf
                    pcm_8k, phone_ratecv_state = audioop.ratecv(
                        msg, 2, 1, 16000, 8000, phone_ratecv_state)
                    phone_pcmu_buf += audioop.lin2ulaw(pcm_8k, 2)
                    # Send in 160-byte (20ms) chunks as required by Telnyx
                    while len(phone_pcmu_buf) >= 160:
                        chunk = phone_pcmu_buf[:160]
                        phone_pcmu_buf = phone_pcmu_buf[160:]
                        payload = base64.b64encode(chunk).decode("utf-8")
                        for p_ws, sid in list(phone_streams):
                            try:
                                await p_ws.send(json.dumps({
                                    "event": "media",
                                    "media": {"payload": payload},
                                }))
                            except Exception as e:
                                print(f"[PHONE SEND ERR] {e}")
                                phone_streams.discard((p_ws, sid))

                if pkt_count % 100 == 0:
                    print(f"[AUDIO] pkts={pkt_count} B={len(browsers)} P={len(phone_streams)} q={audio_queue.qsize()}")

            else:
                try:
                    d = json.loads(msg)
                    if d.get("type") == "esp32_hello":
                        edge_nodes.add(websocket)
                    elif d.get("type") == "command" and edge_nodes:
                        websockets.broadcast(edge_nodes, msg)
                    elif d.get("type") == "update_config":
                        new_conf = d.get("config", {})
                        config_path = "/Users/garycolonna/lcl-asr-live-audio/bridge_config.json"
                        with open(config_path, "w") as f:
                            json.dump(new_conf, f, indent=4)
                        print(f"[CONFIG] Updated: {new_conf}")
                        await websocket.send(json.dumps({"type": "config_status", "ok": True}))
                except json.JSONDecodeError:
                    pass

    except asyncio.TimeoutError:
        print(f"[TIMEOUT] {addr}")
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[ERROR] {addr}: {e}", file=sys.stderr)
    finally:
        browsers.discard(websocket)
        edge_nodes.discard(websocket)
        print(f"[DISC] {addr} pkts={pkt_count} | B:{len(browsers)} E:{len(edge_nodes)}")

# ── Main ───────────────────────────────────────────────────────────────────────

async def main():
    global audio_queue
    audio_queue = asyncio.Queue(maxsize=500)  # ~500 ESP32 packets buffer
    print("--- LCL SOVEREIGN HUB V16 (DEEPGRAM NOVA-3) ONLINE ---")
    asyncio.create_task(deepgram_engine())

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)

    # Port 8765 — WSS (TLS) for browsers + Telnyx (HTTPS pages require wss://)
    # Port 8766 — plain WS for ESP32 (microcontroller can't do TLS on-device)
    async with websockets.serve(
        handler, "0.0.0.0", 8765,
        ssl=ssl_ctx,
        ping_interval=None, ping_timeout=None,
        close_timeout=10, max_size=2**20, compression=None,
    ), websockets.serve(
        handler, "0.0.0.0", 8766,
        ssl=None,
        ping_interval=None, ping_timeout=None,
        close_timeout=10, max_size=2**20, compression=None,
    ):
        print("Listening on wss://0.0.0.0:8765 (browsers/Telnyx) | ws://0.0.0.0:8766 (ESP32)")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
