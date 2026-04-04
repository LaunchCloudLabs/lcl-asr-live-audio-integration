import asyncio
import websockets
import os
import wave
import json
import sys
import numpy as np
from scipy.fftpack import dct
from faster_whisper import WhisperModel

# LCL Sovereign Hub V14 - "The Diarizer"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GAIN_BOOST       = 15.0
ACCUMULATION_SECS = 2.0
SAMPLE_RATE      = 16000
BYTES_PER_SAMPLE = 2
ACCUMULATION_BYTES = int(SAMPLE_RATE * BYTES_PER_SAMPLE * ACCUMULATION_SECS)
SPEAKER_THRESHOLD = 0.84   # cosine sim above this = same speaker
SPEAKER_COLORS    = [
    "#00ff41", "#ff6b35", "#a855f7", "#00d4ff",
    "#ff2d55", "#ffcc00", "#00ffcc", "#ff007f",
    "#7fff00", "#ff9500"
]  # 10 speaker slots

print("Spawning Whisper AI...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")

browsers   = set()
edge_nodes = set()
audio_accumulator = bytearray()
audio_lock = asyncio.Lock()

# ── Speaker tracker ────────────────────────────────────────────────────────────

def extract_mfcc_embedding(audio_np, sr=16000, n_mfcc=13, n_filt=26):
    """Return a normalised n_mfcc-dim vector representing the speaker's voice."""
    audio_f = audio_np.astype(np.float32) / 32768.0
    if np.max(np.abs(audio_f)) < 1e-6:
        return None                          # silence — skip

    # Pre-emphasis
    sig = np.append(audio_f[0], audio_f[1:] - 0.97 * audio_f[:-1])

    # Framing
    fl = int(0.025 * sr)                     # 25 ms frame
    fh = int(0.010 * sr)                     # 10 ms hop
    n_frames = 1 + (len(sig) - fl) // fh
    if n_frames < 4:
        return None
    idx = (np.arange(fl)[None, :] +
           np.arange(n_frames)[:, None] * fh)
    frames = sig[idx] * np.hamming(fl)

    # Power spectrum
    NFFT   = 512
    ps     = (1.0 / NFFT) * np.abs(np.fft.rfft(frames, NFFT)) ** 2

    # Mel filterbank
    low_m  = 0
    high_m = 2595 * np.log10(1 + sr / 2 / 700)
    mel_p  = np.linspace(low_m, high_m, n_filt + 2)
    hz_p   = (700 * (10 ** (mel_p / 2595) - 1))
    bins   = np.floor((NFFT + 1) * hz_p / sr).astype(int)

    fb = np.zeros((n_filt, NFFT // 2 + 1))
    for m in range(1, n_filt + 1):
        for k in range(bins[m - 1], bins[m]):
            fb[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1] + 1e-9)
        for k in range(bins[m], bins[m + 1]):
            fb[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m] + 1e-9)

    log_fb = 20 * np.log10(np.maximum(np.dot(ps, fb.T), 1e-9))
    mfcc   = dct(log_fb, type=2, axis=1, norm='ortho')[:, 1:n_mfcc + 1]
    emb    = np.mean(mfcc, axis=0)
    norm   = np.linalg.norm(emb)
    return emb / norm if norm > 1e-9 else None


class SpeakerTracker:
    def __init__(self):
        self.speakers = []   # list of [name, color, embedding]

    def identify(self, audio_np):
        emb = extract_mfcc_embedding(audio_np)
        if emb is None:
            if self.speakers:
                s = self.speakers[-1]
                return s[0], s[1]
            return "Speaker 1", SPEAKER_COLORS[0]

        if not self.speakers:
            self.speakers.append(["Speaker 1", SPEAKER_COLORS[0], emb])
            return "Speaker 1", SPEAKER_COLORS[0]

        sims    = [float(np.dot(emb, s[2])) for s in self.speakers]
        best_i  = int(np.argmax(sims))
        best_s  = sims[best_i]

        if best_s >= SPEAKER_THRESHOLD:
            # Update running average embedding for that speaker
            self.speakers[best_i][2] = (
                0.9 * self.speakers[best_i][2] + 0.1 * emb)
            norm = np.linalg.norm(self.speakers[best_i][2])
            if norm > 1e-9:
                self.speakers[best_i][2] /= norm
            return self.speakers[best_i][0], self.speakers[best_i][1]

        idx  = len(self.speakers)
        name = f"Speaker {idx + 1}"
        col  = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        self.speakers.append([name, col, emb])
        print(f"[DIA] New speaker detected: {name}")
        return name, col


def boost_bytes(raw: bytes, gain: float) -> bytes:
    """Amplify raw int16 PCM bytes by gain factor, clamped to int16 range."""
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return np.clip(arr * gain, -32768, 32767).astype(np.int16).tobytes()

tracker = SpeakerTracker()


# ── ASR + Diarization engine ───────────────────────────────────────────────────

def _transcribe_blocking(wav_path: str):
    """Run Whisper in a thread so it never blocks the asyncio event loop."""
    segs, _ = model.transcribe(wav_path, beam_size=1, vad_filter=True)
    return [s.text.strip() for s in segs]

async def asr_engine():
    global audio_accumulator
    loop = asyncio.get_event_loop()
    print(f"AI Hub Active. Gain={GAIN_BOOST}x, Window={ACCUMULATION_SECS}s, Diarization=ON")
    while True:
        try:
            await asyncio.sleep(ACCUMULATION_SECS)

            # Grab accumulated audio under the lock
            async with audio_lock:
                if len(audio_accumulator) < ACCUMULATION_BYTES:
                    continue
                raw = bytes(audio_accumulator)
                audio_accumulator = bytearray()

            # Ensure even byte count (int16 = 2 bytes/sample)
            if len(raw) % 2 != 0:
                raw = raw[:-1]

            audio_np = np.frombuffer(raw, dtype=np.int16)
            boosted  = np.clip(audio_np.astype(np.float32) * GAIN_BOOST,
                               -32768, 32767).astype(np.int16)

            with wave.open("hub_stream.wav", "wb") as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                wf.writeframes(boosted.tobytes())

            # *** Run transcription in thread pool — prevents SIGABRT from ctranslate2 ***
            texts = await loop.run_in_executor(
                None, _transcribe_blocking, "hub_stream.wav")

            full_text = " ".join(
                t for t in texts
                if t and t.lower() not in ["you", "you.", "thank you.", ""]
            )

            if full_text:
                speaker, color = tracker.identify(audio_np)
                print(f"[{speaker}] {full_text}")
                msg = json.dumps({
                    "type":    "transcript",
                    "text":    full_text,
                    "speaker": speaker,
                    "color":   color
                })
                if browsers:
                    websockets.broadcast(browsers, msg)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[ASR ERROR] {e}", file=sys.stderr)
            await asyncio.sleep(1)  # brief backoff before retrying

# ── WebSocket handler ──────────────────────────────────────────────────────────

async def handler(websocket):
    global audio_accumulator
    addr = websocket.remote_address
    pkt_count = 0
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
            async with audio_lock: audio_accumulator.extend(first)

        async for msg in websocket:
            if isinstance(msg, bytes):
                pkt_count += 1
                boosted = boost_bytes(msg, GAIN_BOOST)
                if browsers:
                    websockets.broadcast(browsers, boosted)
                async with audio_lock:
                    audio_accumulator.extend(msg)
                    if len(audio_accumulator) > ACCUMULATION_BYTES * 4:
                        audio_accumulator = audio_accumulator[-ACCUMULATION_BYTES:]
                if pkt_count % 100 == 0:
                    print(f"[AUDIO] {addr} pkts={pkt_count} browsers={len(browsers)} accum={len(audio_accumulator)}B")
            else:
                try:
                    d = json.loads(msg)
                    if d.get("type") == "esp32_hello":
                        edge_nodes.add(websocket)
                    elif d.get("type") == "command" and edge_nodes:
                        websockets.broadcast(edge_nodes, msg)
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

async def asr_watchdog():
    """Restart asr_engine automatically if it ever dies."""
    while True:
        task = asyncio.create_task(asr_engine())
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[WATCHDOG] ASR engine died: {e} — restarting in 2s", file=sys.stderr)
            await asyncio.sleep(2)

async def main():
    print("--- LCL SOVEREIGN HUB V14 ONLINE ---")
    asyncio.create_task(asr_watchdog())
    async with websockets.serve(
        handler, "0.0.0.0", 8765,
        ping_interval=None, ping_timeout=None,
        close_timeout=10, max_size=2**20, compression=None,
    ):
        print("Listening on 0.0.0.0:8765 (diarization=ON)")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
