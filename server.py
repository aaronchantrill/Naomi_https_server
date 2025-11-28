#!/usr/bin/env python3
# server.py
import asyncio
import json
import numpy as np
import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import uvicorn
import socket
import torch
import torchaudio

CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

def generate_self_signed_cert(cert_file, key_file):
    # Generate RSA key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    
    # Write the key to file
    with open(key_file, 'wb') as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    # Create cert
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, u"US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"TestState"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u"TestCity"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Naomi"),
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ]
    )
    
    cert = (
        x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(socket.getfqdn())]), critical=False)
            .sign(key, hashes.SHA256())
    )
    
    # Write the cert to file
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

# Ensure cert and key exist
if not (os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)):
    print("Generating certificate....")
    generate_self_signed_cert(CERT_FILE, KEY_FILE)

app = FastAPI()

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Mic -> WebSocket (volume + VAD)</title>
  <style>
    body { font-family: system-ui, -apple-system, Roboto, "Segoe UI", Arial; padding: 24px; }
    #meter { width: 100%; height: 24px; background: #eee; border-radius: 6px; overflow: hidden; margin-top:8px;}
    #meter .level { height:100%; width:0%; background: linear-gradient(90deg,#4caf50,#ff9800,#f44336); transition: width 0.05s; }
    #status { margin-top:12px; }
    #log { margin-top:12px; max-height: 200px; overflow:auto; background:#f7f7f7; padding:8px; border-radius:6px; font-family: monospace; font-size:13px; }
    button { margin-top:12px; }
  </style>
</head>
<body>
  <h1>Mic → WebSocket (volume indicator)</h1>
  <div>
    <button id="startBtn">Start</button>
    <button id="stopBtn" disabled>Stop</button>
  </div>
  <div id="meter"><div class="level" id="level"></div></div>
  <div id="status">Status: <span id="s">idle</span></div>
  <div id="log"></div>

<script>
let ws = null;
let audioCtx = null;
let processor = null;
let source = null;
let stream = null;
let sampleRate = 48000; // will be overwritten by actual AudioContext.sampleRate
const logDiv = document.getElementById('log');

function log(...args){
  const t = new Date().toLocaleTimeString();
  logDiv.innerText = t + "  " + args.join(" ") + "\\n" + logDiv.innerText;
}

function floatTo16BitPCM(float32Array) {
  const l = float32Array.length;
  const out = new DataView(new ArrayBuffer(l * 2));
  let offset = 0;
  for (let i = 0; i < l; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    out.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out.buffer;
}

function updateLevel(rms){
  const pct = Math.min(1, rms * 3); // scale for display
  document.getElementById('level').style.width = (pct*100) + "%";
}

async function start(){
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    document.getElementById('s').innerText = "connected";
    log("WS open");
  };
  ws.onmessage = (evt) => {
    try{
      const data = JSON.parse(evt.data);
      if(data.type === "speech_start"){ log("Speech START detected by server"); }
      if(data.type === "speech_end"){ log("Speech END detected by server"); }
      if(data.type === "vad_prob"){ log("Server VAD prob:", data.prob.toFixed(3)); }
    }catch(e){ logging = evt.data; log("recv text:", evt.data); }
  };
  ws.onclose = () => { document.getElementById('s').innerText = "disconnected"; log("WS closed"); };

  stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  sampleRate = audioCtx.sampleRate;
  source = audioCtx.createMediaStreamSource(stream);

  // ScriptProcessorNode is deprecated but widely supported; for production use AudioWorklet.
  const bufferSize = 4096;
  processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);

  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    // compute RMS for volume meter
    let sum = 0;
    for (let i = 0; i < input.length; i++) { sum += input[i] * input[i]; }
    const rms = Math.sqrt(sum / input.length);
    updateLevel(rms);

    if (ws && ws.readyState === WebSocket.OPEN) {
      // send initial sample rate once
      if(!ws._sent_init){
        ws.send(JSON.stringify({type:"init", sampleRate: sampleRate, channels:1}));
        ws._sent_init = true;
      }
      // convert to 16-bit PCM and send as binary
      const ab = floatTo16BitPCM(input);
      ws.send(ab);
    }
  };

  source.connect(processor);
  processor.connect(audioCtx.destination); // to keep the processor running

  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;
  document.getElementById('s').innerText = "capturing";
  log("Started capturing at sampleRate=" + sampleRate);
}

function stop(){
  if(processor){ processor.disconnect(); processor = null; }
  if(source){ source.disconnect(); source = null; }
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
  if(audioCtx){ audioCtx.close(); audioCtx = null; }
  if(ws){ ws.close(); ws = null; }
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  document.getElementById('s').innerText = "stopped";
  log("Stopped");
}

document.getElementById('startBtn').addEventListener('click', start);
document.getElementById('stopBtn').addEventListener('click', stop);
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)

# Helper: accumulate a short buffer (in samples) and run a simple energy-VAD on it.
def float_from_int16_bytes(b):
    # b is bytes of little-endian int16 PCM
    arr = np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
    return arr

class ConnectionState:
    def __init__(self):
        self.sample_rate = 48000
        self.buffers = []
        self.in_speech = False
        self.last_speech_time = None
        self.speech_start_sent = False

async def handle_audio_and_vad(ws, conn: ConnectionState):
    """
    Continuously process binary messages already collected in conn.buffers.
    We'll run a sliding energy based VAD which is simple but effective.
    """
    # We'll process in short frames of ~0.3s
    frame_seconds = 0.3
    frame_samples = int(conn.sample_rate * frame_seconds)

    while True:
        # wait a bit, process whatever we have
        try:
            await asyncio.sleep(0.05)
        except asyncio.exceptions.CancelledError:
            print("Stream stopped")
        if not conn.buffers:
            continue
        # combine buffers into one numpy array of float32
        data = b"".join(conn.buffers)
        conn.buffers = []
        samples = float_from_int16_bytes(data)
        if samples.size == 0:
            continue

        # process in frames
        i = 0
        while i < samples.size:
            frame = samples[i:i+frame_samples]
            i += frame_samples
            if frame.size == 0:
                break
            # compute RMS
            rms = np.sqrt(np.mean(frame * frame) + 1e-12)
            # simple prob-like metric
            prob = float(min(1.0, (rms / 0.02)))  # tuned scale: adjust threshold if necessary
            # send VAD probability update
            try:
                await ws.send_text(json.dumps({"type":"vad_prob", "prob": prob}))
            except Exception:
                return

            threshold = 0.35  # tune this threshold for speech detection (higher -> fewer false positives)
            now = asyncio.get_event_loop().time()
            if prob >= threshold:
                # consider speech
                if not conn.in_speech:
                    conn.in_speech = True
                    conn.last_speech_time = now
                    try:
                        await ws.send_text(json.dumps({"type":"speech_start"}))
                    except Exception:
                        return
                else:
                    conn.last_speech_time = now
            else:
                # if previously in speech, consider speech ended after short silence
                if conn.in_speech and (now - (conn.last_speech_time or now)) > 0.5:
                    conn.in_speech = False
                    try:
                        await ws.send_text(json.dumps({"type":"speech_end"}))
                    except Exception:
                        return

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conn = ConnectionState()
    # we'll create a background task to process buffers and run VAD
    vad_task = asyncio.create_task(handle_audio_and_vad(websocket, conn))

    try:
        while True:
            msg = await websocket.receive()
            # msg could be text or bytes
            if "text" in msg:
                text = msg["text"]
                try:
                    data = json.loads(text)
                    if data.get("type") == "init":
                        conn.sample_rate = int(data.get("sampleRate", conn.sample_rate))
                        # adjust frame sizes if needed in the VAD loop (it's reading conn.sample_rate directly)
                except Exception:
                    # ignore plain text
                    pass
            elif "bytes" in msg:
                b = msg["bytes"]
                # append raw int16 pcm bytes to buffer - handle as little endian int16
                conn.buffers.append(b)
            else:
                # closed or other
                break
    except Exception as e:
        print("websocket connection closed or error:", e)
    finally:
        vad_task.cancel()
        try:
            await vad_task
        except Exception:
            pass
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8443,
        ssl_certfile=CERT_FILE,
        ssl_keyfile=KEY_FILE,
        log_level="info"
    )
