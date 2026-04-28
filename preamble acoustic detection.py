#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, math, time, queue, struct
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, correlate
from scipy.io import wavfile

# если нужен RS-код: pip install reedsolo
try:
    import reedsolo
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False

# если нужен микрофон: pip install sounddevice
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

# =========================
# ПАРАМЕТРЫ
# =========================
FS            = 48000       # Гц
ORDER_GOLAY   = 10          # генерируем A,B длины 2^order
F_LOW, F_HIGH = 300, 4300   # полоса передачи
THRESH_GOLAY  = 2000        # порог метрики (A^2+B^2)
AGC_TARGET    = 0.9
BLOCK_DUR     = 0.05        # 50 ms
TIMEOUT_SEC   = 10.0

# OFDM-параметры
NFFT = 1024
CP   = 128
SUBC_SPACING = FS/NFFT
K_LOW  = math.ceil(F_LOW/SUBC_SPACING)
K_HIGH = math.floor(F_HIGH/SUBC_SPACING)
CARRIERS = np.arange(K_LOW, K_HIGH+1)
M_ACTIVE = len(CARRIERS)

# RS-параметры (50%)
RS_K    = 170
RS_N    = 255
RS_SYM  = RS_N-RS_K

# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def generate_golay(order:int) -> Tuple[np.ndarray,np.ndarray]:
    A = np.array([1],float)
    B = np.array([1],float)
    for _ in range(order):
        A, B = np.concatenate([A,B]), np.concatenate([A,-B])
    return A, B

def design_bpf(fs,f1,f2,order=6):
    nyq = fs/2
    return butter(order,[f1/nyq,f2/nyq],btype='band')

b_bpf,a_bpf = design_bpf(FS,F_LOW,F_HIGH,6)

def apply_agc(x:np.ndarray,target=AGC_TARGET,eps=1e-12)->np.ndarray:
    pk = np.max(np.abs(x))
    return x if pk<eps else x*(target/pk)

def detect_golay(win:A,B):
    cA = correlate(win,A,mode='valid')
    cB = correlate(win,B,mode='valid')
    M  = cA**2 + cB**2
    idx= np.argmax(M)
    return idx, M[idx]

def bytes_to_bits(b:bytes)->np.ndarray:
    return np.unpackbits(np.frombuffer(b,dtype=np.uint8))

def bits_to_bytes(bv:np.ndarray)->bytes:
    pad = (-len(bv))%8
    if pad: bv = np.concatenate([bv, np.zeros(pad,dtype=np.uint8)])
    return np.packbits(bv).tobytes()

# DQPSK
def dqpsk_mod(bits:str)->np.ndarray:
    map_d = {"00":0,"01":math.pi/2,"11":math.pi,"10":-math.pi/2}
    phase=0; syms=[]
    for i in range(0,len(bits),2):
        d = map_d[bits[i:i+2]]
        phase = (phase+d+math.pi)%(2*math.pi)-math.pi
        syms.append(np.exp(1j*phase))
    return np.array(syms)

def dqpsk_demod(symbols:np.ndarray)->str:
    bits=[]; prev=1+0j
    for s in symbols:
        d = np.angle(s*np.conj(prev))
        prev = s
        if   d< -3*math.pi/4: bits+=['1','1']
        elif d< -math.pi/4:   bits+=['1','0']
        elif d<  math.pi/4:   bits+=['0','0']
        elif d< 3*math.pi/4:  bits+=['0','1']
        else:                  bits+=['1','1']
    return ''.join(bits)

# OFDM-упаковка
def build_herm(Xpos:np.ndarray)->np.ndarray:
    X = np.zeros(NFFT,dtype=complex)
    X[CARRIERS] = Xpos
    X[-CARRIERS] = np.conj(Xpos)
    return X

def ofdm_ifft(data_syms:np.ndarray)->np.ndarray:
    frames=[];ptr=0
    Ns = len(data_syms)//M_ACTIVE
    for i in range(Ns):
        chunk = data_syms[i*M_ACTIVE:(i+1)*M_ACTIVE]
        X = build_herm(chunk)
        t = np.fft.ifft(X)
        frames.append(np.concatenate([t[-CP:],t]))
    return np.concatenate(frames)

def ofdm_fft(rx:np.ndarray)->np.ndarray:
    blk = CP+NFFT; ptr=0; syms=[]
    while ptr+blk<=len(rx):
        seg=rx[ptr+CP:ptr+CP+NFFT]
        S = np.fft.fft(seg)
        syms.append(S[CARRIERS])
        ptr+=blk
    return np.vstack(syms) if syms else np.empty((0,M_ACTIVE),complex)

# RS-код
def rs_encode(data:bytes)->bytes:
    if not RS_AVAILABLE: raise RuntimeError("RS not installed")
    rs = reedsolo.RSCodec(RS_SYM)
    out=bytearray()
    for i in range(0,len(data),RS_K):
        blk=data[i:i+RS_K];
        if len(blk)<RS_K: blk+=b'\0'*(RS_K-len(blk))
        out+=rs.encode(blk)
    return bytes(out)

def rs_decode(data:bytes,orig_len:int)->bytes:
    if not RS_AVAILABLE: raise RuntimeError("RS not installed")
    rs=reedsolo.RSCodec(RS_SYM); out=bytearray()
    for i in range(0,len(data),RS_N):
        blk=data[i:i+RS_N]
        if len(blk)<RS_N: blk+=b'\0'*(RS_N-len(blk))
        dec=rs.decode(blk)[0]
        out+=dec
    return bytes(out[:orig_len])

# =========================
#  GОЛЕЙ-ПРЕАМБУЛА
# =========================
A,B = generate_golay(ORDER_GOLAY)
A_f = lfilter(b_bpf,a_bpf,A)
B_f = lfilter(b_bpf,a_bpf,B)
A_f/=np.max(np.abs(A_f))
B_f/=np.max(np.abs(B_f))
PREAMBLE = np.concatenate([A_f,B_f])

# =========================
#  СБОРКА TX
# =========================
def build_packet_wav(text:str, use_rs:bool)->Tuple[str,dict]:
    data=text.encode('utf-8')
    hdr = struct.pack('>I?',len(data), use_rs)
    payload = data
    if use_rs:
        payload = rs_encode(payload)
    bits = ''.join(f"{b:08b}" for b in hdr+payload)
    syms = dqpsk_mod(bits)
    # OFDM
    ofdm = ofdm_ifft(syms)
    tx = np.concatenate([PREAMBLE, ofdm]).astype(float)
    tx = apply_agc(tx,AGC_TARGET)
    wavfile.write('tx.wav',FS,(tx*32767).astype(np.int16))
    return 'tx.wav',{'len_hdr':len(hdr),'use_rs':use_rs,'symbols':len(syms)}

# =========================
#  RX listener & decoder
# =========================
class Listener:
    def __init__(self,timeout=TIMEOUT_SEC):
        self.timeout=timeout
        self.buf=np.zeros(0,float)
        self.q=queue.Queue()
    def _cb(self,indata,frames,ti,st):
        self.q.put(indata[:,0].copy())
    def start(self):
        if not SD_AVAILABLE: raise RuntimeError("sd not installed")
        self.stream=sd.InputStream(samplerate=FS,channels=1,callback=self._cb)
        self.stream.start()
    def stop(self):
        self.stream.stop(); self.stream.close()
    def listen(self)->Tuple[int,np.ndarray]:
        print("Listening for Golay…")
        t0=time.time()
        Nw=len(PREAMBLE)
        while time.time()-t0<self.timeout:
            try: blk=self.q.get(timeout=0.1)
            except queue.Empty: continue
            self.buf=np.concatenate([self.buf,blk])
            if len(self.buf)>Nw: self.buf=self.buf[-Nw:]
            # фильтрация + AGC
            bf=lfilter(b_bpf,a_bpf,self.buf)
            bf=apply_agc(bf,AGC_TARGET)
            idx,m=detect_golay(bf,A_f,B_f)
            print(f"\rMetric={m:.0f}",end='')
            if m>THRESH_GOLAY:
                samp=len(self.buf)-Nw+idx
                print(f"\nDetected at sample offset={idx}, global sample TBD")
                return idx,self.buf.copy()
        print("\nTimeout")
        return None,None

# =========================
#  Main
# =========================
def main():
    use_rs = (input("Use RS? [y/N]: ").lower()=='y')
    path,meta=build_packet_wav(input("Enter text: "),use_rs)
    print("TX saved to",path, meta)
    if not SD_AVAILABLE:
        print("Install sounddevice; play tx.wav and rerun to decode.")
        return
    L=Listener()
    L.start()
    idx,buf=L.listen()
    L.stop()
    if idx is None: return
    # избавляемся от преамбулы
    data_wave=buf[idx+len(PREAMBLE):]
    # демодуляция OFDM
    S = ofdm_unpack_frames(data_wave)
    if S.size==0:
        print("No OFDM symbols")
        return
    # DQPSK демод
    # берём первый поднесущие линии по time
    # конкатенируем все поднесущие подряд
    flat = S.flatten()
    bits = dqpsk_demod(flat)
    # вырезаем заголовок
    hdr_bits = bits[:32]  # 4 байта *8
    hdr = bits_to_bytes(np.array(list(map(int,hdr_bits))))
    length,rsf = struct.unpack('>I?',hdr)
    pl_bits = bits[32:32+length*8]
    data = bits_to_bytes(np.array(list(map(int,pl_bits))))
    if rsf:
        data=rs_decode(data,length)
    print("Decoded:",data.decode('utf-8',errors='replace'))

if __name__=='__main__':
    main()

