"""
cdsr_core.py — Modul inti Indonesian Connected Digit Speech Recognition (CDSR).

Satu sumber kode untuk NOTEBOOK (pelatihan & evaluasi) dan DASHBOARD (inferensi),
sehingga cara audio diproses saat latih dan saat dipakai dijamin sama persis.

Isi modul:
  1. Kosakata & parsing transkrip dari nama file   -> parse_filename()
  2. Pemuatan audio                                 -> load_audio()
  3. Fitur untuk model CTC (log-mel per frame)      -> ctc_features(), robust_features()
  3b. Penjernihan suara (noise reduction + VAD)     -> estimate_snr(), enhance_audio()
  4. Segmentasi kata + fitur per segmen             -> segment_utterance(), segment_features()
  5. Model deep learning CTC (PyTorch) + decoding   -> CTCNet, RobustCTCNet, ctc_greedy_decode()
  6. Metrik urutan (WER/SER, alignment)             -> align(), error_counts()
  7. Recognizer: memuat model & mengenali audio     -> Recognizer
"""
from __future__ import annotations

import io
import json
import os
import re
import time

import joblib
import librosa
import numpy as np
from scipy import ndimage, signal

# =============================================================================
# 1. KOSAKATA & PARSING TRANSKRIP
# =============================================================================
SR = 16000                     # sample rate dataset (16 kHz)
# 11 token: '0' diucapkan "nol", 'K' diucapkan "kosong" (keduanya bernilai 0)
TOKENS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'K']
TOKEN_TO_ID = {t: i for i, t in enumerate(TOKENS)}
TOKEN_WORDS = {'0': 'nol', '1': 'satu', '2': 'dua', '3': 'tiga', '4': 'empat',
               '5': 'lima', '6': 'enam', '7': 'tujuh', '8': 'delapan',
               '9': 'sembilan', 'K': 'kosong'}

# Nama file: <PENUTUR>_<URUTAN DIGIT><PENANDA REKAMAN>, mis. MWS_985K3A
_NAME_RE = re.compile(r'^([A-Za-z]{3})_([0-9K]+)([A-Z]?)$')


def parse_filename(file_name: str):
    """Mengurai nama file menjadi (penutur, gender, daftar token).

    Contoh: 'MWS_985K3A.wav' -> ('MWS', 'M', ['9','8','5','K','3']).
    Mengembalikan None bila nama tidak sesuai pola.
    """
    stem = os.path.splitext(os.path.basename(file_name))[0]
    m = _NAME_RE.match(stem)
    if not m:
        return None
    speaker = m.group(1).upper()
    return speaker, speaker[0], list(m.group(2))


def to_value(tokens):
    """Nilai angka yang dipakai layanan: 'K' (kosong) dan '0' (nol) sama-sama 0."""
    return ['0' if t == 'K' else t for t in tokens]


def to_words(tokens):
    """Kalimat konfirmasi, mis. ['9','1','K'] -> 'sembilan satu kosong'."""
    return ' '.join(TOKEN_WORDS[t] for t in tokens)


# =============================================================================
# 2. PEMUATAN AUDIO
# =============================================================================
def load_audio(source) -> np.ndarray:
    """Memuat audio (path, bytes, atau file-like) -> mono 16 kHz, puncak dinormalkan ke 1."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, _ = librosa.load(source, sr=SR, mono=True)
    peak = np.max(np.abs(y)) if y.size else 0.0
    return (y / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


# =============================================================================
# 3. FITUR UNTUK MODEL CTC
# =============================================================================
N_MELS, N_FFT, HOP = 40, 400, 160          # jendela 25 ms, geser 10 ms
N_MELS_ROBUST = 64                          # model robust memakai resolusi frekuensi lebih tinggi


def ctc_features(y: np.ndarray) -> np.ndarray:
    """Log-mel 40 dimensi per frame + normalisasi rata-rata/varians per ucapan.

    Hasil berbentuk (jumlah_frame, 40): urutan waktu DIPERTAHANKAN, inilah yang
    memungkinkan model mengenali beberapa digit dalam satu file.
    """
    y, _ = librosa.effects.trim(y, top_db=35)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS)
    feat = np.log(mel + 1e-6).T
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-5)
    return feat.astype(np.float32)


def robust_features(y: np.ndarray) -> np.ndarray:
    """Log-mel 64 dimensi (60–7600 Hz) + CMVN per ucapan, untuk CRNN-CTC Robust.

    Tidak memotong hening di sini: pemotongan dilakukan oleh enhance_audio() yang
    memakai ambang relatif terhadap derau (lebih andal untuk rekaman mikrofon).
    """
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                         n_mels=N_MELS_ROBUST, fmin=60, fmax=7600)
    feat = np.log(mel + 1e-6).T
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-5)
    return feat.astype(np.float32)


# =============================================================================
# 3b. PENJERNIHAN SUARA (untuk rekaman mikrofon yang tidak sebersih data studio)
# =============================================================================
# Data latih direkam di studio: hening di antara kata benar-benar senyap. Rekaman
# mikrofon laptop/HP punya derau latar (kipas, AC, jalanan, orang bicara), dengung
# listrik, klik saat tombol rekam ditekan, dan hening panjang di awal/akhir.
# Langkah penjernihan:
#   1. condition_audio : buang DC + high-pass 70 Hz (dengung, getaran meja)
#   2. spectral_denoise: spectral subtraction — profil derau diambil dari frame
#                        paling senyap, lalu dikurangi dari tiap frame (dengan
#                        penghalusan agar tidak muncul "musical noise")
#   3. trim_speech     : potong hening awal/akhir dengan ambang relatif terhadap
#                        derau (bukan terhadap puncak), klik pendek diabaikan
#   4. normalisasi puncak (robust) seperti data latih
ENH_NFFT, ENH_HOP = 512, 128
ENH_SNR_DB = 30.0                # di bawah ini rekaman dianggap "bising" (mode otomatis)
_HP_SOS = signal.butter(4, 70, btype='highpass', fs=SR, output='sos')


def _frame_db(y: np.ndarray) -> np.ndarray:
    """Energi per frame 25 ms (geser 10 ms) dalam dB."""
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    return 20 * np.log10(rms + 1e-8)


def estimate_snr(y: np.ndarray) -> float:
    """Perkiraan kasar SNR (dB): energi bagian ucapan vs bagian paling senyap.

    Data studio biasanya > 40 dB; rekaman ruangan tenang 25–40 dB; bising < 20 dB.
    """
    if len(y) < 2 * N_FFT:
        return 0.0
    db = _frame_db(y)
    return float(np.percentile(db, 95) - np.percentile(db, 10))


def condition_audio(y: np.ndarray) -> np.ndarray:
    """Buang DC offset + high-pass 70 Hz."""
    y = y.astype(np.float64) - float(np.mean(y)) if y.size else y.astype(np.float64)
    if len(y) > 64:
        y = signal.sosfiltfilt(_HP_SOS, y)
    return y.astype(np.float32)


def spectral_denoise(y: np.ndarray, strength: float = 1.0, floor_db: float = -20.0) -> np.ndarray:
    """Spectral subtraction dengan profil derau dari 15% frame paling senyap.

    strength : 1.0 = standar; >1 lebih agresif (lebih bersih, risiko suara 'cempreng').
    floor_db : redaman maksimum, agar derau tidak hilang total (mengurangi artefak).
    """
    if len(y) < 4 * ENH_NFFT:
        return y.astype(np.float32)
    S = librosa.stft(y, n_fft=ENH_NFFT, hop_length=ENH_HOP)
    power = np.abs(S) ** 2
    frame_e = power.sum(0)
    k = max(5, int(0.15 * len(frame_e)))
    noise_psd = power[:, np.argsort(frame_e)[:k]].mean(1, keepdims=True)
    gain = np.sqrt(np.maximum(1.0 - 1.5 * strength * noise_psd / (power + 1e-12), 0.0))
    gain = ndimage.uniform_filter(gain, size=(3, 5))            # halus di frekuensi & waktu
    gain = np.maximum(gain, 10 ** (floor_db / 20))
    out = librosa.istft(S * gain, hop_length=ENH_HOP, length=len(y))
    return out.astype(np.float32)


def trim_speech(y: np.ndarray, margin: float = 0.15):
    """Potong hening awal/akhir. Mengembalikan (audio_terpotong, indeks_awal)."""
    db = _frame_db(y)
    if db.size == 0:
        return y, 0
    noise, peak = np.percentile(db, 10), np.percentile(db, 99)
    active = db > noise + max(6.0, 0.35 * (peak - noise))
    active = ndimage.binary_opening(active, structure=np.ones(5))   # abaikan klik < 50 ms
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return y, 0
    pad = int(margin * SR)
    s = max(0, idx[0] * HOP - N_FFT // 2 - pad)
    e = min(len(y), idx[-1] * HOP + N_FFT // 2 + pad)
    return y[s:e], int(s)


def _peak_normalize(y: np.ndarray) -> np.ndarray:
    """Normalisasi puncak 'robust' (persentil 99.9) agar satu klik keras tidak membuat ucapan pelan."""
    if not y.size:
        return y.astype(np.float32)
    peak = np.percentile(np.abs(y), 99.9)
    if peak <= 0:
        return y.astype(np.float32)
    return np.clip(y / peak, -1.0, 1.0).astype(np.float32)


def enhance_audio(y: np.ndarray, denoise: bool = True, strength: float = 1.0):
    """Pipeline penjernihan lengkap. Mengembalikan (audio, info).

    info = {'offset': indeks awal (sampel) relatif audio asli, 'snr_before', 'snr_after', 'denoised'}
    """
    snr0 = estimate_snr(y)
    z = condition_audio(y)
    if denoise:
        z = spectral_denoise(z, strength)
    snr1 = estimate_snr(z)
    z, off = trim_speech(z)
    return _peak_normalize(z), {'offset': off, 'snr_before': snr0, 'snr_after': snr1,
                                'denoised': bool(denoise)}


# =============================================================================
# 4. SEGMENTASI KATA + FITUR PER SEGMEN (untuk SVM / RF / XGBoost / KNN)
# =============================================================================
TOP_DB_GRID = (20, 25, 30, 35, 40)          # kandidat ambang hening (dB di bawah puncak)
MIN_GAP, MIN_DUR, PAD = 0.08, 0.10, 0.03    # detik


def segment_utterance(y: np.ndarray, top_db: float):
    """Memotong ucapan menjadi segmen kata berdasarkan jeda hening.

    Jeda < MIN_GAP digabung (mis. penutupan konsonan 'p' pada 'empat'),
    segmen < MIN_DUR dibuang (bunyi klik/napas). Mengembalikan [(awal, akhir)] sampel.
    """
    raw = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=128)
    merged = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) < MIN_GAP * SR:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    pad = int(PAD * SR)
    return [(max(0, s - pad), min(len(y), e + pad))
            for s, e in merged if (e - s) >= MIN_DUR * SR]


SEG_FEAT_DIM = 183


def segment_features(seg: np.ndarray) -> np.ndarray:
    """Vektor 183 fitur untuk satu segmen kata.

    78 = rata-rata & simpangan baku dari MFCC, delta, delta-delta (13 x 3 x 2)
   104 = bentuk MFCC sepanjang waktu, diringkas ke 8 titik waktu (13 x 8)
     1 = durasi segmen (detik)
    """
    if len(seg) < N_FFT:
        seg = np.pad(seg, (0, N_FFT - len(seg)))
    mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13, n_fft=N_FFT, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc, mode='nearest')
    d2 = librosa.feature.delta(mfcc, order=2, mode='nearest')
    stack = np.vstack([mfcc, d1, d2])
    src = np.linspace(0, 1, mfcc.shape[1])
    shape = np.array([np.interp(np.linspace(0, 1, 8), src, row) for row in mfcc]).ravel()
    return np.concatenate([stack.mean(1), stack.std(1), shape,
                           [len(seg) / SR]]).astype(np.float32)


def extract_all(path: str) -> dict:
    """Satu kali baca audio -> semua fitur yang dibutuhkan metode lama (untuk cache)."""
    y = load_audio(path)
    segs = {}
    for db in TOP_DB_GRID:
        iv = segment_utterance(y, db)
        feats = (np.stack([segment_features(y[s:e]) for s, e in iv])
                 if iv else np.zeros((0, SEG_FEAT_DIM), np.float32))
        segs[db] = feats
    return {'ctc': ctc_features(y).astype(np.float16), 'segs': segs,
            'duration': len(y) / SR}


# =============================================================================
# 5. MODEL DEEP LEARNING CTC
# =============================================================================
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:          # dashboard tetap jalan tanpa PyTorch (metode CTC dinonaktifkan)
    TORCH_OK = False

BLANK = 0                    # indeks 0 = blank CTC; token ke-i -> indeks i+1
N_OUT = len(TOKENS) + 1

if TORCH_OK:
    class CTCNet(nn.Module):
        """2 lapis konvolusi 1D -> 2 lapis BiLSTM -> peluang tiap token per frame.
        Dilatih pada audio studio bersih (baseline). Pooling memperpendek urutan 2x."""

        def __init__(self, n_in=N_MELS, hidden=128, n_out=N_OUT):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_in, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
                nn.MaxPool1d(2))
            self.rnn = nn.LSTM(128, hidden, num_layers=2, batch_first=True,
                               bidirectional=True, dropout=0.3)
            self.out = nn.Linear(2 * hidden, n_out)

        def forward(self, x):                      # x: (batch, frame, 40)
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.rnn(h)
            return self.out(h).log_softmax(-1)     # (batch, frame/2, N_OUT)

    class RobustCTCNet(nn.Module):
        """CRNN-CTC tahan derau.

        4 lapis konvolusi 2D membaca spektrogram sebagai 'gambar' waktu x frekuensi,
        sehingga pola formant tetap dikenali walau sebagian pita frekuensi tertutup
        derau; 3 lapis BiGRU memberi konteks urutan. Dilatih dengan augmentasi derau
        (lihat notebook) dan memakai fitur dari enhance_audio() + robust_features().
        """

        def __init__(self, n_in=N_MELS_ROBUST, hidden=192, n_out=N_OUT, channels=64):
            super().__init__()
            c = channels

            def block(i, o):
                return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]

            self.conv = nn.Sequential(
                *block(1, c // 2), *block(c // 2, c // 2), nn.MaxPool2d((2, 2)),   # frek/2, waktu/2
                *block(c // 2, c), *block(c, c), nn.MaxPool2d((2, 1)),              # frek/4
                nn.Dropout(0.1))
            self.proj = nn.Sequential(nn.Linear(c * (n_in // 4), 256), nn.ReLU(), nn.Dropout(0.2))
            self.rnn = nn.GRU(256, hidden, num_layers=3, batch_first=True,
                              bidirectional=True, dropout=0.3)
            self.out = nn.Linear(2 * hidden, n_out)

        def forward(self, x):                      # x: (batch, frame, 64)
            h = self.conv(x.unsqueeze(1).transpose(2, 3))       # (B, C, F/4, T/2)
            b, c, f, t = h.shape
            h = self.proj(h.permute(0, 3, 1, 2).reshape(b, t, c * f))
            h, _ = self.rnn(h)
            return self.out(h).log_softmax(-1)     # (batch, frame/2, N_OUT)

    def out_lengths(lengths):
        """Panjang keluaran model setelah pooling waktu 2x (sama untuk kedua model)."""
        return lengths // 2


def ctc_greedy_decode(logp: np.ndarray):
    """Decoding greedy: ambil token terbaik per frame, gabungkan pengulangan, buang blank.

    Mengembalikan (token, keyakinan per token, frame tengah per token).
    """
    prob = np.exp(logp)
    best = prob.argmax(1)
    tokens, confs, frames = [], [], []
    prev, run = BLANK, []
    for t, k in enumerate(np.append(best, BLANK)):
        if k != prev and prev != BLANK and run:
            tokens.append(TOKENS[prev - 1])
            confs.append(float(prob[run, prev].max()))
            frames.append(int(np.mean(run)))
            run = []
        if k != BLANK and t < len(best):
            run = run + [t] if k == prev else [t]
        prev = k
    return tokens, confs, frames


# =============================================================================
# 6. METRIK URUTAN
# =============================================================================
def align(ref, hyp):
    """Alignment Levenshtein -> daftar pasangan (ref, hyp); None = sisipan/hapusan."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            pairs.append((ref[i - 1], None)); i -= 1
        else:
            pairs.append((None, hyp[j - 1])); j -= 1
    return pairs[::-1]


def error_counts(ref, hyp):
    """(substitusi, hapusan, sisipan) antara urutan referensi dan hipotesis."""
    S = D = I = 0
    for r, h in align(ref, hyp):
        if r is None:
            I += 1
        elif h is None:
            D += 1
        elif r != h:
            S += 1
    return S, D, I


# =============================================================================
# 7. RECOGNIZER — dipakai notebook (evaluasi derau, uji kecepatan) dan dashboard
# =============================================================================
ROBUST = 'CRNN-CTC Robust'
METHOD_INFO = {
    ROBUST: 'Deep learning tahan derau: CNN 2D + BiGRU + CTC, dilatih dengan ribuan ucapan '
            'yang dicampur derau nyata (ESC-50), suara orang lain, gema, dan simulasi mikrofon HP; '
            'input dijernihkan dulu (noise reduction).',
    'CNN-BiLSTM-CTC': 'Deep learning end-to-end (CNN 1D + BiLSTM + CTC) yang dilatih pada audio '
                      'studio bersih: membaca seluruh rekaman tanpa memotong audio.',
    'SVM': 'Audio dipotong per kata pada jeda hening; tiap potongan diklasifikasi '
           'Support Vector Machine (kernel RBF).',
    'Random Forest': 'Audio dipotong per kata; tiap potongan diklasifikasi oleh '
                     'ratusan pohon keputusan yang memberi suara mayoritas.',
    'XGBoost': 'Audio dipotong per kata; tiap potongan diklasifikasi gradient '
               'boosting (pohon dibangun bertahap memperbaiki kesalahan sebelumnya).',
    'KNN': 'Audio dipotong per kata; tiap potongan diberi label mayoritas dari '
           'k contoh latih yang paling mirip.',
}
SEGMENT_METHODS = ['SVM', 'Random Forest', 'XGBoost', 'KNN']
CTC_METHODS = ['CNN-BiLSTM-CTC', ROBUST]
MODEL_FILES = {'SVM': 'svm.joblib', 'Random Forest': 'random_forest.joblib',
               'XGBoost': 'xgboost.joblib', 'KNN': 'knn.joblib',
               'CNN-BiLSTM-CTC': 'ctc.pt', ROBUST: 'crnn_robust.pt'}
ENHANCE_MODES = {'auto': 'Otomatis', 'on': 'Selalu dijernihkan', 'off': 'Mati (audio asli)'}


class Recognizer:
    """Memuat model (dari folder artefak atau objek di memori) dan mengenali urutan digit.

    Mode penjernihan (`enhance`):
      'off'  : audio diproses persis seperti saat model dilatih (tanpa penjernihan)
      'on'   : audio selalu dijernihkan dulu (enhance_audio)
      'auto' : model CTC mencoba versi asli dan versi dijernihkan, lalu memilih hasil
               dengan keyakinan tertinggi; metode potongan memakai penjernihan bila
               SNR < ENH_SNR_DB.
    """

    def __init__(self, artifact_dir: str | None = None, models: dict | None = None,
                 top_db: float | None = None, device: str = 'cpu'):
        self.dir = artifact_dir
        self.config = {}
        if artifact_dir:
            with open(os.path.join(artifact_dir, 'config.json')) as f:
                self.config = json.load(f)
        self.top_db = top_db if top_db is not None else self.config.get('segment_top_db', 30)
        self.device = device
        self._models = dict(models or {})

    def available(self):
        """Metode yang modelnya tersedia (CTC hanya bila PyTorch terpasang)."""
        out = list(self._models)
        if self.dir:
            for name, fn in MODEL_FILES.items():
                if name in out or not os.path.exists(os.path.join(self.dir, 'models', fn)):
                    continue
                if name in CTC_METHODS and not TORCH_OK:
                    continue
                out.append(name)
        return out

    def _model(self, name):
        if name not in self._models:
            path = os.path.join(self.dir, 'models', MODEL_FILES[name])
            if name in CTC_METHODS:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                cls = RobustCTCNet if ckpt.get('kind') == 'crnn_robust' else CTCNet
                net = cls(**ckpt['arch'])
                net.load_state_dict(ckpt['state_dict'])
                self._models[name] = net.to(self.device).eval()
            else:
                self._models[name] = joblib.load(path)
        return self._models[name]

    # ---- varian audio yang dicoba -------------------------------------------------
    def _variants(self, y, method, enhance, snr):
        """Daftar (nama, audio, offset) yang akan dikenali."""
        if method == ROBUST:
            plain = enhance_audio(y, denoise=False)      # model robust selalu lewat VAD+normalisasi
            clean = lambda: enhance_audio(y, denoise=True)
        else:
            plain = (y, {'offset': 0})                   # persis seperti data latih
            clean = lambda: enhance_audio(y, denoise=True)
        if enhance == 'off':
            cand = [('asli', plain)]
        elif enhance == 'on':
            cand = [('dijernihkan', clean())]
        elif method in CTC_METHODS:
            cand = [('asli', plain), ('dijernihkan', clean())]
        else:
            cand = [('dijernihkan', clean())] if snr < ENH_SNR_DB else [('asli', plain)]
        return [(n, a, info['offset']) for n, (a, info) in cand]

    def _run_ctc(self, method, y):
        net = self._model(method)
        feat = robust_features(y) if method == ROBUST else ctc_features(y)
        with torch.no_grad():
            p = next(net.parameters())
            logp = net(torch.from_numpy(feat)[None].to(p.device))[0].cpu().numpy()
        tokens, confs, frames = ctc_greedy_decode(logp)
        start = 0 if method == ROBUST else librosa.effects.trim(y, top_db=35)[1][0]
        times = [(start + f * 2 * HOP) / SR for f in frames]
        return tokens, confs, times, None

    def _run_segments(self, method, y):
        iv = segment_utterance(y, self.top_db)
        tokens, confs, times = [], [], []
        if iv:
            X = np.stack([segment_features(y[s:e]) for s, e in iv])
            model = self._model(method)
            proba = model.predict_proba(X)
            idx = proba.argmax(1)
            tokens = [TOKENS[int(model.classes_[i])] for i in idx]
            confs = proba.max(1).astype(float).tolist()
            times = [(s + e) / 2 / SR for s, e in iv]
        return tokens, confs, times, [(s / SR, e / SR) for s, e in iv]

    def recognize(self, y: np.ndarray, method: str, enhance: str = 'auto') -> dict:
        """Mengenali satu ucapan. Mengembalikan token, angka, kalimat, keyakinan, waktu,
        varian audio yang dipakai, perkiraan SNR, dan audio hasil pemrosesan."""
        self._model(method)                     # muat model dulu agar waktu = waktu mengenali saja
        t0 = time.perf_counter()
        snr = estimate_snr(y)
        best = None
        for name, audio, off in self._variants(y, method, enhance, snr):
            run = self._run_ctc if method in CTC_METHODS else self._run_segments
            tokens, confs, times, segs = run(method, audio)
            score = float(np.mean(confs)) if confs else 0.0
            if best is None or score > best[0]:
                o = off / SR
                best = (score, name, audio, tokens, confs, [t + o for t in times],
                        None if segs is None else [(s + o, e + o) for s, e in segs])
        _, name, audio, tokens, confs, times, segs = best
        return {'method': method, 'tokens': tokens, 'digits': ''.join(to_value(tokens)),
                'words': to_words(tokens), 'confidence': confs, 'times': times,
                'segments': segs, 'variant': name, 'snr_db': snr, 'audio': audio,
                'latency_ms': (time.perf_counter() - t0) * 1000}
