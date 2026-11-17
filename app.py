"""
Dashboard Pengenal Angka Suara Bahasa Indonesia (Connected Digit Speech Recognition).

Jalankan:  streamlit run app.py
Model & hasil evaluasi dibaca dari folder ./artifacts (hasil ekspor notebook).

Struktur:
  - Pemuatan artefak & helper tampilan
  - Sidebar  : pilih metode (dengan rekomendasi), skenario layanan, ambang keyakinan
  - Tab 1    : rekam / unggah / contoh -> deteksi angka, konfirmasi, mode bertahap, riwayat
  - Tab 2    : perbandingan metode (akurasi + selang kepercayaan, uji signifikansi, dst.)
  - Tab 3    : panduan pemakaian untuk layanan nasabah
  - Tab 4    : tentang data, metode, dan keterbatasan
"""
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import cdsr_core as core

ART_DIR = Path(__file__).parent / 'artifacts'
PRIMARY, OK, WARN, BAD = '#0B5394', '#1E8E3E', '#E37400', '#C5221F'

st.set_page_config(page_title='Pengenal Angka Suara', page_icon='🎙️', layout='wide')
st.markdown(f"""
<style>
.hero {{background: linear-gradient(120deg, {PRIMARY} 0%, #1C7ED6 100%); color: white;
        padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}}
.hero h1 {{color: white; margin: 0; font-size: 1.9rem;}}
.hero p {{margin: .3rem 0 0 0; opacity: .92;}}
.digits {{display: flex; flex-wrap: wrap; gap: .45rem; margin: .4rem 0 .2rem 0;}}
.chip {{min-width: 3.1rem; text-align: center; padding: .45rem .2rem .25rem .2rem; border-radius: 10px;
        font-size: 2.1rem; font-weight: 700; line-height: 1; border: 2px solid;}}
.chip small {{display: block; font-size: .62rem; font-weight: 500; margin-top: .25rem;}}
.gap {{width: .7rem;}}
.card {{border: 1px solid #DDE3EA; border-radius: 12px; padding: .9rem 1.1rem; background: #FAFBFD;}}
</style>""", unsafe_allow_html=True)


# =============================================================================
# PEMUATAN ARTEFAK
# =============================================================================
@st.cache_resource(show_spinner='Memuat model...')
def get_recognizer():
    return core.Recognizer(str(ART_DIR))


@st.cache_data
def get_metrics():
    with open(ART_DIR / 'metrics.json', encoding='utf-8') as f:
        return json.load(f)


def artifacts_ready():
    return all((ART_DIR / f).exists() for f in ('config.json', 'metrics.json', 'models'))


if not artifacts_ready():
    st.markdown('<div class="hero"><h1>🎙️ Pengenal Angka Suara</h1>'
                '<p>Model belum terpasang.</p></div>', unsafe_allow_html=True)
    st.error('Folder `artifacts/` belum berisi model.')
    st.markdown("""
1. Jalankan notebook `ADL_DASHBOARD_3_SPEECH_RECOGNITION_FIXED.ipynb` di Colab sampai selesai.
2. Unduh `dashboard_artifacts.zip` dari folder Drive `Tugas kuliah/ADL/`.
3. Ekstrak isinya ke folder `artifacts/` di sebelah `app.py` (harus ada `config.json`, `metrics.json`, `models/`).
4. Muat ulang halaman ini.
""")
    st.stop()

REC = get_recognizer()
M = get_metrics()
AVAILABLE = [m for m in M['ranking'] if m in REC.available()]
BEST = M['recommended']['method']


def version_warnings():
    """Model scikit-learn/xgboost hanya aman dibuka dengan versi yang sama seperti saat dilatih."""
    import importlib
    out = []
    for pkg, trained in REC.config.get('versions', {}).items():
        mod = {'scikit-learn': 'sklearn'}.get(pkg, pkg)
        try:
            now = importlib.import_module(mod).__version__
        except ImportError:
            continue
        if now.split('+')[0] != trained.split('+')[0]:
            out.append(f'{pkg}: dilatih {trained}, terpasang {now}')
    return out


# =============================================================================
# HELPER TAMPILAN
# =============================================================================
def pct(x, d=1):
    return f'{x * 100:.{d}f}%'


def method_label(m):
    star = '⭐ ' if m == BEST else ''
    return f"{star}{m} — akurasi digit {pct(M['methods'][m]['digit_acc'])}"


def digit_chips(tokens, confs, thr):
    """Digit sebagai 'chip' berwarna: hijau = yakin, oranye = perlu dicek. Dikelompokkan per 4."""
    html = ['<div class="digits">']
    for k, (t, c) in enumerate(zip(tokens, confs)):
        if k and k % 4 == 0:
            html.append('<div class="gap"></div>')
        color = OK if c >= thr else WARN
        html.append(f'<div class="chip" style="border-color:{color};color:{color}">'
                    f'{core.to_value([t])[0]}<small>{core.TOKEN_WORDS[t]} · {c:.0%}</small></div>')
    html.append('</div>')
    return ''.join(html)


def waveform_figure(y, result):
    """Gelombang suara + posisi tiap digit yang terdeteksi."""
    step = max(1, len(y) // 4000)
    t = np.arange(0, len(y), step) / core.SR
    fig = go.Figure(go.Scatter(x=t, y=y[::step], mode='lines', line=dict(color='#7A8CA5', width=1),
                               hoverinfo='skip', name='gelombang'))
    labels = core.to_value(result['tokens'])
    if result['segments']:
        for (s, e), lab in zip(result['segments'], labels + [''] * len(result['segments'])):
            fig.add_vrect(x0=s, x1=e, fillcolor=PRIMARY, opacity=.13, line_width=0,
                          annotation_text=lab, annotation_position='top',
                          annotation_font=dict(size=16, color=PRIMARY))
    else:
        for tm, lab in zip(result['times'], labels):
            fig.add_vline(x=tm, line=dict(color=PRIMARY, dash='dot'), annotation_text=lab,
                          annotation_font=dict(size=16, color=PRIMARY))
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis_title='detik', yaxis_visible=False)
    return fig


def sample_tokens(path):
    """Transkrip contoh dari nama file; akhiran '_bising' (contoh berderau) diabaikan."""
    parsed = core.parse_filename(path.stem.removesuffix('_bising'))
    return parsed[2] if parsed else None


def quality_label(snr):
    """Label kualitas rekaman dari perkiraan SNR."""
    if snr >= core.ENH_SNR_DB:
        return '🟢 jernih'
    if snr >= 18:
        return '🟡 agak bising'
    return '🔴 bising'


def length_rule(scenario):
    return SCENARIOS[scenario][:2]


# =============================================================================
# SIDEBAR
# =============================================================================
SCENARIOS = {   # nama: (panjang minimum, panjang maksimum, keterangan)
    'Umum (panjang bebas)': (1, 99, 'Tanpa aturan panjang.'),
    'Nomor antrean (3 digit)': (3, 3, 'Contoh: 123'),
    'Nomor rekening (10 digit)': (10, 10, 'Ucapkan bertahap, mis. 3-3-4 digit.'),
    'Nomor HP (10–13 digit)': (10, 13, 'Ucapkan bertahap, mis. 4-4-4 digit.'),
    'Nomor tiket pengaduan (6 digit)': (6, 6, 'Contoh: 240117'),
}

with st.sidebar:
    st.header('⚙️ Pengaturan')
    method = st.selectbox('Metode pengenalan', AVAILABLE, index=AVAILABLE.index(BEST) if BEST in AVAILABLE else 0,
                          format_func=method_label,
                          help='⭐ = direkomendasikan berdasarkan WER terendah pada 4.004 ucapan uji.')
    mm = M['methods'][method]
    c1, c2 = st.columns(2)
    c1.metric('Akurasi digit', pct(mm['digit_acc']), help='Persentase digit yang benar pada data uji.')
    c2.metric('Urutan utuh benar', pct(mm['seq_acc']), help='Persentase rekaman yang SEMUA digitnya benar.')
    st.caption(f"WER {pct(mm['wer'], 2)} (CI 95%: {pct(mm['ci'][0], 2)}–{pct(mm['ci'][1], 2)}) · "
               f"±{mm['latency_ms']:.0f} ms per rekaman")
    if 'wer_noisy' in mm:
        st.caption(f"Kondisi bising: akurasi digit {pct(1 - mm['wer_noisy'])} · "
                   f"simulasi mic HP/laptop {pct(1 - mm['wer_mic'])}")
    st.caption(core.METHOD_INFO.get(method, mm['description']))
    if method != BEST:
        st.info(f"Rekomendasi: **{BEST}** (akurasi digit {pct(M['methods'][BEST]['digit_acc'])}).")

    st.divider()
    st.subheader('🔇 Penjernihan suara')
    enh_mode = st.radio('Mode penjernihan', list(core.ENHANCE_MODES), format_func=core.ENHANCE_MODES.get,
                        index=0, label_visibility='collapsed',
                        help='Otomatis: sistem mencoba audio asli dan audio yang sudah dibersihkan dari derau, '
                             'lalu memakai hasil dengan keyakinan tertinggi. Disarankan untuk rekaman langsung '
                             'dari mikrofon laptop/HP.')
    st.caption('Membuang dengung, desis/derau latar, klik tombol rekam, dan hening di awal/akhir rekaman '
               'sebelum dikenali.')

    st.divider()
    st.subheader('🏦 Skenario layanan')
    scenario = st.selectbox('Jenis nomor', list(SCENARIOS))
    min_len, max_len = length_rule(scenario)
    st.caption(SCENARIOS[scenario][2])
    stepwise = st.toggle('Mode bertahap (gabungkan beberapa rekaman)', value=max_len > 7,
                         help='Model dilatih pada ucapan maksimal 7 digit. Nomor panjang lebih akurat '
                              'bila diucapkan per kelompok lalu digabung.')
    thr = st.slider('Ambang keyakinan', 0.50, 0.95, 0.70, 0.05,
                    help='Digit dengan keyakinan di bawah ambang ditandai oranye dan pengguna diminta mengecek.')

    warns = version_warnings()
    if warns:
        st.warning('Versi pustaka berbeda dari saat pelatihan; hasil bisa berbeda:\n\n- ' + '\n- '.join(warns))

# =============================================================================
# HEADER
# =============================================================================
st.markdown(f"""<div class="hero"><h1>🎙️ Pengenal Angka Suara Bahasa Indonesia</h1>
<p>Ucapkan deretan angka — misalnya <i>"sembilan satu kosong tiga dua"</i> — dan sistem menuliskannya menjadi
<b>91032</b>. Dirancang untuk membantu layanan nasabah mencatat nomor dengan cepat dan terkonfirmasi.
Metode aktif: <b>{method}</b>{' ⭐' if method == BEST else ''}.</p></div>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault('results', {})
ss.setdefault('history', [])
ss.setdefault('buffer', [])
ss.setdefault('input_key', 0)

tab_detect, tab_compare, tab_service, tab_about = st.tabs(
    ['🎙️ Deteksi Angka', '📊 Perbandingan Metode', '🏦 Panduan Layanan', 'ℹ️ Tentang'])

# =============================================================================
# TAB 1 — DETEKSI
# =============================================================================
with tab_detect:
    left, right = st.columns([1, 1.35], gap='large')

    with left:
        st.subheader('1 · Masukkan suara')
        src_rec, src_up, src_ex = st.tabs(['🎤 Rekam langsung', '📁 Unggah file', '🧪 Contoh dataset'])
        audio_bytes, truth, source = None, None, None
        with src_rec:
            st.caption('Klik mikrofon, ucapkan angka dengan jelas, lalu klik lagi untuk berhenti.')
            rec = st.audio_input('Rekam suara', key=f"rec_{ss.input_key}", label_visibility='collapsed')
            if rec is not None:
                audio_bytes, source = rec.getvalue(), 'rekaman'
        with src_up:
            up = st.file_uploader('Unggah audio', type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
                                  key=f"up_{ss.input_key}", label_visibility='collapsed')
            if up is not None:
                audio_bytes, source = up.getvalue(), up.name
        with src_ex:
            samples = sorted((ART_DIR / 'samples').glob('*.wav')) if (ART_DIR / 'samples').exists() else []
            samples = [p for p in samples if sample_tokens(p)]
            if samples:
                pick = st.selectbox('Pilih rekaman uji', samples, format_func=lambda p: (
                    f"{len(sample_tokens(p))} digit — {p.stem}"
                    + (' (versi bising)' if p.stem.endswith('_bising') else '')))
                if st.toggle('Pakai contoh ini', key=f"ex_{ss.input_key}"):
                    audio_bytes, source = pick.read_bytes(), pick.name
                    truth = sample_tokens(pick)
            else:
                st.caption('Tidak ada contoh audio di artifacts/samples.')

        if audio_bytes:
            st.audio(audio_bytes)
        st.markdown('**Tips:** ucapkan per kata dengan jelas, beri jeda singkat antar-angka, dan dekatkan '
                    'mikrofon (±20–30 cm). Di ruangan bising, biarkan *penjernihan suara* pada mode Otomatis.')

    with right:
        st.subheader('2 · Hasil deteksi')
        if not audio_bytes:
            st.info('Belum ada suara. Rekam, unggah file, atau pilih contoh di sebelah kiri.')
        else:
            key = hashlib.md5(audio_bytes).hexdigest()
            try:
                y = core.load_audio(io.BytesIO(audio_bytes))
            except Exception as err:
                st.error(f'File audio tidak bisa dibaca ({err}). Coba format WAV.')
                st.stop()
            dur = len(y) / core.SR
            if dur < 0.3:
                st.warning('Rekaman terlalu pendek. Silakan rekam ulang.')
            elif dur > 20:
                st.warning('Rekaman lebih dari 20 detik. Ucapkan nomor per kelompok agar lebih akurat.')

            if (key, method, enh_mode) not in ss.results:
                ss.results[(key, method, enh_mode)] = REC.recognize(y, method, enhance=enh_mode)
            r = ss.results[(key, method, enh_mode)]
            toks, confs = r['tokens'], r['confidence']

            used_clean = r['variant'] == 'dijernihkan'
            st.caption(f"Kualitas rekaman: **{quality_label(r['snr_db'])}** (perkiraan SNR {r['snr_db']:.0f} dB) · "
                       f"hasil memakai audio **{'yang dijernihkan' if used_clean else 'asli'}**")
            if r['snr_db'] < 18 and enh_mode == 'off':
                st.info('Rekaman cukup bising. Aktifkan **penjernihan suara** di sidebar untuk hasil lebih akurat.')

            if not toks:
                st.error('Tidak ada angka yang terdeteksi. Pastikan suara cukup keras dan berisi angka.')
            else:
                st.markdown(digit_chips(toks, confs, thr), unsafe_allow_html=True)
                st.markdown(f'🔊 Terdengar: **"{r["words"]}"**')

                problems = []
                low = [k + 1 for k, c in enumerate(confs) if c < thr]
                if low:
                    problems.append(f'digit ke-{", ".join(map(str, low))} kurang yakin')
                if not stepwise and not (min_len <= len(toks) <= max_len):
                    problems.append(f'{scenario.split(" (")[0].lower()} harus {min_len}'
                                    f'{"" if min_len == max_len else f"–{max_len}"} digit, terdeteksi {len(toks)}')
                if truth is not None:
                    ok = core.to_value(truth) == core.to_value(toks)
                    (st.success if ok else st.error)(
                        f'Jawaban sebenarnya: **{"".join(core.to_value(truth))}** '
                        f'("{core.to_words(truth)}") — {"✅ tepat" if ok else "❌ berbeda"}')
                if problems:
                    st.warning('Mohon periksa: ' + '; '.join(problems) + '. Minta nasabah mengulang bila ragu.')
                else:
                    st.success(f'Nomor **{r["digits"]}** siap dikonfirmasi ke nasabah.')

                k1, k2, k3 = st.columns(3)
                k1.metric('Jumlah digit', len(toks))
                k2.metric('Keyakinan rata-rata', f'{np.mean(confs):.0%}')
                k3.metric('Waktu proses', f"{r['latency_ms']:.0f} ms")

                b1, b2, b3 = st.columns(3)
                label_ok = '➕ Tambahkan ke nomor' if stepwise else '✅ Benar, simpan'
                if b1.button(label_ok, type='primary', width='stretch'):
                    if stepwise:
                        ss.buffer.extend(core.to_value(toks))
                    ss.history.append({'waktu': datetime.now().strftime('%H:%M:%S'), 'skenario': scenario,
                                       'metode': method, 'hasil': r['digits'], 'diucapkan': r['words'],
                                       'keyakinan_min': round(min(confs), 3), 'sumber': source})
                    ss.input_key += 1
                    st.rerun()
                if b2.button('🔁 Ulangi rekaman', width='stretch'):
                    ss.input_key += 1
                    st.rerun()
                if stepwise and b3.button('🗑️ Kosongkan nomor', width='stretch'):
                    ss.buffer = []
                    st.rerun()

            st.plotly_chart(waveform_figure(y, r), width='stretch')

            if used_clean:
                with st.expander('🎧 Dengar audio setelah dijernihkan'):
                    st.audio(r['audio'], sample_rate=core.SR)
                    st.caption('Inilah audio yang dibaca model: derau latar dikurangi, hening di awal/akhir '
                               'dipotong, dan volume disamakan dengan data latih.')

            with st.expander('🔬 Bandingkan semua metode untuk rekaman ini'):
                rows = []
                for m in AVAILABLE:
                    if (key, m, enh_mode) not in ss.results:
                        ss.results[(key, m, enh_mode)] = REC.recognize(y, m, enhance=enh_mode)
                    o = ss.results[(key, m, enh_mode)]
                    row = {'Metode': ('⭐ ' if m == BEST else '') + m, 'Hasil': o['digits'] or '—',
                           'Audio': o['variant'],
                           'Keyakinan min.': min(o['confidence']) if o['confidence'] else 0.0,
                           'Waktu (ms)': round(o['latency_ms'], 1)}
                    if truth is not None:
                        row['Tepat?'] = '✅' if core.to_value(truth) == core.to_value(o['tokens']) else '❌'
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
                             column_config={'Keyakinan min.': st.column_config.ProgressColumn(
                                 format='%.2f', min_value=0, max_value=1)})

        if stepwise:
            st.divider()
            number = ''.join(ss.buffer)
            st.markdown(f'#### 🧾 Nomor terkumpul: `{number or "—"}`')
            if max_len < 99:
                st.progress(min(len(number) / max_len, 1.0), text=f'{len(number)} dari {min_len}'
                            f'{"" if min_len == max_len else f"–{max_len}"} digit')
                if min_len <= len(number) <= max_len:
                    st.success(f'Nomor lengkap: **{number}**. Bacakan ulang ke nasabah untuk konfirmasi.')
                elif len(number) > max_len:
                    st.error('Digit melebihi panjang yang seharusnya. Kosongkan lalu ulangi.')

    if ss.history:
        st.divider()
        st.subheader('📝 Riwayat sesi ini')
        hist = pd.DataFrame(ss.history)
        st.dataframe(hist, hide_index=True, width='stretch')
        st.download_button('⬇️ Unduh riwayat (CSV)', hist.to_csv(index=False).encode('utf-8'),
                           'riwayat_deteksi.csv', 'text/csv')

# =============================================================================
# TAB 2 — PERBANDINGAN METODE
# =============================================================================
with tab_compare:
    rec_info = M['recommended']
    st.success(f"⭐ **Rekomendasi: {rec_info['method']}** — {rec_info['reason']}")
    if rec_info['fastest_equivalent'] != rec_info['method']:
        st.info(f"⚡ Bila kecepatan lebih penting, **{rec_info['fastest_equivalent']}** tidak berbeda "
                f"signifikan secara statistik dan lebih cepat.")

    table = pd.DataFrame([{
        'Metode': ('⭐ ' if m == BEST else '') + m,
        'Akurasi digit': M['methods'][m]['digit_acc'],
        'Urutan utuh benar': M['methods'][m]['seq_acc'],
        'WER': M['methods'][m]['wer'],
        'CI 95%': f"{pct(M['methods'][m]['ci'][0], 2)} – {pct(M['methods'][m]['ci'][1], 2)}",
        'Penutur baru (WER)': M['methods'][m]['wer_unseen'],
        'Digit tertukar': M['methods'][m]['sub'], 'Terlewat': M['methods'][m]['del'],
        'Tambahan': M['methods'][m]['ins'], 'Waktu (ms)': round(M['methods'][m]['latency_ms'], 1),
        **({'WER bising (rata-rata)': M['methods'][m]['wer_noisy'],
            'WER mic HP/laptop': M['methods'][m]['wer_mic']} if 'wer_noisy' in M['methods'][m] else {}),
    } for m in M['ranking']])
    acc_col = lambda t: st.column_config.ProgressColumn(t, format='percent', min_value=0, max_value=1)
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        'Akurasi digit': acc_col('Akurasi digit'), 'Urutan utuh benar': acc_col('Urutan utuh benar'),
        'WER': st.column_config.NumberColumn(format='percent'),
        'Penutur baru (WER)': st.column_config.NumberColumn(format='percent'),
        'WER bising (rata-rata)': st.column_config.NumberColumn(format='percent'),
        'WER mic HP/laptop': st.column_config.NumberColumn(format='percent')})

    if M.get('conditions') and all('robustness' in M['methods'][m] for m in M['ranking']):
        st.markdown('##### Ketahanan terhadap derau (meniru rekaman mikrofon langsung)')
        rb = pd.DataFrame([{'Metode': m, 'Kondisi': c, 'Penjernihan': {'auto': 'Otomatis', 'off': 'Mati'}[mode],
                            'Akurasi digit': 1 - M['methods'][m]['robustness'][mode][c]}
                           for m in M['ranking'] for mode in ('auto', 'off') for c in M['conditions']])
        fig = px.line(rb, x='Kondisi', y='Akurasi digit', color='Metode', line_dash='Penjernihan', markers=True,
                      category_orders={'Kondisi': M['conditions']},
                      title='Akurasi digit pada data uji yang dicampur derau (derau tidak pernah dipakai latih)')
        fig.update_layout(yaxis_tickformat='.0%', height=420)
        st.plotly_chart(fig, width='stretch')
        st.caption(f"{M['dataset'].get('n_noisy_eval', '')} ucapan uji per kondisi · sumber derau: "
                   f"{M['dataset'].get('noise_test', '-')}. Garis putus-putus = tanpa penjernihan.")

    g1, g2 = st.columns(2)
    with g1:
        acc = pd.DataFrame([{'Metode': m, 'Akurasi': 1 - M['methods'][m]['wer'],
                             'atas': M['methods'][m]['wer'] - M['methods'][m]['ci'][0],
                             'bawah': M['methods'][m]['ci'][1] - M['methods'][m]['wer']}
                            for m in M['ranking']])
        fig = px.bar(acc, x='Akurasi', y='Metode', orientation='h', error_x='atas', error_x_minus='bawah',
                     color=acc.Metode.eq(BEST).map({True: 'Direkomendasikan', False: 'Lainnya'}),
                     color_discrete_map={'Direkomendasikan': OK, 'Lainnya': '#9AA9C2'},
                     title='Akurasi digit pada data uji (garis = selang kepercayaan 95%)')
        fig.update_layout(xaxis_tickformat='.0%', legend_title='', height=360,
                          yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')
    with g2:
        sc = pd.DataFrame([{'Metode': m, 'Akurasi digit': M['methods'][m]['digit_acc'],
                            'Waktu (ms)': M['methods'][m]['latency_ms']} for m in M['ranking']])
        fig = px.scatter(sc, x='Waktu (ms)', y='Akurasi digit', text='Metode', size=[18] * len(sc),
                         title='Akurasi vs kecepatan (kiri-atas = terbaik)')
        fig.update_traces(textposition='top center', marker_color=PRIMARY)
        fig.update_layout(yaxis_tickformat='.0%', height=360)
        st.plotly_chart(fig, width='stretch')

    g3, g4 = st.columns(2)
    with g3:
        ln = pd.DataFrame([{'Metode': m, 'Jumlah digit': int(k), 'WER': v}
                           for m in M['ranking'] for k, v in M['methods'][m]['per_length'].items()])
        fig = px.line(ln, x='Jumlah digit', y='WER', color='Metode', markers=True,
                      title='Kesalahan menurut panjang nomor yang diucapkan')
        fig.update_layout(yaxis_tickformat='.0%', height=340)
        st.plotly_chart(fig, width='stretch')
    with g4:
        sp = pd.DataFrame({m: M['methods'][m]['per_split'] for m in M['ranking']}).T
        fig = px.imshow(sp * 100, text_auto='.1f', color_continuous_scale='RdYlGn_r', aspect='auto',
                        title='WER (%) per subset uji — makin hijau makin baik')
        fig.update_layout(height=340, coloraxis_colorbar_title='WER %')
        st.plotly_chart(fig, width='stretch')

    st.markdown('##### Apakah perbedaannya nyata atau kebetulan?')
    sig = pd.DataFrame([{'Metode': s['method'], f'Selisih WER vs {BEST}': s['diff'],
                         'CI 95%': f"{s['ci'][0] * 100:+.2f}% s/d {s['ci'][1] * 100:+.2f}%",
                         'p-value': s['p'], 'Kesimpulan': s['conclusion']} for s in M['significance']])
    if len(sig):
        st.dataframe(sig, hide_index=True, width='stretch',
                     column_config={f'Selisih WER vs {BEST}': st.column_config.NumberColumn(format='percent'),
                                    'p-value': st.column_config.NumberColumn(format='%.4f')})
    st.caption('Bootstrap berpasangan per penutur: penutur uji di-resampling ribuan kali. Bila selang '
               'kepercayaan selisih tidak memuat 0, metode terbaik benar-benar lebih baik.')

    cm_method = st.selectbox('Confusion matrix untuk metode', M['ranking'],
                             index=M['ranking'].index(method) if method in M['ranking'] else 0)
    cm = M['methods'][cm_method]['confusion']
    fig = px.imshow(np.array(cm['matrix']), x=cm['labels'], y=cm['labels'], text_auto=True,
                    color_continuous_scale='Blues', aspect='auto',
                    labels=dict(x='Terdeteksi', y='Diucapkan', color='Jumlah'),
                    title=f'Digit mana yang sering tertukar — {cm_method} (∅ = terlewat / tambahan)')
    fig.update_layout(height=480)
    st.plotly_chart(fig, width='stretch')

    with st.expander('📖 Cara membaca metrik'):
        st.markdown("""
- **Akurasi digit** — dari 100 digit yang diucapkan, berapa yang tertulis benar.
- **Urutan utuh benar** — persentase rekaman yang *semua* digitnya benar. Untuk nomor rekening, inilah angka yang paling penting.
- **WER (word/digit error rate)** = (digit tertukar + terlewat + tambahan) ÷ jumlah digit diucapkan. Makin kecil makin baik.
- **CI 95%** — rentang WER yang masuk akal bila sistem dipakai oleh penutur lain.
- **Penutur baru** — WER khusus orang yang suaranya tidak ada di data latih.
- *Nol* dan *kosong* dihitung sebagai angka yang sama (0).
""")
    hist = M.get('ctc_history') or []
    if hist:
        with st.expander('📈 Kurva pelatihan CNN-BiLSTM-CTC'):
            h = pd.DataFrame(hist)
            fig = px.line(h, x='epoch', y='WER_val', markers=True, title='WER validasi per epoch')
            fig.update_layout(yaxis_tickformat='.0%', height=300)
            st.plotly_chart(fig, width='stretch')

# =============================================================================
# TAB 3 — PANDUAN LAYANAN
# =============================================================================
with tab_service:
    st.subheader('Bagaimana sistem ini membantu layanan nasabah')
    a, b, c = st.columns(3)
    a.markdown('<div class="card"><b>⏱️ Lebih cepat</b><br>Petugas tidak perlu mengetik ulang nomor yang '
               'disebutkan nasabah; nomor langsung muncul dan bisa dibacakan ulang.</div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><b>✅ Lebih sedikit salah catat</b><br>Digit yang kurang yakin ditandai '
               'oranye sehingga petugas tahu bagian mana yang perlu ditanyakan ulang.</div>', unsafe_allow_html=True)
    c.markdown('<div class="card"><b>♿ Lebih inklusif</b><br>Dataset ini awalnya dibuat untuk sistem '
               'telekomunikasi bagi penyandang disabilitas pendengaran dan wicara.</div>', unsafe_allow_html=True)

    st.markdown("""
#### Alur yang disarankan
1. **Nasabah menyebut nomor** (nomor rekening, HP, tiket) — untuk nomor panjang, per kelompok 3–4 digit.
2. **Sistem menampilkan digit** beserta tingkat keyakinan dan cek panjang nomor.
3. **Petugas/sistem membacakan ulang** kalimat "Anda menyebut: …" untuk dikonfirmasi nasabah.
4. **Bila ada digit oranye atau panjang tidak sesuai** → minta nasabah mengulang. Bila gagal dua kali → petugas mengetik manual.

#### Indikator yang bisa dipantau
| Indikator | Cara mengukur |
|---|---|
| Waktu penanganan (AHT) | Bandingkan durasi pencatatan nomor sebelum vs sesudah memakai sistem |
| Tingkat konfirmasi pertama | % nomor yang langsung dikonfirmasi benar tanpa diulang |
| Tingkat eskalasi manual | % sesi yang akhirnya diketik petugas |

#### Batasan keamanan — WAJIB
- ⛔ **Jangan** gunakan untuk PIN, kata sandi, CVV, atau kode OTP. Data rahasia tidak boleh diucapkan.
- 🔁 Selalu **bacakan ulang** nomor sebelum transaksi; sistem ini alat bantu, bukan pengambil keputusan.
- 🔒 Rekaman diproses di memori aplikasi dan tidak disimpan ke disk oleh dashboard ini.
""")

# =============================================================================
# TAB 4 — TENTANG
# =============================================================================
with tab_about:
    d = M['dataset']
    st.subheader('Data')
    st.markdown(f"""
**INDspeech_DIGIT_CDSR** (Sakti dkk., 2004): ucapan deretan digit Bahasa Indonesia oleh 214 penutur dari berbagai daerah, 16 kHz.
Kosakata: *nol, kosong, satu, …, sembilan* (11 kata), 1–7 digit per ucapan.

| Bagian | Jumlah ucapan |
|---|---|
| Latih | {d['n_train']:,} |
| Validasi (penutur ditahan) | {d['n_val']:,} |
| Uji (test1–test4) | {d['n_test']:,} dari {d['speakers_test']} penutur |
""")
    st.subheader('Metode')
    for m in M['ranking']:
        st.markdown(f"- **{m}**{' ⭐' if m == BEST else ''} — {M['methods'][m]['description']}")
    st.markdown(f"""
Pada metode berbasis potongan, audio dipotong di bagian yang lebih senyap {d['segment_top_db']} dB dari puncak.
Pemotongan ini hanya cocok dengan jumlah digit sebenarnya pada {pct(d['segment_coverage_train'])} ucapan latih,
sehingga ucapan yang diucapkan menyambung tanpa jeda menjadi sumber kesalahan utama.
""")
    st.subheader('Keterbatasan')
    st.markdown(f"""
- Data latih direkam di ruangan tenang. Untuk rekaman mikrofon laptop/HP, dashboard menjernihkan suara
  (spectral subtraction + pemotongan hening) dan model **CRNN-CTC Robust** dilatih dengan augmentasi derau;
  derau yang sangat keras atau orang lain yang bicara bersamaan tetap bisa menurunkan akurasi.
- Ucapan latih paling panjang 7 digit → gunakan **mode bertahap** untuk nomor panjang.
- {len(d['seen_speakers'])} penutur ({', '.join(d['seen_speakers']) or '—'}) muncul di data latih dan uji; WER penutur baru dilaporkan terpisah.
- Prototipe akademik — bukan sistem produksi untuk transaksi keuangan.
""")
    st.caption(f"Model dilatih: {REC.config.get('created', '-')} · Ringkasan evaluasi: {M.get('created', '-')}")
