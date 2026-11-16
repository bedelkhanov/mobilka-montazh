#!/usr/bin/env python3
"""Мобилка монтаж — движок.

Команды (по порядку):
  setup                         — поставить зависимости
  prep VIDEO WORK               — вертикаль 1080x1920 30 fps, звук
  transcribe WORK               — слова с таймкодами (faster-whisper large-v3)
  cut WORK [--speed 1.1] [--drop 12.3-14.0 ...]
                                — вырезать паузы и куски, собрать cut.mp4 + voice.wav, пересчитать слова
  broll WORK URL|FILE --name N [--start s --end s]
                                — живая перебивка из TikTok (или файл из чата) → broll/N.mp4
  analyze WORK                  — где верх головы на каждом кадре (для субтитров и вставок)
  plan WORK                     — черновик plan.json: карточки субтитров по словам
  render WORK [--out F]          — слой текста/вставок (Chrome) + сборка кадра + эффекты → готовый mp4
  sfx [WORK]                    — звуки в папке sfx; с WORK — где они встанут в ролике
  sheet WORK                    — контакт-лист 2 кадра/с для проверки
"""
import argparse, json, math, os, shutil, subprocess, sys, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'          # шрифты и модель — скачиваются командой setup
SFX = ROOT / 'sfx'                # звуковые эффекты Асланбека
OVERLAY = ROOT / 'overlay.html'
W, H, FPS = 1080, 1920, 30
STAGE = (251, 250, 247)            # белый фон вставок #FBFAF7
FX0, FY0, FX1 = 42, 888, 1038      # рамка спикера: x 42–1038, от y 888 до низа
FR = 84                            # скругление верхних углов рамки
HEAD_Y = 768                       # куда ставим верх головы во вставке (над рамкой на 120 px)
SCALE = (FX1 - FX0) / W            # 0.922


def sh(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def ffprobe_dur(p):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(p)], capture_output=True, text=True).stdout
    return float(out.strip())


def find_chrome():
    for p in ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------- setup
def cmd_setup(a):
    need = []
    for mod, pkg in [('faster_whisper', 'faster-whisper'), ('mediapipe', 'mediapipe'), ('playwright', 'playwright'),
                     ('cv2', 'opencv-python-headless'), ('PIL', 'pillow'), ('numpy', 'numpy')]:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need:
        sh([sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages', *need])
    if not find_chrome():
        sh([sys.executable, '-m', 'playwright', 'install', 'chromium'])
    fonts = ASSETS / 'fonts'; models = ASSETS / 'models'
    fonts.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True); SFX.mkdir(exist_ok=True)
    inter = ['InterDisplay-Regular.otf', 'Inter-SemiBold.otf', 'Inter-Medium.otf']
    if not all((fonts / f).exists() for f in inter):
        import zipfile, io, urllib.request
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip').read()))
        for f in inter:
            (fonts / f).write_bytes(z.read(f'extras/otf/{f}'))
    dl = [(fonts / 'NotoSerifDisplay-Italic.ttf', 'https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifdisplay/NotoSerifDisplay-Italic%5Bwdth,wght%5D.ttf'),
          (models / 'selfie_segmenter.tflite', 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite')]
    for p, url in dl:
        if not p.exists():
            sh(['curl', '-sSfL', '-o', str(p), url])
    print('setup ok')


# ---------------------------------------------------------------- prep
def cmd_prep(a):
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    vf = 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1'
    sh(['ffmpeg', '-v', 'error', '-y', '-i', a.video, '-vf', vf, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14',
        '-pix_fmt', 'yuv420p', '-an', str(work / 'src.mp4')])
    sh(['ffmpeg', '-v', 'error', '-y', '-i', a.video, '-vn', '-ac', '2', '-ar', '48000', str(work / 'voice48.wav')])
    sh(['ffmpeg', '-v', 'error', '-y', '-i', a.video, '-vn', '-ac', '1', '-ar', '16000', str(work / 'audio16.wav')])
    print('prep ok', ffprobe_dur(work / 'src.mp4'), 'с')


# ---------------------------------------------------------------- transcribe
def read_wav(p):
    import numpy as np
    wf = wave.open(str(p))
    return np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768


def transcribe(wav, out, prompt='', lang=None):
    from faster_whisper import WhisperModel
    m = WhisperModel('large-v3', device='cpu', compute_type='int8', cpu_threads=os.cpu_count() or 2)
    segs, info = m.transcribe(read_wav(wav), language=lang or None, word_timestamps=True, beam_size=5,
                              vad_filter=False, initial_prompt=prompt or None)
    words = []
    for s in segs:
        for w in s.words:
            words.append({'w': w.word.strip(), 's': round(w.start, 3), 'e': round(w.end, 3), 'p': round(w.probability, 2)})
    json.dump({'lang': info.language, 'words': words}, open(out, 'w'), ensure_ascii=False, indent=1)
    print(info.language, ' '.join(x['w'] for x in words))
    return words


def cmd_transcribe(a):
    work = Path(a.work)
    transcribe(work / 'audio16.wav', work / 'words_src.json', a.prompt, a.lang)


# ---------------------------------------------------------------- cut
def envelope(path):
    import numpy as np
    x = read_wav(path); fr = 160; n = len(x) // fr
    return 20 * np.log10(np.sqrt((x[:n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-9)


def runs(m):
    r = []; i = 0; n = len(m)
    while i < n:
        j = i
        while j < n and m[j] == m[i]:
            j += 1
        r.append([bool(m[i]), i, j]); i = j
    return r


def cmd_cut(a):
    import numpy as np
    work = Path(a.work)
    words = json.load(open(work / 'words_src.json'))['words']
    sh(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'voice48.wav'), '-af', 'highpass=f=3500,highpass=f=3500', '-ac', '1', '-ar', '16000', str(work / 'hp16.wav')])
    e = envelope(work / 'audio16.wav'); h = envelope(work / 'hp16.wav'); n = min(len(e), len(h)); e, h = e[:n], h[:n]
    thr = min(-40.0, max(float(np.percentile(e, 20)) + 15, -50.0))
    hthr = float(np.percentile(h, 20)) + 12
    full = e > thr
    near = np.convolve(full.astype(int), np.ones(61), 'same') > 0          # ±0,3 с вокруг речи
    sp = full | (near & (h > hthr))                                       # шипящие «с», «ш» на краях слов
    for v, i, j in runs(sp):
        if not v and j - i < 8: sp[i:j] = True
    for v, i, j in runs(sp):
        if v and j - i < 6: sp[i:j] = False
    segs = [[i / 100, j / 100] for v, i, j in runs(sp) if v]
    has_word = lambda s, t: any(min(t, w['e']) - max(s, w['s']) > 0.05 for w in words)
    segs = [s for s in segs if has_word(*s)]
    drops = []
    for d in a.drop or []:
        x, y = d.split('-'); drops.append((float(x), float(y)))
    PAD, GAP = a.pad, a.gap
    keep = []
    for s, t in segs:
        # не обрезать слова: тянем край куска до границы слова (не дальше 0,12 с)
        for w in words:
            if w['s'] < t and w['e'] > s:
                if w['e'] > t: t = min(w['e'], t + 0.12)
                if w['s'] < s: s = max(w['s'], s - 0.08)
        s = max(0, s - PAD); t = t + PAD
        if keep and s - keep[-1][1] < GAP: keep[-1][1] = max(keep[-1][1], t)
        else: keep.append([s, t])
    # вырезать выброшенные куски (оговорки, дубли, лишние фразы)
    out = []
    for s, t in keep:
        cur = [[s, t]]
        for x, y in drops:
            nxt = []
            for p, q in cur:
                if y <= p or x >= q: nxt.append([p, q])
                else:
                    if x > p: nxt.append([p, x])
                    if y < q: nxt.append([y, q])
            cur = nxt
        out += [c for c in cur if c[1] - c[0] > 0.08]
    keep = out
    json.dump(keep, open(work / 'edl.json', 'w'))
    sp_ = a.speed
    fa = ';'.join(f'[0:a]atrim={s:.3f}:{t:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.015,afade=t=out:st={t - s - 0.015:.3f}:d=0.015[a{i}]' for i, (s, t) in enumerate(keep))
    fa += ';' + ''.join(f'[a{i}]' for i in range(len(keep))) + f'concat=n={len(keep)}:v=0:a=1' + (f',atempo={sp_}' if sp_ != 1 else '') + '[out]'
    (work / 'afc.txt').write_text(fa)
    sh(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'voice48.wav'), '-filter_complex_script', str(work / 'afc.txt'), '-map', '[out]', str(work / 'voice.wav')])
    fv = ';'.join(f'[0:v]trim={s:.4f}:{t:.4f},setpts=PTS-STARTPTS[v{i}]' for i, (s, t) in enumerate(keep))
    fv += ';' + ''.join(f'[v{i}]' for i in range(len(keep))) + f'concat=n={len(keep)}:v=1:a=0,setpts=PTS/{sp_},fps=30[out]'
    (work / 'vfc.txt').write_text(fv)
    sh(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'src.mp4'), '-filter_complex_script', str(work / 'vfc.txt'), '-map', '[out]',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '14', '-pix_fmt', 'yuv420p', str(work / 'cut.mp4')])
    sh(['ffmpeg', '-v', 'error', '-y', '-i', str(work / 'voice.wav'), '-ac', '1', '-ar', '16000', str(work / 'voice16.wav')])
    # слова исходника переносим на новую шкалу (распознавание исходника точнее, чем нарезки)
    cum = [0.0]
    for s_, t_ in keep: cum.append(cum[-1] + t_ - s_)
    ws = []
    for w in words:
        best, ov = None, 0.0
        for i, (s_, t_) in enumerate(keep):
            o = min(w['e'], t_) - max(w['s'], s_)
            if o > ov: best, ov = i, o
        if best is None: continue
        s_, t_ = keep[best]
        ns = (cum[best] + max(w['s'], s_) - s_) / sp_; ne = (cum[best] + min(w['e'], t_) - s_) / sp_
        ws.append({'w': w['w'], 's': round(ns, 3), 'e': round(ne, 3), 'p': w.get('p', 1)})
    ev = envelope(work / 'voice16.wav')
    for w in ws:
        i, j = int(w['s'] * 100), int(w['e'] * 100); k = i
        while k < j and k < len(ev) and ev[k] < thr: k += 1
        if k < j: w['s'] = round(min(k / 100, w['e'] - 0.08), 3)
    for a_, b_ in zip(ws, ws[1:]):          # порядок и минимальная длина
        if b_['s'] < a_['s'] + 0.12: b_['s'] = round(a_['s'] + 0.12, 3)
        if b_['e'] < b_['s'] + 0.05: b_['e'] = round(b_['s'] + 0.05, 3)
    if ws: ws[0]['s'] = 0.0                  # первое слово — с первого кадра
    lang = json.load(open(work / 'words_src.json')).get('lang', 'ru')
    json.dump({'lang': lang, 'words': ws}, open(work / 'words.json', 'w'), ensure_ascii=False, indent=1)
    print(' '.join(w['w'] for w in ws))
    dur = ffprobe_dur(work / 'voice.wav')
    wpm = len(ws) / max(1e-6, (ws[-1]['e'] - ws[0]['s'])) * 60 if ws else 0
    print(f'cut ok: {len(keep)} кусков, {dur:.1f} с, темп {wpm:.0f} слов/мин (норма 175–215)')


# ---------------------------------------------------------------- segmentation helpers
_seg = None


def segmenter():
    global _seg
    if _seg is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        _seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(ASSETS / 'models' / 'selfie_segmenter.tflite')), output_confidence_masks=True))
    return _seg


def person_mask(rgb_small):
    import mediapipe as mp, numpy as np
    r = segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_small)))
    return r.confidence_masks[-1].numpy_view().copy()


def frames(path, w=W, h=H, fps=None, ss=None, vf=None):
    import numpy as np
    vf = vf or (f'scale={w}:{h}' + (f',fps={fps}' if fps else ''))
    pre = ['-ss', f'{ss:.3f}'] if ss else []
    p = subprocess.Popen(['ffmpeg', '-v', 'error', *pre, '-i', str(path), '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    try:
        while True:
            b = p.stdout.read(w * h * 3)
            if len(b) < w * h * 3: break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        p.kill(); p.wait()


# ---------------------------------------------------------------- analyze
def cmd_analyze(a):
    import numpy as np
    work = Path(a.work); res = []
    for i, f in enumerate(frames(work / 'cut.mp4', 270, 480, fps=10)):
        m = person_mask(f) > 0.5
        rows = np.where(m.sum(1) > 6)[0]
        top = int(rows[0] * 4) if len(rows) else 600
        res.append([round(i / 10, 2), top])
    tops = np.array([r[1] for r in res], float)
    k = np.ones(5) / 5
    sm = np.convolve(np.pad(tops, 2, mode='edge'), k, 'valid')
    res = [[r[0], int(v)] for r, v in zip(res, sm)]
    json.dump(res, open(work / 'heads.json', 'w'))
    print('analyze ok: верх головы', int(np.median(sm)), 'px из 1920')


# ---------------------------------------------------------------- plan
def cmd_plan(a):
    work = Path(a.work)
    d = json.load(open(work / 'words.json')); ws = d['words']
    dur = ffprobe_dur(work / 'voice.wav')
    cards = []; pend = []
    clean = lambda x: x.strip('.,!?:;«»"—-')
    for w in ws:
        if len(clean(w['w'])) <= 3 and w['w'][-1:] not in ',.?!:;' and len(pend) < 2:
            pend.append(w); continue
        grp = pend + [w]; pend = []
        cards.append({'text': ' '.join(x['w'] for x in grp).lower(), 'start': grp[0]['s'], 'wend': grp[-1]['e'], 'accent': False})
    if pend and cards:
        cards[-1]['text'] += ' ' + ' '.join(x['w'] for x in pend).lower(); cards[-1]['wend'] = pend[-1]['e']
    for i, c in enumerate(cards):
        nxt = cards[i + 1]['start'] if i + 1 < len(cards) else dur
        c['end'] = round(nxt if nxt - c['wend'] < 0.45 else c['wend'] + 0.15, 3)
        c['start'] = round(c['start'], 3); c.pop('wend')
    end = min(dur, (ws[-1]['e'] + 0.3) if ws else dur)
    plan = {'lang': d.get('lang', 'ru'), 'end': round(end, 3), 'captions': cards, 'inserts': []}
    p = work / 'plan.json'
    if p.exists() and not a.force:
        print('plan.json уже есть (—force чтобы перезаписать)'); return
    json.dump(plan, open(p, 'w'), ensure_ascii=False, indent=1)
    print(f'plan ok: {len(cards)} карточек. Теперь отметь accent и добавь inserts.')


# ---------------------------------------------------------------- broll
def cmd_broll(a):
    """Живая перебивка: ссылка TikTok (или другой сайт, который берёт yt-dlp) либо файл из чата → broll/NAME.mp4 (1080x1920, 30 fps, без звука)."""
    work = Path(a.work).resolve(); bd = work / 'broll'; bd.mkdir(parents=True, exist_ok=True)
    raw = bd / f'{a.name}_src.mp4'
    if Path(a.src).exists():
        shutil.copy(a.src, raw)
    else:
        try:
            import yt_dlp  # noqa
        except ImportError:
            sh([sys.executable, '-m', 'pip', 'install', '-q', '--break-system-packages', 'yt-dlp'])
        if raw.exists(): raw.unlink()
        # без водяного знака TikTok: формат "download" — с вотермаркой, его не берём
        sh([sys.executable, '-m', 'yt_dlp', '-q', '--no-warnings', '--no-playlist', '-f', 'best[format_id!=download][vcodec!=none]/best',
            '-o', str(raw), a.src])
    cut = []
    if a.start: cut += ['-ss', str(a.start)]
    if a.end: cut += ['-to', str(a.end)]
    out = bd / f'{a.name}.mp4'
    vf = 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1'
    sh(['ffmpeg', '-v', 'error', '-y', *cut, '-i', str(raw), '-vf', vf, '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '16', '-pix_fmt', 'yuv420p', str(out)])
    d = ffprobe_dur(out)
    sheet = bd / f'{a.name}_sheet.jpg'
    sh(['ffmpeg', '-v', 'error', '-y', '-i', str(out), '-vf', f'fps={min(2, 16 / max(d, 1)):.3f},scale=180:-2,drawtext=text=%{{pts}}:x=4:y=4:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.5,tile=8x2',
        '-frames:v', '1', str(sheet)])
    print(f'broll ok: {out.relative_to(work)} — {d:.1f} с. Кадры: {sheet}')


# ---------------------------------------------------------------- render
def export_thumbs(work, plan):
    """data.thumbs: [секунды в cut.mp4] → пути к jpg"""
    tdir = work / 'thumbs'; tdir.mkdir(exist_ok=True)
    for k, ins in enumerate(plan.get('inserts', [])):
        th = ins.get('data', {}).get('thumbs')
        if not th: continue
        paths = []
        for j, t in enumerate(th):
            if isinstance(t, str): paths.append(t); continue
            out = tdir / f'i{k}_{j}.jpg'
            sh(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-i', str(work / 'cut.mp4'), '-frames:v', '1', '-vf', 'scale=360:640', '-q:v', '3', str(out)])
            paths.append(out.as_uri())
        ins['data']['thumbs'] = paths


def render_overlay(work, plan, nf):
    from playwright.sync_api import sync_playwright
    od = work / 'ov'
    if od.exists(): shutil.rmtree(od)
    od.mkdir()
    with sync_playwright() as pw:
        kw = {'executable_path': find_chrome()} if find_chrome() else {}
        br = pw.chromium.launch(args=['--allow-file-access-from-files', '--disable-web-security'], **kw)
        pg = br.new_page(viewport={'width': 360, 'height': 640}, device_scale_factor=3)
        pg.goto(OVERLAY.as_uri())
        pg.evaluate('document.fonts.ready')
        pg.evaluate('(p) => init(p)', plan)
        pg.evaluate('document.fonts.ready')
        for n in range(nf):
            pg.evaluate(f'seek({n / FPS})')
            pg.screenshot(path=str(od / f'{n:05d}.png'), omit_background=True)
            if n % 150 == 0: print('overlay', n, '/', nf, flush=True)
        br.close()


def ease(x):
    x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3


BR_X0, BR_Y0, BR_X1, BR_Y1, BR_R = 141, 276, 939, 630, 48     # карточка перебивки во вставке-карточке


def _broll_next(work, ins, t, w, h):
    """Следующий кадр перебивки нужного размера (поток открывается на входе во вставку)."""
    import numpy as np
    d = ins.get('data', {}) or {}
    if '_it' not in ins:
        f = d.get('file') or f"broll/{d.get('name', 'b1')}.mp4"
        f = Path(f) if Path(f).is_absolute() else work / f
        z = float(d.get('zoom', 1.0)); fy = float(d.get('y', 0.5))
        sw, shh = int(round(w * z / 2) * 2), int(round(h * z / 2) * 2)
        vf = (f'scale={sw}:{shh}:force_original_aspect_ratio=increase,'
              f'crop={w}:{h}:(iw-{w})/2:(ih-{h})*{fy},fps=30,setsar=1')
        ins['_it'] = frames(f, w, h, ss=float(d.get('from', 0)) + (t - ins['start']), vf=vf); ins['_last'] = None
    try:
        ins['_last'] = next(ins['_it'])
    except StopIteration:
        pass                                   # клип короче вставки — держим последний кадр
    return ins['_last'] if ins['_last'] is not None else np.zeros((h, w, 3), np.uint8)


def broll_frame(work, ins, t, fr, stage, fmask, shadow, fw, fh):
    """mode=full — перебивка на весь экран (медленный наезд); mode=card — белый фон, спикер в рамке, видео в карточке сверху."""
    import numpy as np, cv2
    d = ins.get('data', {}) or {}; lt = t - ins['start']; dur = ins['end'] - ins['start']
    if d.get('mode', 'full') == 'full':
        f = _broll_next(work, ins, t, W, H).astype(np.float32) / 255
        k = 1 + 0.05 * min(1, lt / max(dur, 0.1))                       # наезд 1.00 → 1.05
        if k > 1.001:
            M = np.float32([[k, 0, (1 - k) * W / 2], [0, k, (1 - k) * H / 2]])
            f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR)
        return f
    # карточка: та же сцена, что у остальных вставок, + видео в скруглённой карточке
    pa = ease(lt / 0.25); pf = ease(lt / 0.4); pc = ease((lt - 0.1) / 0.35)
    base = fr.astype(np.float32) / 255
    can = stage.copy() * (1 - shadow * pf)
    sw_, shh = fw, int(round(H * SCALE))
    sp = cv2.resize(fr, (sw_, shh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    ty = int(round(HEAD_Y - ins['_head'] * SCALE)) + int(round(180 * (1 - pf)))
    rows = np.clip(np.arange(FY0, H) - ty, 0, shh - 1)
    a_ = fmask * pf
    can[FY0:, FX0:FX1] = can[FY0:, FX0:FX1] * (1 - a_) + sp[rows] * a_
    top_rows = np.arange(0, FY0) - ty; okt = (top_rows >= 0) & (top_rows < shh)
    if okt.any():
        m = person_mask(cv2.resize(fr, (256, 455), interpolation=cv2.INTER_AREA))
        m = np.clip((cv2.GaussianBlur(cv2.resize(m, (sw_, shh)), (0, 0), 2.0) - 0.35) / 0.3, 0, 1)
        ys = np.where(okt)[0]; al = m[top_rows[ys]][..., None] * pf
        can[ys, FX0:FX1] = can[ys, FX0:FX1] * (1 - al) + sp[top_rows[ys]] * al
    cw, ch = BR_X1 - BR_X0, BR_Y1 - BR_Y0
    v = _broll_next(work, ins, t, cw, ch).astype(np.float32) / 255
    if '_cmask' not in ins:
        cm = np.zeros((ch, cw), np.uint8)
        cv2.rectangle(cm, (BR_R, 0), (cw - 1 - BR_R, ch - 1), 255, -1); cv2.rectangle(cm, (0, BR_R), (cw - 1, ch - 1 - BR_R), 255, -1)
        for cx in (BR_R, cw - 1 - BR_R):
            for cy in (BR_R, ch - 1 - BR_R): cv2.circle(cm, (cx, cy), BR_R, 255, -1, lineType=cv2.LINE_AA)
        sh_ = np.zeros((H, W), np.float32); sh_[BR_Y0 + 24:BR_Y1 + 24, BR_X0 + 10:BR_X1 - 10] = 0.16
        ins['_cmask'] = (cm.astype(np.float32) / 255)[..., None]; ins['_cshadow'] = cv2.GaussianBlur(sh_, (0, 0), 26)[..., None]
    dy = int(round(-40 * (1 - pc)))
    can = can * (1 - ins['_cshadow'] * pc)
    y0 = BR_Y0 + dy; y1 = y0 + ch
    a2 = ins['_cmask'] * pc
    can[y0:y1, BR_X0:BR_X1] = can[y0:y1, BR_X0:BR_X1] * (1 - a2) + v * a2
    return base * (1 - pa) + can * pa


def cmd_render(a):
    import numpy as np, cv2
    from PIL import Image
    work = Path(a.work).resolve()
    plan = json.load(open(work / 'plan.json'))
    plan['heads'] = json.load(open(work / 'heads.json')) if (work / 'heads.json').exists() else []
    end = plan.get('end') or ffprobe_dur(work / 'voice.wav')
    nf = int(round(end * FPS))
    export_thumbs(work, plan)
    json.dump(plan, open(work / 'plan_render.json', 'w'), ensure_ascii=False)
    if not a.skip_overlay:
        render_overlay(work, plan, nf)
    # аудио
    mix = work / 'mix.wav'
    audio_mix(work, end, plan, mix, a.sfx_boost)
    json.dump(plan.get('_sfx_used', []), open(work / 'sfx_used.json', 'w'), ensure_ascii=False)
    # заготовки рамки
    fw, fh = FX1 - FX0, H - FY0
    fmask = np.zeros((fh, fw), np.uint8)
    cv2.rectangle(fmask, (0, FR), (fw - 1, fh - 1), 255, -1); cv2.rectangle(fmask, (FR, 0), (fw - 1 - FR, fh - 1), 255, -1)
    for cx in (FR, fw - 1 - FR): cv2.circle(fmask, (cx, FR), FR, 255, -1, lineType=cv2.LINE_AA)
    fmask = (fmask.astype(np.float32) / 255)[..., None]
    shadow = np.zeros((H, W), np.float32); shadow[FY0 + 36:, FX0:FX1] = 0.14
    shadow = cv2.GaussianBlur(shadow, (0, 0), 30)[..., None]
    ins_list = sorted(plan.get('inserts', []), key=lambda x: x['start'])
    heads = plan['heads']
    def head_at(t):
        if not heads: return 600
        i = min(range(len(heads)), key=lambda k: abs(heads[k][0] - t)); return heads[i][1]
    for ins in ins_list:
        hs = [h[1] for h in heads if ins['start'] - 0.2 <= h[0] <= ins['end']] or [head_at(ins['start'])]
        ins['_head'] = float(np.median(hs))
    out = a.out or str(work / 'final.mp4')
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-i', str(mix), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    stage = np.empty((H, W, 3), np.float32); stage[:] = np.array(STAGE, np.float32) / 255
    for n, fr in enumerate(frames(work / 'cut.mp4')):
        if n >= nf: break
        t = n / FPS
        base = fr.astype(np.float32) / 255
        ins = next((i for i in ins_list if i['start'] <= t < i['end']), None)
        if ins and ins['type'] == 'broll':
            base = broll_frame(work, ins, t, fr, stage, fmask, shadow, fw, fh)
            ins = None
        if ins:
            lt = t - ins['start']
            pa = ease(lt / 0.25); pf = ease(lt / 0.4)
            can = stage.copy()
            can = can * (1 - shadow * pf)
            sw, shh = fw, int(round(H * SCALE))
            sp = cv2.resize(fr, (sw, shh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            ty = int(round(HEAD_Y - ins['_head'] * SCALE)) + int(round(180 * (1 - pf)))
            # видео внутри рамки
            y0 = FY0; rows = np.arange(y0, H) - ty
            ok = (rows >= 0) & (rows < shh)
            part = np.zeros((fh, fw, 3), np.float32)
            part[ok] = sp[rows[ok]]
            if not ok.all(): part[~ok] = sp[np.clip(rows[~ok], 0, shh - 1)]
            a_ = fmask * pf
            can[FY0:, FX0:FX1] = can[FY0:, FX0:FX1] * (1 - a_) + part * a_
            # вырезанная голова над рамкой
            top_rows = np.arange(0, FY0) - ty
            okt = (top_rows >= 0) & (top_rows < shh)
            if okt.any():
                small = cv2.resize(fr, (256, 455), interpolation=cv2.INTER_AREA)
                m = person_mask(small)
                m = cv2.resize(m, (sw, shh), interpolation=cv2.INTER_LINEAR)
                m = np.clip((cv2.GaussianBlur(m, (0, 0), 2.0) - 0.35) / 0.3, 0, 1)
                ys = np.where(okt)[0]
                src_rows = top_rows[ys]
                pix = sp[src_rows]; al = m[src_rows][..., None] * pf
                reg = can[ys, FX0:FX1]
                can[ys, FX0:FX1] = reg * (1 - al) + pix * al
            base = base * (1 - pa) + can * pa
        ovp = work / 'ov' / f'{n:05d}.png'
        if ovp.exists():
            ov = np.asarray(Image.open(ovp).convert('RGBA'), np.float32) / 255
            if ov.shape[:2] != (H, W): ov = cv2.resize(ov, (W, H))
            al = ov[..., 3:4]
            base = base * (1 - al) + ov[..., :3] * al
        enc.stdin.write((np.clip(base, 0, 1) * 255).astype(np.uint8).tobytes())
        if n % 150 == 0: print('frame', n, '/', nf, flush=True)
    enc.stdin.close(); enc.wait()
    print('готово:', out)


def sfx_meta():
    p = SFX / 'sfx.json'
    return {k: v for k, v in json.load(open(p)).items() if not k.startswith('_')} if p.exists() else {}


def peak_db(p):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(p), '-af', 'volumedetect', '-f', 'null', '-'], capture_output=True, text=True).stderr
    for ln in r.splitlines():
        if 'max_volume' in ln: return float(ln.split(':')[1].split()[0])
    return -6.0


# Где во вставке что появляется (секунды от начала вставки) — те же тайминги, что в overlay.html.
CLICK, TAP, TICKS = 'click.wav', 'iphone_tab.mp3', 'ticks.wav'


def insert_sfx(ins):
    d = ins.get('data', {}) or {}; ty = ins['type']; ev = [(0.05, CLICK)]          # вход вставки — всегда птык
    if ty == 'title':
        ev = [(l.get('at', 0.2 + i * 0.55), CLICK) for i, l in enumerate(d.get('lines', []))]
    elif ty == 'dark_card':
        ev += [(l.get('at', 0.3 + i * 0.5), CLICK) for i, l in enumerate(d.get('lines', []))]
    elif ty == 'cards':
        ev += [(0.39, CLICK)]
    elif ty == 'marker':
        ev += [(0.6, CLICK)]
    elif ty == 'checklist':
        a0, st = d.get('first', 0.7), d.get('step', 0.5)
        ev += [(a0 + i * st, CLICK) for i in range(min(4, len(d.get('items', []))))]
    elif ty == 'chat':
        ev += [(0.4, TAP), (1.6, TAP)]
    elif ty == 'notifications':
        ev = [(n.get('at', 0.3 + i * 0.6), TAP) for i, n in enumerate(d.get('items', [])[:3])]
    elif ty == 'chart':
        ev += [(0.3, TICKS), (1.0, CLICK)]
    elif ty == 'before_after':
        ev += [(0.9, TICKS), (1.55, CLICK)]
    elif ty == 'ring':
        ev += [(0.35, TICKS), (1.75, CLICK)]
    elif ty == 'funnel':
        ev = [(0.2 + i * 0.4, CLICK) for i in range(len(d.get('rows', [])[:4]))]
    elif ty == 'price':
        ev += [(0.5, CLICK), (0.9, TAP)]
    elif ty == 'ab':
        ev += [(0.45, TICKS), (1.3, CLICK)]
    elif ty == 'calendar':
        n = len(d.get('days', [1] * 7)); ev += [(0.4, TICKS), (0.4 + (n - 1) * 0.2, CLICK)]
    elif ty == 'map':
        ev = [(0.25, CLICK), (0.55, TICKS), (1.0, CLICK)]
    elif ty == 'folder':
        ev += [(0.9, CLICK)]
    elif ty == 'timeline':
        ev += [(0.6, TICKS)]
    elif ty == 'broll':
        ev = [(0.0, CLICK)]
    return [(ins['start'] + t, f) for t, f in ev if ins['start'] + t < ins['end'] - 0.1]


def auto_sfx(plan):
    """Звуки на всё, что появляется: вставки по элементам + акцентные слова. Между звуками ≥ 0,12 с."""
    ev = []
    for ins in plan.get('inserts', []): ev += insert_sfx(ins)
    ins_at = lambda t: any(i['start'] <= t < i['end'] for i in plan.get('inserts', []))
    for c in plan.get('captions', []):
        if c.get('accent'): ev.append((c['start'], CLICK))                     # акцент — птык, в том числе внутри вставки
    ev.sort(); out = []
    for t, f in ev:
        if out and t - out[-1]['t'] < 0.12:
            if f != CLICK: out[-1] = {'t': out[-1]['t'], 'file': f}           # тап/тики важнее птыка
            continue
        out.append({'t': round(t, 3), 'file': f})
    return out


def audio_mix(work, end, plan, out, boost=0.0):
    """Голос не обрабатываем: без нормализации, компрессии и ускорения.
    Сверху звуковые эффекты: plan.sfx = "auto" (по умолчанию) или [{"t": сек, "file": "click.wav", "gain": дБ?}].
    Громкость эффекта без gain — по пику голоса: пик эффекта = пик голоса + rel из sfx/sfx.json + boost."""
    v = str(work / 'voice.wav')
    sfx = plan.get('sfx', 'auto')
    if sfx == 'auto': sfx = auto_sfx(plan)
    plan['_sfx_used'] = sfx
    if not sfx:
        sh(['ffmpeg', '-v', 'error', '-y', '-i', v, '-af', f'atrim=0:{end}', '-c:a', 'pcm_s16le', str(out)]); return
    meta = sfx_meta(); vpk = peak_db(v); pk = {}
    cmd = ['ffmpeg', '-v', 'error', '-y', '-i', v]
    fc = [f'[0:a]atrim=0:{end},asetpts=PTS-STARTPTS[v]']; labels = ['[v]']
    for i, x in enumerate(sfx):
        f = Path(x['file'])
        if not f.is_absolute(): f = SFX / f
        if not f.exists(): print('нет эффекта', f); continue
        m = meta.get(f.name, {})
        if 'gain' in x: g = x['gain']
        else:
            if f not in pk: pk[f] = peak_db(f)
            g = vpk + m.get('rel', -3) + boost - pk[f]
        on = m.get('onset', 0.0)
        cmd += ['-i', str(f)]; k = len(labels)
        ms = max(0, int(round(x['t'] * 1000)))
        fc.append(f'[{k}:a]atrim=start={on},asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo,volume={g:.1f}dB,adelay={ms}|{ms}[s{k}]')
        labels.append(f'[s{k}]')
    fc.append(''.join(labels) + f'amix=inputs={len(labels)}:normalize=0:duration=first[o]')
    sh(cmd + ['-filter_complex', ';'.join(fc), '-map', '[o]', '-c:a', 'pcm_s16le', str(out)])
    print(f'эффекты: {len(labels) - 1} шт. (пик голоса {vpk:.1f} дБ, голос не тронут)')


def cmd_sfx(a):
    d = SFX; meta = sfx_meta()
    fs = sorted(p for p in d.rglob('*') if p.suffix.lower() in ('.wav', '.mp3', '.m4a', '.aif', '.aiff', '.ogg'))
    if not fs: print('в папке sfx пока нет эффектов'); return
    for f in fs: print(f'{f.relative_to(d)}\t{ffprobe_dur(f):.2f} с\t{meta.get(f.name, {}).get("use", "")}')
    if a.work:
        plan = json.load(open(Path(a.work) / 'plan.json')); s = plan.get('sfx', 'auto')
        s = auto_sfx(plan) if s == 'auto' else s
        print(f'\nв ролике {len(s)} звуков:'); [print(f"  {x['t']:6.2f}  {x['file']}") for x in s]


def cmd_sheet(a):
    work = Path(a.work); f = a.file or str(work / 'final.mp4')
    sh(['ffmpeg', '-v', 'error', '-y', '-i', f, '-vf', 'fps=2,scale=216:-2,tile=10x5', '-frames:v', '1', str(work / 'sheet.jpg')])
    print(work / 'sheet.jpg')


def main():
    ap = argparse.ArgumentParser(description='Мобилка монтаж')
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('setup')
    p = sp.add_parser('prep'); p.add_argument('video'); p.add_argument('work')
    p = sp.add_parser('transcribe'); p.add_argument('work'); p.add_argument('--prompt', default=''); p.add_argument('--lang', default=None, help='ru или kk')
    p = sp.add_parser('cut'); p.add_argument('work'); p.add_argument('--speed', type=float, default=1.0)
    p.add_argument('--drop', nargs='*'); p.add_argument('--pad', type=float, default=0.06); p.add_argument('--gap', type=float, default=0.16)
    p.add_argument('--prompt', default=''); p.add_argument('--lang', default=None)
    p = sp.add_parser('broll'); p.add_argument('work'); p.add_argument('src', help='ссылка TikTok или путь к файлу')
    p.add_argument('--name', default='b1'); p.add_argument('--start', type=float); p.add_argument('--end', type=float)
    p = sp.add_parser('analyze'); p.add_argument('work')
    p = sp.add_parser('plan'); p.add_argument('work'); p.add_argument('--force', action='store_true')
    p = sp.add_parser('render'); p.add_argument('work'); p.add_argument('--out'); p.add_argument('--skip-overlay', action='store_true')
    p.add_argument('--sfx-boost', type=float, default=0.0, help='дБ ко всем эффектам: +3 громче, -3 тише')
    p = sp.add_parser('sfx'); p.add_argument('work', nargs='?')
    p = sp.add_parser('sheet'); p.add_argument('work'); p.add_argument('--file')
    a = ap.parse_args()
    {'setup': cmd_setup, 'prep': cmd_prep, 'transcribe': cmd_transcribe, 'cut': cmd_cut, 'analyze': cmd_analyze,
     'plan': cmd_plan, 'broll': cmd_broll, 'render': cmd_render, 'sheet': cmd_sheet, 'sfx': cmd_sfx}[a.cmd](a)


if __name__ == '__main__':
    main()
