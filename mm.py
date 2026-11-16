#!/usr/bin/env python3
"""Мобилка монтаж — движок.

Команды (по порядку):
  setup                         — поставить зависимости
  prep VIDEO WORK               — вертикаль 1080x1920 30 fps, звук
  transcribe WORK               — слова с таймкодами (faster-whisper large-v3)
  cut WORK [--speed 1.1] [--drop 12.3-14.0 ...] [--keep 5.2]
                                — вырезать паузы и куски, собрать cut.mp4 + voice.wav, пересчитать слова
  broll WORK URL|FILE --name N [--start s --end s]
                                — живая перебивка из TikTok (или файл из чата) → broll/N.mp4
  verify WORK                   — заново распознать нарезку: паразиты, повторы, паузы, съеденные слова
  analyze WORK                  — где верх головы на каждом кадре (для субтитров и вставок)
  plan WORK                     — черновик plan.json: карточки субтитров по словам
  preview WORK                  — кадр каждой вставки (голова вырезана ровно? субтитр под лицом?) → previews.jpg
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
          (models / 'selfie_segmenter.tflite', 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite'),
          (models / 'blaze_face_short_range.tflite', 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite')]
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


# Подсказка Whisper писать речь дословно — с «ээ», «ну», повторами (иначе он их «причёсывает» и мы их не найдём).
VERBATIM = {'ru': 'Ээ, ну, я, я хотел... эм, сказать, что, ну, вот. ', 'kk': 'Ээ, мм, мен, мен айтайын... әә, яғни. '}


def transcribe(wav, out, prompt='', lang=None):
    from faster_whisper import WhisperModel
    m = WhisperModel('large-v3', device='cpu', compute_type='int8', cpu_threads=os.cpu_count() or 2)
    segs, info = m.transcribe(read_wav(wav), language=lang or None, word_timestamps=True, beam_size=5,
                              vad_filter=False, initial_prompt=VERBATIM.get(lang or 'ru', '') + (prompt or ''))
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


# ---------------------------------------------------------------- чистка речи: паразиты, запинки, повторы
FILLERS = {'э', 'ээ', 'эээ', 'эм', 'эмм', 'м', 'мм', 'ммм', 'хм', 'а-а', 'ааа', 'ә', 'әә', 'әәә', 'ы', 'ыы'}
SOFT = {'ну', 'вот', 'короче', 'типа', 'значит', 'собственно', 'енді', 'яғни'}     # решать по смыслу — только в отчёте
PUN = '.,!?:;«»"…()[]'


def norm(w):
    return w.lower().strip(PUN).strip('-—').replace('ё', 'е')


def find_junk(words):
    """[(индекс, причина)] — что выкинуть: звуки-паразиты, оборванные слова, повтор после запинки (оставляем последний дубль)."""
    drop = {}
    for i, w in enumerate(words):
        n = norm(w['w'])
        if n in FILLERS or not n: drop[i] = 'паразит «%s»' % w['w']
        elif w['w'].rstrip(PUN).endswith('-'): drop[i] = 'оборвано «%s»' % w['w']
    idx = [i for i in range(len(words)) if i not in drop]
    tok = [norm(words[i]['w']) for i in idx]
    j = 0
    while j < len(tok):
        hit = False
        for n in (6, 5, 4, 3, 2, 1):
            if j + 2 * n <= len(tok) and tok[j:j + n] == tok[j + n:j + 2 * n]:
                for k in range(j, j + n): drop[idx[k]] = 'повтор «%s»' % ' '.join(tok[j:j + n])
                j += n; hit = True; break
        if hit: continue
        # оборванный старт: «хо… хочу», «я хоте… я хотел»
        if j + 1 < len(tok) and 1 <= len(tok[j]) <= 5 and tok[j + 1] != tok[j] and tok[j + 1].startswith(tok[j]) and len(tok[j + 1]) > len(tok[j]):
            drop[idx[j]] = 'запинка «%s»' % tok[j]
        j += 1
    return sorted(drop.items())


def junk_ranges(words, junk, keep_t=()):
    """Интервалы исходника для --drop: от начала мусорного слова до начала следующего нормального."""
    bad = {i for i, _ in junk if not any(words[i]['s'] - 0.05 <= t <= words[i]['e'] + 0.05 for t in keep_t)}
    out = []
    for i in sorted(bad):
        prev_e = max([words[k]['e'] for k in range(i) if k not in bad] or [0])
        nxt = next((words[k]['s'] for k in range(i + 1, len(words)) if k not in bad), words[i]['e'] + 0.15)
        x, y = max(prev_e + 0.02, words[i]['s'] - 0.03), max(nxt - 0.04, words[i]['e'])
        if out and x <= out[-1][1] + 0.01: out[-1][1] = max(out[-1][1], y)
        else: out.append([x, y])
    return [(round(x, 3), round(y, 3)) for x, y in out], bad


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
    if not a.no_auto:
        junk = find_junk(words)
        rng, bad = junk_ranges(words, junk, a.keep or ())
        drops += rng
        if bad:
            print('авточистка (секунды исходника; ошибка — верни через --keep СЕК):')
            for i, why in junk:
                if i in bad: print(f"  {words[i]['s']:7.2f}  {why}")
        soft = [(w['s'], w['w']) for w in words if norm(w['w']) in SOFT]
        if soft: print('проверь по смыслу (слова-паразиты, если лишние — --drop):', ', '.join(f'{t:.2f} «{w}»' for t, w in soft))
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
    # не резать внутри слова: тишина посреди слова (взрывные «т», «п», «к») — склеиваем куски обратно
    merged = []
    for s_, t_ in keep:
        if merged:
            g0, g1 = merged[-1][1], s_
            if g1 - g0 < 0.6 and any(w['s'] + 0.04 < g0 and w['e'] - 0.04 > g1 for w in words):
                merged[-1][1] = t_; continue
        merged.append([s_, t_])
    keep = merged
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
    json.dump({'keep': keep, 'speed': sp_}, open(work / 'cutmap.json', 'w'))
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


# ---------------------------------------------------------------- verify: повторная проверка чистой речи
def cut_to_src(work, tau):
    m = json.load(open(work / 'cutmap.json')); x = tau * m['speed']; acc = 0.0
    for s_, t_ in m['keep']:
        if x <= acc + (t_ - s_): return s_ + x - acc
        acc += t_ - s_
    return m['keep'][-1][1] if m['keep'] else tau


def cmd_verify(a):
    """Заново распознать уже нарезанный голос и найти, что осталось: паразиты, повторы, оборванные слова, паузы, лишние звуки."""
    import difflib
    work = Path(a.work)
    exp = json.load(open(work / 'words.json')); lang = exp.get('lang', 'ru')
    chk = transcribe(work / 'voice16.wav', work / 'words_check.json', a.prompt, lang)
    issues = []; sugg = []
    junk = find_junk(chk)
    for i, why in junk:
        issues.append((chk[i]['s'], why))
    rng, _ = junk_ranges(chk, junk)
    sugg += [(x, y) for x, y in rng]
    for x, y in zip(chk, chk[1:]):
        if y['s'] - x['e'] > a.gap: issues.append((x['e'], f"пауза {y['s'] - x['e']:.2f} с"))
    for w in chk:
        if w['p'] < 0.35 and len(norm(w['w'])) > 1: issues.append((w['s'], f"нечётко «{w['w']}» — проверь на слух"))
    A = [norm(w['w']) for w in exp['words']]; B = [norm(w['w']) for w in chk]
    cm = json.load(open(work / 'cutmap.json')); acc = 0.0; joints = []
    for s_, t_ in cm['keep']:
        acc += t_ - s_; joints.append(acc / cm['speed'])
    near_joint = lambda w: any(w['s'] - 0.15 <= j <= w['e'] + 0.15 for j in joints[:-1])
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if op == 'insert':
            issues.append((chk[j1]['s'], 'лишнее в нарезке: «%s»' % ' '.join(B[j1:j2])))
            sugg.append((chk[j1]['s'] - 0.03, (chk[j2]['s'] - 0.04) if j2 < len(chk) else chk[j2 - 1]['e'] + 0.05))
        elif op == 'replace' and i2 - i1 == 1 and j2 - j1 == 1 and B[j1] and (B[j1] in A[i1] or A[i1] in B[j1]) and near_joint(chk[j1]):
            issues.append((chk[j1]['s'], 'край слова срезан: «%s» звучит как «%s» — проверь стык' % (A[i1], B[j1])))
        elif op == 'delete': issues.append((exp['words'][i1]['s'], 'съедено/обрезано: «%s»' % ' '.join(A[i1:i2])))
        elif op == 'replace' and (i2 - i1 > 1 or j2 - j1 > 1): issues.append((chk[j1]['s'], 'звучит иначе: «%s» → «%s»' % (' '.join(A[i1:i2]), ' '.join(B[j1:j2]))))
    issues.sort()
    dur = ffprobe_dur(work / 'voice.wav')
    wpm = len(chk) / max(1e-6, chk[-1]['e'] - chk[0]['s']) * 60 if chk else 0
    first = chk[0]['s'] if chk else 0
    if first > 0.25: issues.insert(0, (0.0, f'первое слово только на {first:.2f} с — срезать начало'))
    print(f'\nverify: {dur:.1f} с, темп {wpm:.0f} слов/мин')
    if not issues:
        print('ЧИСТО: лишних слов, повторов и пауз не найдено.'); return
    print('нашёл (время ролика → время исходника для --drop/--keep):')
    for t, why in issues:
        print(f'  {t:6.2f} → {cut_to_src(work, t):7.2f}  {why}')
    if sugg:
        sugg.sort(); m = []
        for x, y in sugg:
            if m and x <= m[-1][1] + 0.02: m[-1][1] = max(m[-1][1], y)
            else: m.append([x, y])
        print('предлагаю добавить к cut: --drop ' + ' '.join(f'{cut_to_src(work, x):.2f}-{cut_to_src(work, y):.2f}' for x, y in m))
    print('Исправь cut (--drop/--keep) и запусти verify ещё раз, пока не будет ЧИСТО.')


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
    """heads.json: [время, верх головы, подбородок] (px кадра 1080x1920) — для субтитров и вставок."""
    import numpy as np, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    fm = ASSETS / 'models' / 'blaze_face_short_range.tflite'
    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(fm)))) if fm.exists() else None
    work = Path(a.work); tops, chins, ts = [], [], []
    for i, f in enumerate(frames(work / 'cut.mp4', 270, 480, fps=10)):
        m = person_mask(f) > 0.5
        rows = np.where(m.sum(1) > 6)[0]
        top = rows[0] * 4 if len(rows) else 600
        chin = np.nan
        if det:
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
            if r.detections:
                b = max(r.detections, key=lambda d: d.bounding_box.width).bounding_box
                chin = (b.origin_y + b.height * 0.9) * 4         # низ рамки детектора чуть ниже подбородка
                hf = (b.origin_y - b.height * 0.55) * 4          # верх волос по лицу
                if abs(top - hf) > b.height * 4 * 0.3: top = hf  # рука на голове / шапка фона — верим лицу
        ts.append(round(i / 10, 2)); tops.append(top); chins.append(chin)
    tops = np.array(tops, float); chins = np.array(chins, float)
    if np.isnan(chins).all(): chins = tops + 560
    else:
        ok = ~np.isnan(chins); chins = np.interp(np.arange(len(chins)), np.where(ok)[0], chins[ok])
    sm = lambda v: np.convolve(np.pad(v, 2, mode='edge'), np.ones(5) / 5, 'valid')
    if det: det.close()
    res = [[t, int(x), int(c)] for t, x, c in zip(ts, sm(tops), sm(chins))]
    json.dump(res, open(work / 'heads.json', 'w'))
    print('analyze ok: верх головы', int(np.median(tops)), 'px, подбородок', int(np.median(chins)), 'px из 1920')


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
CHIN_MAX = 1460      # во вставке подбородок не ниже — под ним субтитр (до y 1600, дальше зона ника Instagram)
CAP_MAX = 1600
ANIM_END = 660       # низ полосы анимации + отступ: голова не заходит выше
_MASKS = {}


def round_mask(w, h, r, bottom=False):
    """Маска скругления: верхние углы (рамка спикера) или все четыре (карточка)."""
    import numpy as np, cv2
    key = (w, h, r, bottom)
    if key not in _MASKS:
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(m, (0, r), (w - 1, (h - 1 - r) if bottom else h - 1), 255, -1)
        for cx in (r, w - 1 - r):
            cv2.circle(m, (cx, r), r, 255, -1, lineType=cv2.LINE_AA)
            if bottom: cv2.circle(m, (cx, h - 1 - r), r, 255, -1, lineType=cv2.LINE_AA)
        _MASKS[key] = (m.astype(np.float32) / 255)[..., None]
    return _MASKS[key]


def ins_geometry(ins, mode):
    """Где стоит спикер во вставке. pop — голова вырезана и выходит над рамкой; box — голова целиком внутри рамки.
    Подбородок не ниже CHIN_MAX, чтобы субтитр встал под лицом; при крупном лице рамка в box становится уже и выше."""
    face = max(200.0, ins['_chin'] - ins['_head'])
    if mode == 'pop':
        S = SCALE; top = min(HEAD_Y, CHIN_MAX - face * S)
        if top < ANIM_END: return None                       # голове не хватает места — только box
        g = dict(mode='pop', S=S, ft=FY0, top=top)
    else:
        S = SCALE; ft = min(FY0, CHIN_MAX - face * S - 40)
        if ft < ANIM_END + 40:
            ft = ANIM_END + 40; S = min(SCALE, (CHIN_MAX - 40 - ft) / face)
        g = dict(mode='box', S=S, ft=ft, top=ft + 40)
    w = int(W * g['S']) // 2 * 2; w = min(w, FX1 - FX0)
    g['x0'] = (W - w) // 2; g['x1'] = g['x0'] + w
    g['chin'] = g['top'] + face * g['S']
    g['capb'] = int(min(g['chin'] + 150, CAP_MAX))
    return g


def head_quality(work, ins):
    """Насколько ровно вырезаются волосы (по 5 моментам вставки, по 2 соседних кадра):
    jag — рваность контура, flick — дрожание контура между кадрами, soft — доля «полупрозрачных» пикселей у волос."""
    import numpy as np, cv2
    t0, t1 = ins['start'] + 0.15, ins['end'] - 0.15
    ts = np.linspace(t0, t1, 5) if t1 > t0 else [t0]
    head = int(ins['_head']); band = slice(max(0, head - 30), min(H, head + 160))
    jag, flick, soft, edge = [], [], [], []
    for t in ts:
        outs = []
        for k, fr in enumerate(frames(work / 'cut.mp4', ss=float(t))):
            m = person_mask(cv2.resize(fr, (256, 455), interpolation=cv2.INTER_AREA))
            m = cv2.GaussianBlur(cv2.resize(m, (W, H)), (0, 0), 2.0)[band]
            cols = np.where((m > 0.5).any(0))[0]
            if len(cols) < 40: break
            y = np.array([np.argmax(m[:, c] > 0.5) for c in cols], float)
            sm = cv2.GaussianBlur(y.reshape(1, -1), (0, 0), 14).ravel()
            jag.append(float(np.mean(np.abs(y - sm))))
            edge.append(int(((m[:, :12] > 0.5).any(1) | (m[:, -12:] > 0.5).any(1)).sum()))   # рука/плечо у края — срежется ровной линией
            body = m > 0.15
            soft.append(float(((m > 0.15) & (m < 0.85)).sum() / max(1, body.sum())))
            outs.append(dict(zip(cols, y)))
            if k == 1: break
        if len(outs) == 2:
            cc = set(outs[0]) & set(outs[1])
            if cc: flick.append(float(np.mean([abs(outs[0][c] - outs[1][c]) for c in cc])))
    q = dict(jag=round(float(np.median(jag)) if jag else 99, 2), flick=round(float(np.median(flick)) if flick else 99, 2),
             soft=round(float(np.median(soft)) if soft else 1, 3), edge=int(max(edge) if edge else 0))
    q['ok'] = q['jag'] <= 4.0 and q['flick'] <= 6.0 and q['soft'] <= 0.22 and q['edge'] <= 10
    return q


def plan_inserts(work, plan, head_opt='auto', verbose=True):
    """Для каждой вставки: голова (median), подбородок, режим головы и геометрия, низ субтитра."""
    import numpy as np
    heads = plan.get('heads') or []
    def near(t, k):
        if not heads: return 600 if k == 1 else 1160
        i = min(range(len(heads)), key=lambda j: abs(heads[j][0] - t)); h = heads[i]
        return h[k] if len(h) > k else h[1] + 560
    for ins in plan.get('inserts', []):
        if ins['type'] == 'broll' and (ins.get('data') or {}).get('mode', 'full') == 'full':
            continue
        hs = [h for h in heads if ins['start'] - 0.2 <= h[0] <= ins['end']]
        ins['_head'] = float(np.median([h[1] for h in hs])) if hs else near(ins['start'], 1)
        ins['_chin'] = float(np.median([h[2] if len(h) > 2 else h[1] + 560 for h in hs])) if hs else near(ins['start'], 2)
        want = (ins.get('data') or {}).get('head') or ins.get('head') or plan.get('head') or head_opt
        g = None; why = ''
        if want in ('auto', 'pop'):
            g = ins_geometry(ins, 'pop')
            if g is None: why = 'крупное лицо — голове не хватает места над рамкой'
            elif want == 'auto':
                if '_q' not in ins: ins['_q'] = head_quality(work, ins)
                q = ins['_q']
                if not q['ok']:
                    g = None
                    why = ('рука или плечо у края кадра' if q['edge'] > 10 else 'волосы вырезаются неровно') + ' %s' % {k: v for k, v in q.items() if k != 'ok'}
        if g is None: g = ins_geometry(ins, 'box')
        ins['_g'] = g; ins['_capb'] = g['capb']
        if verbose: print(f"  вставка {ins['type']:<13} {ins['start']:6.2f}: голова {g['mode']}" + (f' ({why})' if why and g['mode'] == 'box' else '') +
                          (f", рамка {g['x1'] - g['x0']}px с y {int(g['ft'])}" if g['mode'] == 'box' else '') + f", субтитр до y {g['capb']}")


def speaker_stage(fr, ins, lt, stage):
    """Белый фон + спикер в рамке (pop: голова над рамкой) — общая сцена для всех вставок."""
    import numpy as np, cv2
    g = ins['_g']; pf = ease(lt / 0.4)
    x0, x1, ft, S = g['x0'], g['x1'], int(round(g['ft'])), g['S']
    fw, fh = x1 - x0, H - ft
    key = ('sh', x0, ft)
    if key not in _MASKS:
        s_ = np.zeros((H, W), np.float32); s_[ft + 36:, x0:x1] = 0.14
        _MASKS[key] = cv2.GaussianBlur(s_, (0, 0), 30)[..., None]
    can = stage * (1 - _MASKS[key] * pf)
    sw = max(int(round(W * S)), fw); shh = int(round(H * sw / W))
    sp = cv2.resize(fr, (sw, shh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    ox = (sw - fw) // 2                                            # видео по центру рамки
    ty = int(round(g['top'] - ins['_head'] * S)) + int(round(180 * (1 - pf)))
    rows = np.arange(ft, H) - ty
    rows = np.where(rows >= shh, 2 * shh - rows - 2, rows)            # видео короче рамки — низ отражаем, а не тянем полосами
    rows = np.clip(rows, 0, shh - 1)
    part = sp[rows][:, ox:ox + fw]
    a_ = round_mask(fw, fh, FR) * pf
    can[ft:, x0:x1] = can[ft:, x0:x1] * (1 - a_) + part * a_
    if g['mode'] == 'pop':
        top_rows = np.arange(0, ft) - ty; okt = (top_rows >= 0) & (top_rows < shh)
        if okt.any():
            m = person_mask(cv2.resize(fr, (256, 455), interpolation=cv2.INTER_AREA))
            m = np.clip((cv2.GaussianBlur(cv2.resize(m, (sw, shh)), (0, 0), 2.0) - 0.35) / 0.3, 0, 1)
            ys = np.where(okt)[0]; src = top_rows[ys]
            al = m[src][:, ox:ox + fw][..., None] * pf
            can[ys, x0:x1] = can[ys, x0:x1] * (1 - al) + sp[src][:, ox:ox + fw] * al
    return can


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


def compose(work, fr, ins, t, stage):
    """Кадр ролика во время вставки (без слоя текста)."""
    import numpy as np, cv2
    lt = t - ins['start']; d = ins.get('data', {}) or {}
    base = fr.astype(np.float32) / 255
    if ins['type'] == 'broll' and d.get('mode', 'full') == 'full':
        f = _broll_next(work, ins, t, W, H).astype(np.float32) / 255
        k = 1 + 0.05 * min(1, lt / max(ins['end'] - ins['start'], 0.1))     # наезд 1.00 → 1.05
        if k > 1.001:
            f = cv2.warpAffine(f, np.float32([[k, 0, (1 - k) * W / 2], [0, k, (1 - k) * H / 2]]), (W, H), flags=cv2.INTER_LINEAR)
        return f
    if '_g' not in ins:
        return base
    pa = ease(lt / 0.25)
    can = speaker_stage(fr, ins, lt, stage)
    if ins['type'] == 'broll':                        # карточка с живым видео над спикером
        pc = ease((lt - 0.1) / 0.35); cw, ch = BR_X1 - BR_X0, BR_Y1 - BR_Y0
        v = _broll_next(work, ins, t, cw, ch).astype(np.float32) / 255
        key = ('bsh',)
        if key not in _MASKS:
            s_ = np.zeros((H, W), np.float32); s_[BR_Y0 + 24:BR_Y1 + 24, BR_X0 + 10:BR_X1 - 10] = 0.16
            _MASKS[key] = cv2.GaussianBlur(s_, (0, 0), 26)[..., None]
        can = can * (1 - _MASKS[key] * pc)
        y0 = BR_Y0 + int(round(-40 * (1 - pc)))
        a2 = round_mask(cw, ch, BR_R, bottom=True) * pc
        can[y0:y0 + ch, BR_X0:BR_X1] = can[y0:y0 + ch, BR_X0:BR_X1] * (1 - a2) + v * a2
    return base * (1 - pa) + can * pa


def load_plan(work, head_opt='auto'):
    plan = json.load(open(work / 'plan.json'))
    plan['heads'] = json.load(open(work / 'heads.json')) if (work / 'heads.json').exists() else []
    plan['inserts'] = sorted(plan.get('inserts', []), key=lambda x: x['start'])
    print('вставки:')
    plan_inserts(work, plan, head_opt)
    return plan


def cmd_preview(a):
    """По одному кадру на каждую вставку (через 1 с после входа) → previews.jpg. Смотреть волосы, рамку, лицо."""
    import numpy as np
    from PIL import Image, ImageDraw
    work = Path(a.work).resolve(); plan = load_plan(work, a.head)
    stage = np.empty((H, W, 3), np.float32); stage[:] = np.array(STAGE, np.float32) / 255
    tiles = []
    for ins in plan['inserts']:
        t = min(ins['start'] + 1.0, ins['end'] - 0.05)
        fr = next(frames(work / 'cut.mp4', ss=t))
        img = Image.fromarray((np.clip(compose(work, fr, ins, t, stage), 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        if '_capb' in ins: dr.rectangle([160, ins['_capb'] - 100, 920, ins['_capb']], outline=(255, 60, 60), width=4)   # где будет субтитр
        img = img.resize((360, 640)); ImageDraw.Draw(img).text((8, 8), f"{ins['type']} {ins['start']:.1f}s {ins.get('_g', {}).get('mode', '')}", fill=(255, 0, 0))
        tiles.append(img)
        ins.pop('_it', None)
    if not tiles: print('вставок нет'); return
    sheet = Image.new('RGB', (370 * len(tiles), 640), 'white')
    for i, im in enumerate(tiles): sheet.paste(im, (i * 370, 0))
    sheet.save(work / 'previews.jpg', quality=88)
    print(work / 'previews.jpg', '— красная рамка: где встанет субтитр')


def cmd_render(a):
    import numpy as np, cv2
    from PIL import Image
    work = Path(a.work).resolve()
    plan = load_plan(work, a.head)
    end = plan.get('end') or ffprobe_dur(work / 'voice.wav')
    nf = int(round(end * FPS))
    export_thumbs(work, plan)
    json.dump({k: v for k, v in plan.items()}, open(work / 'plan_render.json', 'w'), ensure_ascii=False, default=str)
    if not a.skip_overlay:
        render_overlay(work, json.load(open(work / 'plan_render.json')), nf)
    mix = work / 'mix.wav'
    audio_mix(work, end, plan, mix, a.sfx_boost)
    json.dump(plan.get('_sfx_used', []), open(work / 'sfx_used.json', 'w'), ensure_ascii=False)
    ins_list = plan['inserts']
    out = a.out or str(work / 'final.mp4')
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-i', str(mix), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    stage = np.empty((H, W, 3), np.float32); stage[:] = np.array(STAGE, np.float32) / 255
    for n, fr in enumerate(frames(work / 'cut.mp4')):
        if n >= nf: break
        t = n / FPS
        ins = next((i for i in ins_list if i['start'] <= t < i['end']), None)
        base = compose(work, fr, ins, t, stage) if ins else fr.astype(np.float32) / 255
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
    p = sp.add_parser('cut'); p.add_argument('work'); p.add_argument('--speed', type=float, default=1.1)
    p.add_argument('--no-auto', action='store_true', help='не вырезать паразиты и повторы автоматически')
    p.add_argument('--keep', nargs='*', type=float, help='секунды исходника: слово в этом месте не трогать')
    p.add_argument('--drop', nargs='*'); p.add_argument('--pad', type=float, default=0.06); p.add_argument('--gap', type=float, default=0.16)
    p.add_argument('--prompt', default=''); p.add_argument('--lang', default=None)
    p = sp.add_parser('broll'); p.add_argument('work'); p.add_argument('src', help='ссылка TikTok или путь к файлу')
    p.add_argument('--name', default='b1'); p.add_argument('--start', type=float); p.add_argument('--end', type=float)
    p = sp.add_parser('verify'); p.add_argument('work'); p.add_argument('--prompt', default=''); p.add_argument('--gap', type=float, default=0.4)
    p = sp.add_parser('analyze'); p.add_argument('work')
    p = sp.add_parser('plan'); p.add_argument('work'); p.add_argument('--force', action='store_true')
    p = sp.add_parser('render'); p.add_argument('work'); p.add_argument('--out'); p.add_argument('--head', default='auto', choices=['auto', 'pop', 'box'])
    p.add_argument('--skip-overlay', action='store_true')
    p.add_argument('--sfx-boost', type=float, default=0.0, help='дБ ко всем эффектам: +3 громче, -3 тише')
    p = sp.add_parser('sfx'); p.add_argument('work', nargs='?')
    p = sp.add_parser('preview'); p.add_argument('work'); p.add_argument('--head', default='auto', choices=['auto', 'pop', 'box'])
    p = sp.add_parser('sheet'); p.add_argument('work'); p.add_argument('--file')
    a = ap.parse_args()
    {'setup': cmd_setup, 'prep': cmd_prep, 'transcribe': cmd_transcribe, 'cut': cmd_cut, 'verify': cmd_verify, 'analyze': cmd_analyze,
     'plan': cmd_plan, 'broll': cmd_broll, 'render': cmd_render, 'preview': cmd_preview, 'sheet': cmd_sheet, 'sfx': cmd_sfx}[a.cmd](a)


if __name__ == '__main__':
    main()
