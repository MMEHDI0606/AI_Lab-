"""app.py — RACE Reading Comprehension Quiz Generator (Streamlit UI)."""
import os
import sys
import time
import random
import traceback

import streamlit as st
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────
_BASE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_BASE)
_SRC   = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Evict any cached HuggingFace 'evaluate' module so our local one loads
for _mod in list(sys.modules.keys()):
    if _mod == 'evaluate' or _mod.startswith('evaluate.'):
        del sys.modules[_mod]

try:
    from inference import predict_answer, generate_distractors, get_hints
except Exception as _e:
    st.error(f"Failed to load inference module: {_e}")
    st.stop()

# ── Pre-load test set for random samples ───────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_test_df():
    path = os.path.join(_ROOT, 'data', 'raw', 'test.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ── Metric results (from executed notebook) ────────────────────────────────
MODEL_A_METRICS = {
    'Logistic Regression': {'bleu': 0.3027, 'rouge_1': 0.4733, 'rouge_l': 0.4664, 'meteor': 0.4072},
    'SVM (Calibrated)':    {'bleu': 0.3018, 'rouge_1': 0.4725, 'rouge_l': 0.4655, 'meteor': 0.4062},
    'Random Forest':       {'bleu': 0.2573, 'rouge_1': 0.4268, 'rouge_l': 0.4199, 'meteor': 0.3595},
    'Ensemble (Soft Vote)':{'bleu': 0.2735, 'rouge_1': 0.4469, 'rouge_l': 0.4401, 'meteor': 0.3787},
}
MODEL_B_METRICS = {
    'Distractor Generation': {'bleu': 0.0034, 'rouge_1': 0.0720, 'rouge_l': 0.0708, 'meteor': 0.0289},
}
UNSUP_METRICS = {
    'K-Means Silhouette':    0.0947,
    'K-Means Purity':        0.7469,
    'GMM Silhouette':        0.0130,
    'GMM Log-Likelihood': 5208.3940,
    'Label Propagation F1':  0.4303,
}

# ── Session state init ──────────────────────────────────────────────────────
def _init_state():
    defaults = {
        'screen':          'input',
        'article':         '',
        'question':        '',
        'options':         [],
        'correct_letter':  '',
        'distractors':     [],
        'hints':           [],
        'chosen':          None,
        'checked':         False,
        'hints_opened':    set(),
        'reveal_shown':    False,
        'log':             [],
        'total_answered':  0,
        'total_bleu':      [],
        'inference_times': [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='RACE Quiz Generator',
    page_icon='📚',
    layout='wide',
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.title('📚 RACE Quiz Generator')
    st.markdown('---')
    nav = st.radio(
        'Navigate',
        ['Article Input', 'Quiz', 'Hints', 'Analytics'],
        index=['input', 'quiz', 'hints', 'analytics'].index(st.session_state.screen)
              if st.session_state.screen in ['input', 'quiz', 'hints', 'analytics'] else 0,
    )
    screen_map = {
        'Article Input': 'input',
        'Quiz':          'quiz',
        'Hints':         'hints',
        'Analytics':     'analytics',
    }
    st.session_state.screen = screen_map[nav]
    st.markdown('---')
    st.caption('⚠️ AI-generated content — errors possible. Not for use in real exams without human review.')

# ═══════════════════════════════════════════════════════════════
# SCREEN 1 — Article Input
# ═══════════════════════════════════════════════════════════════
if st.session_state.screen == 'input':
    st.title('Reading Comprehension Quiz Generator')
    st.subheader('Step 1 — Provide a reading passage')

    col1, col2 = st.columns([3, 1])
    with col1:
        article_input = st.text_area(
            'Paste a reading passage:',
            value=st.session_state.article,
            height=300,
            placeholder='Paste any English passage here...',
        )
    with col2:
        st.write('')
        st.write('')
        if st.button('🎲 Load random RACE sample'):
            test_df = _load_test_df()
            if test_df is not None:
                row = test_df.sample(1, random_state=random.randint(0, 9999)).iloc[0]
                st.session_state.article         = str(row['article'])
                st.session_state.question        = str(row['question'])
                st.session_state.correct_letter  = str(row['answer'])
                article_input = st.session_state.article
                st.rerun()
            else:
                st.error('test.csv not found. Place data files in data/raw/.')

    # Clear stale RACE question/answer if the article was manually edited
    if article_input != st.session_state.article:
        st.session_state.question = ''
        st.session_state.correct_letter = ''
    st.session_state.article = article_input

    if st.button('🚀 Generate Quiz', type='primary'):
        if not st.session_state.article.strip():
            st.error('Please paste a passage or load a random sample first.')
        else:
            with st.spinner('Generating quiz...'):
                t0 = time.time()
                try:
                    article = st.session_state.article

                    # Try to find a matching RACE row first
                    test_df = _load_test_df()
                    matched_row = None
                    if test_df is not None:
                        match = test_df[test_df['article'] == article]
                        if not match.empty:
                            matched_row = match.iloc[0]

                    if matched_row is not None:
                        # ── RACE sample path: use real question + real options ──
                        question       = str(matched_row['question'])
                        opts           = [str(matched_row['A']), str(matched_row['B']),
                                          str(matched_row['C']), str(matched_row['D'])]
                        correct_letter = str(matched_row['answer'])
                        correct_answer = opts['ABCD'.index(correct_letter)]
                        distractors    = [o for o in opts if o != correct_answer]
                    else:
                        # ── Custom article path: extract a meaningful answer phrase ──
                        # Use the most content-rich sentence fragment (5-8 words, not opening words)
                        question = st.session_state.question or 'What is the main topic discussed in the passage?'
                        import re as _re
                        sents = [s.strip() for s in _re.split(r'[.!?]', article) if len(s.split()) >= 6]
                        if sents:
                            # Pick the sentence with the most unique content words
                            stop = {'the','a','an','is','was','are','were','of','in','to','and','or','it','that','this','for','with','by','on','at','from','as','but','not','be','have','has','had','they','their','which','who','what'}
                            best = max(sents, key=lambda s: len(set(s.lower().split()) - stop))
                            words = [w for w in best.split() if w.lower() not in stop]
                            # Take a 3-5 word span from the middle of the best sentence
                            mid = len(words) // 2
                            span = words[max(0, mid-1):mid+3]
                            correct_answer = ' '.join(span) if span else best.split()[0]
                        else:
                            words = article.split()
                            correct_answer = words[len(words)//2] if words else 'unknown'
                        distractors = generate_distractors(article, question, correct_answer)
                        opts = [correct_answer] + distractors[:3]
                        random.shuffle(opts)
                        correct_letter = 'ABCD'[opts.index(correct_answer)]

                    hints   = get_hints(article, question)
                    elapsed = time.time() - t0

                    st.session_state.options        = opts
                    st.session_state.correct_letter = correct_letter
                    st.session_state.question       = question
                    st.session_state.distractors    = distractors
                    st.session_state.hints          = hints
                    st.session_state.chosen         = None
                    st.session_state.checked        = False
                    st.session_state.hints_opened   = set()
                    st.session_state.reveal_shown   = False
                    st.session_state.inference_times.append(elapsed)
                    st.session_state.screen = 'quiz'
                    st.rerun()
                except Exception:
                    st.error(f'Inference failed:\n```\n{traceback.format_exc()}\n```')

# ═══════════════════════════════════════════════════════════════
# SCREEN 2 — Quiz View
# ═══════════════════════════════════════════════════════════════
elif st.session_state.screen == 'quiz':
    st.title('Quiz')
    st.caption('⚠️ AI-generated — errors possible')

    if not st.session_state.options:
        st.warning('No quiz loaded. Go back to Article Input to generate one.')
        if st.button('← Back to Input'):
            st.session_state.screen = 'input'
            st.rerun()
    else:
        st.subheader('Question')
        st.write(st.session_state.question or '*(AI-generated question — see passage above)*')

        with st.expander('Show passage', expanded=False):
            st.write(st.session_state.article[:1500] + ('...' if len(st.session_state.article) > 1500 else ''))

        opts = st.session_state.options
        choice = st.radio(
            'Choose the correct answer:',
            options=['A', 'B', 'C', 'D'],
            format_func=lambda x: f'{x}. {opts["ABCD".index(x)]}',
            index=None if st.session_state.chosen is None else 'ABCD'.index(st.session_state.chosen),
        )
        if choice:
            st.session_state.chosen = choice

        col_check, col_hint, col_back = st.columns([1, 1, 1])
        with col_check:
            if st.button('✅ Check Answer', type='primary'):
                if not st.session_state.chosen:
                    st.warning('Please select an option first.')
                else:
                    st.session_state.checked = True
                    st.session_state.total_answered += 1
                    st.rerun()
        with col_hint:
            if st.button('💡 Get Hints'):
                st.session_state.screen = 'hints'
                st.rerun()
        with col_back:
            if st.button('← New Question'):
                st.session_state.screen = 'input'
                st.rerun()

        if st.session_state.checked and st.session_state.chosen:
            correct = st.session_state.correct_letter
            chosen  = st.session_state.chosen
            if chosen == correct:
                st.success('🎉 Correct!')
            else:
                st.error(f'❌ Incorrect — correct answer was **{correct}. {opts["ABCD".index(correct)]}**')

            with st.expander('Why?'):
                # Show cosine-style explanation using keyword overlap
                correct_opt = opts['ABCD'.index(correct)]
                art_words = set(st.session_state.article.lower().split())
                opt_words = set(correct_opt.lower().split())
                shared = art_words & opt_words
                st.write(
                    f'The correct option shares **{len(shared)} keywords** with the passage, '
                    f'including: *{", ".join(list(shared)[:8])}*.'
                )

# ═══════════════════════════════════════════════════════════════
# SCREEN 3 — Hint Panel
# ═══════════════════════════════════════════════════════════════
elif st.session_state.screen == 'hints':
    st.title('Hints')

    if not st.session_state.hints:
        st.warning('No hints available. Generate a quiz first.')
        if st.button('← Back to Input'):
            st.session_state.screen = 'input'
            st.rerun()
    else:
        st.write('Open hints one by one — from general to specific.')

        for i, (label, hint) in enumerate([
            ('Hint 1 — General clue',   st.session_state.hints[0]),
            ('Hint 2 — More specific',  st.session_state.hints[1]),
            ('Hint 3 — Near-explicit',  st.session_state.hints[2]),
        ]):
            with st.expander(label, expanded=False):
                st.write(hint)
                st.session_state.hints_opened.add(i)

        all_opened = len(st.session_state.hints_opened) >= 3
        if all_opened:
            if st.button('🔓 Reveal Answer', type='primary'):
                st.session_state.reveal_shown = True
                st.rerun()

        if st.session_state.reveal_shown:
            correct = st.session_state.correct_letter
            opts    = st.session_state.options
            if correct and opts:
                st.info(f'Answer: **{correct}. {opts["ABCD".index(correct)]}**')

        st.markdown('---')
        if st.button('← Back to Quiz'):
            st.session_state.screen = 'quiz'
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# SCREEN 4 — Analytics Dashboard
# ═══════════════════════════════════════════════════════════════
elif st.session_state.screen == 'analytics':
    st.title('Analytics Dashboard')

    # ── Model A Performance ──────────────────────────────────────
    st.subheader('Model A — Answer Verification')
    st.caption('BLEU / ROUGE-1 / ROUGE-L / METEOR on test set (from notebook)')

    rows_a = []
    for model, m in MODEL_A_METRICS.items():
        rows_a.append({'Model': model, 'BLEU': m['bleu'], 'ROUGE-1': m['rouge_1'],
                        'ROUGE-L': m['rouge_l'], 'METEOR': m['meteor']})
    df_a = pd.DataFrame(rows_a).set_index('Model')
    st.dataframe(df_a.style.format('{:.4f}'), use_container_width=True)

    st.bar_chart(df_a[['BLEU', 'ROUGE-L', 'METEOR']])

    # ── Model B Performance ──────────────────────────────────────
    st.subheader('Model B — Distractor Generation')
    rows_b = []
    for task, m in MODEL_B_METRICS.items():
        rows_b.append({'Task': task, 'BLEU': m['bleu'], 'ROUGE-1': m['rouge_1'],
                        'ROUGE-L': m['rouge_l'], 'METEOR': m['meteor']})
    df_b = pd.DataFrame(rows_b).set_index('Task')
    st.dataframe(df_b.style.format('{:.4f}'), use_container_width=True)

    # ── Unsupervised metrics ─────────────────────────────────────
    st.subheader('Unsupervised / Semi-Supervised (diagnostics)')
    df_u = pd.DataFrame([
        {'Metric': k, 'Value': v} for k, v in UNSUP_METRICS.items()
    ]).set_index('Metric')
    st.dataframe(df_u.style.format('{:.4f}'), use_container_width=True)

    # ── Session Stats ────────────────────────────────────────────
    st.subheader('Session Stats')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric('Questions Answered', st.session_state.total_answered)
    with c2:
        avg_t = (
            sum(st.session_state.inference_times) / len(st.session_state.inference_times) * 1000
            if st.session_state.inference_times else 0.0
        )
        st.metric('Avg Inference Time', f'{avg_t:.0f} ms')
    with c3:
        st.metric('Sessions with Hints', len(st.session_state.hints_opened))

    if st.session_state.log:
        if st.button('📥 Export session log to CSV'):
            log_df = pd.DataFrame(st.session_state.log)
            csv = log_df.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', csv, 'session_log.csv', 'text/csv')
