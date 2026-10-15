import {
  MODELS,
  DEFAULT_MODEL,
  makeTokenizer,
  bm25Search,
  buildPrompt,
  parseAnswer,
  citedIndex,
  askGemini,
  redactKey,
} from './rag.js'

const TOP_K = 3

// The key lives only in this variable. It is never written to storage, the DOM, the URL or the console.
let apiKey = ''
let corpus = null
let index = null
let examples = null
let tokenize = null
let busy = false

const $ = (id) => document.getElementById(id)

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag)
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') node.className = v
    else if (k === 'text') node.textContent = v
    else node.setAttribute(k, v)
  }
  for (const c of children) node.append(c)
  return node
}

async function loadJson(path) {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`Could not load ${path}`)
  return res.json()
}

// ---- mode ----

function setMode() {
  const live = apiKey !== ''
  const badge = $('mode-badge')
  badge.textContent = live
    ? 'Live mode: your key is held in memory, answers come from Gemini'
    : 'Mock mode: precomputed examples, no key'
  badge.className = `badge ${live ? 'badge-live' : 'badge-mock'}`
  $('clear-key').hidden = !live
  $('ask-hint').textContent = live
    ? 'Your question and the retrieved passages are sent to Google with your key.'
    : 'Without a key you get the retrieved passages only. For an answer, pick a precomputed example or add your key.'
}

// ---- retrieval over the built-in sample corpus (in the browser) ----

function retrieve(question) {
  const hits = bm25Search(index, tokenize(question), TOP_K)
  return hits.map(([i, score], r) => ({ rank: r + 1, title: corpus.chunks[i].title, text: corpus.chunks[i].text, score }))
}

// ---- rendering ----

function highlight(text, needle) {
  const frag = document.createDocumentFragment()
  const at = needle ? text.toLowerCase().indexOf(needle.toLowerCase()) : -1
  if (at < 0) {
    frag.append(text)
    return frag
  }
  frag.append(text.slice(0, at), el('mark', { text: text.slice(at, at + needle.length) }), text.slice(at + needle.length))
  return frag
}

function renderResult({ question, tag, live, answer, citedRank, chunks, notes = [], outcome = null, error = null }) {
  const box = $('result')
  box.hidden = false
  box.replaceChildren()
  box.append(el('h2', {}, 'Result ', el('span', { class: `tag${live ? ' live' : ''}`, text: tag })))
  box.append(el('p', { class: 'question', text: question }))
  if (error) {
    box.append(el('p', { class: 'error', role: 'alert', text: error }))
  } else if (answer === undefined) {
    // retrieval only
  } else if (answer === null) {
    box.append(el('p', { class: 'answer none', text: 'No answer: the system abstained because the passages do not answer this question.' }))
  } else {
    const p = el('p', { class: 'answer' }, answer)
    if (citedRank) p.append(' ', el('span', { class: 'cite', text: `[${citedRank}]` }))
    box.append(p)
  }
  for (const n of notes) box.append(el('p', { class: 'meta', text: n }))
  if (chunks.length === 0) {
    if (!error) box.append(el('p', { class: 'meta', text: 'No passage in the sample corpus shares a word with this question.' }))
  } else {
    box.append(el('h3', { text: `Retrieved passages (top ${chunks.length})` }))
    const list = el('ol', { class: 'passages' })
    for (const c of chunks) {
      const cited = citedRank === c.rank
      const li = el('li', { class: `passage${cited ? ' cited' : ''}`, id: `passage-${c.rank}` })
      li.append(el('div', { class: 'head' }, el('span', { class: 'n', text: `[${c.rank}]` }), el('span', { text: c.title.replaceAll('_', ' ') }), cited ? el('span', { text: 'cited' }) : ''))
      const p = el('p')
      p.append(cited && answer ? highlight(c.text, answer) : c.text)
      li.append(p)
      list.append(li)
    }
    box.append(list)
  }
  if (outcome) box.append(outcome)
  box.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
}

// ---- actions ----

function showExample(ex, button) {
  for (const b of document.querySelectorAll('.chip')) b.setAttribute('aria-pressed', String(b === button))
  const gold = ex.gold.length ? ex.gold.join(' | ') : '(none: the question is unanswerable)'
  const outcome = el(
    'div',
    { class: 'outcome' },
    el('div', {}, el('strong', { text: 'Evaluation outcome: ' }), ex.outcome_label),
    el('div', {}, el('strong', { text: 'Gold answer: ' }), gold),
  )
  renderResult({
    question: ex.question,
    tag: 'Precomputed, not a live model',
    live: false,
    answer: ex.answer,
    citedRank: ex.cited_rank,
    chunks: ex.retrieved,
    notes: [
      `Produced by the committed evaluation run of ${examples.meta.date}: ${examples.meta.pipeline}. Retrieval for this example ran over the full 35-article corpus, so the passages may come from articles that are not in this page's sample.`,
    ],
    outcome,
  })
}

async function ask(question) {
  const q = question.trim()
  if (!q || busy) return
  const chunks = retrieve(q)
  if (apiKey === '') {
    renderResult({
      question: q,
      tag: 'Retrieval only (no key)',
      live: false,
      chunks,
      notes: ['Retrieval ran in your browser (BM25 over the sample corpus). No key is set, so no answer is generated. Pick a precomputed example or paste your own Gemini key.'],
    })
    return
  }
  if (chunks.length === 0) {
    renderResult({ question: q, tag: 'Live', live: true, chunks, notes: ['Nothing was sent to Google because no passage matched.'] })
    return
  }
  busy = true
  $('ask').disabled = true
  const model = $('model').value
  try {
    const raw = await askGemini({ key: apiKey, model, prompt: buildPrompt(q, chunks.map((c) => c.text)) })
    const parsed = parseAnswer(raw)
    const idx = parsed.text === null ? null : citedIndex(parsed.citations, chunks.length)
    renderResult({
      question: q,
      tag: `Answer from ${model}`,
      live: true,
      answer: parsed.text,
      citedRank: idx === null ? null : idx + 1,
      chunks,
      notes: [
        idx === null && parsed.text !== null
          ? 'The model did not cite one of the passages above, so treat this answer with extra care.'
          : 'Generated by a language model: check the cited passage before relying on it.',
      ],
    })
  } catch (err) {
    renderResult({ question: q, tag: 'Live', live: true, chunks, error: redactKey(err?.message ?? 'Request failed.', apiKey) })
  } finally {
    busy = false
    $('ask').disabled = false
  }
}

function buildExamples() {
  const wrap = $('examples')
  wrap.replaceChildren()
  $('examples-note').textContent = `Precomputed from the committed evaluation of ${examples.meta.date}. They show real outcomes, including failures; none of them calls a model when you click.`
  for (const ex of examples.examples) {
    const b = el('button', { type: 'button', class: 'chip', 'aria-pressed': 'false', text: ex.question })
    b.addEventListener('click', () => {
      $('question').value = ex.question
      showExample(ex, b)
    })
    wrap.append(b)
  }
}

async function init() {
  const sel = $('model')
  for (const m of MODELS) sel.append(el('option', { value: m.id, text: m.label }))
  sel.value = DEFAULT_MODEL
  setMode()
  $('key-form').addEventListener('submit', (ev) => {
    ev.preventDefault()
    const input = $('key-input')
    const v = input.value.trim()
    if (!v) return
    apiKey = v
    input.value = ''
    setMode()
  })
  $('clear-key').addEventListener('click', () => {
    apiKey = ''
    $('key-input').value = ''
    setMode()
  })
  $('ask-form').addEventListener('submit', (ev) => {
    ev.preventDefault()
    if (!corpus) return
    for (const b of document.querySelectorAll('.chip')) b.setAttribute('aria-pressed', 'false')
    ask($('question').value)
  })
  try {
    const [c, i, e] = await Promise.all([loadJson('data/sample_corpus.json'), loadJson('data/bm25_index.json'), loadJson('data/examples.json')])
    corpus = c
    index = i.index
    tokenize = makeTokenizer(i.stopwords)
    examples = e
    $('question').placeholder = 'Type a question about the articles listed below'
    $('attribution').textContent = c.meta.attribution
    buildExamples()
  } catch {
    renderResult({ question: 'Demo data', tag: 'Error', live: false, chunks: [], error: 'Could not load the demo data files.' })
  }
}

init()
