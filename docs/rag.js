// Pure functions shared by the demo page and the Node parity tests.
// They mirror src/rag_qa_eval/bm25.py and src/rag_qa_eval/llm.py exactly; tests/test_demo_parity.py checks that.

export const GEMINI_HOST = 'generativelanguage.googleapis.com'
export const MODELS = [
  { id: 'gemini-3.5-flash-lite', label: 'Gemini 3.5 Flash-Lite (fastest, lowest cost)' },
  { id: 'gemini-3.8-flash', label: 'Gemini 3.8 Flash (stronger, costs more)' },
]
export const DEFAULT_MODEL = MODELS[0].id

export const PROMPT_HEADER =
  'Answer the question using only the numbered passages below. ' +
  'Reply with the shortest span that fully answers it, copied from a passage, ' +
  'followed by the passage number in square brackets, for example: Normandy [2]. ' +
  'If the passages do not contain the answer, reply exactly: UNANSWERABLE'

// ---- BM25 (same tokenizer, same Lucene-style idf as the Python implementation) ----

export function makeTokenizer(stopwords) {
  const stop = new Set(stopwords)
  return function tokenize(text, removeStopwords = true) {
    const folded = text.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase()
    const tokens = folded.match(/[a-z0-9]+/g) || []
    return removeStopwords ? tokens.filter((t) => !stop.has(t)) : tokens
  }
}

export function bm25Scores(index, queryTokens) {
  const { n, avgdl, k1, b, doc_len: docLen, postings } = index
  const scores = new Float64Array(n)
  for (const term of new Set(queryTokens)) {
    const post = postings[term]
    if (!post) continue
    const [ids, tfs] = post
    const idf = Math.log(1 + (n - ids.length + 0.5) / (ids.length + 0.5))
    for (let i = 0; i < ids.length; i++) {
      const d = ids[i]
      const norm = k1 * (1 - b + (b * docLen[d]) / (avgdl || 1))
      scores[d] += (idf * tfs[i] * (k1 + 1)) / (tfs[i] + norm)
    }
  }
  return scores
}

// Returns [[docIndex, score], ...] sorted by score desc then index asc, positive scores only.
export function bm25Search(index, queryTokens, k) {
  const scores = bm25Scores(index, queryTokens)
  const order = Array.from(scores.keys())
    .filter((i) => scores[i] > 0)
    .sort((a, b) => scores[b] - scores[a] || a - b)
  return order.slice(0, k).map((i) => [i, scores[i]])
}

// ---- Prompt and answer parsing (mirror of build_prompt / parse_llm_answer) ----

const collapse = (s) => s.split(/\s+/).filter(Boolean).join(' ')

export function buildPrompt(question, chunks) {
  const passages = chunks.map((c, i) => `[${i + 1}] ${collapse(c)}`).join('\n')
  return `${PROMPT_HEADER}\n\nPassages:\n${passages}\n\nQuestion: ${question}\nAnswer:`
}

export function parseAnswer(raw) {
  const s = (raw ?? '').trim()
  if (!s || /^unanswerable\W*$/i.test(s)) return { text: null, citations: [] }
  const citations = []
  for (const m of s.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
    for (const n of m[1].split(/\s*,\s*/)) citations.push(Number(n))
  }
  let text = s.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, '').trim()
  text = text.replace(/^(answer\s*:\s*)/i, '')
  text = text.trim().replace(/^["']+|["']+$/g, '').trim().replace(/\.+$/, '').trim()
  return { text: text || null, citations }
}

// Index (zero-based) of the first citation that points at a passage we sent, else null.
export function citedIndex(citations, nChunks) {
  const c = citations.find((n) => n >= 1 && n <= nChunks)
  return c === undefined ? null : c - 1
}

// ---- Gemini request (key goes in a header, never in the URL) ----

export function redactKey(text, key) {
  const s = String(text ?? '')
  return key ? s.split(key).join('[redacted]') : s
}

export function buildGeminiRequest({ key, model, prompt }) {
  return {
    url: `https://${GEMINI_HOST}/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0, maxOutputTokens: 256 },
      }),
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
    },
  }
}

export async function askGemini({ key, model, prompt, signal, fetchImpl = globalThis.fetch }) {
  const trimmed = (key ?? '').trim()
  if (!trimmed) throw new Error('Paste your Gemini API key first.')
  const { url, init } = buildGeminiRequest({ key: trimmed, model, prompt })
  let res
  try {
    res = await fetchImpl(url, { ...init, signal })
  } catch (err) {
    if (err?.name === 'AbortError') throw new Error('Cancelled.')
    throw new Error('Could not reach the Gemini API. Check your connection and try again.')
  }
  let data = null
  try {
    data = await res.json()
  } catch {
    data = null
  }
  if (!res.ok) {
    const message = redactKey(data?.error?.message ?? '', trimmed)
    if ((res.status === 400 && /api[ _-]?key/i.test(message)) || res.status === 401 || res.status === 403) {
      throw new Error('Google rejected the API key or the request (HTTP ' + res.status + '). Check the key and your account.')
    }
    if (res.status === 429) throw new Error('Google reports the quota or rate limit was reached (HTTP 429).')
    throw new Error(`Gemini API error (HTTP ${res.status}). ${message}`.trim())
  }
  const parts = data?.candidates?.[0]?.content?.parts ?? []
  const text = parts.map((p) => p.text ?? '').join('')
  if (!text) {
    const reason = data?.promptFeedback?.blockReason ?? 'empty response'
    throw new Error(`Gemini returned no text (${reason}).`)
  }
  return text
}
