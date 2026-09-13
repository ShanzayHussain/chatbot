import { useState, useRef, useEffect } from 'react'

const API_URL = 'http://localhost:8000/chat'

const STARTER_PROMPTS = [
  'Does TCS deliver on holidays?',
  'How long do I have to file a claim?',
  "What's TCS's head office address?",
]

function Message({ role, text, sources, refused }) {
  if (role === 'user') {
    return (
      <div className="row row--user">
        <div className="bubble bubble--user">{text}</div>
      </div>
    )
  }

  return (
    <div className="row row--bot">
      <div className={`bubble bubble--bot ${refused ? 'bubble--refused' : ''}`}>
        {refused && <span className="stamp">Outside TCS scope</span>}
        <p>{text}</p>
        {sources && sources.length > 0 && (
          <div className="sources">
            {sources.map((s) => (
              <span className="source-pill" key={s}>
                {s.replaceAll('_', ' ')}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      text: "Hi — I'm the TCS assistant. Ask me about tracking, services, rates, or policies.",
      sources: [],
      refused: false,
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage(text) {
    const trimmed = text.trim()
    if (!trimmed || loading) return

    setMessages((prev) => [...prev, { role: 'user', text: trimmed }])
    setInput('')
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed }),
      })
      if (!res.ok) throw new Error(`Server responded ${res.status}`)
      const data = await res.json()

      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: data.answer, sources: data.sources, refused: data.refused },
      ])
    } catch (err) {
      setError('Could not reach the TCS assistant backend. Is the API running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header__stub">
          <div className="header__brand">
            <span className="header__mark">TCS</span>
            <span className="header__divider" />
            <span className="header__label">Assistant</span>
          </div>
          <p className="header__sub">Tracking, services, rates &amp; policies only</p>
        </div>
        <div className="perforation" aria-hidden="true" />
      </header>

      <main className="chat" ref={scrollRef}>
        {messages.map((m, i) => (
          <Message key={i} {...m} />
        ))}

        {loading && (
          <div className="row row--bot">
            <div className="bubble bubble--bot bubble--loading">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
            </div>
          </div>
        )}

        {error && <div className="error-banner">{error}</div>}
      </main>

      {messages.length === 1 && (
        <div className="starters">
          {STARTER_PROMPTS.map((p) => (
            <button key={p} className="starter-chip" onClick={() => sendMessage(p)}>
              {p}
            </button>
          ))}
        </div>
      )}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          sendMessage(input)
        }}
      >
        <input
          className="composer__input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your shipment..."
          disabled={loading}
        />
        <button className="composer__send" type="submit" disabled={loading || !input.trim()}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path
              d="M4 12L20 4L14 20L11 13L4 12Z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </form>
    </div>
  )
}