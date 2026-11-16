"use client";

import {
  ArrowDownLeft, ArrowRight, ArrowUp, BookOpen, Check, CircleHelp, Compass,
  FileText, Layers, Library, LoaderCircle, MessageSquare, PanelLeft, Paperclip,
  Plus, Sparkles, Square, UploadCloud, X,
} from "lucide-react";
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

type Mode = "plan" | "tutor" | "quiz" | "research";
type Pending = { interrupt_id: string } & (
  | { kind: "approval"; topic: string; steps: string[] }
  | { kind: "answer" | "clarification"; question: string }
);
type Source = { source_id: string; source: string; page: number | null; excerpt: string };
type Result = {
  id: string;
  pending: Pending | null;
  declined: boolean;
  closed?: boolean;
  attempt?: { question: string; answer: string };
  plan?: { topic: string; steps: string[] };
  lesson?: { status: string; explanation: string; worked_example?: string; practice_question?: string; sources: Source[] };
  feedback?: { verdict: string; feedback: string; next_step: string };
  research?: { findings: { summary: string; source_urls: string[] }[]; uncertainties: string[] };
};
type Turn = { message: string; result: Result };
type Session = { title: string; result: Result; mode?: Mode; message?: string; history?: Turn[] };

const modes = [
  { id: "plan", title: "Study plan", description: "Map out what to learn", icon: Layers, instruction: "Create a study plan for" },
  { id: "tutor", title: "Explain a concept", description: "Work through an idea", icon: BookOpen, instruction: "Explain this concept:" },
  { id: "quiz", title: "Quiz me", description: "Check what you know", icon: CircleHelp, instruction: "Quiz me on" },
  { id: "research", title: "Deep research", description: "Investigate with sources", icon: Compass, instruction: "Research this using authoritative sources:" },
] as const;

const prompts = [
  { label: "SQL joins", detail: "Explain a concept", mode: "tutor", icon: BookOpen },
  { label: "How photosynthesis works", detail: "Build a study plan", mode: "plan", icon: Layers },
  { label: "Python decorators", detail: "Test your knowledge", mode: "quiz", icon: CircleHelp },
] as const;
const storageKey = "learn-a-topic-sessions";
const activeSessionKey = "learn-a-topic-active-session";
const endRequest = /^(?:please\s+)?(?:(?:we need to|i want to)\s+)?end\s+(?:(?:this|the)\s+)?(?:chat|session)[.!]?$/i;

async function post(action: string, payload: unknown): Promise<Result> {
  const response = await fetch(`/api/study/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const json = await response.json().catch(() => null);
  if (!response.ok || !json) {
    throw new Error(json?.error ?? "The local workflow server is unavailable. Start it and try again.");
  }
  return json;
}

function encodeFile(file: File) {
  return new Promise<{ name: string; data: string }>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.onload = () => resolve({ name: file.name, data: String(reader.result).split(",", 2)[1] });
    reader.readAsDataURL(file);
  });
}

function submitOnEnter(event: KeyboardEvent<HTMLTextAreaElement>) {
  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }
}

export default function StudyWorkspace() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("plan");
  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("beginner");
  const [files, setFiles] = useState<File[]>([]);
  const [notesOnly, setNotesOnly] = useState(false);
  const [answer, setAnswer] = useState("");
  const [followUp, setFollowUp] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const topicInput = useRef<HTMLTextAreaElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const current = sessions.find(({ result }) => result.id === active);
  const result = current?.result;
  const citedSources = [...new Map(
    [...(current?.history ?? []).map((turn) => turn.result), result]
      .flatMap((turn) => turn?.lesson?.sources ?? [])
      .map((source) => [JSON.stringify([source.source, source.page, source.excerpt]), source] as const),
  ).values()];
  const selected = modes.find((item) => item.id === mode)!;
  const activity = modes.find((item) => item.id === current?.mode);
  const webSources = [...new Set(
    [...(current?.history ?? []).map((turn) => turn.result), result]
      .flatMap((turn) => turn?.research?.findings.flatMap((finding) => finding.source_urls) ?? [])
      .filter(isWebSource),
  )];

  useEffect(() => {
    if (!viewport.current) return;
    viewport.current.scrollTop = current ? viewport.current.scrollHeight : 0;
  }, [current, busy, error]);

  useEffect(() => {
    function closeOnEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") {
        setHistoryOpen(false);
        setSourcesOpen(false);
      }
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  useEffect(() => {
    try {
      const stored = JSON.parse(sessionStorage.getItem(storageKey) ?? "[]");
      if (Array.isArray(stored)) {
        setSessions(stored);
        const savedActive = sessionStorage.getItem(activeSessionKey);
        if (stored.some((session) => session.result.id === savedActive)) setActive(savedActive);
      }
    } catch {}
    setReady(true);
  }, []);

  useEffect(() => {
    if (ready) {
      try {
        sessionStorage.setItem(storageKey, JSON.stringify(sessions.slice(0, 20)));
        if (active) sessionStorage.setItem(activeSessionKey, active);
        else sessionStorage.removeItem(activeSessionKey);
      } catch {}
    }
  }, [sessions, active, ready]);

  function newSession() {
    setActive(null);
    setTopic("");
    setAnswer("");
    setFollowUp("");
    setError("");
    setFiles([]);
    setNotesOnly(false);
    setMode("plan");
    setHistoryOpen(false);
    setSourcesOpen(false);
  }

  function addFiles(incoming: File[]) {
    setError("");
    if (incoming.some((file) => !/\.(txt|md|pdf)$/i.test(file.name) || file.size > 8 * 1024 * 1024)) {
      setError("Choose TXT, Markdown, or PDF files of up to 8 MB each.");
      return;
    }
    const unique = new Map([...files, ...incoming].map((file) => [file.name, file]));
    if (unique.size > 4) {
      setError("Attach up to four source files per session.");
      return;
    }
    setFiles([...unique.values()]);
  }

  async function start(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!topic.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const next = await post("start", {
        request: `${selected.instruction} ${topic.trim()}`,
        level, notes_only: notesOnly, files: await Promise.all(files.map(encodeFile)),
      });
      setSessions((previous) => [{ title: topic.trim(), result: next, mode }, ...previous]);
      setActive(next.id);
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function sendChatAction(action: "resume" | "follow-up" | "end", payload: object = {}, message = "") {
    if (!current || busy || current.result.closed) return;
    setBusy(true);
    setError("");
    try {
      const next = await post(action, { ...payload, id: current.result.id });
      setSessions((previous) => previous.map((session) => {
        if (session.result.id !== next.id) return session;
        if (action === "follow-up") {
          const previousTurn = { message: session.message ?? session.title, result: session.result };
          return { ...session, message, result: next, history: [...(session.history ?? []), previousTurn] };
        }
        return { ...session, result: next };
      }));
      setAnswer("");
      setFollowUp("");
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function resume(reply: string | boolean) {
    if (!result?.pending || (typeof reply === "string" && !reply.trim())) return;
    if (typeof reply === "string" && endRequest.test(reply.trim())) {
      void sendChatAction("end");
      return;
    }
    void sendChatAction("resume", { interrupt_id: result.pending.interrupt_id, reply });
  }

  function submitFollowUp(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = followUp.trim();
    if (!message || !result || result.pending) return;
    if (endRequest.test(message)) {
      void sendChatAction("end");
      return;
    }
    void sendChatAction("follow-up", { request: message }, message);
  }

  return (
    <div className="workspace">
      <button type="button" className="scrim history-scrim" hidden={!historyOpen} onClick={() => setHistoryOpen(false)} aria-label="Close session history"/>
      <aside id="session-history" className="sidebar" data-open={historyOpen} aria-label="Session history">
        <div className="sidebar-heading">
          <div className="brand"><StudyMark/><span>Learn a Topic<small>Study workspace</small></span></div>
          <button type="button" className="icon-button sidebar-close" aria-label="Close session history" onClick={() => setHistoryOpen(false)}><X size={17}/></button>
        </div>
        <button type="button" className="new-button" onClick={newSession} disabled={busy}><Plus size={17}/> New session</button>
        <nav className="session-list" aria-label="Study sessions">
          <h2>Recent sessions</h2>
          {sessions.length ? sessions.map((session) => (
            <button type="button" key={session.result.id} disabled={busy}
              className={active === session.result.id ? "session active" : "session"}
              aria-current={active === session.result.id ? "true" : undefined}
              title={session.title}
              onClick={() => { setActive(session.result.id); setAnswer(""); setFollowUp(""); setError(""); setHistoryOpen(false); }}>
              <MessageSquare size={15}/><span>{session.title}</span>
            </button>
          )) : <p className="empty-history">Your study chats will appear here.</p>}
        </nav>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="topbar-title">
            <button type="button" className="icon-button history-toggle" aria-controls="session-history" aria-expanded={historyOpen}
              aria-label="Open session history" onClick={() => { setSourcesOpen(false); setHistoryOpen(!historyOpen); }}><PanelLeft size={18}/></button>
            <span className="header-crumb">{activity?.title ?? (current ? "Study session" : "Study workspace")}</span>
            {current && <><span className="breadcrumb-divider">/</span><span className="header-topic" title={current.title}>{current.title}</span></>}
          </div>
          <div className="header-actions">
            <button type="button" className="secondary sources-toggle" aria-expanded={sourcesOpen} aria-controls="study-sources"
              onClick={() => { setHistoryOpen(false); setSourcesOpen(!sourcesOpen); }}><Library size={16}/><span>Sources</span>{citedSources.length + webSources.length > 0 && <span className="source-count">{citedSources.length + webSources.length}</span>}</button>
            {current && !result?.closed && <button type="button" className="secondary end-chat" disabled={busy}
              onClick={() => void sendChatAction("end")}><Square size={13}/> End chat</button>}
          </div>
        </header>

        <div className="content" data-sources={sourcesOpen}>
          <main className="center">
            {!current ? (
              <div className="scroll-area welcome-scroll" ref={viewport}>
                <div className="welcome">
                  <section className="intro">
                    <StudyMark large/>
                    <h1>What do you want to learn?</h1>
                    <p>Pick a study activity, then send a topic or question.</p>
                  </section>
                  <form className="composer" onSubmit={start} onDragOver={(event) => event.preventDefault()}
                    onDrop={(event) => { event.preventDefault(); if (!busy) addFiles(Array.from(event.dataTransfer.files)); }}>
                    {files.length > 0 && <ul className="files" aria-label="Attached files">{files.map((file) =>
                      <li key={file.name}><FileText size={14}/><span title={file.name}>{file.name}</span>
                        <button type="button" aria-label={"Remove " + file.name} disabled={busy} onClick={() => {
                          setFiles(files.filter((item) => item.name !== file.name)); if (files.length === 1) setNotesOnly(false);
                        }}><X size={14}/></button></li>,
                    )}</ul>}
                    <label htmlFor="topic" className="sr-only">Topic or question</label>
                    <textarea id="topic" ref={topicInput} value={topic} onChange={(event) => setTopic(event.target.value)}
                      onKeyDown={submitOnEnter} placeholder="Ask a question or type a topic…" rows={2} maxLength={3900} required disabled={busy}/>
                    <div className="composer-footer">
                      <div className="composer-controls">
                        <button type="button" className="icon-button attach-button" disabled={busy} aria-label="Add study files"
                          title="Add study files" onClick={() => fileInput.current?.click()}><Paperclip size={18}/></button>
                        <label className="select-control"><selected.icon size={15}/><span className="sr-only">Study activity</span>
                          <select aria-label="Study activity" value={mode} onChange={(event) => setMode(event.target.value as Mode)} disabled={busy}>
                            {modes.map((item) => <option key={item.id} value={item.id}>{item.title}</option>)}
                          </select>
                        </label>
                        <label className="select-control level-control"><span className="sr-only">Level</span>
                          <select aria-label="Level" value={level} onChange={(event) => setLevel(event.target.value)} disabled={busy}>
                            <option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option>
                          </select>
                        </label>
                      </div>
                      <button type="submit" className="send-button" disabled={busy || !topic.trim()} aria-label="Start session"
                        title="Start session">{busy ? <LoaderCircle className="spin" size={19}/> : <ArrowUp size={20}/>}</button>
                    </div>
                    {files.length > 0 && <label className="only"><input type="checkbox" checked={notesOnly} disabled={busy}
                      onChange={(event) => setNotesOnly(event.target.checked)}/> Use attached sources only</label>}
                  </form>
                  {error && <p className="error" role="alert">{error}</p>}
                  <section className="starter-section" aria-labelledby="starter-title">
                    <h2 id="starter-title">Try a topic</h2>
                    <div className="suggestions">{prompts.map(({ label, detail, mode: activityMode, icon: Icon }) => (
                      <button type="button" key={label} disabled={busy} onClick={() => { setTopic(label); setMode(activityMode); topicInput.current?.focus(); }}>
                        <Icon size={16}/><span><strong>{label}</strong><small>{detail}</small></span><ArrowDownLeft size={14} className="starter-arrow"/>
                      </button>
                    ))}</div>
                  </section>
                  <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
                </div>
              </div>
            ) : (
              <>
                <div className="scroll-area" ref={viewport}>
                  <div className="conversation" aria-label="Conversation">
                    <h1 className="sr-only">{current.title}</h1>
                    {[...(current.history ?? []), { message: current.message ?? current.title, result: current.result }].map((turn, index, turns) => {
                      const latest = index === turns.length - 1;
                      return <section className="study-turn" key={index} aria-label={"Turn " + (index + 1)}>
                        <p className="chat-message"><span className="sr-only">You: </span>{turn.message}</p>
                        <div className="agent-reply">
                          <div className="agent-byline"><StudyMark/><span>Learn a Topic</span></div>
                          <StudyTurn result={turn.result} busy={busy || !latest} answer={latest ? answer : undefined}
                            setAnswer={latest ? setAnswer : undefined} resume={latest ? resume : undefined}/>
                        </div>
                      </section>;
                    })}
                    {busy && <p className="work-status" role="status"><LoaderCircle className="spin" size={17}/> Continuing your session…</p>}
                    {error && <p className="error" role="alert">{error}</p>}
                  </div>
                </div>
                {result?.closed ? <p className="chat-closed" role="status">This chat has ended.</p>
                  : result && !result.pending && (
                    <div className="reply-dock">
                      <form className="composer reply-composer" onSubmit={submitFollowUp}>
                        <label htmlFor="follow-up" className="sr-only">Follow-up question</label>
                        <textarea id="follow-up" value={followUp} onChange={(event) => setFollowUp(event.target.value)}
                          onKeyDown={submitOnEnter} placeholder="Ask a follow-up question…" rows={2} maxLength={4000} required disabled={busy}/>
                        <button className="send-button" type="submit" disabled={busy || !followUp.trim()} aria-label="Send follow-up">
                          {busy ? <LoaderCircle className="spin" size={18}/> : <ArrowUp size={20}/>}
                        </button>
                      </form>
                      <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
                    </div>
                  )}
              </>
            )}
          </main>

          {!current && <input type="file" ref={fileInput} accept=".pdf,.txt,.md" multiple hidden
            onChange={(event) => { addFiles(Array.from(event.target.files ?? [])); event.target.value = ""; }}/>}
          <button type="button" className="scrim sources-scrim" hidden={!sourcesOpen} onClick={() => setSourcesOpen(false)} aria-label="Close sources"/>
          <aside className="sources" id="study-sources" hidden={!sourcesOpen} aria-label="Study sources">
            <div className="source-title"><Library size={18}/><div><h2>Study sources</h2>
              <p>{current ? "Evidence cited in this chat" : "Add material for this session"}</p></div>
              <button type="button" className="icon-button" aria-label="Close study sources" onClick={() => setSourcesOpen(false)}><X size={17}/></button>
            </div>
            {!current ? <>
              <button type="button" className="drop" disabled={busy} onClick={() => fileInput.current?.click()}
                onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); if (!busy) addFiles(Array.from(event.dataTransfer.files)); }}>
                <UploadCloud size={23}/><strong>Add study files</strong><small>PDF, TXT or Markdown</small>
              </button>
              {files.length > 0 && <p className="no-evidence">{files.length} attached</p>}
            </> : citedSources.length || webSources.length ? <div className="evidence">
              {citedSources.map((source, index) => <details key={index}><summary><FileText size={16}/><span>{source.source}{source.page ? " · p. " + source.page : ""}</span></summary><p>{source.excerpt}</p></details>)}
              {webSources.map((url) => <SourceLink key={url} url={url}/>)}
            </div> : <p className="no-evidence">Cited files and websites will appear here.</p>}
          </aside>
        </div>
      </div>
    </div>
  );
}

function StudyMark({ large = false }: { large?: boolean }) {
  return <span className={large ? "study-mark large-mark" : "study-mark"} aria-hidden="true"><BookOpen size={large ? 26 : 16}/><Sparkles className="mark-spark" size={large ? 15 : 10}/></span>;
}

function isWebSource(url: string) {
  try {
    const protocol = new URL(url).protocol;
    return protocol === "https:" || protocol === "http:";
  } catch {
    return false;
  }
}

function SourceLink({ url }: { url: string }) {
  return <a className="citation" href={url} title={url} target="_blank" rel="noreferrer"><FileText size={13}/>{new URL(url).hostname.replace(/^www\./, "")}<ArrowRight size={13}/></a>;
}

function StudyTurn({ result, busy = false, answer = "", setAnswer, resume }: {
  result: Result;
  busy?: boolean;
  answer?: string;
  setAnswer?: (answer: string) => void;
  resume?: (reply: string | boolean) => void;
}) {
  return <>
    {result?.plan && <section className="result-card">
      <h2><Layers size={20}/> Study plan</h2>
      <ol>{result.plan.steps.map((step, index) => <li key={index}>{step.replace(/^\s*\d+[.)]\s+/, "")}</li>)}</ol>
      {result.pending?.kind === "approval" && <div className="actions">
        <button type="button" className="primary" disabled={busy} onClick={() => resume?.(true)}>Approve plan <ArrowRight size={16}/></button>
        <button type="button" className="secondary" disabled={busy} onClick={() => resume?.(false)}>Decline</button>
      </div>}
      {result.declined && <p>Plan declined.</p>}
    </section>}
    {result?.lesson && <section className="result-card">
      <h2><BookOpen size={20}/> {result.lesson.status === "unverified" ? "Evidence needed" : "Lesson"}</h2>
      <p>{result.lesson.explanation}</p>
      {result.lesson.worked_example && <><h3>Worked example</h3><p className="example">{result.lesson.worked_example}</p></>}
      {!result.pending && !result.feedback && result.lesson.practice_question && <><h3>Try it yourself</h3><p>{result.lesson.practice_question}</p></>}
    </section>}
    {result?.pending && result.pending.kind !== "approval" && <form className="result-card" onSubmit={(event) => { event.preventDefault(); resume?.(answer); }}>
      <h2><CircleHelp size={20}/> {result.pending.kind === "answer" ? "Practice question" : "A quick question"}</h2>
      <label htmlFor="answer">{result.pending.question}</label>
      <textarea id="answer" className="answer" value={answer} onChange={(event) => setAnswer?.(event.target.value)}
        onKeyDown={submitOnEnter} placeholder="Write your response…" maxLength={4000} disabled={busy} required/>
      <button className="primary" type="submit" disabled={busy || !answer.trim()}>Send response <ArrowUp size={16}/></button>
    </form>}
    {result?.feedback && <section className="result-card">
      <h2><Check size={20}/> Practice feedback</h2>
      {result.attempt && <><p>{result.attempt.question}</p><h3>Your answer</h3><p>{result.attempt.answer}</p></>}
        <h3 className="verdict">{result.feedback.verdict.replaceAll("_", " ")}</h3>
      <p>{result.feedback.feedback}</p><h3>Next step</h3><p>{result.feedback.next_step}</p>
    </section>}
    {result?.research && <section className="result-card">
      <h2><Compass size={20}/> Research findings</h2>
      {result.research.findings.map((finding, index) => <div className="research-finding" key={index}>
        <p>{finding.summary}</p>
        {finding.source_urls.filter(isWebSource).map((url) =>
          <SourceLink url={url} key={url}/>,
        )}
      </div>)}
      {result.research.uncertainties.length > 0 && <><h3>What remains unclear</h3><ul>{result.research.uncertainties.map((item, index) => <li key={index}>{item}</li>)}</ul></>}
    </section>}
  </>;
}
