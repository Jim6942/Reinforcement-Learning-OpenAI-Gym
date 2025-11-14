import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./ui.css";

type Mode = "human" | "agent" | "duel";

type NewSession = {
  session_id: string;
  obs: number[];
  frame: string;
  done: boolean;
};

type StepResp = {
  obs: number[];
  reward?: number;
  done: boolean;
  frame: string;
  action?: number;
  steps?: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

function useKeys() {
  const keys = useRef<Record<string, boolean>>({});
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      keys.current[e.key] = e.type === "keydown";
    };
    window.addEventListener("keydown", h);
    window.addEventListener("keyup", h);
    return () => {
      window.removeEventListener("keydown", h);
      window.removeEventListener("keyup", h);
    };
  }, []);
  return keys;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("human");
  const [status, setStatus] = useState("Booting…");

  const [sid, setSid] = useState<string | null>(null);
  const [img, setImg] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [stepR, setStepR] = useState(0);
  const [epR, setEpR] = useState(0);

  const [sidH, setSidH] = useState<string | null>(null);
  const [sidA, setSidA] = useState<string | null>(null);
  const [imgH, setImgH] = useState<string | null>(null);
  const [imgA, setImgA] = useState<string | null>(null);
  const [doneH, setDoneH] = useState(false);
  const [doneA, setDoneA] = useState(false);
  const [epRH, setEpRH] = useState(0);
  const [epRA, setEpRA] = useState(0);
  const [duelSeed, setDuelSeed] = useState<number | null>(null);

  const [fps, setFps] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);

  const [autoAgentDemo, setAutoAgentDemo] = useState(false);

  const keys = useKeys();
  const lastAction = useRef(0);
  const running = useRef(true);
  const singleLoopBusy = useRef(false);
  const duelLoopBusy = useRef(false);

  const http = axios.create({
    baseURL: API_BASE,
    timeout: 40000,
    headers: { "Content-Type": "application/json" },
  });

  const computeAction = () => {
    const A = !!keys.current["a"] || !!keys.current["A"];
    const W = !!keys.current["w"] || !!keys.current["W"];
    const D = !!keys.current["d"] || !!keys.current["D"];
    if (W) lastAction.current = 2;
    else if (D && !A) lastAction.current = 3;
    else if (A && !D) lastAction.current = 1;
    else lastAction.current = 0;
    return lastAction.current;
  };

  const createSession = async (seed?: number): Promise<NewSession> => {
    let attempt = 0;
    let delay = 400;
    while (attempt < 4) {
      try {
        const { data } = await http.post<NewSession>("/session/new", {
          env_id: "LunarLander-v3",
          seed,
          render_w: 400,
          render_h: 300,
        });
        return data;
      } catch {
        attempt += 1;
        if (attempt >= 4) throw new Error("init failed");
        await new Promise((r) => setTimeout(r, delay));
        delay *= 2;
      }
    }
    throw new Error("unreachable");
  };

  const repeatRef = useRef(1);
  const targetMs = 55;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

  useEffect(() => {
    running.current = true;
    const init = async () => {
      try {
        setStatus("Creating session…");
        const data = await createSession();
        setSid(data.session_id);
        setImg(data.frame);
        setDone(false);
        setStepR(0);
        setEpR(0);
        lastAction.current = 0;
        repeatRef.current = 1;
        setStatus("Ready");
      } catch (e: any) {
        setStatus(e?.message ?? "Init error");
      }
    };
    void init();
    return () => {
      running.current = false;
    };
  }, []);

  useEffect(() => {
    if (mode === "duel") return;
    if (!sid || done) return;
    if (singleLoopBusy.current) return;
    singleLoopBusy.current = true;
    let cancel = false;
    const loop = async () => {
      while (running.current && !cancel && sid && !done && mode !== "duel") {
        try {
          const t0 = performance.now();
          const repeat = repeatRef.current;
          let resp: StepResp;
          if (mode === "human") {
            const action = computeAction();
            const { data } = await http.post<StepResp>("/step", { session_id: sid, action, repeat });
            resp = data;
          } else {
            const { data } = await http.post<StepResp>("/agent_step", { session_id: sid, repeat });
            resp = data;
          }
          const dt = performance.now() - t0;
          setLatencyMs(dt);
          const used = resp.steps ?? repeat;
          const est = used > 0 && dt > 0 ? Math.round((used * 1000) / dt) : 0;
          setFps(est);
          setImg(resp.frame);
          if (typeof resp.reward === "number") {
            setStepR(resp.reward);
            setEpR((x) => x + resp.reward!);
          }
          if (resp.done) {
            if (mode === "agent" && autoAgentDemo && sid) {
              try {
                setStatus("Agent demo: resetting…");
                const { data: resetData } = await http.post<{ frame: string; done: boolean }>("/reset", {
                  session_id: sid,
                });
                setImg(resetData.frame);
                setDone(false);
                setStepR(0);
                setEpR(0);
                lastAction.current = 0;
                repeatRef.current = 1;
                setStatus("Agent demo: running…");
                continue;
              } catch (err: any) {
                setDone(true);
                setStatus(err?.message ?? "Auto reset failed");
                break;
              }
            } else {
              setDone(true);
              setStatus("Episode finished");
              break;
            }
          } else {
            setStatus("Running…");
          }
          const ideal = targetMs;
          if (dt > ideal * 1.25) repeatRef.current = clamp(repeat + 1, 1, 6);
          else if (dt < ideal * 0.75) repeatRef.current = clamp(repeat - 1, 1, 6);
        } catch (e: any) {
          const d = e?.response?.data?.detail ?? e?.message ?? "unknown";
          if (String(d).toLowerCase().includes("session not found")) {
            try {
              setStatus("Recovering session…");
              const s = await createSession();
              setSid(s.session_id);
              setImg(s.frame);
              setDone(false);
              setStepR(0);
              setEpR(0);
              repeatRef.current = 1;
              continue;
            } catch (er: any) {
              setStatus(er?.message ?? "Recovery failed");
              break;
            }
          } else {
            setStatus(`Loop error: ${d}`);
            await new Promise((r) => setTimeout(r, 120));
          }
        }
      }
      singleLoopBusy.current = false;
    };
    void loop();
    return () => {
      cancel = true;
      singleLoopBusy.current = false;
    };
  }, [mode, sid, done, autoAgentDemo]);

  useEffect(() => {
    if (mode !== "duel") return;
    if (!sidH || !sidA) return;
    if (duelLoopBusy.current) return;
    duelLoopBusy.current = true;
    let cancel = false;
    const loop = async () => {
      while (running.current && !cancel && mode === "duel" && sidH && sidA && !(doneH && doneA)) {
        try {
          const t0 = performance.now();
          const repeat = repeatRef.current;
          const reqs: Promise<{ data: StepResp }>[] = [];
          if (!doneH) {
            const action = computeAction();
            reqs.push(http.post<StepResp>("/step", { session_id: sidH, action, repeat }));
          }
          if (!doneA) {
            reqs.push(http.post<StepResp>("/agent_step", { session_id: sidA, repeat }));
          }
          const res = await Promise.all(reqs);
          const dt = performance.now() - t0;
          setLatencyMs(dt);
          const used = repeat;
          const est = used > 0 && dt > 0 ? Math.round((used * 1000) / dt) : 0;
          setFps(est);
          let i = 0;
          if (!doneH) {
            const { data } = res[i++];
            setImgH(data.frame);
            if (typeof data.reward === "number") setEpRH((r) => r + data.reward!);
            if (data.done) setDoneH(true);
          }
          if (!doneA) {
            const { data } = res[i++];
            setImgA(data.frame);
            if (typeof data.reward === "number") setEpRA((r) => r + data.reward!);
            if (data.done) setDoneA(true);
          }
          setStatus(doneH && doneA ? "Duel finished" : "Duel running…");
        } catch (e: any) {
          const d = e?.response?.data?.detail ?? e?.message ?? "unknown";
          if (String(d).toLowerCase().includes("session not found")) {
            try {
              setStatus("Recovering duel…");
              const seed = duelSeed ?? Math.floor(Math.random() * 1e9);
              const [h, a] = await Promise.all([createSession(seed), createSession(seed)]);
              setSidH(h.session_id);
              setImgH(h.frame);
              setDoneH(false);
              setEpRH(0);
              setSidA(a.session_id);
              setImgA(a.frame);
              setDoneA(false);
              setEpRA(0);
              repeatRef.current = 1;
              continue;
            } catch (er: any) {
              setStatus(er?.message ?? "Recovery failed");
              break;
            }
          } else {
            setStatus(`Loop error: ${d}`);
            await new Promise((r) => setTimeout(r, 120));
          }
        }
      }
      duelLoopBusy.current = false;
    };
    void loop();
    return () => {
      cancel = true;
      duelLoopBusy.current = false;
    };
  }, [mode, sidH, sidA, doneH, doneA, duelSeed]);

  const restartSingle = async () => {
    if (!sid) return;
    setStatus("Resetting…");
    const { data } = await http.post<{ frame: string; done: boolean }>("/reset", { session_id: sid });
    setImg(data.frame);
    setDone(false);
    setStepR(0);
    setEpR(0);
    lastAction.current = 0;
    repeatRef.current = 1;
    setStatus("Ready");
  };

  const startDuel = async () => {
    setStatus("Creating duel sessions…");
    setImgH(null);
    setImgA(null);
    setDoneH(false);
    setDoneA(false);
    setEpRH(0);
    setEpRA(0);
    const seed = Math.floor(Math.random() * 1e9);
    setDuelSeed(seed);
    const [h, a] = await Promise.all([createSession(seed), createSession(seed)]);
    setSidH(h.session_id);
    setImgH(h.frame);
    setSidA(a.session_id);
    setImgA(a.frame);
    setMode("duel");
    repeatRef.current = 1;
    setStatus("Duel ready");
  };

  const restartDuel = async () => {
    if (!sidH || !sidA) return;
    setStatus("Resetting duel…");
    const seed = Math.floor(Math.random() * 1e9);
    setDuelSeed(seed);
    const [h, a] = await Promise.all([createSession(seed), createSession(seed)]);
    setSidH(h.session_id);
    setImgH(h.frame);
    setDoneH(false);
    setEpRH(0);
    setSidA(a.session_id);
    setImgA(a.frame);
    setDoneA(false);
    setEpRA(0);
    repeatRef.current = 1;
    setStatus("Duel ready");
  };

  const duelOver = mode === "duel" && doneH && doneA;
  const winner = duelOver ? (Math.abs(epRH - epRA) < 1e-6 ? "Tie" : epRH > epRA ? "Human" : "Agent") : "";

  return (
    <div className="app">
      <h1 className="title">LunarLander — Gym Session</h1>

      <div className="toolbar">
        <div className="left">
          <button className="btn" disabled={mode === "human"} onClick={() => setMode("human")}>
            Human
          </button>
          <button className="btn" disabled={mode === "agent"} onClick={() => setMode("agent")}>
            Agent
          </button>
          <button className="btn" onClick={restartSingle} disabled={mode === "duel"}>
            Restart
          </button>
        </div>
        <div className="status">
          Status: <strong>{status}</strong>
        </div>
        <div className="right">
          <button
            className="btn"
            disabled={mode === "duel"}
            onClick={() => {
              if (!autoAgentDemo) {
                setMode("agent");
              }
              setAutoAgentDemo((v) => !v);
            }}
          >
            {autoAgentDemo ? "Stop Agent Demo" : "Run Agent Demo"}
          </button>
          <button className="btn" onClick={startDuel} disabled={mode === "duel"}>
            Start Duel
          </button>
          <button className="btn" onClick={restartDuel} disabled={mode !== "duel"}>
            Restart Duel
          </button>
        </div>
      </div>

      <div className="layout">
        <div className="stage">
          {mode !== "duel" ? (
            <>
              <div className={`frame ${mode}`}>
                {img ? (
                  <img src={`data:image/webp;base64,${img}`} width={800} height={600} />
                ) : (
                  <div className="loading">Loading…</div>
                )}
                <div className={`corner ${mode}`}>{mode === "human" ? "Human" : "Agent"}</div>
              </div>
              <div className="row">
                <span className="chip">Step {stepR.toFixed(2)}</span>
                <span className="chip">Episode {epR.toFixed(2)}</span>
                <span className="chip">Repeat {repeatRef.current}</span>
              </div>
            </>
          ) : (
            <>
              <div className="duel">
                <div className="frame human">
                  {imgH ? (
                    <img src={`data:image/webp;base64,${imgH}`} width={520} height={390} />
                  ) : (
                    <div className="loading small">Loading…</div>
                  )}
                  <div className="corner human">Human</div>
                  {duelOver && winner === "Human" && <div className="ribbon human">Winner: Human</div>}
                </div>
                <div className="frame agent">
                  {imgA ? (
                    <img src={`data:image/webp;base64,${imgA}`} width={520} height={390} />
                  ) : (
                    <div className="loading small">Loading…</div>
                  )}
                  <div className="corner agent">Agent</div>
                  {duelOver && winner === "Agent" && <div className="ribbon agent">Winner: Agent</div>}
                </div>
              </div>
              <div className="row">
                <span className="chip">Human {epRH.toFixed(2)}</span>
                <span className="chip">Agent {epRA.toFixed(2)}</span>
                <span className="chip">Seed {duelSeed ?? "-"}</span>
              </div>
            </>
          )}
        </div>

        <aside className="sidebar">
          <div className="panel">
            <div className="panel-head">
              <div>Now Playing</div>
              <span className={`pill ${mode}`}>{mode === "duel" ? "Duel" : mode === "human" ? "Human" : "Agent"}</span>
            </div>
            <div className="grid">
              <div className="metric">
                <div className="label">FPS</div>
                <div className="value">{fps}</div>
              </div>
              <div className="metric">
                <div className="label">Latency</div>
                <div className="value">{Math.round(latencyMs)} ms</div>
              </div>
              <div className="metric">
                <div className="label">Repeat</div>
                <div className="value">{repeatRef.current}</div>
              </div>
              <div className="metric span2">
                <div className="label">Status</div>
                <div className="value small">{status}</div>
              </div>
            </div>
          </div>

          {mode !== "duel" ? (
            <div className="panel">
              <div className="panel-title">Session</div>
              <div className="kv">
                <span>Episode</span>
                <b>{epR.toFixed(2)}</b>
              </div>
              <div className="kv">
                <span>Last step</span>
                <b>{stepR.toFixed(2)}</b>
              </div>
              <div className="kv">
                <span>Session</span>
                <b>{sid?.slice(0, 8) ?? "-"}</b>
              </div>
              <div className="help">Controls: A=left, W=main, D=right</div>
            </div>
          ) : (
            <div className="panel">
              <div className="panel-title">Duel</div>
              <div className="kv">
                <span>Human</span>
                <b>{epRH.toFixed(2)}</b>
              </div>
              <div className="kv">
                <span>Agent</span>
                <b>{epRA.toFixed(2)}</b>
              </div>
              <div className="kv">
                <span>Seed</span>
                <b>{duelSeed ?? "-"}</b>
              </div>
              <div className="help">Controls: A=left, W=main, D=right</div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
