// src/App.tsx
import { useEffect, useRef, useState } from "react";
import axios from "axios";

const API = "/api";

type Mode = "human" | "agent" | "duel";

type NewSessionResp = {
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
};

export default function App() {
  const [mode, setMode] = useState<Mode>("human");
  const [status, setStatus] = useState("Booting…");

  // single-session state (human OR agent)
  const [sid, setSid] = useState<string | null>(null);
  const [img, setImg] = useState<string | null>(null);
  const [over, setOver] = useState<boolean>(false);
  const [reward, setReward] = useState<number>(0);
  const [episodeReward, setEpisodeReward] = useState<number>(0);

  // duel state
  const [sidH, setSidH] = useState<string | null>(null);
  const [sidA, setSidA] = useState<string | null>(null);
  const [imgH, setImgH] = useState<string | null>(null);
  const [imgA, setImgA] = useState<string | null>(null);
  const [overH, setOverH] = useState(false);
  const [overA, setOverA] = useState(false);
  const [epRH, setEpRH] = useState(0);
  const [epRA, setEpRA] = useState(0);
  const [duelSeed, setDuelSeed] = useState<number | null>(null);

  const keys = useRef<Record<string, boolean>>({});
  const raf = useRef<number>(0);
  const lastAction = useRef<number>(0);

  const http = axios.create({
    baseURL: API,
    timeout: 15000,
    headers: { "Content-Type": "application/json" },
  });

  // keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      keys.current[e.key] = e.type === "keydown";
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKey);
    };
  }, []);

  // boot a single session (default page load)
  useEffect(() => {
    const boot = async () => {
      setStatus("Creating session…");
      try {
        const { data } = await http.post<NewSessionResp>("/session/new", {
          env_id: "LunarLander-v3",
        });
        setSid(data.session_id);
        setImg(data.frame);
        setOver(false);
        setReward(0);
        setEpisodeReward(0);
        lastAction.current = 0;
        setStatus("Ready");
      } catch (e: any) {
        console.error(e);
        setStatus(`Init failed: ${e?.response?.data?.detail ?? e?.message ?? "unknown"}`);
      }
    };
    void boot();

  }, []);

  useEffect(() => {
    let running = true;
    let last = performance.now();
    const targetMs = 1000 / 30;

    const tick = async (t: number) => {
      if (!running) return;
      if (t - last < targetMs) {
        raf.current = requestAnimationFrame(tick);
        return;
      }
      last = t;

      try {
        if (mode === "duel") {
          if (!sidH || !sidA || (overH && overA)) {
            raf.current = requestAnimationFrame(tick);
            return;
          }

          // derive human action from keys (A/W/D)
          const A = !!keys.current["a"] || !!keys.current["A"];
          const W = !!keys.current["w"] || !!keys.current["W"];
          const D = !!keys.current["d"] || !!keys.current["D"];
          if (W) lastAction.current = 2;
          else if (D && !A) lastAction.current = 3;
          else if (A && !D) lastAction.current = 1;
          else lastAction.current = 0;

          // step both sides in parallel where not done
          const reqs: Promise<any>[] = [];
          if (!overH) {
            reqs.push(
              http.post<StepResp>("/step", { session_id: sidH, action: lastAction.current })
            );
          }
          if (!overA) {
            reqs.push(http.post<StepResp>("/agent_step", { session_id: sidA }));
          }
          const results = await Promise.all(reqs);

          // map back
          let i = 0;
          if (!overH) {
            const { data } = results[i++] as { data: StepResp };
            setImgH(data.frame);
            if (typeof data.reward === "number") setEpRH((r) => r + data.reward!);
            if (data.done) setOverH(true);
          }
          if (!overA) {
            const { data } = results[i++] as { data: StepResp };
            setImgA(data.frame);
            if (typeof data.reward === "number") setEpRA((r) => r + data.reward!);
            if (data.done) setOverA(true);
          }

          // show status when both done
          if (!overH || !overA) {
            setStatus("Duel running…");
          } else {
            setStatus("Duel finished");
          }
        } else {
          // single-session
          if (!sid || over) {
            raf.current = requestAnimationFrame(tick);
            return;
          }

          let resp: StepResp | undefined;
          if (mode === "human") {
            const A = !!keys.current["a"] || !!keys.current["A"];
            const W = !!keys.current["w"] || !!keys.current["W"];
            const D = !!keys.current["d"] || !!keys.current["D"];
            if (W) lastAction.current = 2;
            else if (D && !A) lastAction.current = 3;
            else if (A && !D) lastAction.current = 1;
            else lastAction.current = 0;

            const { data } = await http.post<StepResp>("/step", {
              session_id: sid,
              action: lastAction.current,
            });
            resp = data;
          } else {
            const { data } = await http.post<StepResp>("/agent_step", { session_id: sid });
            resp = data;
          }

          if (resp) {
            setImg(resp.frame);
            if (typeof resp.reward === "number") {
              setReward(resp.reward);
              setEpisodeReward((r) => r + resp!.reward!);
            }
            if (resp.done) {
              setOver(true);
              setStatus("Episode finished");
            } else {
              setStatus("Running…");
            }
          }
        }
      } catch (e: any) {
        console.error(e);
        setStatus(`Loop error: ${e?.response?.data?.detail ?? e?.message ?? "unknown"}`);
      }

      raf.current = requestAnimationFrame(tick);
    };

    raf.current = requestAnimationFrame(tick);
    return () => {
      running = false;
      cancelAnimationFrame(raf.current);
    };
  }, [mode, sid, over, sidH, sidA, overH, overA]);

  // actions
  const restartSingle = async () => {
    if (!sid) return;
    setStatus("Resetting…");
    const { data } = await http.post<{ frame: string; done: boolean }>("/reset", {
      session_id: sid,
    });
    setImg(data.frame);
    setOver(false);
    setReward(0);
    setEpisodeReward(0);
    lastAction.current = 0;
    setStatus("Ready");
  };

  const startDuel = async () => {
    setStatus("Creating duel sessions…");
    setImgH(null); setImgA(null);
    setOverH(false); setOverA(false);
    setEpRH(0); setEpRA(0);

    // one shared seed for same terrain
    const seed = Math.floor(Math.random() * 1e9);
    setDuelSeed(seed);

    const [h, a] = await Promise.all([
      http.post<NewSessionResp>("/session/new", { env_id: "LunarLander-v3", seed }),
      http.post<NewSessionResp>("/session/new", { env_id: "LunarLander-v3", seed }),
    ]);
    setSidH(h.data.session_id); setImgH(h.data.frame);
    setSidA(a.data.session_id); setImgA(a.data.frame);
    setMode("duel");
    setStatus("Duel ready");
  };

  const restartDuel = async () => {
    if (!sidH || !sidA) return;
    setStatus("Resetting duel…");
    const seed = Math.floor(Math.random() * 1e9);
    setDuelSeed(seed);

    const [h, a] = await Promise.all([
      http.post<NewSessionResp>("/session/new", { env_id: "LunarLander-v3", seed }),
      http.post<NewSessionResp>("/session/new", { env_id: "LunarLander-v3", seed }),
    ]);
    setSidH(h.data.session_id); setImgH(h.data.frame); setOverH(false); setEpRH(0);
    setSidA(a.data.session_id); setImgA(a.data.frame); setOverA(false); setEpRA(0);
    setStatus("Duel ready");
  };

  // winner banner
  const duelOver = mode === "duel" && overH && overA;
  let winnerText = "";
  if (duelOver) {
    if (Math.abs(epRH - epRA) < 1e-6) winnerText = "Tie!";
    else winnerText = epRH > epRA ? "Winner: Human 🎉" : "Winner: Agent 🤖";
  }

  const Score = ({ label, value }: { label: string; value: number }) => (
    <div style={{ padding: "4px 8px", border: "1px solid #444", borderRadius: 8 }}>
      {label}: <strong>{value.toFixed(2)}</strong>
    </div>
  );

  return (
    <div style={{ padding: 16, fontFamily: "system-ui, sans-serif", color: "#eee" }}>
      <h1>LunarLander — Gym Session</h1>

      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
        <button disabled={mode === "human"} onClick={() => setMode("human")}>Human</button>
        <button disabled={mode === "agent"} onClick={() => setMode("agent")}>Agent</button>
        <button onClick={restartSingle} disabled={mode === "duel"}>Restart</button>

        <span style={{ marginLeft: 8, opacity: 0.85 }}>Status: <strong>{status}</strong></span>

        <span style={{ marginLeft: "auto" }}>
          <button onClick={startDuel} disabled={mode === "duel"}>Start Duel</button>
          <button onClick={restartDuel} disabled={mode !== "duel"} style={{ marginLeft: 8 }}>Restart Duel</button>
        </span>
      </div>

      {mode !== "duel" ? (
        <>
          <div
            style={{
              width: 800,
              height: 600,
              border: "1px solid #444",
              display: "grid",
              placeItems: "center",
              background: "#000",
              borderRadius: 8,
            }}
          >
            {img ? (
              <img src={`data:image/png;base64,${img}`} width={800} height={600} alt="Gym render" />
            ) : (
              <span>Loading…</span>
            )}
          </div>

          <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
            <Score label="Step reward" value={reward} />
            <Score label="Episode reward" value={episodeReward} />
            <div>Mode: <strong>{mode}</strong></div>
            <div>Controls: <code>A</code>=left, <code>W</code>=main, <code>D</code>=right</div>
          </div>
        </>
      ) : (
        <>
          {/* Duel layout */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <div style={{ marginBottom: 6, opacity: 0.8 }}>Human</div>
              <div
                style={{
                  width: 520,
                  height: 390,
                  border: "1px solid #444",
                  display: "grid",
                  placeItems: "center",
                  background: "#000",
                  borderRadius: 8,
                }}
              >
                {imgH ? (
                  <img src={`data:image/png;base64,${imgH}`} width={520} height={390} alt="Human render" />
                ) : (
                  <span>Loading…</span>
                )}
              </div>
            </div>

            <div>
              <div style={{ marginBottom: 6, opacity: 0.8 }}>Agent</div>
              <div
                style={{
                  width: 520,
                  height: 390,
                  border: "1px solid #444",
                  display: "grid",
                  placeItems: "center",
                  background: "#000",
                  borderRadius: 8,
                }}
              >
                {imgA ? (
                  <img src={`data:image/png;base64,${imgA}`} width={520} height={390} alt="Agent render" />
                ) : (
                  <span>Loading…</span>
                )}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 16, marginTop: 10, alignItems: "center" }}>
            <Score label="Human reward" value={epRH} />
            <Score label="Agent reward" value={epRA} />
            <div>Seed: <code>{duelSeed ?? "-"}</code></div>
            {duelOver && (
              <div style={{ marginLeft: 16, padding: "4px 10px", borderRadius: 8, background: "#262", color: "#fff" }}>
                {winnerText}
              </div>
            )}
            <div style={{ marginLeft: "auto" }}>
              Controls: <code>A</code>=left, <code>W</code>=main, <code>D</code>=right
            </div>
          </div>
        </>
      )}
    </div>
  );
}
