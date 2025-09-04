"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { appDataDir, join } from "@tauri-apps/api/path";
import { useFileContext } from "@/context/file-context";
import type { VideoCuttingSettings } from "@/forms/video-cutting";
import {useSnapsContext} from "@/context/snaps-context";

/** ----------------- Types ----------------- */

export type AnalyzerStatus = "idle" | "starting" | "running" | "done" | "error" | "cancelled";

export interface AnalyzerEvent {
    type: string;
    [key: string]: any;
}

export interface AnalyzerPaths {
    filePath: string;
    outputDir: string;
}

export interface AnalyzerOptions {
    /** Base URL of your FastAPI server, defaults to http://localhost:8008 */
    apiBase?: string;
    /** Called when a fatal error happens (e.g., failed to start) */
    onError?: (err: Error) => void;
}

export interface AnalyzerStore {
    status: AnalyzerStatus;
    jobId: string | null;
    events: AnalyzerEvent[];
    overall: number; // 0..1
    pct: number;     // 0..100 rounded
    paths: AnalyzerPaths | null;
    analyze: (fileName: string, settings: VideoCuttingSettings) => Promise<void>;
    cancel: () => Promise<void>;
    reset: () => void;
}

/** ----------------- Context ----------------- */

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8008";

const AnalyzerCtx = createContext<AnalyzerStore | null>(null);

export function AnalyzerProvider({
                                     children,
                                     apiBase,
                                     onError,
                                 }: React.PropsWithChildren<AnalyzerOptions>) {
    const { selectedFile } = useFileContext();
    const { refetch } = useSnapsContext()

    const API_BASE = apiBase ?? DEFAULT_API_BASE;

    const [status, setStatus] = useState<AnalyzerStatus>("idle");
    const [jobId, setJobId] = useState<string | null>(null);
    const [events, setEvents] = useState<AnalyzerEvent[]>([]);
    const [overall, setOverall] = useState<number>(0);
    const [paths, setPaths] = useState<AnalyzerPaths | null>(null);

    const wsRef = useRef<WebSocket | null>(null);

    const pct = useMemo(() => Math.round(Math.max(0, Math.min(1, overall)) * 100), [overall]);

    const openWebSocket = useCallback(
        (id: string) => {
            const ws = new WebSocket(`${API_BASE.replace(/^http/, "ws")}/ws/progress/${id}`);
            wsRef.current = ws;

            ws.onopen = () => setStatus("running");

            ws.onmessage = async (ev) => {
                try {
                    const e: AnalyzerEvent = JSON.parse(ev.data);
                    setEvents((prev) => (prev.length > 500 ? prev.slice(-500) : prev).concat(e));

                    if ("approx_percent" in e) {
                        const raw = Number(e.approx_percent);
                        if (!Number.isNaN(raw)) {
                            const normalized = raw > 1 ? raw / 100 : raw;
                            const clamped = Math.max(0, Math.min(1, normalized));
                            setOverall(clamped);
                        }
                    }

                    if (e.type === "error") {
                        setStatus("error");
                        onError?.(new Error(String(e.message || "Unknown error")));
                    }
                    if (e.type === "cancelled") setStatus("cancelled");
                    if (e.type === "completed") setStatus("done");

                    if (e.type === "summary" && selectedFile) {
                        try {
                            refetch()
                        } catch (err) {
                            console.error("Failed to list snaps:", err);
                        }
                    }
                } catch (err) {
                    console.error("Bad WS message", err);
                }
            };

            ws.onclose = () => {
                wsRef.current = null;
            };
        },
        [API_BASE, selectedFile, refetch, onError]
    );

    const analyze = useCallback(
        async (fileName: string, settings: VideoCuttingSettings) => {
            setStatus("starting");
            setEvents([]);
            setOverall(0);
            setJobId(null);

            const appData = await appDataDir();
            const filePath = await join(appData, "videos", fileName);
            const outputDir = await join(appData, "extracted", fileName.replace(/\.[^/.]+$/, ""));
            setPaths({ filePath, outputDir });

            try {
                const res = await fetch(`${API_BASE}/v1/cut_video`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ file_path: filePath, output_dir: outputDir, ...settings }),
                });
                if (!res.ok) {
                    const txt = await res.text();
                    throw new Error(`Start failed: ${txt}`);
                }
                const data = await res.json();
                const id = data.job_id as string;
                setJobId(id);
                openWebSocket(id);
            } catch (err: any) {
                console.error(err);
                setStatus("error");
                onError?.(new Error(String(err?.message ?? err)));
            }
        },
        [API_BASE, openWebSocket, onError]
    );

    const cancel = useCallback(async () => {
        if (!jobId) return;
        try {
            await fetch(`${API_BASE}/v1/cut_video/${jobId}/cancel`, { method: "POST" });
            wsRef.current?.close();
            setStatus("cancelled");
        } catch (err) {
            console.error(err);
        }
    }, [API_BASE, jobId]);

    const reset = useCallback(() => {
        wsRef.current?.close();
        setStatus("idle");
        setEvents([]);
        setOverall(0);
        setJobId(null);
        setPaths(null);
    }, []);

    // Cleanup WS on unmount
    useEffect(() => {
        return () => wsRef.current?.close();
    }, []);

    const value = useMemo<AnalyzerStore>(
        () => ({ status, jobId, events, overall, pct, paths, analyze, cancel, reset }),
        [status, jobId, events, overall, pct, paths, analyze, cancel, reset]
    );

    return <AnalyzerCtx.Provider value={value}>{children}</AnalyzerCtx.Provider>;
}

/** Consumer hook (reads the ONE shared instance) */
export function useAnalyzerStore(): AnalyzerStore {
    const ctx = useContext(AnalyzerCtx);
    if (!ctx) throw new Error("useAnalyzerStore must be used within <AnalyzerProvider>");
    return ctx;
}
