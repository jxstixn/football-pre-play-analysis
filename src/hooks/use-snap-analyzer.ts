import {useCallback, useRef, useState} from "react";
import {useSnapsContext} from "@/context/snaps-context";
import {appDataDir, join} from "@tauri-apps/api/path";

export type SnapAnalyzerStatus = "idle" | "starting" | "running" | "done" | "error" | "cancelled";

export type SnapAnalyzerEvent = {
    type: "error";
    message: string;
} | {
    type: "stage";
    stage: {
        index: number;
        total: number;
        name: string;
    }
    message: string;
} | {
    type: "result",
    status: string;
    errors: string[],
    message: string,
}

export interface SnapAnalyzerPaths {
    snapPath: string;
    outputPath: string;
}

export interface SnapAnalyzerOptions {
    apiBase?: string;
    onError?: (error: Error) => void;
    onResult?: (artifacts: string[]) => void;
}

export interface SnapAnalyzer {
    status: SnapAnalyzerStatus;
    jobId: string | null;
    events: SnapAnalyzerEvent[];
    progress: number; // 0 to 100
    paths: SnapAnalyzerPaths | null;
    analyze: (snapPath: string) => Promise<void>;
    cancel: () => Promise<void>;
    reset: () => void;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8008";

export function useSnapAnalyzer({apiBase, onError}: SnapAnalyzerOptions): SnapAnalyzer {
    const API_BASE = apiBase ?? DEFAULT_API_BASE;

    const [status, setStatus] = useState<SnapAnalyzerStatus>("idle");
    const [jobId, setJobId] = useState<string | null>(null);
    const [events, setEvents] = useState<SnapAnalyzerEvent[]>([]);
    const [progress, setProgress] = useState<number>(0);
    const [paths, setPaths] = useState<SnapAnalyzerPaths | null>(null);

    const {refetch} = useSnapsContext()

    const wsRef = useRef<WebSocket | null>(null);

    const openWebSocket = useCallback(
        (id: string) => {
            const ws = new WebSocket(`${API_BASE.replace(/^http/, "ws")}/ws/progress/${id}`);
            wsRef.current = ws;

            ws.onopen = () => setStatus("running");

            ws.onmessage = async (ev) => {
                try {
                    const e: SnapAnalyzerEvent = JSON.parse(ev.data);
                    setEvents((prev) => (prev.length > 500 ? prev.slice(-500) : prev).concat(e));
                    console.log("WS message", e);
                    switch (e.type) {
                        case "error": {
                            setStatus("error");
                            onError?.(new Error(e.message));
                            ws.close();
                            break;
                        }
                        case "stage": {
                            const raw = Number(e.stage.index) / Number(e.stage.total + 1);
                            if (!Number.isNaN(raw)) {
                                const clamped = Math.max(0, Math.min(1, raw));
                                setProgress(Math.round(clamped * 100));
                            }
                            break;
                        }
                        case "result": {
                            setStatus("done");
                            setProgress(100);
                            refetch()
                            break;
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
        [API_BASE, refetch, onError]
    );

    const analyze = useCallback(
        async (snapPath: string) => {
            if (status === "running" || status === "starting") {
                throw new Error("Analysis already in progress");
            }

            setStatus("starting");
            setEvents([]);
            setProgress(0);
            setPaths(null);
            setJobId(null);

            const dir = await appDataDir()
            const inputPath = await join(dir, snapPath);
            const relativePath = snapPath
                .replace("snaps", "analysis")      // swap "snaps" → "analysis"
                .replace(/(\.[^/.]+)$/, "")        // drop extension
                .replace(/[\\/][^\\/]+$/, "");     // drop last segment (file/folder name)
            const outputDir = await join(dir, relativePath);
            setPaths({snapPath: inputPath, outputPath: outputDir});

            try {
                const res = await fetch(`${API_BASE}/v1/analyze-snap`, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({file_path: inputPath, output_dir: outputDir}),
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
                console.error("Failed to start analysis:", err);
                setStatus("error");
                onError?.(err instanceof Error ? err : new Error(String(err)));
            }
        },
        [API_BASE, openWebSocket, onError, status]
    );

    const cancel = useCallback(async () => {
        if (!jobId) return;
        try {
            await fetch(`${API_BASE}/v1/analyze-snap/${jobId}/cancel`, {method: "POST"});
            wsRef.current?.close();
            setStatus("cancelled");
        } catch (err) {
            console.error("Failed to cancel analysis:", err);
            onError?.(err instanceof Error ? err : new Error(String(err)));
        }
    }, [API_BASE, jobId, onError]);

    const reset = useCallback(() => {
        if (status === "running" || status === "starting") {
            throw new Error("Cannot reset while analysis is in progress");
        }
        setStatus("idle");
        setJobId(null);
        setEvents([]);
        setProgress(0);
        setPaths(null);
    }, [status]);

    return {
        status,
        jobId,
        events,
        progress,
        paths,
        analyze,
        cancel,
        reset,
    };
}