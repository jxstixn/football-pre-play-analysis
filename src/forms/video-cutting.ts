import {z} from "zod";

export const VideoCuttingSchema = z.object({
    scene_threshold: z.number().min(0).max(1)
        .describe("ffmpeg scene-change sensitivity"),
    min_play_seconds: z.number().min(0)
        .describe("Ignore clips shorter than this"),
    max_clips: z.number().min(0)
        .describe("0 = no limit; otherwise stop after N clips"),
    snap_at: z.number().min(0)
        .describe("First snapshot (s) into each clip"),
    snap_step: z.number().min(0)
        .describe("Spacing (s) between snapshots"),
    num_snap_frames: z.number().min(1)
        .describe("Snapshots per clip"),
});

export type VideoCuttingSettings = z.infer<typeof VideoCuttingSchema>;