import {exists} from '@tauri-apps/plugin-fs';
import {convertFileSrc} from "@tauri-apps/api/core";
import {appDataDir, join} from "@tauri-apps/api/path";
import {openPath} from "@tauri-apps/plugin-opener";
import {invoke} from "@tauri-apps/api/core";
import {Snap} from "@/hooks/use-snaps";

export type FileItem = {
    name: string;
    path: string;
}

export async function openFolder(folderPath: string): Promise<void> {
    const dir = await appDataDir()
    const fullPath = await join(dir, folderPath);
    await openPath(fullPath);
}

export async function fetchFilesFromDir(folderPath: string): Promise<FileItem[]> {
    const dir = await appDataDir();
    const fullPath = await join(dir, folderPath);
    const dirExists = await exists(fullPath);

    if (!dirExists) {
        console.warn("Directory does not exist:", fullPath);
        return [];
    }

    try {
        const entries: string[] = await invoke("custom_list_dir", {path: fullPath});

        return await Promise.all(
            entries.map(async (entry) => {
                const filePath = await join(folderPath, entry);
                return {
                    name: entry,
                    path: filePath,
                } satisfies FileItem;
            })
        );
    } catch (error) {
        console.error("Error reading directory:", error);
        return [];
    }
}

export async function getFileURL(filePath: string): Promise<string | null> {
    const dir = await appDataDir()
    const joinedPath = await join(dir, filePath);
    const fileExists = await exists(joinedPath);
    if (!fileExists) {
        console.warn("File does not exist:", joinedPath);
        return null;
    }

    try {
        return convertFileSrc(joinedPath);
    } catch (error) {
        console.error("Error converting file path to URL:", error);
        return null;
    }
}

export async function deleteFile(filePath: string): Promise<void> {
    const dir = await appDataDir()
    const joinedPath = await join(dir, filePath);
    const fileExists = await exists(joinedPath);
    if (!fileExists) {
        console.warn("File does not exist:", joinedPath);
        return;
    }

    try {
        await invoke("custom_remove", {path: joinedPath});
    } catch (error) {
        console.error("Error deleting file:", error);
    }
}

export type AnalysisResult = {
    player_detection: Snap | null;
    top_down_perspective: Snap | null;
}

export async function getAnalysisResults(snapPath: string): Promise<AnalysisResult> {
    const dir = await appDataDir()
    // append _player_detection and _top_down_perspective before the file extension
    const playerDetectionPath = snapPath
        .replace("snaps", "analysis")
        .replace(/(\.[^/.]+)$/, "_player_detection$1");
    const topDownPerspectivePath = snapPath
        .replace("snaps", "analysis")
        .replace(/(\.[^/.]+)$/, "_top_down_perspective$1");

    console.log("Checking for analysis files:", playerDetectionPath, topDownPerspectivePath);

    const [playerDetectionExists, topDownPerspectiveExists] = await Promise.all([
        exists(await join(dir, playerDetectionPath)),
        exists(await join(dir, topDownPerspectivePath))
    ]);

    return {
        player_detection: playerDetectionExists ? { name: playerDetectionPath.split('/').pop() || '', path: playerDetectionPath, url: await getFileURL(playerDetectionPath), analysis: { player_detection: null, top_down_perspective: null } } : null,
        top_down_perspective: topDownPerspectiveExists ? { name: topDownPerspectivePath.split('/').pop() || '', path: topDownPerspectivePath, url: await getFileURL(topDownPerspectivePath), analysis: { player_detection: null, top_down_perspective: null } } : null,
    };
}