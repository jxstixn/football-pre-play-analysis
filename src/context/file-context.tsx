"use client"

import React, {createContext, useContext, useEffect, useState, PropsWithChildren} from "react";
import {FileItem} from "@/lib/file-utils";
import {useFiles} from "@/hooks/use-files";

type FileContextType = {
    files: FileItem[]
    selectedFile: FileItem | null
    setSelectedFile: React.Dispatch<React.SetStateAction<FileItem | null>>
    loading: boolean
    error?: string,
    refetch: () => void
}

export const FileContext = createContext<FileContextType | undefined>(undefined);

export const FileProvider = ({ dir = "videos", children }: PropsWithChildren<{ dir?: string }>) => {
    const { data = [], isLoading, isError, error, refetch } = useFiles(dir);
    const [selectedFile, setSelectedFile] = useState<FileItem | null>(null);

    useEffect(() => {
        if (!data.length) return setSelectedFile(null);
        setSelectedFile(prev => (prev && data.some(d => d.path === prev.path)) ? prev : data[0]);
    }, [data]);

    const value = {
        files: data,
        selectedFile,
        setSelectedFile,
        loading: isLoading,
        error: isError ? (error as Error).message : undefined,
        refetch
    }

    return (
        <FileContext.Provider value={value}>
            {children}
        </FileContext.Provider>
    )
}

export const useFileContext = () => {
    const ctx = useContext(FileContext);
    if (ctx === undefined) {
        throw new Error("useFileContext must be used within a <FileProvider>.");
    }
    return ctx;
};


// const [loading, setLoading] = useState(true);
// const [snapsLoading, setSnapsLoading] = useState(false);
// const [snaps, setSnaps] = useState<string[]>([]);
// const [files, setFiles] = useState<string[]>([])
// const [selectedFile, setSelectedFile] = useState<string | null>(null)
//
// const getFileUrl = async (fileName: string) => {
//     const dir = await appDataDir();
//     const filePath = await join(dir, 'videos', fileName);
//     return convertFileSrc(filePath);
// }
//
// const fetchSnaps = async (videoFile: string) => {
//     setSnapsLoading(true);
//     try {
//         const dir = await appDataDir();
//         const snapsDir = await join(dir, 'extracted', videoFile.replace(/\.[^/.]+$/, ""), 'snaps');
//         const dirExists = await exists(snapsDir);
//         if (!dirExists) {
//             setSnaps([]);
//             return;
//         }
//         const entries: string[] = await invoke("custom_list_dir", {path: snapsDir});
//         const snapUrls = entries
//             .filter((e) => e.match(/\.(jpe?g)$/i))
//             .map(async (e) => convertFileSrc(await join(snapsDir, e)));
//         snapUrls.sort();
//
//         setSnaps(await Promise.all(snapUrls));
//     } catch (error) {
//         console.error("Error reading snaps directory:", error);
//         setSnaps([]);
//     } finally {
//         setSnapsLoading(false);
//     }
// }
//
// useEffect(() => {
//     if (selectedFile) {
//         fetchSnaps(selectedFile)
//     } else {
//         setSnaps([]);
//     }
// }, [selectedFile]);
//
// useEffect(() => {
//     setLoading(true);
//     fetchFilesFromDir("videos")
//         .then((fetchedFiles) => {
//             setFiles(fetchedFiles);
//             if (fetchedFiles.length > 0) {
//                 setSelectedFile(fetchedFiles[0]);
//             }
//         })
//         .catch((error) => {
//             console.error("Error fetching files:", error);
//         })
//         .finally(() => {
//             setLoading(false);
//         });
// }, []);
//
// const deleteSnap = async (snapPath: string) => {
//     try {
//         const dir = await appDataDir();
//         const fullSnapPath = await join(dir, snapPath);
//         console.log("Deleting snap at path:", fullSnapPath);
//         const pathExists = await exists(fullSnapPath);
//         if (pathExists) {
//             await invoke("custom_remove", {path: fullSnapPath});
//             // Refresh snaps list
//             if (selectedFile) {
//                 await fetchSnaps(selectedFile);
//             }
//         } else {
//             console.warn("Snap path does not exist:", fullSnapPath);
//         }
//     } catch (error) {
//         console.error("Error deleting snap:", error);
//     }
// };
