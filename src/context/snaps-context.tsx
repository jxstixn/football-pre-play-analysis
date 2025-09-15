"use client"

import React, {createContext, useContext, useState, PropsWithChildren, useMemo} from "react";
import {Snap, useSnaps} from "@/hooks/use-snaps";
import {FileItem} from "@/lib/file-utils";

type SnapsContextType = {
    snaps: Snap[]
    selectedSnap: Snap | null
    setSelectedSnap: React.Dispatch<React.SetStateAction<Snap | null>>
    loading: boolean
    error?: string,
    refetch: () => void
}

export const SnapsContext = createContext<SnapsContextType | undefined>(undefined);

export const SnapsProvider = ({ selectedFile, children }: PropsWithChildren<{ selectedFile: FileItem }>) => {
    const dir = selectedFile.name.replace(/\.[^/.]+$/, "")
    const { data = [], isLoading, isError, error, refetch } = useSnaps(dir);
    const [selectedSnapInternal, setSelectedSnap] = useState<Snap | null>(null);

    const selectedSnap = useMemo(() => {
        if (selectedSnapInternal) {
            return data.find(snap => snap.path === selectedSnapInternal.path) || null;
        }
        return null;
    }, [selectedSnapInternal, data]);

    const value = {
        snaps: data,
        selectedSnap,
        setSelectedSnap,
        loading: isLoading,
        error: isError ? (error as Error).message : undefined,
        refetch
    }

    return (
        <SnapsContext.Provider value={value}>
            {children}
        </SnapsContext.Provider>
    )
}

export const useSnapsContext = () => {
    const ctx = useContext(SnapsContext);
    if (ctx === undefined) {
        throw new Error("useSnapsContext must be used within a <FileProvider>.");
    }
    return ctx;
};

