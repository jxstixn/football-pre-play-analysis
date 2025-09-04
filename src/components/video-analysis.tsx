"use client";
import React from "react";
import {AnalyzerProvider} from "@/context/analyzer-context";
import {useFileContext} from "@/context/file-context";
import {SnapsCard} from "@/components/snaps-card";
import {VideoAnalysisSettings} from "@/components/video-analysis-settings";
import {SnapsProvider} from "@/context/snaps-context";
import {PreviewCard} from "@/components/preview-card";

export function VideoAnalysis() {
    const {selectedFile} = useFileContext();

    if (!selectedFile) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">Please select a video file to analyze.</p>
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-4 overflow-y-scroll max-h-full">
            <SnapsProvider selectedFile={selectedFile}>
                <AnalyzerProvider>
                    <PreviewCard/>
                    <VideoAnalysisSettings/>
                    <SnapsCard className={"col-span-1 md:col-span-3 max-h-[600px]"}/>
                </AnalyzerProvider>
            </SnapsProvider>
        </div>
    );
}
