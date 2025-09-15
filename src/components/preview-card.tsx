import React from "react";
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card";
import {Badge} from "@/components/ui/badge";
import {Progress} from "@/components/ui/progress";
import {useAnalyzerStore} from "@/context/analyzer-context";
import {useFileContext} from "@/context/file-context";
import {getFileURL} from "@/lib/file-utils";
import {useEffect, useState} from "react";

export const PreviewCard = () => {
    const {selectedFile} = useFileContext();
    const {status, pct} = useAnalyzerStore();
    const [videoSrc, setVideoSrc] = useState<string | null>(null);

    // Load video preview URL when selection changes
    useEffect(() => {
        if (selectedFile) {
            getFileURL(selectedFile.path).then((url) => {
                setVideoSrc(url);
            });
        } else {
            setVideoSrc(null);
        }
    }, [selectedFile]);

    if (!selectedFile) return null;

    return (
        <Card className={"col-span-1 md:col-span-2 max-h-[600px]"}>
            <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>{selectedFile.name}</CardTitle>
                <Badge
                    variant={status === "done" ? "default" : status === "error" ? "destructive" : status === "running" ? "success" : "secondary"}>
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                </Badge>
            </CardHeader>
            <CardContent className="space-y-4">
                {videoSrc ? (
                    <div className="flex justify-center max-h-[440px]">
                        <video key={videoSrc} controls className="h-auto max-h-[440px] rounded-xl">
                            <source src={videoSrc}/>
                            Your browser does not support the video tag.
                        </video>
                    </div>
                ) : (
                    <div className="flex items-center justify-center h-48 bg-muted rounded-lg">
                        <span className="text-sm text-muted-foreground">No preview available</span>
                    </div>
                )}
                <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm text-muted-foreground">
                        <span>Overall progress</span>
                        <span>{pct}%</span>
                    </div>
                    <Progress value={pct} className="h-2"/>
                </div>
            </CardContent>
        </Card>
    )
}