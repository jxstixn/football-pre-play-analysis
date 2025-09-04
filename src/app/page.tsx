"use client"
import {ResizableHandle, ResizablePanel, ResizablePanelGroup} from "@/components/ui/resizable";
import {FileList} from "@/components/file-list";
import {VideoAnalysis} from "@/components/video-analysis";

export default function Home() {
    return (<>
        <ResizablePanelGroup direction="horizontal">
            <ResizablePanel className={"p-4"} minSize={25} defaultSize={25}>
                <FileList/>
            </ResizablePanel>
            <ResizableHandle/>
            <ResizablePanel minSize={40} defaultSize={75}>
                    <VideoAnalysis/>
            </ResizablePanel>
        </ResizablePanelGroup>
    </>)
}
