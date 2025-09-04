import React, {FC, useMemo, useState} from "react";
import {Card, CardHeader, CardTitle} from "@/components/ui/card";
import {Dialog, DialogContent, DialogHeader, DialogTitle} from "@/components/ui/dialog";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger
} from "@/components/ui/alert-dialog";
import {useFileContext} from "@/context/file-context";
import {Spinner} from "@/components/ui/spinner";
import {ScrollArea} from "@/components/ui/scroll-area";
import {cn} from "@/lib/utils";
import {Button} from "@/components/ui/button";
import {type Snap, useSnaps} from "@/hooks/use-snaps";
import {XIcon} from "lucide-react";
import {useSnapsContext} from "@/context/snaps-context";
import {Tabs, TabsContent, TabsList, TabsTrigger} from "@/components/ui/tabs";

interface SnapsCardProps extends React.HTMLAttributes<HTMLDivElement> {
    // You can add custom props here if needed
}

export const SnapsCard = React.forwardRef<HTMLDivElement, SnapsCardProps>(
    ({className}, ref) => {
        const {snaps, loading} = useSnapsContext()
        const [open, setOpen] = useState(false);
        const [selectedSnap, setSelectedSnap] = useState<Snap | null>(null);

        if (loading) {
            return (
                <div className={"flex flex-col gap-1 h-full w-full"}>
                    <h1 className="scroll-m-20 text-2xl font-bold tracking-tight text-balance">
                        Files
                    </h1>
                    <div className={"flex flex-col h-full items-center justify-center"}>
                        <Spinner/>
                    </div>
                </div>
            )
        }

        return (
            <Card ref={ref} className={cn("gap-0", className)}>
                <CardHeader>
                    <CardTitle>Snaps</CardTitle>
                </CardHeader>
                <ScrollArea className={"overflow-auto"}>
                    {snaps.length === 0 ? (
                        <div className="text-sm text-muted-foreground text-center">No snaps available.</div>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 mt-6 mx-6">
                            {snaps.map((snap, i) => <SnapItem key={i} snap={snap} index={i} onClick={(snap) => {
                                setSelectedSnap(snap);
                                setOpen(true);
                            }}/>)}
                        </div>
                    )}
                </ScrollArea>
                <SnapViewer snap={selectedSnap} open={open} onOpenChange={setOpen}/>
            </Card>
        );
    })

interface SnapItemProps {
    snap: Snap;
    index: number;
    onClick: (snap: Snap) => void;
}

const SnapItem: FC<SnapItemProps> = ({snap, index, onClick}) => {
    const {selectedFile} = useFileContext();
    if (!selectedFile) return null;
    const {deleteSnap} = useSnaps(selectedFile.name.replace(/\.[^/.]+$/, ""));

    return (
        <div className={"relative grid gap-1 hover:scale-105 transition-transform cursor-pointer group"}>
            <AlertDialog>
                <AlertDialogTrigger asChild>
                    <Button
                        className={"absolute -top-3 -right-3 z-10 p-1 size-6 rounded-full bg-black/70 hover:bg-black/100 group-hover:opacity-100 opacity-0 cursor-pointer"}
                        size={"icon"}>
                        <XIcon className={"size-3"} color={"white"}/>
                    </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                        <AlertDialogDescription>
                            This action cannot be undone. This will permanently delete the snap from your device.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={() => deleteSnap.mutate(snap.path)}
                        >Continue</AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
            <img src={snap.url ?? ""} alt={snap.name}
                 onClick={() => onClick(snap)}
                 className="rounded-md object-fill"/>
            <div className="text-xs text-muted-foreground">Snap {index + 1}</div>
        </div>
    )
}

interface SnapViewerProps {
    snap: Snap | null;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

const SnapViewer: FC<SnapViewerProps> = ({snap, open, onOpenChange}) => {
    if (!snap) return null;

    const analysisAvailable = useMemo(() => {
        return snap.analysis && snap.analysis.player_detection && snap.analysis.player_detection.url;
    }, [snap]);

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Snap Viewer</DialogTitle>
                </DialogHeader>
                <div className={"grid gap-4"}>
                    <Tabs defaultValue="snap">
                        <TabsList className={"w-full"}>
                            <TabsTrigger value="snap">Snap</TabsTrigger>
                            <TabsTrigger value="player_detections" disabled={!analysisAvailable}>Player
                                Detections</TabsTrigger>
                            <TabsTrigger value="top_down_perspective" disabled={!analysisAvailable}>Top-Down Perspective</TabsTrigger>
                        </TabsList>
                        <TabsContent value="snap">
                            <img src={snap.url ?? ""} alt="Snap"
                                 className="aspect-video w-full object-cover rounded-md"/>
                        </TabsContent>
                        <TabsContent value="player_detections">
                            <img src={snap.analysis.player_detection?.url ?? ""} alt="Snap"
                                 className="aspect-video w-full object-cover rounded-md"/>
                        </TabsContent>
                        <TabsContent className={"justify-items-center"} value="top_down_perspective">
                            <img src={snap.analysis.top_down_perspective?.url ?? ""} alt="Snap"
                                 className="max-h-96 object-cover rounded-md"/>
                        </TabsContent>
                    </Tabs>
                    {!analysisAvailable && <Button>
                        Analyze Snap
                    </Button>
                    }
                </div>
            </DialogContent>
        </Dialog>
    )
}