import React, {FC, useEffect, useRef} from "react";
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Button} from "@/components/ui/button";
import {Form, FormControl, FormField, FormItem, FormLabel, FormMessage} from "@/components/ui/form";
import {Tooltip, TooltipContent, TooltipTrigger} from "@/components/ui/tooltip";
import {Input} from "@/components/ui/input";
import {zodResolver} from "@hookform/resolvers/zod";
import {useForm, useFormContext} from "react-hook-form";
import {VideoCuttingSchema, type VideoCuttingSettings} from "@/forms/video-cutting";
import {CircleQuestionMarkIcon} from "lucide-react";
import {useFileContext} from "@/context/file-context";
import {useAnalyzerStore} from "@/context/analyzer-context";

export const VideoAnalysisSettings: FC = () => {
    const {selectedFile} = useFileContext();
    const {status, analyze, cancel, events} = useAnalyzerStore();
    const [mode, setMode] = React.useState<"log" | "settings">("settings");
    const firstRun = useRef(true);
    const endRef = useRef<HTMLDivElement | null>(null);

    const form = useForm<VideoCuttingSettings>({
        resolver: zodResolver(VideoCuttingSchema),
        defaultValues: {
            scene_threshold: 0.25,
            min_play_seconds: 3,
            max_clips: 10,
            snap_at: 0.5,
            snap_step: 0.4,
            num_snap_frames: 3,
        }
    });

    useEffect(() => {
        // Smooth on subsequent updates; instant on first render to avoid flicker
        endRef.current?.scrollIntoView({behavior: firstRun.current ? "auto" : "smooth"});
        firstRun.current = false;
    }, [events]);

    if (!selectedFile) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">Please select a video file to analyze.</p>
            </div>
        );
    }

    return (
        <Card className={"col-span-1 max-h-[600px]"}>
            <CardHeader>
                <CardTitle>{mode === "settings" ? "Analysis Settings" : "Analysis Log"}</CardTitle>
            </CardHeader>
            <Form {...form}>
                <form className={"grid gap-4"}>
                    {mode === "settings" && <VideoAnalysisSettingsMenu/>}
                    {mode === "log" &&
                        <ScrollArea className={"overflow-auto px-6 min-h-full max-h-[calc(600px-11rem)]"}>
                            {events.map((e, i) => (
                                <pre key={i}
                                     className="text-[10px] leading-tight whitespace-pre-wrap break-words">
                                            {JSON.stringify(e, null, 2)}
                                        </pre>
                            ))}
                            <div ref={endRef}/>
                        </ScrollArea>
                    }
                    {(() => {
                        switch (status) {
                            case "running":
                            case "starting":
                                return <Button className={"mx-6"} variant="outline" onClick={cancel}>Cancel</Button>;
                            case "error":
                                return <Button className={"mx-6"} variant="destructive"
                                               onClick={() => setMode("settings")}>Error - Back to Settings</Button>;
                            case "done":
                                return <Button className={"mx-6"} onClick={() => setMode("settings")}>Analysis Complete
                                    - Back to Settings</Button>;
                            case "cancelled":
                                return <Button className={"mx-6"} variant="outline" onClick={() => setMode("settings")}>Analysis
                                    Cancelled - Back to Settings</Button>;
                            case "idle":
                                return <Button className={"mx-6"}
                                               onClick={(evt) => {
                                                   form.handleSubmit((data) => {
                                                       analyze(selectedFile.name, data)
                                                           .then(() => setMode("log"))
                                                           .catch(console.error);
                                                   })(evt)
                                               }}>Analyze
                                    Video</Button>
                            default:
                                return null;
                        }
                    })()}
                </form>
            </Form>
        </Card>
    );
}

const VideoAnalysisSettingsMenu: FC = () => {
    const form = useFormContext<VideoCuttingSettings>();

    return (
        <CardContent className={"grid gap-4"}>
            <FormField
                control={form.control}
                name="scene_threshold"
                render={({field}) => (
                    <FormItem>
                        <FormLabel className={"items-center"}>
                            Scene Threshold
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <CircleQuestionMarkIcon className={"text-muted-foreground size-4"}/>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p>The threshold for detecting scene changes. A lower value makes the
                                        detection more sensitive.</p>
                                </TooltipContent>
                            </Tooltip>
                        </FormLabel>
                        <FormControl>
                            <Input {...field} type={"number"} min={0} step={0.01} placeholder="0.25" onChange={(e) => {
                                const val = parseFloat(e.target.value);
                                if (!isNaN(val)) {
                                    field.onChange(val);
                                } else {
                                    field.onChange(0);
                                }
                            }}/>
                        </FormControl>
                        <FormMessage/>
                    </FormItem>
                )}
            />
            <FormField
                control={form.control}
                name="min_play_seconds"
                render={({field}) => (
                    <FormItem>
                        <FormLabel className={"items-center"}>
                            Min Play Seconds
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <CircleQuestionMarkIcon className={"text-muted-foreground size-4"}/>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p>Ignore clips shorter than this duration (in seconds).</p>
                                </TooltipContent>
                            </Tooltip>
                        </FormLabel>
                        <FormControl>
                            <Input {...field} type={"number"} min={0} step={1} placeholder="3" onChange={(e) => {
                                const val = parseInt(e.target.value);
                                if (!Number.isNaN(val)) {
                                    field.onChange(val);
                                } else {
                                    field.onChange(0);
                                }
                            }} />
                        </FormControl>
                        <FormMessage/>
                    </FormItem>
                )}
            />
            <FormField
                control={form.control}
                name="max_clips"
                render={({field}) => (
                    <FormItem>
                        <FormLabel className={"items-center"}>
                            Max Clips
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <CircleQuestionMarkIcon className={"text-muted-foreground size-4"}/>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p>Maximum number of clips to extract. Set to 0 for no limit.</p>
                                </TooltipContent>
                            </Tooltip>
                        </FormLabel>
                        <FormControl>
                            <Input {...field} type={"number"} min={0} step={1} placeholder="10" onChange={(e) => {
                                const val = parseInt(e.target.value);
                                if (!Number.isNaN(val)) {
                                    field.onChange(val);
                                } else {
                                    field.onChange(0);
                                }
                            }} />
                        </FormControl>
                        <FormMessage/>
                    </FormItem>
                )}
            />
            <FormField
                control={form.control}
                name="snap_at"
                render={({field}) => (
                    <FormItem>
                        <FormLabel className={"items-center"}>
                            Snap At (s)
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <CircleQuestionMarkIcon className={"text-muted-foreground size-4"}/>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p> Time (in seconds) into each clip to take the first snapshot.</p>
                                </TooltipContent>
                            </Tooltip>
                        </FormLabel>
                        <FormControl>
                            <Input type={"number"} min={0} step={0.1} placeholder="0.5" {...field} />
                        </FormControl>
                        <FormMessage/>
                    </FormItem>
                )}
            />
            <FormField
                control={form.control}
                name="snap_step"
                render={({field}) => (
                    <FormItem>
                        <FormLabel className={"items-center"}>
                            Snap Step (s)
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <CircleQuestionMarkIcon className={"text-muted-foreground size-4"}/>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p> Time interval (in seconds) between consecutive snapshots.</p>
                                </TooltipContent>
                            </Tooltip>
                        </FormLabel>
                        <FormControl>
                            <Input type={"number"} min={0} step={0.1} placeholder="0.4" {...field} />
                        </FormControl>
                        <FormMessage/>
                    </FormItem>
                )}
            />
            <FormField
                control={form.control}
                name="num_snap_frames"
                render={({field}) => (
                    <FormItem>
                        <FormLabel className={"items-center"}>
                            Number of Snap Frames
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <CircleQuestionMarkIcon className={"text-muted-foreground size-4"}/>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p>Number of snapshots to take per clip.</p>
                                </TooltipContent>
                            </Tooltip>
                        </FormLabel>
                        <FormControl>
                            <Input type={"number"} min={1} step={1} placeholder="3" {...field} />
                        </FormControl>
                        <FormMessage/>
                    </FormItem>
                )}
            />
        </CardContent>
    );
}