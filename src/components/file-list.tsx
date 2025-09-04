import {Spinner} from "@/components/ui/spinner";
import {useFileContext} from "@/context/file-context";
import {Tooltip, TooltipContent, TooltipTrigger} from "@/components/ui/tooltip";
import {Button} from "@/components/ui/button";
import {openFolder} from "@/lib/file-utils";
import {FolderOpenIcon, RefreshCw, Trash} from "lucide-react";
import {
    AlertDialog, AlertDialogAction, AlertDialogCancel,
    AlertDialogContent, AlertDialogDescription, AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
    AlertDialogTrigger
} from "@/components/ui/alert-dialog";
import React from "react";
import {useFiles} from "@/hooks/use-files";

export const FileList = () => {
    const {files, selectedFile, setSelectedFile, refetch, loading} = useFileContext();
    const {remove} = useFiles()

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
        <div className="flex flex-col gap-1">
            <div className={"flex flex-row justify-between"}>
                <h1 className="scroll-m-20 text-2xl font-bold tracking-tight text-balance">
                    Files
                </h1>
                <div className={"flex flex-row"}>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                size={"icon"}
                                variant={"ghost"}
                                onClick={refetch}
                            >
                                <RefreshCw/>
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                            <>Place video files here</>
                        </TooltipContent>
                    </Tooltip>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                size={"icon"}
                                variant={"ghost"}
                                onClick={() => openFolder("videos")}
                            >
                                <FolderOpenIcon/>
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                            <>Place video files here</>
                        </TooltipContent>
                    </Tooltip>
                </div>
            </div>
            <div className={"grid overflow-y-auto"}>
                {files.length === 0
                    ? <div
                        className={"flex flex-col h-full items-center justify-center text-center text-muted-foreground"}>
                        <p>No files uploaded yet.</p>
                        <p className={"text-sm"}>Please add video files to get started.</p>
                    </div>
                    : files.map((file, index) => (
                        <div
                            key={index}
                            className={`flex flex-row items-center overflow-x-hidden rounded-lg hover:bg-muted transition-colors justify-between pl-2 ${selectedFile === file ? "bg-muted" : ""}`}
                            onClick={() => {
                                setSelectedFile(file);
                            }}
                        >
                            <span className={"text-sm truncate"}>{file.name}</span>
                            <AlertDialog>
                                <AlertDialogTrigger asChild>
                                    <Button
                                        size={"icon"}
                                        variant={"ghost"}
                                    >
                                        <Trash/>
                                    </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent>
                                    <AlertDialogHeader>
                                        <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                                        <AlertDialogDescription>
                                            This action cannot be undone. This will permanently delete the
                                            file <strong>{file.name}</strong> from your system.
                                        </AlertDialogDescription>
                                    </AlertDialogHeader>
                                    <AlertDialogFooter>
                                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                                        <AlertDialogAction onClick={() => remove.mutate(file.path)}
                                        >Continue</AlertDialogAction>
                                    </AlertDialogFooter>
                                </AlertDialogContent>
                            </AlertDialog>
                        </div>
                    ))}
            </div>
        </div>
    );

}