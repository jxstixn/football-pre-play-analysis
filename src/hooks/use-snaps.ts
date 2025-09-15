import {useQuery, useMutation, useQueryClient} from "@tanstack/react-query";
import {fetchFilesFromDir, deleteFile, getFileURL, SnapAnalysisResult, getAnalysisResults} from "@/lib/file-utils";

export type Snap = {
    name: string;
    path: string;
    url: string | null;
    analysis: SnapAnalysisResult
}

const key = (dir: string) => ["files", dir];

export function useSnaps(video: string) {
    const qc = useQueryClient();
    const dir = `extracted/${video}/snaps`;

    const snapsQuery = useQuery<Snap[]>({
        queryKey: key(dir),
        queryFn: () => fetchFilesFromDir(dir).then(
            async (files) => {
                return await Promise.all(files.map(async (file) => {
                    const url = await getFileURL(file.path);
                    const analysis = await getAnalysisResults(file.path);
                    return {
                        ...file,
                        url,
                        analysis
                    } as Snap;
                }));
            }
        )
    });

    const deleteSnap = useMutation({
        mutationFn: (path: string) => deleteFile(path),
        onSuccess: () => qc.invalidateQueries({queryKey: key(dir)}),
    });

    return {
        ...snapsQuery,          // { data, isLoading, isError, error, refetch, ... }
        deleteSnap,                 // remove.mutate(path)
    };
}
