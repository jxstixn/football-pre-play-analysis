import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {fetchFilesFromDir, deleteFile, FileItem} from "@/lib/file-utils";

const key = (dir: string) => ["files", dir];

export function useFiles(dir = "videos") {
    const qc = useQueryClient();

    const filesQuery = useQuery<FileItem[]>({
        queryKey: key(dir),
        queryFn: () => fetchFilesFromDir(dir),
    });

    const remove = useMutation({
        mutationFn: (path: string) => deleteFile(path),
        onSuccess: () => qc.invalidateQueries({ queryKey: key(dir) }),
    });

    return {
        ...filesQuery,          // { data, isLoading, isError, error, refetch, ... }
        remove,                 // remove.mutate(path)
    };
}
