"use client"
import "./globals.css";
import React from "react";
import {QueryClientProvider} from "@tanstack/react-query";
import {queryClient} from "@/lib/query-client";
import {ThemeProvider} from "@/components/theme-provider";
import {FileProvider} from "@/context/file-context";

export default function RootLayout({children}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" suppressHydrationWarning>
        <body>
        <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
        >
            <QueryClientProvider client={queryClient}>
                <FileProvider>
                    <main className={`grid overflow-y-auto min-h-screen h-screen max-h-screen`}>
                        {children}
                    </main>
                </FileProvider>
            </QueryClientProvider>
        </ThemeProvider>
        </body>
        </html>
    )
}