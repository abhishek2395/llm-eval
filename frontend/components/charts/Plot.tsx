"use client";

import dynamic from "next/dynamic";

/** Client-only Plotly — plotly.js can't render during SSR. */
export const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => (
    <div className="flex h-48 items-center justify-center font-mono text-[0.7rem] text-mute">
      loading chart…
    </div>
  ),
});
