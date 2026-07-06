/**
 * plotly.ts — shared dark layout for every chart (parity with V1 PLOTLY_LAYOUT).
 */

import type { Layout, Config } from "plotly.js";
import { PALETTE } from "./colors";

export const MONO = "IBM Plex Mono, monospace";

export function baseLayout(overrides: Partial<Layout> = {}): Partial<Layout> {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: MONO, color: PALETTE.text, size: 12 },
    margin: { l: 8, r: 8, t: 36, b: 8 },
    legend: {
      bgcolor: "rgba(0,0,0,0)",
      bordercolor: PALETTE.border,
      borderwidth: 1,
      font: { size: 11 },
    },
    xaxis: {
      gridcolor: PALETTE.border,
      linecolor: PALETTE.border,
      tickfont: { size: 10 },
    },
    yaxis: {
      gridcolor: PALETTE.border,
      linecolor: PALETTE.border,
      tickfont: { size: 10 },
    },
    ...overrides,
  };
}

export const PLOT_CONFIG: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};
