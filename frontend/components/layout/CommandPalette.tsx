"use client";

/**
 * CommandPalette — ⌘K. Navigate, run models (incl. "surprise me"),
 * export, and search the prompt library.
 */

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";
import {
  Dices,
  Download,
  FileText,
  GitCompareArrows,
  LayoutDashboard,
  ListChecks,
  Play,
  Search,
} from "lucide-react";
import { MODEL_ALIASES, shortName } from "@/lib/aliases";
import { api, exportUrls } from "@/lib/api";
import type { Prompt } from "@/lib/types";

export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [prompts, setPrompts] = useState<Prompt[]>([]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  useEffect(() => {
    if (open && !prompts.length) {
      api.prompts().then(setPrompts).catch(() => {});
    }
  }, [open, prompts.length]);

  const go = useCallback(
    (path: string) => {
      setOpen(false);
      router.push(path);
    },
    [router],
  );

  const runModels = (models: string[]) =>
    go(`/?run=${encodeURIComponent(models.join(","))}`);

  const surpriseMe = async () => {
    try {
      const cat = await api.catalog({ tier: "lt1", limit: 200 });
      const pool = cat.models.filter((m) => !m.is_free);
      const picks: string[] = [];
      while (picks.length < 3 && pool.length) {
        const i = Math.floor(Math.random() * pool.length);
        picks.push(pool.splice(i, 1)[0].id);
      }
      if (picks.length) runModels(picks);
    } catch {
      setOpen(false);
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 pt-[15vh]"
      onClick={() => setOpen(false)}
    >
      <div onClick={(e) => e.stopPropagation()} className="w-full max-w-xl">
        <Command
          label="Command palette"
          className="overflow-hidden rounded-xl border border-line bg-bg2 shadow-2xl"
        >
          <div className="flex items-center gap-2 border-b border-line px-4">
            <Search size={15} className="text-mute" />
            <Command.Input
              autoFocus
              placeholder="Run a model, jump to a page, export…"
              className="w-full bg-transparent py-3.5 font-mono text-[0.85rem] text-ink outline-none placeholder:text-mute"
            />
            <kbd className="pill pill-mute">esc</kbd>
          </div>
          <Command.List className="max-h-[50vh] overflow-y-auto p-2">
            <Command.Empty className="px-3 py-6 text-center font-mono text-[0.75rem] text-mute">
              Nothing matches.
            </Command.Empty>

            <Command.Group
              heading="Navigate"
              className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[0.6rem] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.12em] [&_[cmdk-group-heading]]:text-mute"
            >
              {[
                { label: "Dashboard", path: "/", icon: LayoutDashboard },
                { label: "Compare", path: "/compare", icon: GitCompareArrows },
                { label: "Prompt library", path: "/prompts", icon: ListChecks },
                { label: "Report & export", path: "/report", icon: FileText },
              ].map(({ label, path, icon: Icon }) => (
                <Command.Item
                  key={path}
                  onSelect={() => go(path)}
                  className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[0.82rem] text-ink data-[selected=true]:bg-bg3"
                >
                  <Icon size={15} className="text-mute" /> {label}
                </Command.Item>
              ))}
            </Command.Group>

            <Command.Group
              heading="Evaluate"
              className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[0.6rem] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.12em] [&_[cmdk-group-heading]]:text-mute"
            >
              <Command.Item
                onSelect={() => void surpriseMe()}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[0.82rem] text-amber data-[selected=true]:bg-bg3"
              >
                <Dices size={15} /> Surprise me — eval 3 random catalog models
              </Command.Item>
              {Object.keys(MODEL_ALIASES).map((m) => (
                <Command.Item
                  key={m}
                  value={`run ${m} ${MODEL_ALIASES[m]}`}
                  onSelect={() => runModels([m])}
                  className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[0.82rem] text-ink data-[selected=true]:bg-bg3"
                >
                  <Play size={13} className="text-mute" /> Run {shortName(m)}
                  <span className="ml-auto font-mono text-[0.65rem] text-mute">{m}</span>
                </Command.Item>
              ))}
            </Command.Group>

            <Command.Group
              heading="Export"
              className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[0.6rem] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.12em] [&_[cmdk-group-heading]]:text-mute"
            >
              <Command.Item
                onSelect={() => {
                  window.open(exportUrls.html, "_blank");
                  setOpen(false);
                }}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[0.82rem] text-ink data-[selected=true]:bg-bg3"
              >
                <Download size={15} className="text-mute" /> Open HTML report
              </Command.Item>
              <Command.Item
                onSelect={() => {
                  window.open(exportUrls.json, "_blank");
                  setOpen(false);
                }}
                className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[0.82rem] text-ink data-[selected=true]:bg-bg3"
              >
                <Download size={15} className="text-mute" /> Download JSON export
              </Command.Item>
            </Command.Group>

            {prompts.length > 0 && (
              <Command.Group
                heading="Prompts"
                className="[&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:font-mono [&_[cmdk-group-heading]]:text-[0.6rem] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-[0.12em] [&_[cmdk-group-heading]]:text-mute"
              >
                {prompts.map((p) => (
                  <Command.Item
                    key={p.id}
                    value={`prompt ${p.id} ${p.category} ${p.prompt}`}
                    onSelect={() => go(`/?focus=${encodeURIComponent(p.id)}`)}
                    className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[0.82rem] text-ink data-[selected=true]:bg-bg3"
                  >
                    <ListChecks size={13} className="text-mute" />
                    <span className="font-mono text-[0.75rem]">{p.id}</span>
                    <span className="truncate text-[0.75rem] text-mute">
                      {p.prompt.slice(0, 60)}
                    </span>
                  </Command.Item>
                ))}
              </Command.Group>
            )}
          </Command.List>
        </Command>
      </div>
    </div>
  );
}
