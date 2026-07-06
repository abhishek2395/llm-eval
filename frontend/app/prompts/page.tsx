"use client";

/**
 * Prompt library — add / edit / delete prompts, tag category + difficulty,
 * CSV import, and named prompt sets.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pencil, Plus, Save, Trash2, Upload, X } from "lucide-react";
import { api } from "@/lib/api";
import type { Prompt, PromptSet } from "@/lib/types";

const DIFFICULTIES = ["easy", "medium", "hard"];

const EMPTY: Omit<Prompt, "id"> & { id?: string } = {
  category: "general",
  difficulty: "medium",
  prompt: "",
  ground_truth: "",
};

function diffPill(d: string) {
  return d === "hard" ? "pill-red" : d === "medium" ? "pill-amber" : "pill-green";
}

/** Minimal CSV parser (handles quoted fields + escaped quotes). */
function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let cur = "", row: string[] = [], inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"' && text[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') inQ = false;
      else cur += c;
    } else if (c === '"') inQ = true;
    else if (c === ",") { row.push(cur); cur = ""; }
    else if (c === "\n" || c === "\r") {
      if (cur !== "" || row.length) { row.push(cur); rows.push(row); row = []; cur = ""; }
      if (c === "\r" && text[i + 1] === "\n") i++;
    } else cur += c;
  }
  if (cur !== "" || row.length) { row.push(cur); rows.push(row); }
  if (rows.length < 2) return [];
  const header = rows[0].map((h) => h.trim().toLowerCase());
  return rows.slice(1).map((r) =>
    Object.fromEntries(header.map((h, i) => [h, (r[i] ?? "").trim()])),
  );
}

export default function PromptsPage() {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [sets, setSets] = useState<PromptSet[]>([]);
  const [editing, setEditing] = useState<string | null>(null); // id or "new"
  const [form, setForm] = useState(EMPTY);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [setName, setSetName] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const [p, s] = await Promise.all([api.prompts(), api.sets()]);
    setPrompts(p);
    setSets(s);
  }, []);

  useEffect(() => {
    refresh().catch((e) => setStatus(`Backend unreachable: ${e.message}`));
  }, [refresh]);

  const categories = useMemo(
    () => [...new Set(prompts.map((p) => p.category))].sort(),
    [prompts],
  );

  const flash = (msg: string) => {
    setStatus(msg);
    setTimeout(() => setStatus(null), 4000);
  };

  const submit = async () => {
    try {
      if (editing === "new") {
        await api.addPrompt(form);
        flash("Prompt added.");
      } else if (editing) {
        await api.updatePrompt(editing, form);
        flash("Prompt updated.");
      }
      setEditing(null);
      setForm(EMPTY);
      await refresh();
    } catch (e) {
      flash(e instanceof Error ? e.message : String(e));
    }
  };

  const remove = async (id: string) => {
    await api.deletePrompt(id);
    flash(`Deleted ${id}.`);
    await refresh();
  };

  const importCsv = async (file: File) => {
    const rows = parseCsv(await file.text());
    const items = rows
      .filter((r) => r.prompt)
      .map((r) => ({
        id: r.id || undefined,
        category: r.category || "imported",
        difficulty: DIFFICULTIES.includes(r.difficulty) ? r.difficulty : "medium",
        prompt: r.prompt,
        ground_truth: r.ground_truth || "",
      }));
    if (!items.length) {
      flash("No usable rows — need columns: prompt, category, difficulty, ground_truth, id?");
      return;
    }
    const created = await api.bulkImport(items);
    flash(`Imported ${created.length} prompts (${items.length - created.length} duplicates skipped).`);
    await refresh();
  };

  const toggleSelect = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const saveSet = async () => {
    if (!setName.trim() || selected.size === 0) return;
    await api.saveSet({ name: setName.trim(), prompt_ids: [...selected] });
    flash(`Saved set “${setName.trim()}” (${selected.size} prompts).`);
    setSetName("");
    setSelected(new Set());
    await refresh();
  };

  return (
    <div className="mx-auto max-w-[1100px]">
      <header className="pb-2 pt-1">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-amber">
          LLM Eval Framework
        </div>
        <h1 className="text-[1.6rem] font-semibold leading-tight">Prompt Library</h1>
        <div className="text-[0.82rem] text-mute">
          {prompts.length} prompts · {categories.length} categories · {sets.length}{" "}
          saved sets · leave ground truth empty for no-ground-truth judging
        </div>
      </header>

      {status && (
        <div className="mb-3 rounded-lg border border-amber/30 bg-amber/5 px-4 py-2 font-mono text-[0.75rem] text-amber">
          {status}
        </div>
      )}

      <hr className="hdivider" />

      {/* Toolbar */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <button
          className="panel-btn panel-btn-primary flex items-center gap-1.5"
          onClick={() => {
            setEditing("new");
            setForm(EMPTY);
          }}
        >
          <Plus size={13} /> New prompt
        </button>
        <button
          className="panel-btn flex items-center gap-1.5"
          onClick={() => fileRef.current?.click()}
        >
          <Upload size={13} /> Import CSV
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void importCsv(f);
            e.target.value = "";
          }}
        />
        <span className="flex-1" />
        {selected.size > 0 && (
          <>
            <input
              className="panel-input w-44 font-mono text-[0.75rem]"
              placeholder="Set name — e.g. Coding Suite"
              value={setName}
              onChange={(e) => setSetName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void saveSet()}
            />
            <button
              className="panel-btn flex items-center gap-1.5"
              disabled={!setName.trim()}
              onClick={() => void saveSet()}
            >
              <Save size={13} /> Save set ({selected.size})
            </button>
          </>
        )}
      </div>

      {/* Saved sets */}
      {sets.length > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <span className="font-mono text-[0.65rem] uppercase tracking-[0.1em] text-mute">
            sets:
          </span>
          {sets.map((s) => (
            <span key={s.name} className="pill pill-blue inline-flex items-center gap-1.5">
              {s.name} · {s.prompt_ids.length}
              <button
                aria-label={`select set ${s.name}`}
                className="underline decoration-dotted"
                onClick={() => setSelected(new Set(s.prompt_ids))}
              >
                select
              </button>
              <button
                aria-label={`delete set ${s.name}`}
                className="hover:text-red"
                onClick={() => api.deleteSet(s.name).then(refresh)}
              >
                <X size={11} />
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Editor */}
      {editing && (
        <div className="card mb-4">
          <div className="card-title">
            {editing === "new" ? "New prompt" : `Editing ${editing}`}
          </div>
          <div className="mb-3 flex flex-wrap gap-2">
            <input
              className="panel-input w-44 font-mono text-[0.75rem]"
              placeholder="category"
              list="categories"
              value={form.category}
              onChange={(e) => setForm({ ...form, category: e.target.value })}
            />
            <datalist id="categories">
              {categories.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
            <select
              className="panel-input font-mono text-[0.75rem]"
              value={form.difficulty}
              onChange={(e) => setForm({ ...form, difficulty: e.target.value })}
            >
              {DIFFICULTIES.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </div>
          <textarea
            className="panel-input mb-2 w-full font-sans text-[0.85rem]"
            rows={3}
            placeholder="The prompt text…"
            value={form.prompt}
            onChange={(e) => setForm({ ...form, prompt: e.target.value })}
          />
          <textarea
            className="panel-input mb-3 w-full font-sans text-[0.85rem]"
            rows={3}
            placeholder="Ground truth / expected answer (optional — empty = no-ground-truth judging)"
            value={form.ground_truth}
            onChange={(e) => setForm({ ...form, ground_truth: e.target.value })}
          />
          <div className="flex gap-2">
            <button
              className="panel-btn panel-btn-primary"
              disabled={!form.prompt.trim()}
              onClick={() => void submit()}
            >
              {editing === "new" ? "Add prompt" : "Save changes"}
            </button>
            <button className="panel-btn" onClick={() => setEditing(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Prompt list */}
      <div className="flex flex-col gap-2">
        {prompts.map((p) => (
          <div
            key={p.id}
            className={`rounded-lg border bg-bg2 px-4 py-3 transition-colors ${
              selected.has(p.id) ? "border-blue/50" : "border-line"
            }`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="checkbox"
                checked={selected.has(p.id)}
                onChange={() => toggleSelect(p.id)}
                className="accent-[--blue]"
                aria-label={`select ${p.id}`}
              />
              <span className="font-mono text-[0.8rem] font-semibold">{p.id}</span>
              <span className="pill pill-mute">{p.category}</span>
              <span className={`pill ${diffPill(p.difficulty)}`}>{p.difficulty}</span>
              {!p.ground_truth && (
                <span className="pill pill-blue">no ground truth</span>
              )}
              <span className="flex-1" />
              <button
                aria-label={`edit ${p.id}`}
                className="text-mute hover:text-ink"
                onClick={() => {
                  setEditing(p.id);
                  setForm({ ...p });
                }}
              >
                <Pencil size={14} />
              </button>
              <button
                aria-label={`delete ${p.id}`}
                className="text-mute hover:text-red"
                onClick={() => void remove(p.id)}
              >
                <Trash2 size={14} />
              </button>
            </div>
            <div className="mt-1.5 text-[0.85rem] leading-relaxed">{p.prompt}</div>
            {p.ground_truth && (
              <div className="mt-1 text-[0.78rem] leading-relaxed text-mute">
                GT: {p.ground_truth}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
