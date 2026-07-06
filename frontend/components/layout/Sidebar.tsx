"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FlaskConical,
  LayoutDashboard,
  GitCompareArrows,
  ListChecks,
  FileText,
} from "lucide-react";
import { api } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/compare", label: "Compare", icon: GitCompareArrows },
  { href: "/prompts", label: "Prompts", icon: ListChecks },
  { href: "/report", label: "Report", icon: FileText },
];

export function Sidebar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      api
        .health()
        .then((h) => {
          if (!cancelled) {
            setHealth(h);
            setHealthError(false);
          }
        })
        .catch(() => {
          if (!cancelled) setHealthError(true);
        });
    check();
    const t = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  return (
    <aside className="sticky top-0 flex h-screen w-56 flex-shrink-0 flex-col border-r border-line bg-bg2 px-4 py-6 max-lg:w-16 max-lg:px-2">
      {/* Wordmark */}
      <Link href="/" className="mb-8 flex items-center gap-2.5 px-2">
        <FlaskConical size={20} className="flex-shrink-0 text-amber" />
        <div className="max-lg:hidden">
          <div className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-amber">
            LLM Eval
          </div>
          <div className="text-[0.7rem] text-mute">Framework v2</div>
        </div>
      </Link>

      {/* Nav */}
      <nav className="flex flex-col gap-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[0.82rem] transition-colors ${
                active
                  ? "bg-bg3 text-ink"
                  : "text-mute hover:bg-bg3/60 hover:text-ink"
              }`}
            >
              <Icon size={16} className="flex-shrink-0" />
              <span className="max-lg:hidden">{label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="flex-1" />

      {/* API status */}
      <div className="px-1 max-lg:px-0">
        {healthError ? (
          <span className="pill pill-red">⚠ API offline</span>
        ) : health ? (
          health.api_key_set ? (
            <span className="pill pill-green">● key set</span>
          ) : (
            <span className="pill pill-red">⚠ no key</span>
          )
        ) : (
          <span className="pill pill-mute">connecting…</span>
        )}
        {health && (
          <div className="mt-2 font-mono text-[0.62rem] leading-relaxed text-mute max-lg:hidden">
            {health.models_evaluated} models · {health.prompts} prompts
            <br />
            backend v{health.version}
          </div>
        )}
      </div>
    </aside>
  );
}
