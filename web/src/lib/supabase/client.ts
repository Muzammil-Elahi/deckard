import { createBrowserClient } from "@supabase/ssr";
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "./types";

export function createSupabaseBrowserClient(): SupabaseClient<Database> | null {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseAnonKey) {
    if (typeof window !== "undefined") {
      console.warn(
        "Missing Supabase browser credentials. Check NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY."
      );
    }
    return null;
  }

  return createBrowserClient<Database>(supabaseUrl, supabaseAnonKey);
}
