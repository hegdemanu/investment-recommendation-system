// Standard Deno dependencies
export * as path from "https://deno.land/std@0.220.1/path/mod.ts";
export * as fs from "https://deno.land/std@0.220.1/fs/mod.ts";
export * as log from "https://deno.land/std@0.220.1/log/mod.ts";
export { join } from "https://deno.land/std@0.220.1/path/mod.ts";

// Express-like server for Deno
export { Application, Router, send } from "https://deno.land/x/oak@v12.6.2/mod.ts";
export type { RouterContext } from "https://deno.land/x/oak@v12.6.2/mod.ts";

// CORS middleware
export { oakCors } from "https://deno.land/x/cors@v1.2.2/mod.ts";

// Database - SQLite for simplicity
export { DB } from "https://deno.land/x/sqlite@v3.8/mod.ts"; 