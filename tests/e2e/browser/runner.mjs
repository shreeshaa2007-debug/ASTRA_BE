// node runner.mjs — runs $E2E_SCRIPT (default export: async (b, args) => result) in headless Chrome and
// prints one line, `E2E_RESULT {json}`, that the Python tests parse. Arguments arrive as JSON in $E2E_ARGS.
import { pathToFileURL } from 'node:url';
import { launch } from './lib.mjs';

const env = process.env;
let b;
const result = {};
try {
  b = await launch({ chrome: env.E2E_CHROME, profile: env.E2E_PROFILE, port: Number(env.E2E_DEBUG_PORT), artifacts: env.E2E_ARTIFACTS });
  const script = await import(pathToFileURL(env.E2E_SCRIPT).href);
  Object.assign(result, (await script.default(b, JSON.parse(env.E2E_ARGS || '{}'))) ?? {});
} catch (err) {
  result.error = String(err?.stack ?? err);
  try { await b?.shot('failure'); } catch { /* the page may be gone */ }
} finally {
  result.problems = b?.problems ?? [];
  result.requests = b?.requests ?? [];
  await b?.close();
}
console.log('E2E_RESULT ' + JSON.stringify(result));
process.exit(0);
