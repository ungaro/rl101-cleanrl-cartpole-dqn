#!/usr/bin/env node
// Pre-process markdown by rendering mermaid blocks with beautiful-mermaid
// (zinc-light theme) into PNG images, replacing each block with an image
// reference. quadrantChart blocks are left in place — they're not yet in
// beautiful-mermaid's diagram set, so mermaid-filter handles them
// downstream.
//
// Usage:
//   node render-mermaid.mjs INPUT.md > OUTPUT.md
//
// Side effect: writes <hash>.png files to ./mermaid-beautiful/ relative
// to the working directory.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);

// Resolve the npm-global install for beautiful-mermaid (ESM-only) and sharp.
const NPM_PREFIX = process.env.NPM_PREFIX
  || `${process.env.HOME}/.nvm/versions/node/${process.versions.node ? 'v' + process.versions.node : ''}/lib/node_modules`;

async function loadModules() {
  // Prefer npm-global; fall back to node_modules in CWD.
  const candidates = [
    `${process.env.HOME}/.nvm/versions/node/v${process.versions.node.split('.')[0]}.${process.versions.node.split('.')[1]}.${process.versions.node.split('.')[2]}/lib/node_modules`,
    `${process.env.HOME}/.nvm/versions/node/v${process.versions.node}/lib/node_modules`,
  ];
  let beautiful, sharp;
  for (const dir of candidates) {
    try {
      beautiful = await import(`${dir}/beautiful-mermaid/dist/index.js`);
      sharp = require(`${dir}/sharp`);
      return {beautiful, sharp};
    } catch (e) { /* try next */ }
  }
  // Last resort: project-local resolve
  beautiful = await import('beautiful-mermaid');
  sharp = require('sharp');
  return {beautiful, sharp};
}

const SUPPORTED_RX = /^\s*(flowchart|graph|stateDiagram|sequenceDiagram|classDiagram|erDiagram|xychart-beta)\b/m;

async function main() {
  const input = process.argv[2];
  if (!input) {
    console.error('Usage: render-mermaid.mjs INPUT.md');
    process.exit(1);
  }
  const {beautiful, sharp} = await loadModules();
  const THEME = beautiful.THEMES['zinc-light'];

  const outDir = path.resolve('mermaid-beautiful');
  fs.mkdirSync(outDir, {recursive: true});

  const text = fs.readFileSync(input, 'utf8');
  const blocks = [];
  const transformed = text.replace(/```mermaid\n([\s\S]*?)\n```/g, (full, code) => {
    if (!SUPPORTED_RX.test(code)) return full;
    const hash = crypto.createHash('md5').update(code).digest('hex').slice(0, 12);
    const png = path.join(outDir, `${hash}.png`);
    blocks.push({hash, code, png});
    // Center the image and let LaTeX scale it down if needed.
    return `\n![](${path.relative(path.dirname(input), png)}){ width=90% }\n`;
  });

  // Render each block — cache by content hash so unchanged blocks reuse PNGs.
  for (const b of blocks) {
    if (fs.existsSync(b.png)) continue;
    try {
      const svg = beautiful.renderMermaidSVG(b.code, THEME);
      // Sharp wants width >= 1; we render at 2x to keep edges crisp.
      await sharp(Buffer.from(svg), {density: 192}).png().toFile(b.png);
    } catch (e) {
      // If beautiful-mermaid throws on a block, write a debug file and
      // leave the block in place by removing the replacement. We can't
      // easily un-replace at this point — log loudly so the build can
      // fall back to mermaid-filter for the offender.
      console.error(`[render-mermaid] failed: ${b.hash}: ${e.message}`);
      console.error(`--- offending block ---\n${b.code}\n-----------------------`);
      process.exit(2);
    }
  }

  process.stdout.write(transformed);
}

await main();
