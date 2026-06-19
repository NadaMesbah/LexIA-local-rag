'use client';

import { useState, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import Box from '@mui/material/Box';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';

// Recursively extract the raw text of a node (for the per-block copy button).
function nodeText(children: ReactNode): string {
  if (children == null) return '';
  if (typeof children === 'string' || typeof children === 'number') return String(children);
  if (Array.isArray(children)) return children.map(nodeText).join('');
  if (typeof children === 'object' && 'props' in (children as any)) {
    return nodeText((children as any).props?.children);
  }
  return '';
}

function CodeBlock({ children }: { children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    const text = nodeText(children);
    const done = () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    };
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(done).catch(() => {});
    else done();
  };
  return (
    <Box sx={{ position: 'relative', my: 1.25 }}>
      <Tooltip title={copied ? 'Copié' : 'Copier'}>
        <IconButton
          size="small"
          onClick={copy}
          sx={{
            position: 'absolute',
            top: 6,
            right: 6,
            color: copied ? 'success.light' : 'grey.400',
            bgcolor: 'rgba(255,255,255,0.06)',
            '&:hover': { bgcolor: 'rgba(255,255,255,0.16)', color: '#fff' },
          }}
        >
          {copied ? <CheckRoundedIcon fontSize="small" /> : <ContentCopyRoundedIcon fontSize="small" />}
        </IconButton>
      </Tooltip>
      <pre>{children}</pre>
    </Box>
  );
}

export default function Markdown({ children }: { children: string }) {
  return (
    <Box className="md" sx={{ fontSize: 14, color: 'text.primary', wordBreak: 'break-word' }}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={{
          pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
        }}
      >
        {normalizeMath(children)}
      </ReactMarkdown>
    </Box>
  );
}

// Qwen emits LaTeX with \( \) and \[ \] delimiters, but remark-math only knows
// $…$ / $$…$$ (and markdown would otherwise strip the backslashes). Convert them
// so equations render. Code blocks are left untouched.
function normalizeMath(src: string): string {
  if (!src) return src;
  const blocks: string[] = [];
  // protect fenced + inline code from the delimiter swap
  let s = src.replace(/```[\s\S]*?```/g, (m) => `\u0000C${blocks.push(m) - 1}\u0000`);
  s = s.replace(/`[^`\n]*`/g, (m) => `\u0000C${blocks.push(m) - 1}\u0000`);
  s = s
    .replace(/\\\[/g, () => '$$')
    .replace(/\\\]/g, () => '$$')
    .replace(/\\\(/g, () => '$')
    .replace(/\\\)/g, () => '$');
  s = s.replace(/\u0000C(\d+)\u0000/g, (_, i) => blocks[Number(i)]);
  return s;
}