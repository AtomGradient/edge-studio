// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';

interface MarkdownContentProps {
  content: string;
  className?: string;
}

export default function MarkdownContent({ content, className = '' }: MarkdownContentProps) {
  return (
    <div className={`markdown-content ${className}`}>
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        h1: ({ children }) => <h1 className="mb-2 mt-3 text-lg font-bold first:mt-0">{children}</h1>,
        h2: ({ children }) => <h2 className="mb-2 mt-3 text-base font-bold first:mt-0">{children}</h2>,
        h3: ({ children }) => <h3 className="mb-1.5 mt-2.5 text-sm font-semibold first:mt-0">{children}</h3>,
        h4: ({ children }) => <h4 className="mb-1 mt-2 text-sm font-medium first:mt-0">{children}</h4>,
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="mb-2 ml-4 list-disc space-y-0.5 last:mb-0">{children}</ul>,
        ol: ({ children }) => <ol className="mb-2 ml-4 list-decimal space-y-0.5 last:mb-0">{children}</ol>,
        li: ({ children }) => <li>{children}</li>,
        blockquote: ({ children }) => (
          <blockquote className="mb-2 border-l-3 border-stone-300 pl-3 italic text-stone-600 dark:border-stone-600 dark:text-stone-400 last:mb-0">
            {children}
          </blockquote>
        ),
        code: ({ className: codeClass, children }) => {
          const isBlock = codeClass?.startsWith('language-');
          if (isBlock) {
            return (
              <code className={`${codeClass ?? ''} block`}>{children}</code>
            );
          }
          return (
            <code className="rounded bg-stone-200/70 px-1 py-0.5 text-[0.85em] dark:bg-stone-700/70">
              {children}
            </code>
          );
        },
        pre: ({ children }) => (
          <pre className="mb-2 overflow-x-auto rounded-lg bg-stone-100 p-3 text-xs dark:bg-stone-800/80 last:mb-0">
            {children}
          </pre>
        ),
        table: ({ children }) => (
          <div className="mb-2 overflow-x-auto last:mb-0">
            <table className="min-w-full border-collapse text-sm">{children}</table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-stone-100 dark:bg-stone-800">{children}</thead>
        ),
        th: ({ children }) => (
          <th className="border border-stone-300 px-2 py-1 text-left font-medium dark:border-stone-600">{children}</th>
        ),
        td: ({ children }) => (
          <td className="border border-stone-300 px-2 py-1 dark:border-stone-600">{children}</td>
        ),
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline dark:text-blue-400">
            {children}
          </a>
        ),
        hr: () => <hr className="my-3 border-stone-200 dark:border-stone-700" />,
      }}
    >
      {content}
    </ReactMarkdown>
    </div>
  );
}
