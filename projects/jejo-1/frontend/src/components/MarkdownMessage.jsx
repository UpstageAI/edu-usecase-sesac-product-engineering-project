function parseInline(text, keyPrefix) {
  const nodes = [];
  const pattern =
    /(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*)/g;
  let lastIndex = 0;
  let match;
  let cursor = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    if (match[2] && match[3]) {
      nodes.push(
        <a
          key={`${keyPrefix}-link-${cursor}`}
          href={match[3]}
          target="_blank"
          rel="noreferrer"
        >
          {match[2]}
        </a>
      );
    } else if (match[4]) {
      nodes.push(<code key={`${keyPrefix}-code-${cursor}`}>{match[4]}</code>);
    } else if (match[5]) {
      nodes.push(<strong key={`${keyPrefix}-bold-${cursor}`}>{match[5]}</strong>);
    } else if (match[6]) {
      nodes.push(<em key={`${keyPrefix}-italic-${cursor}`}>{match[6]}</em>);
    } else {
      nodes.push(match[0]);
    }

    lastIndex = pattern.lastIndex;
    cursor += 1;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

function renderParagraph(text, keyPrefix) {
  const lines = text.split("\n");
  const children = [];
  lines.forEach((line, idx) => {
    children.push(...parseInline(line, `${keyPrefix}-line-${idx}`));
    if (idx < lines.length - 1) {
      children.push(<br key={`${keyPrefix}-br-${idx}`} />);
    }
  });
  return children;
}

function isUnorderedListLine(line) {
  return /^[-*+]\s+/.test(line);
}

function isOrderedListLine(line) {
  return /^\d+\.\s+/.test(line);
}

export default function MarkdownMessage({ text }) {
  const source = String(text || "").replace(/\r\n/g, "\n");
  const lines = source.split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const language = trimmed.slice(3).trim();
      const codeLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) {
        i += 1;
      }
      blocks.push(
        <pre key={`code-${blocks.length}`} className="md-pre">
          <code className={language ? `language-${language}` : ""}>
            {codeLines.join("\n")}
          </code>
        </pre>
      );
      continue;
    }

    if (/^#{1,6}\s+/.test(trimmed)) {
      const level = Math.min(6, Math.max(1, trimmed.match(/^#+/)[0].length));
      const content = trimmed.replace(/^#{1,6}\s+/, "");
      const key = `h-${blocks.length}`;
      const children = parseInline(content, key);
      if (level === 1) blocks.push(<h1 key={key}>{children}</h1>);
      else if (level === 2) blocks.push(<h2 key={key}>{children}</h2>);
      else if (level === 3) blocks.push(<h3 key={key}>{children}</h3>);
      else if (level === 4) blocks.push(<h4 key={key}>{children}</h4>);
      else if (level === 5) blocks.push(<h5 key={key}>{children}</h5>);
      else blocks.push(<h6 key={key}>{children}</h6>);
      i += 1;
      continue;
    }

    if (trimmed.startsWith(">")) {
      const quoteLines = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      blocks.push(
        <blockquote key={`quote-${blocks.length}`}>
          {renderParagraph(quoteLines.join("\n"), `quote-${blocks.length}`)}
        </blockquote>
      );
      continue;
    }

    if (isUnorderedListLine(trimmed)) {
      const items = [];
      while (i < lines.length && isUnorderedListLine(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*+]\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ul key={`ul-${blocks.length}`}>
          {items.map((item, idx) => (
            <li key={`ul-${blocks.length}-${idx}`}>
              {parseInline(item, `ul-${blocks.length}-${idx}`)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    if (isOrderedListLine(trimmed)) {
      const items = [];
      while (i < lines.length && isOrderedListLine(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ol key={`ol-${blocks.length}`}>
          {items.map((item, idx) => (
            <li key={`ol-${blocks.length}-${idx}`}>
              {parseInline(item, `ol-${blocks.length}-${idx}`)}
            </li>
          ))}
        </ol>
      );
      continue;
    }

    const paragraphLines = [line];
    i += 1;
    while (i < lines.length) {
      const probe = lines[i];
      const probeTrimmed = probe.trim();
      if (!probeTrimmed) {
        i += 1;
        break;
      }
      if (
        probeTrimmed.startsWith("```") ||
        /^#{1,6}\s+/.test(probeTrimmed) ||
        probeTrimmed.startsWith(">") ||
        isUnorderedListLine(probeTrimmed) ||
        isOrderedListLine(probeTrimmed)
      ) {
        break;
      }
      paragraphLines.push(probe);
      i += 1;
    }

    blocks.push(
      <p key={`p-${blocks.length}`}>
        {renderParagraph(paragraphLines.join("\n"), `p-${blocks.length}`)}
      </p>
    );
  }

  return <div className="markdown-body">{blocks}</div>;
}
