document.querySelectorAll('.example').forEach(button => {
    const panelId = button.getAttribute('aria-controls');
    const panel = document.getElementById(panelId);
    const closeButton = panel?.querySelector('.close');

    if (panel) {
        button.addEventListener('click', () => {
            const isExpanded = button.getAttribute('aria-expanded') === 'true';
            button.setAttribute('aria-expanded', String(!isExpanded));
            button.textContent = isExpanded ? 'Show example' : 'Hide example';
            panel.hidden = isExpanded;
        });
    }

    if (closeButton && panel) {
        closeButton.addEventListener('click', () => {
            panel.hidden = true;
            button.setAttribute('aria-expanded', 'false');
            button.textContent = 'Show example';
        });
    }
});

// Dedent all <pre> elements
function dedent(text) {
    const lines = text.split('\n');
    const nonEmptyLines = lines.filter(line => line.trim());
    if (nonEmptyLines.length === 0) return text;
    const minIndent = Math.min(...nonEmptyLines.map(line => line.length - line.trimStart().length));
    return lines.map(line => line.startsWith(' '.repeat(minIndent)) ? line.slice(minIndent) : line).join('\n');
}

document.querySelectorAll('pre').forEach(pre => {
    pre.textContent = dedent(pre.textContent);
});
