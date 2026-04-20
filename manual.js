// Dedent all <pre> elements
function dedent(text) {
    const lines = text.split('\n');
    const nonEmptyLines = lines.filter(line => line.trim());
    if (nonEmptyLines.length === 0) return text;
    const minIndent = Math.min(...nonEmptyLines.map(line => line.length - line.trimStart().length));
    return lines.map(line => line.startsWith(' '.repeat(minIndent)) ? line.slice(minIndent) : line).join('\n');
}

window.addEventListener('load', () => {
    document.querySelectorAll('pre').forEach(pre => {
        pre.textContent = dedent(pre.textContent);
    });

    // Ensure all collapsed elements are hidden by default unless aria-expanded is true
    document.querySelectorAll('.collapse').forEach(collapse => {
        const button = document.querySelector(`[aria-controls="${collapse.id}"]`);
        const isExpanded = button && button.getAttribute('aria-expanded') === 'true';
        collapse.classList.toggle('expanded', isExpanded);
        collapse.hidden = !isExpanded;
    });

    // Handle example button clicks
    document.querySelectorAll('.example').forEach(button => {
        const panelId = button.getAttribute('aria-controls');
        const panel = document.getElementById(panelId);
        const closeButton = panel?.querySelector('.close');

        if (panel) {
            button.addEventListener('click', () => {
                const isExpanded = button.getAttribute('aria-expanded') === 'true';
                if (!isExpanded) {
                    document.querySelectorAll('.collapse.expanded').forEach(other => {
                        if (other !== panel) {
                            other.classList.remove('expanded');
                            other.hidden = true;
                            const otherButton = document.querySelector(`[aria-controls="${other.id}"]`);
                            if (otherButton) {
                                otherButton.setAttribute('aria-expanded', 'false');
                                otherButton.textContent = 'Show example';
                            }
                        }
                    });
                }
                button.setAttribute('aria-expanded', String(!isExpanded));
                button.textContent = isExpanded ? 'Show example' : 'Hide example';
                panel.classList.toggle('expanded', !isExpanded);
                panel.hidden = isExpanded;
            });
        }

        if (closeButton && panel) {
            closeButton.addEventListener('click', () => {
                panel.classList.remove('expanded');
                panel.hidden = true;
                button.setAttribute('aria-expanded', 'false');
                button.textContent = 'Show example';
            });
        }
    });
});

// END
