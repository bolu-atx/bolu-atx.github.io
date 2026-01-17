function setTheme(theme) {
    document.getElementById('dark-theme-css').disabled = theme !== 'dark';
    localStorage.setItem('theme', theme);
    // add .dark to all elements with the following classes and tags
    const classes = ["site-title", "site-nav", "page-link", "site-header-bottom", "site-header", "site-footer", "menu-icon", "theme-toggle"];
    const tags = ["a", "body", "pre", "code"];

    if (theme === 'dark') {
        classes.forEach(className => {
            const elements = document.getElementsByClassName(className);
            for (let i = 0; i < elements.length; i++) {
                elements[i].classList.add('dark');
            }
        });

        tags.forEach(domType => {
            const elements = document.getElementsByTagName(domType);
            for (let i = 0; i < elements.length; i++) {
                elements[i].classList.add('dark');
            }
        })
    }
    else {
        classes.forEach(className => {
            const elements = document.getElementsByClassName(className);
            for (let i = 0; i < elements.length; i++) {
                elements[i].classList.remove('dark');
            }
        })

        tags.forEach(tag => {
            const elements = document.getElementsByTagName(tag);
            for (let i = 0; i < elements.length; i++) {
                elements[i].classList.remove('dark');
            }
        })
    }

    // Refresh Mermaid diagrams if available
    if (typeof window.refreshMermaid === 'function') {
        window.refreshMermaid();
    }

    // Dispatch custom event for other components (e.g., D3 animations)
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

function toggleTheme() {
    const currentTheme = localStorage.getItem('theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    console.log("toggling theme...", currentTheme, newTheme)
    setTheme(newTheme);
}

document.addEventListener('DOMContentLoaded', (event) => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
});