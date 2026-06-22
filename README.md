# Adam Razali — ePortfolio

A personal portfolio website built as a fast, zero-build static site. No frameworks,
no build step — just HTML, CSS, and a little JavaScript. It deploys anywhere that
serves static files (Vercel, GitHub Pages, Netlify).

## 🗂 Structure

```
index.html            # All page content (edit text/projects here)
assets/
  css/style.css       # Styles + light/dark theme
  js/main.js          # Theme toggle, mobile nav, scroll animations
  favicon.svg         # Site icon
semester 3/ ...        # Coursework (linked from the site)
semester 5/ ...
```

## ✏️ How to update it later

Everything is plain HTML — open `index.html` and edit the text directly.

- **Add a project:** copy one `<article class="project"> … </article>` block in the
  Projects section and change the title, description, tags, and links.
- **Add a course:** copy a `<a class="course"> … </a>` block in the Education section.
- **Change skills:** edit the `<li>` chips in the Skills section.
- **Update links:** the GitHub, email, and LinkedIn links live in the hero and contact
  sections (search for `TODO` to find the LinkedIn placeholder).

## 🔍 Preview locally

Just open `index.html` in your browser. Or run a tiny local server:

```bash
# Python
python -m http.server 8000
# then visit http://localhost:8000
```

## 🚀 Deploy

**Vercel** — import the GitHub repo at [vercel.com/new](https://vercel.com/new),
keep the defaults (framework: *Other*, no build command, output: root), and deploy.

**GitHub Pages** — in the repo go to *Settings → Pages*, set the source to the
`main` branch / root, and save. Your site goes live at
`https://adamrazal1.github.io/eportfolio/`.

Either way, every push to `main` redeploys automatically.
