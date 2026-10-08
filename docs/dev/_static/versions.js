// One manifest serves every immutable snapshot; preserve page and anchor when present.
(async () => {
  const meta = name => document.querySelector(`meta[name="${name}"]`)?.content;
  if (!meta("bloq-doc-root")) return;
  const docsRoot = new URL(meta("bloq-doc-root"), location.href);
  const siteRoot = new URL("../../", docsRoot);
  const header = document.querySelector(".md-header__inner");
  if (!header) return;
  const editor = document.createElement("a");
  editor.className = "bloq-header-link"; editor.href = new URL("editor/", siteRoot).href;
  editor.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5v14l11-7z"/></svg>Open Editor';
  const select = document.createElement("select");
  select.className = "bloq-version-select"; select.setAttribute("aria-label", "Version");
  const current = docsRoot.pathname.split("/").at(-2) || "dev";
  select.add(new Option(current === "dev" ? `Development (${meta("bloq-package-version")})` : current, current));
  header.querySelector(".md-header__title").after(select);
  const palette = header.querySelector('[data-md-component="palette"]');
  const search = header.querySelector(".md-search");
  if (palette && search) search.after(palette);
  const source = header.querySelector(".md-header__source");
  if (source) source.before(editor);
  else header.append(editor);
  if (current !== "dev") {
    const note = document.createElement("div");
    note.className = "bloq-version-note";
    note.textContent = `${current} snapshot. The editor runs the current development build.`;
    document.querySelector(".md-header")?.insertAdjacentElement("afterend", note);
  }
  try {
    const response = await fetch(new URL("versions.json", siteRoot));
    if (!response.ok) return;
    const manifest = await response.json();
    select.replaceChildren();
    for (const version of manifest.versions) select.add(new Option(version.label, version.name));
    select.value = current;
    select.addEventListener("change", () => {
      const version = manifest.versions.find(entry => entry.name === select.value);
      if (!version) return;
      const page = meta("bloq-page");
      const exists = version.pages.includes(page);
      const target = new URL(version.path + (exists ? page : "index.html"), siteRoot);
      if (exists) target.hash = location.hash;
      location.assign(target.href);
    });
  } catch { /* The current label remains useful during an offline preview. */ }
})();
