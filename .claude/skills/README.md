# Thermite Claude Code skill

`thermite/` is a [Claude Code](https://claude.com/claude-code) **skill**: a usage
guide that teaches an agent how to write code *with* the Thermite SIMD library
(generic `*Vector` programming, the math/policy system, masks, dispatch, slice
iteration, the composite types `Dual`/`Compensated`, and the companion crates).
Once installed, it auto-loads when you work on Thermite-related code, or you can
invoke it explicitly with `/thermite`.

`thermite.zip` is a prebuilt bundle of that skill directory, ready to drop into
your own project.

## Install (download the zip)

1. **Get `thermite.zip`** — from the latest GitHub release, or download
   [`.claude/skills/thermite.zip`](https://github.com/raygon-renderer/thermite/blob/master/.claude/skills/thermite.zip)
   from the repo (use the "Download raw file" button).

2. **Unzip it into a `.claude/skills/` directory.** The archive roots at
   `thermite/`, so it lands as `.claude/skills/thermite/`. Choose the scope:

   - **This project only:** unzip into your project's `.claude/skills/`.
   - **All your projects:** unzip into `~/.claude/skills/` (your home directory).

   ```bash
   # macOS / Linux (project scope)
   unzip thermite.zip -d .claude/skills/
   ```
   ```powershell
   # Windows PowerShell (project scope)
   Expand-Archive thermite.zip -DestinationPath .claude/skills/
   ```

3. **Use it.** Start (or `/reload`) Claude Code in that project. The skill is
   discovered automatically and triggers on Thermite work; or run `/thermite`.

That's it — the skill is plain Markdown, no build step or dependencies.

## Install (as a plugin)

If you'd rather manage it as a versioned, updatable plugin:

```text
/plugin marketplace add raygon-renderer/thermite
/plugin install thermite@thermite
```

`/plugin marketplace update` pulls later revisions.

## Already in this repo?

If you're working inside the Thermite repo itself, the skill at
`.claude/skills/thermite/` is discovered automatically — no install needed.

---

*Maintainers:* regenerate `thermite.zip` after editing the skill with
`just bundle-skill`.
