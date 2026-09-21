---
name: system-tools
description: "Install a missing command-line tool: detect it, ask the user, install ffmpeg, imagemagick, poppler, tesseract or pandoc via brew, apt or winget, then verify it works."
---

# Getting a command-line tool the machine does not have

Read this guide when a command you need is missing — the shell said
`command not found` (or Windows said `is not recognized`) — and when a task
turns out to need media or document tooling: converting a video or audio file,
making a thumbnail, extracting text from a PDF, OCR, format conversion.

**This product ships no bundled binaries and installs nothing at first run.**
Tooling a task actually needs is acquired here, on demand, on the user's machine.
That is deliberate: nothing is downloaded into their app-support directory for
them, and nothing is installed that no task asked for.

## The rule that matters

**Never install anything silently.** The harness approving your tool call is not
the user approving a change to their machine; those are two consents and the
console guide already carries that rule — do not restate it, follow it. Say what
you are about to install and why, in plain language, with the exact command, and
get their answer **before** anything privileged runs.

And the sentence that decides most of these cases: **if the change is not worth
the user's approval, do not ask for it.** Installing a package manager and then a
60 MB media framework to perform one image resize is a bad trade. See
"When installing ffmpeg is the wrong answer" below.

## The shape of the operation

### 1. Detect: is it there, and does it actually work

```sh
if command -v ffmpeg >/dev/null 2>&1; then echo "present: $(command -v ffmpeg)"; else echo "MISSING"; fi
```

`command -v` is not the same question as "it works", and the gap between them is
where a fix goes wrong:

- It answers *is a file by that name executable somewhere on `PATH`*. It says
  nothing about whether the tool can do the task.
- It is **silent** — it prints a path, and nothing at all when absent — which is
  exactly why it is written with `>/dev/null` and tested for its exit status
  rather than read for a word.
- A present-but-broken tool is the common case: a Homebrew tool whose dynamic
  libraries moved, a `.local/bin` symlink into a directory that no longer exists.
  Verify by running the thing, not by finding it.

**A shell that answered `command not found` is only evidence about THAT shell.**
The console surface starts the user's own shell, and what is on its `PATH`
depends on how that shell was started: a login shell reads `~/.zprofile`,
`~/.bash_profile` or `~/.profile`, while a non-login one may read none of them —
and on Windows a process sees the `PATH` its parent had at launch. **Measured in
the Console on a macOS host with Homebrew installed:** the surface's `bash`
reported `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, `command -v ffmpeg` said
`MISSING`, and `/opt/homebrew/bin/ffmpeg` ran fine from the very same surface.
`ffmpeg` was installed the whole time.

So check it twice before installing anything:

```sh
command -v ffmpeg                    # this shell's PATH
zsh -lc 'command -v ffmpeg'          # macOS: what a LOGIN shell finds
/opt/homebrew/bin/ffmpeg -version     # macOS: what the manager installed
```

A tool that exists but is not on this shell's `PATH` is a **one-line fix, not an
install**, and it is the first thing to rule out — see "When the installer
finishes" under each platform below for the fix.

### 2. Decide: install, or route around it

Three questions, in this order:

1. **Is the task real and wanted?** Installing is a side effect on someone else's
   machine; it has to buy them something they asked for.
2. **Is there a cheaper path?** A tool they already have (see the alternatives
   column in the media table), a pure-Python package, or asking them to hand you
   the converted file. Prefer what is already on the machine.
3. **Is this a package-manager install or a one-file download?** A package
   manager is almost always right: it is signed, versioned, updatable and
   uninstallable by the user later. A raw binary you download and `chmod +x` is
   the fallback for a machine where the user cannot or will not grant root.

### 3. Explain: what they are being asked to allow, in their words

Not the command line — the consequence. Name the package, what it is for, how
big it is, and what it will change. For example:

> I need FFmpeg to convert your video. It is not installed on this machine. I
> would install it with Homebrew (`brew install ffmpeg`, about 100 MB with its
> dependencies). It will be added to Homebrew's own folder and uninstallable
> later with `brew uninstall ffmpeg`. Homebrew on this machine is already set
> up, so no administrator password is needed. Shall I go ahead?

If a package manager is not present yet, **say so in the same breath** and
explain the extra step it needs — it is a bigger change than one package and the
user is entitled to hear that before they agree to the small one.

### 4. Approve: `ask`, with the exact command

Use `ask` with the exact command and what it will change, every time, before
anything privileged runs. Do not type a password yourself: if the command needs
one, it must be the user's to type (see the platform sections). Where a stored
secret exists for it, pass it by name with `secret_ref` — never as text.

Do not batch several packages into one approval to save a round trip, and do not
sneak an install into a longer command the user already approved. Approval is per
change.

### 5. Run it in the Console

The Console is where an install belongs. It drives a real pty, so a package
manager's progress output, its prompts and its `sudo`/UAC interaction arrive
where the user can see and answer them — which a non-interactive `bash` call
cannot host at all. Read `guide://console` first if you have not used a surface
before.

Two console facts that decide how you drive it:

- **stdout and stderr are one stream.** A pty has a single channel, so do not
  try to separate a warning from progress in the output.
- **An idle surface is not a finished one.** There is no in-band signal for "the
  program is waiting for input", so an install that has stopped printing may be
  running, downloading, or sitting at a prompt. Read it, and if the last line is
  a question, that question is the user's to answer, not yours.

```sh
# macOS, once: does the tool respond through the shell's own PATH now?
zsh -lc 'ffmpeg -version' || echo 'PATH fix needed in this surface'
eval "$(brew shellenv)"        # for the rest of THIS surface
```

That last line is what to run in a console surface after a Homebrew install; it
is a session-local fix. Making it permanent means a line in the user's own
`~/.zprofile`, which is a change to their shell config — ask, do not do it
silently. On Linux a missing `PATH` entry is the same problem with the same
answer (`source ~/.profile`, or a new login shell). On Windows the fix is
usually a **new shell**, because a process keeps the `PATH` it was launched
with.

### 6. Verify it works, then report what changed

Run the command again from step 1, then prove it does the task, not just that it
exists — a version banner only proves the binary loads. Convert something. The
verification command per tool is in the tables below.

Then tell the user what changed: the package, the command that ran, the version
now installed, and how to undo it (`brew uninstall ffmpeg`, `sudo apt remove
ffmpeg`, `winget uninstall …`). A user who agreed to a change is owed the receipt
for it.

## macOS

Homebrew is the right manager: it installs into its own prefix with **no `sudo`
after the initial setup**, it is the same tool the user will use themselves, and
`brew uninstall` reverses it.

After installing, `command -v` in the SAME surface may still say nothing — see
step 1. `brew install` symlinks into `/opt/homebrew/bin` (or `/usr/local/bin`)
and tells the user their shell config needs the prefix on `PATH`; the tool works
from that prefix immediately, and the surface needs `eval "$(brew shellenv)"`
to see it. Verify against the absolute path or after that line, and never treat
the unchanged `command -v` as a failed install.

```sh
brew install ffmpeg
/opt/homebrew/bin/ffmpeg -version | head -1   # works immediately
eval "$(brew shellenv)" && ffmpeg -version | head -1   # works through PATH too
```

### When `brew` is not installed at all

This is the part that is never documented for a non-technical user, and it is the
one step in this guide that is genuinely intrusive, so do not start it without
saying all of it up front:

- Homebrew's installer needs **Command Line Tools for Xcode**. If they are
  missing, macOS shows its own dialog (*"The 'xcode-select' command requires the
  command line developer tools. Would you like to install the tools now?"*) and
  the install stalls behind it until the user clicks. The agent cannot click it.
- The install itself asks for the **user's administrator password**.
- It is a large download and a real change to their system.

The vendor's own instructions are the authority, and the command is on
<https://brew.sh> (this is what that page gives you today):

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

The script prints what it will do and pauses for confirmation before doing it.
That confirmation, the Xcode dialog and the password are all the user's. Hand the
surface to them, tell them what each one is, and wait — do not look for a flag
that answers them for the user.

**When the installer finishes, `brew` may still be "not found" in your shell.**
That is not a failed install: Homebrew installs into `/opt/homebrew` (Apple
Silicon) or `/usr/local` (Intel), and the shell only finds it after the shell
config exports the prefix. The installer prints the exact line for the user's
shell; it is the same one Homebrew's own installation page documents:

```sh
eval "$(/opt/homebrew/bin/brew shellenv)"
```

For the rest of that surface, prefix it once and keep working; for the user's own
terminals, the line belongs in their `~/.zprofile`, which is a change to their
shell config — tell them rather than doing it silently.

## Linux

Use the distribution's own manager. It is signed, it is what `man` and the
distribution's documentation assume, and the user can remove the package later.

| distribution | install | check it works |
|---|---|---|
| Debian, Ubuntu | `sudo apt update && sudo apt install ffmpeg` | `ffmpeg -version \| head -1` |
| Fedora, RHEL | `sudo dnf install ffmpeg` | `rpm -q ffmpeg` |
| Arch, Manjaro | `sudo pacman -S ffmpeg` | `pacman -Q ffmpeg` |
| openSUSE | `sudo zypper install ffmpeg` | `rpm -q ffmpeg` |

`ffmpeg` is not in every distribution's default repositories — Fedora and openSUSE
need RPM Fusion, and on some distributions it lives in `ffmpeg-free`. If the
package is not found, say so and let the user decide whether adding a repository
is wanted; do not add one on your own initiative.

**After the install, check the same shell again.** A distribution package drops
its binary in `/usr/bin` or `/usr/local/bin`, which is normally already on
`PATH` — so `command -v` finding it is the ordinary result — but a static or
configured-on-a-different-prefix install may land elsewhere. If the package
manager says the package is installed and `command -v` still says nothing, read
where the files went (`dpkg -L ffmpeg | grep bin/`, `rpm -ql ffmpeg`) before
concluding anything.

**The `sudo` prompt.** These commands need root, so the surface will print, on
its own line:

```
[sudo] password for <user>:
```

The user types it. Keystrokes into a pty are never recorded, so the password does
not enter the surface's output or your record — but the assistant is not
authorised to type it either, whether or not it could. If they have stored it,
`console_input {surface, secret_ref: "SUDO_PASSWORD"}` supplies it without you
ever seeing the value; otherwise hand the surface to them and wait.

**When the user will not or cannot grant root.** A static build into the user's
own `~/.local/bin` needs no privileges at all and is the honest alternative:

```sh
mkdir -p ~/.local/bin
# the distribution's own -static or -noprefix link, or the project's release page
curl -fsSL <url> -o ~/.local/bin/ffmpeg && chmod +x ~/.local/bin/ffmpeg
command -v ffmpeg || echo 'export PATH="$HOME/.local/bin:$PATH"' # if it is not already on PATH
```

Say plainly that this route gets no updates and no provenance checking, and that
`~/.local/bin` has to be on their `PATH`. It is a reasonable choice, not a
preferred one.

## Windows

Windows matters here for one reason the other two platforms do not have: **an
elevated command raises a UAC consent dialog, and the Console surface cannot
answer it.** The user clicks Allow or the install does not happen. Write the
procedure around that, never through it.

Available managers, in the order to try them:

| manager | install | notes |
|---|---|---|
| `winget` | `winget install --id Gyan.FFmpeg -e` | Ships with modern Windows 10 and 11. `-e` is exact-id matching, which matters: a bare query can match several packages and winget stops to ask. |
| `choco` | `choco install ffmpeg` | Needs an elevated shell **to install Chocolatey itself**, and its own install raises UAC. |
| `scoop` | `scoop install ffmpeg` | Installs per-user, no elevation — **the one path that fits a non-elevated console surface**. Needs PowerShell, and its install is `Set-ExecutionPolicy`-scoped to the current process. |

**What the user actually sees.** Run any of these in a console surface as the
user's own shell and one of three things happens, all of them fine and only one
of them entirely yours:

- The installer is a per-user one and completes in the surface. Nothing to click.
- Windows raises the UAC dialog (*"Do you want to allow this app to make changes
  to your device?"*) — a **secure desktop** prompt, drawn by the OS, on top of
  everything, which the pty can neither see nor answer.
- The package manager itself asks a question (`winget` needs
  `--accept-package-agreements` for some packages; `scoop` may ask which bucket
  to use).

In the second and third cases the agent's job is to stop: say what is on their
screen, that they need to click or answer it themselves, wait, and then read the
surface for the result. Do not retry the command to "make it take", do not
attempt elevation yourself, and do not conclude the install failed while the
dialog is still up. And note that winget's `--scope user` is not a promise —
Microsoft documents that an EXE-based installer in user scope *may still require
UAC authorization*.

```powershell
# detect (PowerShell)
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { ffmpeg -version | Select-Object -First 1 } else { "MISSING" }
```

`winget` itself is part of the App Installer and is normally already there on
Windows 11 and current Windows 10. If it is genuinely absent, it comes from the
Microsoft Store, which is a user action, not a command you can run for them.

**After the install, open a new shell before verifying.** A Windows process keeps
the `PATH` it was launched with, so a freshly installed tool usually does not
appear in the surface that installed it — `scoop` updates the user `PATH`, which
takes effect for processes started afterwards. That is a new surface, not a
failed install, and it is the Windows form of the macOS `PATH` trap above.

## Media and document tooling

These are the tools a media or document task actually reaches for. Package names
differ per platform — installing the wrong one is the most common failure in this
guide — so the table is by tool, not by task.

| tool | what it is for | macOS | Debian/Ubuntu | Fedora/RHEL | Arch | Windows | provenance check |
|---|---|---|---|---|---|---|---|
| ffmpeg | convert, cut, encode, mux video and audio; thumbnails | `brew install ffmpeg` | `sudo apt install ffmpeg` | `sudo dnf install ffmpeg` | `sudo pacman -S ffmpeg` | `winget install --id Gyan.FFmpeg -e` | `ffmpeg -version \| head -1` |
| ffprobe | inspect a file without converting it — codecs, duration, streams | ships with ffmpeg | `sudo apt install ffmpeg` | ships with ffmpeg | ships with ffmpeg | ships with ffmpeg | `ffprobe -version \| head -1` |
| ImageMagick | convert, resize, crop, compose images; PDF-to-image (needs ghostscript) | `brew install imagemagick` | `sudo apt install imagemagick` | `sudo dnf install ImageMagick` | `sudo pacman -S imagemagick` | `winget install --id ImageMagick.ImageMagick -e` | `magick -version \| head -1` |
| poppler | text and images out of PDFs (`pdftotext`, `pdftoppm`, `pdfinfo`) | `brew install poppler` | `sudo apt install poppler-utils` | `sudo dnf install poppler-utils` | `sudo pacman -S poppler` | `winget install --id oschwartz10612.Poppler -e` | `pdftotext -v` (see note) |
| ghostscript | the PostScript/PDF engine ImageMagick and `mutool` lean on | `brew install ghostscript` | `sudo apt install ghostscript` | `sudo dnf install ghostscript` | `sudo pacman -S ghostscript` | **no winget package** — `choco install ghostscript`, or the Artifex installer | `gs --version` |
| pandoc | document format conversion (`docx` ↔ `md`, `html`, `epub`) | `brew install pandoc` | `sudo apt install pandoc` | `sudo dnf install pandoc` | `sudo pacman -S pandoc` | `winget install --id JohnMacFarlane.Pandoc -e` | `pandoc --version \| head -1` |
| tesseract | OCR: text out of a scanned image or PDF page | `brew install tesseract` | `sudo apt install tesseract-ocr` | `sudo dnf install tesseract` | `sudo pacman -S tesseract` | `winget install --id UB-Mannheim.TesseractOCR -e` | `tesseract --version \| head -1` (see note) |
| yt-dlp | download media a site publishes | `brew install yt-dlp` | `sudo apt install yt-dlp` | `sudo dnf install yt-dlp` | `sudo pacman -S yt-dlp` | `winget install --id yt-dlp.yt-dlp -e` | `yt-dlp --version` |

Three caveats in that table are load-bearing:

- **`convert` is not ImageMagick on Windows** — `convert.exe` is a filesystem
  tool. ImageMagick 7's command is `magick`, which is what the row checks.
- **`pdftotext -v` and `tesseract --version` are special.** `pdftotext` has no
  `--version`; `-v` prints a banner to stderr and it treats an unknown option as
  a filename. `tesseract --version` prints a usage block first. Verify these two
  with the real task instead: `pdftotext file.pdf -` and
  `tesseract image.png -` are the honest tests.
- **`ghostscript` has no winget package.** Its `ArtifexSoftware` manifest is
  `mutool`, not Ghostscript; on Windows it is the Artifex installer or Chocolatey.

Package names in the Linux columns are apt/dnf/pacman names, not upstream names:
`imagemagick` is lowercase, the RPM world calls it `ImageMagick`, and tesseract's
package is `tesseract` on RPM and `tesseract-ocr` on Debian. Ask the manager
before installing when unsure — `apt-cache policy <name>`, `dnf info <name>`,
`pacman -Si <name>` — rather than guessing and installing a similarly-named
package that is not the tool.

### When installing ffmpeg is the wrong answer

Do not reach for a package manager when the task is one operation on one file and
the machine already has something that does it:

- **Resize, convert or crop one image on macOS**: `sips` is built in —
  `sips -Z 1200 input.png --out output.png`. No install, no approval.
- **One image conversion anywhere else**: most tasks that reach for ImageMagick
  are a pure-Python package away (`Pillow`), which installs into the session's own
  environment rather than the user's machine.
- **Reading one PDF's text**: `pdftotext` may already be present; if not, `pypdf`
  or `pdfminer.six` reads most PDFs without a system package.

What those do **not** cover is video and audio. There is no built-in media
conversion in this product and no Python package that substitutes for FFmpeg's
codec set, so a video or audio task genuinely needs the install. Say that in one
sentence and get the approval, rather than hunting for a cheaper route that
cannot work.

## When it cannot be done

Say so and stop, with the reason — these are all cases where the honest answer is
cheaper than the workaround:

- **No Console on this host.** The `console` tool is absent when the desktop app
  is not running. You can still detect a missing tool, but the install needs a
  pty because installers prompt; do not fake one with a piped `bash` call, and do
  not install a terminal emulator to stand in. Tell the user what to reopen.
- **The user declines.** Report what the task needed and what it will not be able
  to do without it, and carry on with the rest of the work.
- **The package is not in their distribution's repositories** and adding a
  repository is required. That is a second change to their machine, and it is
  theirs to decide.
- **The tool needs a licence or an account** (some codecs, some commercial PDF
  tooling). Not something an agent can obtain for them.
