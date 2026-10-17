# Canvas feature map

Where each current window and command lives on the shared screen.
Status is `done`, `TODO`, `NEEDS OWNER DECISION`, or `removed (owner decision 2026-10-17)`.

| Feature | Old place | New place | Status |
| --- | --- | --- | --- |
| App name | Window title | Simple title | done |
| About | About button | Overflow Settings opens About | done |
| Start from words | Text tab | Picture setup: Words | done |
| img2img (restyle) | Image tab, Restyle | Picture setup: Restyle, plus Use a picture | done |
| inpaint | Image tab, Inpaint; `generate --mask` | Picture setup: Paint over, plus Edit the mask | done |
| Extend | Image tab, Outpaint | Picture setup: Extend, plus the mask wizard | done |
| structure guidance, pose | Image tab, Pose | Picture setup: Copy the pose, plus Use a picture | done |
| structure guidance, depth | Image tab, Depth | Picture setup: Use a depth map, plus Use a picture | done |
| structure guidance, edges | Image tab, Edge | Picture setup: Use the edges, plus Use a picture | done |
| Instruction editing | Image tab; `generate --edit-instruction` | removed (owner decision 2026-10-17) | removed (owner decision 2026-10-17) |
| Prompt | Prompt box | Simple prompt | done |
| Improve prompt | Enhance | Simple Improve prompt, same field | done |
| Revert | Revert | Simple Revert | done |
| Aspect ratios | Aspect picker | Simple Size chips | done |
| Match a picture's shape | Input size | Use a picture | done |
| 4K via Enlarge | 4K checkbox | Simple Output: 4K | done |
| Install Enlarge | Install Enlarge | 4K chip action | done |
| File type | AVIF, JPEG XL, WebP list | Picture setup: File type | done |
| Generate | Generate | Simple Generate | done |
| Status line | Status label | Result sentence | done |
| Add-ons and the third-party notice | Add-ons checkbox | Add-ons wizard; Expert folder and strength | done |
| Responsibility line | Footer | Footer | done |
| In-app model download | Download question | Manage models (MDL) | done |
| Splash | Startup splash | Startup splash | done |
| Quick picture | Text mode | Picture setup: Quick | done |
| More varied picture | Creative mode on 24 GB | Picture setup: More varied | done |
| Look | (styles were not a separate control) | Simple Look chips | done |
| How many pictures | (not on the old window) | Simple count chips, 1 / 2 / 4 | done |
| History | (not on the old window) | Simple history strip | done |
| Steps, guidance, seed, sampler, negative prompt | (not on the old window) | Expert panel | done |
| VRAM cap | `--vram-cap-gb` | Expert VRAM limit, and the command | done |
| Adult content | (not on the old window) | Expert. It changes only the prompt policy | done |
| Exact size, including 2K | `--output-size 2k` | Expert Exact size, and the command | done |
| `generate --prompt` | Command | Simple prompt, and the command | done |
| `generate --input` | Command | Use a picture, and the command | done |
| `generate --mask` | Command | Edit the mask, and the command | done |
| `generate --aspect` | Command | Size chips, and the command | done |
| `generate --out` | Command | Save, and the command | done |
| `generate --output-size native` | Command | Output Standard, and the command | done |
| `generate --output-size 4k` | Command | Output 4K, and the command | done |
| `generate --seed` | Command | Expert seed, and the command | done |
| `generate --enhance` | Command | Improve prompt, and the command | done |
| `generate --edit-instruction` | Command | removed (owner decision 2026-10-17) | removed (owner decision 2026-10-17) |
| `--selftest` | Command | Check this PC, and the command | done |
| `--deep` | Command | Check every file, and the command | done |
| `--check-update` | Command | Check for an update, and the command | done |
| `--version` | Command | About, and the command | done |
| `--json` | Command | Command only | NEEDS OWNER DECISION |
| `--register-models` | Command | Command only | NEEDS OWNER DECISION |
| `--unregister-models` | Command | Command only | NEEDS OWNER DECISION |
| Manage models | (download question) | Overflow Manage models | done |
