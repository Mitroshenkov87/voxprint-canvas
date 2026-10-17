# Canvas feature map

Where each current window and command lives on the shared screen.
Status is `done`, `TODO`, `NEEDS OWNER DECISION`, `pending Yitro`, or `removed (owner decision 2026-10-17)`.

`pending Yitro` waits for the shared image surface in voxprint-common PR 13. The pin stays at `f4e95cc` until that tag's SHA arrives. The source picture and the mask will opt into the image-surface slots. The generator will receive the live settings `shell.steps`, `shell.guidance`, `shell.sampler`, `shell.seed`, and `shell.resolution`.

| Feature | Old place | New place | Status |
| --- | --- | --- | --- |
| App name | Window title | Simple title | done |
| About | About button | Overflow Settings opens About | done |
| Start from words | Text tab | More, then Picture setup: From text | done |
| img2img (restyle) | Image tab, Restyle | More, then Picture setup: Restyle, plus Use a picture | done |
| inpaint | Image tab, Inpaint; `generate --mask` | More, then Picture setup: Paint over. Edit the mask appears after a picture is chosen | done |
| Extend | Image tab, Outpaint | More, then Picture setup: Extend, plus the mask editor | done |
| structure guidance, pose | Image tab, Pose | More, then Picture setup: Copy the pose, plus Use a picture | done |
| structure guidance, depth | Image tab, Depth | More, then Picture setup: Depth map, plus Use a picture | done |
| structure guidance, edges | Image tab, Edge | More, then Picture setup: Outlines, plus Use a picture | done |
| Instruction editing | Image tab; `generate --edit-instruction` | removed (owner decision 2026-10-17) | removed (owner decision 2026-10-17) |
| Prompt | Prompt box | Simple prompt | done |
| Improve prompt | Enhance | Simple Improve prompt, same field | done |
| Revert | Revert | Simple Revert to my prompt | done |
| Aspect ratios | Aspect picker | Size: Landscape, Square, or Portrait, then the ratio menu | done |
| Match a picture's shape | Input size | Use a picture | done |
| 4K via Enlarge | 4K checkbox | Output: 4K (needs Enlarge) until Enlarge is installed | done |
| Install Enlarge | Install Enlarge | Install, beside that 4K label | done |
| File type | AVIF, JPEG XL, WebP list | More, then Picture setup: File type | done |
| Generate | Generate | Simple Generate | done |
| Status line | Status label | Result sentence | done |
| Add-ons and the third-party notice | Add-ons checkbox | More, then Add-ons; Expert folder and strength | done |
| Responsibility line | Footer | Footer | done |
| In-app model download | Download question | More, then Manage models (MDL) | done |
| Splash | Startup splash | Startup splash | done |
| Quick picture | Text mode | More, then Picture setup: Usual | done |
| More varied picture | Creative mode on 24 GB | More, then Picture setup: Bolder | done |
| Look | (styles were not a separate control) | Simple Look chips | done |
| How many pictures | (not on the old window) | Simple count chips, 1 / 2 / 4 | done |
| Light, medium, strong | (not a separate old control) | Shown on the Output row only when 4K is chosen | done |
| History | (not on the old window) | Simple history strip. A finished picture shows More like this, Save, Open folder, Use prompt again, Use as input, and Delete. Each history item's menu still has Open folder and Delete | done |
| Steps, guidance, seed, sampler, negative prompt | (not on the old window) | Expert panel | done |
| VRAM cap | `--vram-cap-gb` | Expert VRAM limit, and the command | done |
| Adult content | (not on the old window) | Expert. It changes only the prompt policy | done |
| Exact size, including 2K | `--output-size 2k` | Expert Exact size, and the command | done |
| Source picture | Use a picture; `generate --input` | Image surface source-picture slot, opt-in | pending Yitro |
| Mask | Edit the mask; `generate --mask` | Image surface mask slot, opt-in | pending Yitro |
| Structure guidance | Pose, depth, and edges | Image surface source-picture slot, opt-in | pending Yitro |
| Expert steps | Expert panel | `shell.steps` in the live settings passed to the generator | pending Yitro |
| Expert CFG | Expert panel | `shell.guidance` in the live settings passed to the generator | pending Yitro |
| Expert sampler | Expert panel | `shell.sampler` in the live settings passed to the generator | pending Yitro |
| Expert seed | Expert panel | `shell.seed` in the live settings passed to the generator | pending Yitro |
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
| `--selftest` | Command | More, then Check this PC, and the command | done |
| `--deep` | Command | Check every file, and the command | done |
| `--check-update` | Command | Check for an update, and the command | done |
| `--version` | Command | About, and the command | done |
| `--json` | Command | Command only | NEEDS OWNER DECISION |
| `--register-models` | Command | Command only | NEEDS OWNER DECISION |
| `--unregister-models` | Command | Command only | NEEDS OWNER DECISION |
| Manage models | (download question) | More, then Manage models | done |
