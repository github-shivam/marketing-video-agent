# Marketing Video Agent

A LangGraph multi-agent pipeline. Give it a topic or a link; it works out what
you are asking for, writes a full screenplay, renders it into the exact prompt
syntax of the video model, generates the video and hands it back. Posting to
Instagram is an optional last step.

```
python main.py "explain how RAG works to non-technical people"
python main.py https://example.com/product
python main.py "a launch teaser for this" https://example.com --model seedance
```

```
START → classify → research → writer ⇄ critic → dialogue → approve_script
        → video → confirm_video → (post) → END
```

| Node | What it does | Tools |
|---|---|---|
| `classify` | Detects a link in code, fetches the page, then classifies the request into one of six intents (product promo, explainer, how-to, brand story, announcement, entertainment) plus audience, tone, key message and whether a person should speak | `httpx` |
| `research` | Facts, hook angles and filmable visuals for the subject | Bright Data MCP (optional) |
| `writer` | Writes a model-agnostic **screenplay** (shots, camera, action, dialogue, sound), then renders it into the target model's prompt | none |
| `critic` | Scores 0–10 strictly against the intent and the model's rules, loops back to the writer | none |
| `dialogue` | Caption and hashtags (only used if you post) | none |
| `approve_script` | **Human gate** — read the exact prompts for free | — |
| `video` | Sends the approved prompts to the video model, saves the MP4 and the prompts beside it | Veo 3.1 Fast |
| `confirm_video` | **Human gate** — the video is already saved; answer `y` to post it | — |
| `publish` | Instagram Graph API | none (plain code) |

The critic loops back to the writer while `score < SCORE_THRESHOLD` **and**
`iterations < MAX_ITERATIONS`. Rejecting at the prompt gate with feedback sends
it back to the writer and resets the iteration count; rejecting without feedback
ends the run.

## How it works, in plain words

Think of it as a small production team made of AI models, with you as the
producer who signs off twice.

1. **You give it an idea**: a topic ("explain how RAG works") or a link to a
   product page. Optionally add a length (4–30 seconds).
2. **It works out what you want.** The `classify` step decides what kind of video
   this is (product promo, explainer, how-to, and so on), who it is for and what
   tone fits. If you gave a link it reads the page first.
3. **It researches the subject** for facts, hooks and things that can be filmed.
4. **It writes a screenplay, and a critic marks it.** A writer drafts shot-by-shot
   scenes (camera, action, dialogue, sound). A critic scores the draft from 0 to
   10 and sends it back for another pass until it is good enough or the retry
   limit is reached.
5. **You approve the script (gate 1).** You see the exact text that will be sent
   to the video model. This is free, so nothing is spent until you say yes.
6. **It generates the video** with Google Veo 3.1 and saves the MP4 in `output/`.
7. **You decide whether to post (gate 2).** The video is already saved. Say yes to
   post it to Instagram, or no to keep it only.

The text steps run on AWS Bedrock (Claude models). Video generation runs on
Google Vertex AI (Veo). With no Google credentials it falls back to a free
placeholder video made by ffmpeg, so you can try the whole flow at no cost.

## How to run it

### 1. Install

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) on your PATH.

```bash
git clone https://github.com/<your-username>/marketing-video-agent.git
cd marketing-video-agent

python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements-dev.txt
```

### 2. Configure

```bash
cp .env.example .env                # Windows PowerShell: copy .env.example .env
```

Open `.env` and set the minimum below. Leave everything else as it is.

| What | Setting | Notes |
|---|---|---|
| AWS login for the text models | `AWS_PROFILE`, `BEDROCK_TEXT_REGION` | Run `aws sso login --profile <your-profile>` first |
| One model per step | `MODEL_CLASSIFY`, `MODEL_RESEARCH`, `MODEL_WRITER`, `MODEL_CRITIC`, `MODEL_DIALOGUE`, `MODEL_CONVERSE` | Required. The values in `.env.example` work if your AWS account has access to them |
| Safety switch | `DRY_RUN=true` | Keep it `true` until you want a real Instagram post |

Leave `GOOGLE_CLOUD_PROJECT` and `VEO_GCS_BUCKET` blank to use the free stub video.
Fill them in to generate real videos (see *Veo on Vertex AI* below). Keep `.env`
private: it is gitignored and must never be committed.

### 3. Check it works

```bash
pytest
```

All tests use fakes, so this needs no network and no credentials.

### 4. Run it

**From the command line (simplest)**

```bash
python main.py "explain how RAG works to non-technical people"
python main.py https://example.com/product --seconds 20
python main.py "a launch teaser for this" https://example.com --model seedance
```

| Option | Meaning |
|---|---|
| `--seconds N` | Video length, 4–30 (default 15) |
| `--model veo\|seedance` | Which model's prompt style to write (Seedance is a stub for now) |
| `--thread ID` | Name the run. Use the same ID later to resume a paused run |

The program pauses twice and asks you a question in the terminal:

- **Prompt review**: type `y` to generate, `n` to be asked what should change (it
  then rewrites the script), or `q` to quit without generating.
- **Video review**: type `y` to post to Instagram, `n` to keep the video unposted.

The finished video and the prompts used are in `output/`.

**As an HTTP API**

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # put this in .env as API_KEY
python -m uvicorn api:create_app --factory --port 8000
```

Then open http://localhost:8000/docs. See *HTTP API* below for the routes.

**Through WhatsApp**

Run the API above, expose it over HTTPS, and connect it to a WhatsApp number. See
*WhatsApp → Setting it up* below.

### Common problems

| Problem | Fix |
|---|---|
| Error naming a missing `MODEL_...` setting | Set that variable in `.env` |
| AWS credential or "token expired" error | Run `aws sso login --profile <your-profile>` again |
| ffmpeg not found | Install ffmpeg and reopen the terminal so PATH updates |
| Nothing posts to Instagram | Expected while `DRY_RUN=true`. Set `false` only when ready |
| Run stopped part-way | Re-run the same command with the same `--thread ID` |

## How the prompt is written

The LLM writes **content**; code applies **syntax**. The writer fills a
`Screenplay` (segments of timed shots with camera, subject, action, setting,
speaker, dialogue, sfx, ambient, music). `agents/render.py` then turns it into
the format each model documents, so a model can never forget a bracket:

| | Veo 3.1 | Seedance 2.5 |
|---|---|---|
| Generation | 8s base clip + 7s extensions, one prompt each | One pass, up to 30s, one prompt |
| Shots | `[00:00-00:02] Medium shot, …` | `Shot 1 [0-3s]: Medium shot, …` |
| Order inside a shot | cinematography → subject → action → context | camera → subject → action → setting |
| Dialogue | `The presenter says warmly, "…"` | `The presenter says warmly: {…}` |
| Sound | `SFX: …` / `Ambient noise: …` / `Music: …` | `<sfx>` / `<ambient>` / `(music)` |
| Exclusions | none (Google: describe what *is* there) | `No subtitles. No watermarks or logos. No duplicated people.` |

The writer's system prompt also carries the researched guidance
(`agents/prompt_guides.py`): one camera move per shot, identical descriptions
for recurring people, no real names, no reliance on in-frame text, gapless
timelines, a spoken-word budget per segment, and a structural template per
intent. Timelines are repaired in code (`retime`) when a model leaves gaps or
overshoots, rather than discarding the draft.

Sources: Google's Veo 3.1 prompting guide (Google Cloud blog) and ByteDance's
Seedance 2.5 announcement and prompt guidance.

## Why two approval gates

Prompt review is free, video generation is not (~$1.50–2.30 per 15s Veo reel,
roughly $0.15 per generated second).
The first gate shows the exact prompts that will be sent, so money is only ever
spent on text a person has already read. The second gate comes after the video
is saved and decides only one thing: whether it goes to Instagram.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # Windows
# source .venv/bin/activate && pip install -r requirements-dev.txt  # macOS/Linux

cp .env.example .env     # then fill it in
pytest                   # 380 tests, no network, no credentials needed
python main.py "explain how RAG works to non-technical people"
```

Requires Python 3.10+ (developed on 3.14) and `ffmpeg` on PATH for the offline
stub provider.

### Running with no credentials at all

Works out of the box. With Veo not configured (no `GOOGLE_CLOUD_PROJECT` and
`VEO_GCS_BUCKET`) the pipeline uses a stub video provider that renders a real
720×1280 / 15s / H.264 placeholder with ffmpeg, and
with `DRY_RUN=true` nothing is ever sent to Instagram. Only the text model needs
AWS credentials.

## Configuration

Everything lives in `.env`. See `.env.example` for the annotated list.

| Setting | Default | Notes |
|---|---|---|
| `DRY_RUN` | `true` | `false` **really posts to Instagram** |
| `VIDEO_MODEL` | `veo` | `veo` or `seedance`; `--model` overrides per run |
| `ASSET_MODEL` / `UPLOADS_DIR` | `gemini-2.5-flash` / `uploads` | The Vertex AI model that reads uploads, and where files are kept until the run ends |
| `VIDEO_SECONDS_DEFAULT` | `15` | Video length when the person does not say. 4–30s. See *Video length* |
| `VEO_TIMESTAMPS` | `true` | `[00:00-00:02]` shots. Not verified live with extension; set `false` to fall back to `Shot 1:` labels |
| `MODEL_CLASSIFY`, `MODEL_RESEARCH`, `MODEL_WRITER`, `MODEL_CRITIC`, `MODEL_DIALOGUE`, `MODEL_CONVERSE` | — | **Required**, one Bedrock model per node. There is no default |
| `MODEL` | — | Optional shared fallback for any node left blank |
| `BEDROCK_TEXT_REGION` | `us-east-1` | Where Bedrock is called. `AWS_PROFILE` is read by boto3 |
| `SCORE_THRESHOLD` / `MAX_ITERATIONS` | `8` / `3` | Writer⇄critic loop |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` / `VEO_GCS_BUCKET` | — / `us-central1` / — | Veo on Vertex AI. Blank project or bucket = the free stub. See *Veo on Vertex AI* |
| `VEO_MODEL` | `veo-3.1-fast-generate-001` | GA. `veo-3.1-generate-001` is the non-Fast model |
| `VEO_RESOLUTION` | `720p` | Extension is 720p-only |
| `UPLOAD_STRATEGY` | `resumable` | `resumable` sends bytes to Meta; `url` uses S3 |
| `S3_BUCKET` | — | Only needed when `UPLOAD_STRATEGY=url` |

Each node names its own model, so cost and quality can be tuned per step. A
node with no model, and no shared `MODEL`, stops with an error naming the
missing setting rather than running on a default nobody chose. Newer Claude
models on Bedrock are only available through an inference profile, so use the
`us.` form of the ID (e.g. `us.anthropic.claude-sonnet-4-6`), not the bare one.
The research node runs a multi-step tool loop, which is the most demanding use,
so give it a strong model.

## Instagram prerequisites

A real publish needs all four:

1. **An Instagram Business or Creator account.** Personal accounts cannot use
   the Content Publishing API. Convert in the app under
   *Settings → Account type and tools*.
2. **A Facebook Page linked to that account**, via
   *Settings → Business tools and controls → Branded content* or Meta Business Suite.
3. **A Meta app** at [developers.facebook.com](https://developers.facebook.com)
   with the Instagram Graph API product and the **`instagram_content_publish`**
   permission. Also add `instagram_basic` and `pages_read_engagement`. Outside
   of development mode this requires App Review.
4. **A long-lived access token.** Short-lived tokens last an hour; exchange for
   a long-lived one (60 days) and refresh before it expires.

Get your `IG_USER_ID` from:

```
GET https://graph.facebook.com/v25.0/me/accounts?access_token=TOKEN
GET https://graph.facebook.com/v25.0/<PAGE_ID>?fields=instagram_business_account&access_token=TOKEN
```

### Going live

1. Set `GOOGLE_CLOUD_PROJECT` and `VEO_GCS_BUCKET` to switch from the stub to
   real Veo generation (setup below).
2. Set `IG_USER_ID` and `IG_ACCESS_TOKEN`.
3. Set `DRY_RUN=false`.

### Veo on Vertex AI

Veo is reached through Vertex AI with Google Cloud credentials, the way Bedrock
is reached with an AWS profile. **There is no API key.** The SDK finds
Application Default Credentials on its own, and billing goes to your GCP
project. Vertex writes each clip to a Cloud Storage bucket, so a bucket is
required.

Setup, once:

```bash
gcloud auth application-default login            # the "aws sso login" for our code
gcloud auth application-default set-quota-project PROJECT_ID
gcloud services enable aiplatform.googleapis.com
gcloud storage buckets create gs://BUCKET --location=us-central1
```

The identity needs `roles/aiplatform.user` on the project and
`roles/storage.objectAdmin` on the bucket. On a server use a service account
(`GOOGLE_APPLICATION_CREDENTIALS` or an attached identity) instead of a login.
An extension reads its base clip from the bucket and the finished clip is
downloaded from it. Clips stay there until deleted, so add a lifecycle rule
(for example, delete after 7 days):

```bash
echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":7}}]}' > lifecycle.json
gcloud storage buckets update gs://BUCKET --lifecycle-file=lifecycle.json
```

No S3 required. The default `UPLOAD_STRATEGY=resumable` POSTs the local file's
bytes straight to Meta, so the video is never hosted anywhere.

Publishing is capped at **100 posts per rolling 24 hours**, enforced on
`media_publish`.

## How the video reaches Instagram

The video is always written to `OUTPUT_DIR` (default `output/`) and recorded in
`media.local_files`. `UPLOAD_STRATEGY` decides how Instagram gets it:

**`resumable` (default)** — open a session with `upload_type=resumable`, then
POST the bytes to
`https://rupload.facebook.com/ig-api-upload/<VERSION>/<CONTAINER_ID>` with
`Authorization: OAuth <token>`, `offset: 0` and `file_size`. Nothing is hosted,
no S3, no presigned URL, no public exposure of client media. Requires an app
using **Facebook Login for Business** — which the Page-linked setup above gives
you. Apps on Instagram Login cannot use this.

**`url`** — upload to S3 and hand Meta a presigned URL to fetch. Use this if you
are on Instagram Login, or if you want the asset archived in your own bucket
anyway. Needs `S3_BUCKET`, `S3_REGION` and `PRESIGN_EXPIRY`.

Either way the 300 MB ceiling applies; oversized files are rejected locally
before anything is uploaded.

## Video length

The person chooses how long the video is: **4 to 30 seconds, 15 by default**.

| Where | How |
|---|---|
| WhatsApp | Say it in the message: "explain RAG in 20 seconds", "a half minute ad", or just "30 sec". Hinglish works. A link message works too |
| CLI | `python main.py "explain RAG" --seconds 20` |
| HTTP API | `{"request": "...", "seconds": 20}` on `POST /videos` |

Veo cannot make an arbitrary length: it makes a base clip of 4, 6 or 8 seconds
and extends it in steps of exactly 7. `video_plan.py` turns the length asked for
into the generations that make it, and **the video is trimmed to the exact
length**:

| Asked | Generated | Kept |
|---|---|---|
| 5s | 6s | 5s |
| 15s | 8 + 7 | 15s |
| 20s | 8 + 7 + 7 = 22s | 20s |
| 25s | 8 + 7 + 7 + 7 = 29s | 25s |
| 30s | 8 + 7 + 7 + 7 = 29s | **29s** |

- **The story finishes inside the time asked.** The writer is told the exact
  length and which seconds are kept, so the speech and the call to action end
  before the cut. Nothing is chopped mid-sentence.
- **You pay for what is generated**, which can be a few seconds more than is kept.
- **A length that would keep just one second of a final extension** (9, 16, 23 or
  30) is made one second shorter instead: paying for seven seconds to keep one is
  not worth it. A video is never longer than asked, and never more than a second
  shorter.
- **Out of range is clamped, and the person is told** ("The longest video I can
  make is 30 seconds, so I'll make it 30").
- **A longer video needs 720p**, because extension is 720p-only. With
  `VEO_RESOLUTION=1080p` anything over 8s fails before a cent is spent.
- Trimming re-encodes with ffmpeg, so **ffmpeg must be on PATH** wherever the
  server runs. If it is missing the video is delivered untrimmed and its real
  length is reported, not the one asked for.

In WhatsApp the length can also be changed while reviewing the prompts ("make it
25 seconds instead"): the script is rewritten for the new length.

## Uploads: photos, video and audio

A person can send photos, a video clip or a voice note **before** their request,
and the video is made from them: it takes on their colours, mood and setting,
features the person in them, and sounds like the audio.

```
You   : [sends 2 photos of their bakery and the baker]
Bot   : Got your photo. Send more if you like, then tell me what to make...
You   : [sends a voice note with the music they like]
You   : a 20 second ad for my bakery
Bot   : On it. I'll make a 20-second video. I'll take the look, the person and
        the sound from your 2 photos and audio clip. ...
Bot   : *From your files*
        Colours: warm amber, cream and deep teal (#E8A33D #F5E6D0 #1F6F78)
        Mood: warm, calm
        Person: the baker in the green apron (your photos go to the video model...)
        Audio: a slow acoustic folk tune
```

A photo or video with a caption is both at once: the caption is the request.
Up to 6 files of 25 MB each. Files sent while a video is being made are not taken.

| Upload | What is taken from it | How it is used |
|---|---|---|
| Photo, or stills from a video | Colours (measured from the pixels), mood, setting, visual style, and any person (described, never named) | Written into the script's style and continuity |
| The person in them | Up to 3 clear photos of them | Sent to Veo as **subject reference images**, so it keeps their face |
| Video | The same, plus pacing and camera movement, and its sound | As above |
| Audio or a voice note | Mood, genre, tempo, instruments, language, and any words | Steers the script and the music and ambience cues. **It is not mixed into the video** |

**Who reads them.** The `analyze` node (`agents/assets.py`) uses Gemini on Vertex AI
(`ASSET_MODEL`, default `gemini-2.5-flash`): the only model here that takes images,
video and audio natively, and it uses the Google Cloud login Veo already needs. The
colours are not left to it: they are measured with Pillow, because a model describes
a colour well and reports a colour code badly. The node only runs when there are
files, and a failure never fails the run: the video is made from the written
request alone and the person is told.

**The person.** Their description is copied word for word into the script, and the
critic checks it, so they look the same in every shot even if Veo takes no photos.
If Veo **rejects the photos** (its safety filter, or a combination it does not
support) the video is made from the description instead, and the person is told:
*"Veo couldn't use your photos as references (...), so the person comes from a
description and may not look exactly like them."* Any other failure, such as running
out of quota, is not hidden behind that fallback.

- Photos sent to Veo force an **8-second base clip** (a documented Veo rule), so a
  5s video is generated as 8s and trimmed. Nothing else about length changes.
- The model is told never to name or identify anyone from how they look, and
  to treat words in a caption, a photo or a voice note as data, never as
  instructions.
- **Only upload photos of people who agreed to appear.** The bot answers only the
  numbers in `WHATSAPP_ALLOWED_NUMBERS`, and every Veo video carries a SynthID
  watermark, but a photo of someone else is still a photo of someone else.

**Privacy.** Files apply to one video and are **deleted when its run ends**: when
it is posted, declined, cancelled or fails, along with everything made from them
while reading them. Files nobody used are deleted after an hour, and a run's files
after 24 hours at most if the chat is simply abandoned. Photos are sent to Google
Cloud for reading and, if there is a person, to Veo.

## HTTP API

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # put the result in .env as API_KEY
.venv/Scripts/python.exe -m uvicorn api:create_app --factory --port 8000
```

A video takes minutes and pauses twice for a person, so this is a job API
rather than one request that returns a file. Every route except `/health`
needs an `X-API-Key` header, and the server refuses to start without
`API_KEY` set, because every approved request spends Veo money. Interactive
docs are at `/docs`.

| Route | Purpose |
|---|---|
| `POST /videos` `{"request": "...", "model": "veo", "seconds": 20}` | Start a run. `model` and `seconds` are optional. Returns `202` and a `video_id` at once |
| `GET /videos/{id}` | Status, plus the gate's payload (the exact prompts) when paused |
| `POST /videos/{id}/approve` `{"approved": true, "feedback": "...", "seconds": 25}` | Answer the current gate, same as `y`/`n` in the CLI |
| `GET /videos/{id}/video` | Download the finished MP4 |

Statuses: `classifying`, `researching`, `writing`, `awaiting_prompt_approval`,
`generating`, `awaiting_post_decision`, `publishing`, then one of `completed`
(video made, not posted), `posted`, `ended` (declined at the prompt gate),
`failed` or `interrupted` (the server restarted mid-step).

```bash
KEY=...                                  # your API_KEY
curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"request": "explain how RAG works"}' localhost:8000/videos
# -> {"video_id": "veo-3f9a1c2b7d4e", "status_url": "/videos/veo-3f9a1c2b7d4e"}

curl -s -H "X-API-Key: $KEY" localhost:8000/videos/veo-3f9a1c2b7d4e
# -> {"status": "awaiting_prompt_approval", "awaiting": {"prompts": [...]}, ...}

curl -s -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"approved": true}' localhost:8000/videos/veo-3f9a1c2b7d4e/approve
```

State lives in the SQLite checkpointer, not in server memory, so a run paused
at a gate can be answered after a restart. Answering a gate while the run is
still working returns `409`. Run **one** server process: SQLite and the local
`output/` folder are not shared between instances (Postgres checkpoints and S3 are the next step for several instances).

Limits of this version: files are served from local disk, there is one shared
API key rather than per-user auth, and no CORS headers (add
`CORSMiddleware` if a browser frontend on another origin will call it).

## WhatsApp

The same pipeline, driven from a chat. A user messages a topic or a link; the
bot researches and writes, sends the exact prompts, waits for approval, sends
the finished video into the chat, and asks whether to post it. Replies are free
text, not `y`/`n`.

```
You   : explain how RAG works to non-technical people
Bot   : On it. I'll write the script and send you the exact prompts to approve...
Bot   : *Prompt 1 of 2* ...  (then the caption, the critic's score)
Bot   : Should I generate it? Say yes to go ahead, tell me what to change, or say cancel.
You   : looks good but the first line is too long
Bot   : Got it. Rewriting the script with your changes. One moment.
Bot   : (new prompts)  Should I generate it?
You   : haan kar do
Bot   : Great, generating your video now. This usually takes a few minutes.
Bot   : [the video]  Should I post this to Instagram?
You   : not yet, let me show my team first
Bot   : I didn't catch that. Should I post this video to Instagram? ...
You   : no
Bot   : Okay, I won't post it. The video is saved...
```

**How replies are understood.** `agents/converse.py` reads each message *as the
answer to the question just asked* and returns an intent: `approve`, `change`
(with the user's feedback, which goes to the writer exactly like a human
rejection at the CLI), `decline`, `unclear`, or, with nothing open, `request` /
`chat`. It handles Hinglish and typos. Two approvals cost something that cannot
be undone (Veo spend, and a public post), so:

- hedging ("maybe", "later", "let me check") is `unclear` and the bot asks again;
- "yes, but make it shorter" is a `change`, not an approval;
- an intent that does not belong to the question is thrown away, so a model slip
  cannot post a video nobody agreed to post;
- if the model call fails, the user is asked again rather than guessed at.

A link in a message is a request by definition and skips the interpreter.

**How long the video is** is read from the message too. For free text the
interpreter understands the sentence, so "a hook in the first 3 seconds" is not
taken as a 3-second video. A message with a link never reaches the interpreter,
so its length is read by a plain pattern match, which skips a number that follows
"first", "last", "every" or "each". The bot always says what it will make
("On it. I'll make a 20-second video.").

**Who can use it.** `WHATSAPP_ALLOWED_NUMBERS` is required: the bot silently
ignores everyone else. Veo spend and the Instagram account are shared, so an
open number would let strangers spend your money and post as you. `*` opens it
to anyone, deliberately. `WHATSAPP_MAX_RUNS_PER_DAY` (default 10) caps each
number. Each number has its own conversation and cannot answer another's gate.

### Setting it up

Allow 30 to 45 minutes the first time. You need a Meta developer account (log in
at developers.facebook.com with a Facebook account), a phone with WhatsApp, and
the API running somewhere Meta can reach over HTTPS.

**How the pieces connect.** You never connect your computer to WhatsApp directly.
Meta owns the WhatsApp number. You tell Meta two things: "this number belongs to
my app", and "when a message arrives, send it to this web address" (the
*webhook*). Meta then calls your server for every message, and your server
replies by calling Meta back with an access token.

```
Your phone  →  Meta (owns the number)  →  public HTTPS address  →  your server
Your phone  ←  Meta                    ←  your server replies using the access token
```

#### Step 1. Create the Meta app

1. At developers.facebook.com, open **My Apps → Create App**.
2. Give it a plain name (Meta rejects names containing "Meta", "Facebook",
   "Instagram" or "WhatsApp").
3. Choose the use case **Connect with customers through WhatsApp**. If you also
   want Instagram posting, add that use case to the same app later; one app can
   do both.
4. When it asks for a **business portfolio**, pick one or create one. A brand-new
   Facebook account may be told it is "too new to create a business account";
   wait about an hour and try again.

#### Step 2. Claim the free test number

1. Open **Use cases → Connect with customers through WhatsApp → Customize**.
2. Click **Step 1. Try it out** in the left menu. Meta creates a free **test
   WhatsApp Business account** and a **test number** for you.
3. Under **Recipient**, add your own phone and enter the verification code
   WhatsApp sends you. The test number can only message phones you add here, up
   to 5. Skip *Step 2. Production setup* and *Step 3. Business verification* for
   now; they are only for a real number.
4. Send Meta's sample "Hello World" message to confirm your phone receives it.
   That proves the number, the account and your phone are linked.

#### Step 3. Collect five values and put them in `.env`

Never paste these into chat, email or a ticket, and never commit `.env`. Write
each value after the `=` with no quotes, no spaces and no angle brackets.

| Setting | Where to find it |
|---|---|
| `WHATSAPP_PHONE_NUMBER_ID` | On the Step 1 page, next to the test number, labelled **Phone number ID**. Not the phone number itself, and not the "WhatsApp Business account ID" beside it |
| `WHATSAPP_ACCESS_TOKEN` | On the same page. See *Access tokens* below |
| `WHATSAPP_APP_SECRET` | **App settings → Basic → App secret → Show**. This is the app's main secret. The **Instagram app secret** shown on the Instagram page is a different one and will not work here |
| `WHATSAPP_VERIFY_TOKEN` | Any long random string you make up. You will paste the same string into Meta in Step 5 |
| `WHATSAPP_ALLOWED_NUMBERS` | Your own phone, digits only with the country code first (for example, a number written `+91 98xxx xxxxx` becomes `9198xxxxxxxx`). Separate several with commas. `*` allows anyone, which you should not use while the bot can spend money and post for you |

Leave `DRY_RUN=true` for the first tests so nothing is posted to Instagram.

**Access tokens.** The token on the Step 1 page is **temporary and expires in
about an hour**, which is fine for a first try. When it expires, the bot can still
receive your messages but every reply is rejected, and the server log says the
token is invalid or expired. For anything longer, create a **System User** token:

1. At business.facebook.com, open **Business settings** for your portfolio.
2. **Users → System users → Add**, give it a name and the **Admin** role.
3. **Add assets**: give it full control of your app and of your WhatsApp account.
4. **Generate new token**, choose your app, set expiry to **Never**, and tick
   `whatsapp_business_messaging` and `whatsapp_business_management`.
5. Put it in `WHATSAPP_ACCESS_TOKEN` and restart the server. The server reads
   `.env` only at start-up, so any change needs a restart.

#### Step 4. Start the server and a public address

1. Set `API_KEY` in `.env` if you have not already (see *HTTP API*), then start
   the API as described in *How to run it → As an HTTP API*, on port 8000.
   If WhatsApp is only partly configured, the server refuses to start and names
   the missing setting.
2. Meta cannot reach `localhost`, so give the server a public HTTPS address. For
   development, a tunnelling tool such as **ngrok** does this: create a free
   account, install it, and point it at port 8000. It prints an
   `https://…` address. Keep both programs running.
3. Check the address works by opening it with `/health` on the end in a browser.
   You should see a short "status ok" answer.

A free tunnel address can change when the tunnel restarts. If it does, repeat
Step 5 with the new address. A hosted server with a fixed address needs no
tunnel.

#### Step 5. Tell Meta where to send messages

Your server's webhook address is your public address followed by
`/whatsapp/webhook`.

1. In the WhatsApp use case, open **Configuration**. In some dashboard layouts
   this menu is hidden; try the dropdown at the top of the left menu, **Next
   steps**, or **Tools**.
2. Under **Webhook**, click **Edit**. Enter the webhook address as the **Callback
   URL**, and your `WHATSAPP_VERIFY_TOKEN` as the **Verify token**.
3. Click **Verify and save**. Meta calls your server once to check the token; this
   only works while the server and the tunnel are running.
4. Under **Webhook fields**, turn on **messages**.

If the Configuration page cannot be found, the same two things can be done
through Meta's Graph API: register the callback URL, verify token and the
`messages` field on the app's WhatsApp subscription (this needs the app ID and
the app secret), then subscribe your WhatsApp Business account to the app. Check
the result by reading the subscriptions back.

#### Step 6. Test

1. From your phone, open the chat with the **test number** and send a message,
   for example a link or "explain how RAG works to non-technical people". **The
   first message must come from you.** The bot can only reply within 24 hours of
   your last message.
2. A few seconds later you should get "On it…", then the prompts and the
   question "Should I generate it?" about a minute or two after that.
3. For a free first test, answer `cancel`. Generating the video costs real Veo
   credits once you approve.
4. Then try the full flow: ask for a change in your own words, approve, watch the
   video arrive, and answer the post question (nothing is posted while
   `DRY_RUN=true`).

If a message arrives, the server log shows a line for the webhook request. The
tunnel's web interface (ngrok serves one on local port 4040) lists every request
Meta sent, which is the quickest way to see whether Meta is reaching you at all.

#### If something goes wrong

| What you see | Likely cause and fix |
|---|---|
| **Verify and save** fails | The server or tunnel is not running, the address does not end with `/whatsapp/webhook`, or the verify token differs from `WHATSAPP_VERIFY_TOKEN` |
| Your message is received but there is never a reply; the log mentions an invalid or expired token (error 190) | The temporary token expired. Generate a new one (or a System User token), update `.env`, restart the server |
| Meta's requests show `401` in the tunnel list | `WHATSAPP_APP_SECRET` is wrong, most often the Instagram secret was used. Use the one from **App settings → Basic** |
| No request reaches the tunnel at all | The **messages** field is not subscribed, the WhatsApp account is not subscribed to the app, or the callback address is out of date after a tunnel restart |
| The bot never answers your number | Your number is not in `WHATSAPP_ALLOWED_NUMBERS`, or it has spaces, a `+` or dashes. Strangers are ignored by design |
| The bot stops answering after a day or two of silence | The 24-hour window closed. Send it a message first |
| "Account too new to create a business" | Meta's limit on new accounts. Wait an hour and retry |
| The server will not start and names a setting | WhatsApp is partly configured. Fill the setting it names, or clear every `WHATSAPP_*` value to switch the feature off |

#### Before real use

The test number only reaches the phones you added. To let other people message
the bot you need your own business number registered with Meta, business
verification, and a System User token that does not expire. Until then, keep the
allow-list short, because every approved request spends Veo credits and a `yes`
posts to the connected Instagram account.

If a token or secret is ever pasted somewhere it should not be, treat it as
leaked: regenerate it in Meta, update `.env`, and restart the server.

#### How the webhook is protected

Meta signs every delivery with your app secret. An unsigned or wrongly signed
request is answered `401` and starts nothing, and the webhook does not use the API
key header. Numbers outside the allow-list are ignored silently, and each number
is capped by `WHATSAPP_MAX_RUNS_PER_DAY`. Leave every `WHATSAPP_*` setting blank
and the WhatsApp routes do not exist at all; `/videos` and the command line are
unaffected.

### Limits

- **24-hour window.** WhatsApp only allows free-form messages within 24 hours of
  the user's last message. A chat that is answering gates stays inside it; if
  someone walks away overnight, the bot's next push will be rejected (logged).
- **Photos, video and audio are taken as uploads.** Documents, stickers and
  locations get a reply saying what the bot can read.
- **16 MB video cap.** A 15s 720p reel is about 8 MB, but a 30s one can come
  close to the limit. A video over it is compressed for the chat only: Instagram
  still gets the original. If it cannot be made to fit, the bot says so, still
  asks the post question, and tells you where the file is saved.
- **No cancel mid-run.** While a run is working, further messages get a status
  line. Say "cancel" at either gate to stop.
- **One server process**, as with the HTTP API. `conversations.sqlite` sits
  beside `checkpoints.sqlite`; a paused run can be answered after a restart.
- Not exercised against live WhatsApp or live Bedrock. The tests cover the wiring
  with fakes, and the interpreter prompt has not yet been tried on a real model.
  Try a few replies before relying on it.

## Resuming a paused run

State is checkpointed to SQLite, so a run paused at an approval gate survives
the process exiting:

```bash
python main.py "explain RAG" --thread my-run-id   # starts (or resumes)
```

## Testing

```bash
pytest           # 380 tests
pytest -q tests/test_instagram.py
```

Every test runs against fakes — mocked LLMs, a recording video provider, and
`respx` for the Graph API and page fetches. `tests/test_pipeline.py` runs the
real nodes through the real graph and a real SQLite checkpointer, faking only
the LLM boundary. No test touches a model, AWS, or Instagram. The
default test settings deliberately leave `s3_bucket` empty so a test *cannot*
accidentally upload.

Covered: link detection and classification, both prompt formats, the loop
exiting at the threshold and at the iteration cap, all three prompt-approval
branches, both video-confirm branches, checkpoint resume across
instances, the full Reels create → poll → publish sequence, error-body
surfacing, and that a failed container is never published.

## Notable constraints encoded in the code

These were verified against live docs and a real AWS account, and several
contradict what the documentation implies:

- **`duration_seconds` is an `int` in the Python SDK**, though the REST API
  documents it as a string enum.
- **`person_generation` must be `"allow_all"` for text-to-video AND for
  extension.** `allow_adult` is for image-to-video, interpolation and
  reference images. In the EU, UK, CH and MENA only `allow_adult` is
  permitted, which conflicts with text-to-video — those regions need an
  image-to-video flow instead.
- **`seed` exists only on Veo 3 models**, not Veo 3.1, so generations are not
  reproducible.
- **4k is available** (Veo 3.1 & Fast, 8s only, never with extension), but
  Instagram caps at 1080p so it is cost without benefit here.
- **The extension call must restate `aspect_ratio`.** Omit it and the API
  defaults to 16:9, then rejects a 9:16 input with
  `Aspect ratio of the input video must be 16:9, but got: 9:16` — which reads
  like vertical video cannot be extended. It can: verified against the live
  API producing a 15.000s 720x1280 clip. The source and the requested output
  simply have to match.
- **Extension is 720p-only**, needs an 8s base and (on Vertex) a 7s output
  duration, so anything over 8s is 720x1280. For native 1080x1920 set
  `VEO_RESOLUTION=1080p`, which limits videos to 8s. Each extension adds
  exactly 7s and is unavailable on Veo 3.1 Lite. The product stops at 30s, which
  is four generations.
- **Seedance 2.5 is a stub for now.** Its prompt format, 30s single-pass
  duration and exclusions are implemented and tested, but its API is not wired
  in: ByteDance's own announcement lists ModelArk access as "coming soon", and
  third-party hosts (e.g. Together AI, `ByteDance/Seedance-2.5`) do not document
  resolution, aspect ratio or an audio toggle. `--model seedance` runs the full
  flow against the offline ffmpeg stub and shows the real Seedance prompt.
- **Seedance 2.5's timestamps do not apply to 2.0.** Do not mix versions in
  one sequence; 2.0 only follows shot numbers.
- **Veo 3.1 base clips are 4, 6 or 8s** per Google's docs. Everything longer is
  an 8s base plus 7s extensions, trimmed to the length asked for.
- **There is no `on_screen_text` any more.** It never reached the Veo prompt,
  and generated lettering is unreliable on both models.
- **Bright Data's free tier is "Search & Extract" only** (5,000 calls/month):
  `search_engine`, `search_engine_batch`, `scrape_as_markdown`, `scrape_batch`,
  `scrape_as_html`, `extract`, `session_stats`. The `web_data_*` dataset tools
  (Instagram, TikTok, LinkedIn) are Web Scraper API, a separate paid product,
  and are deliberately excluded from `BRIGHTDATA_TOOLS`.
- **Nova Pro intermittently emits a malformed tool-use block**, surfacing as
  `ModelErrorException: Model produced invalid sequence as part of ToolUse`.
  Retried up to 3 times; the critic degrades to human review if it persists.
- **Windows consoles are cp1252** and cannot encode emoji, which captions
  routinely contain. `main.py` forces UTF-8 on stdout/stderr at startup.
- **Veo audio comes from prompt cues:** dialogue in quotes, SFX and ambience
  described. Vertex is sent `generate_audio=True` on every call.
- **Vertex keeps clips in your bucket until you delete them** (the Gemini API
  deleted them after 2 days). Each run writes under its own folder in
  `gs://<bucket>/marketing-agent/`. Add a lifecycle rule.
- **All Veo output carries a SynthID watermark.** Worth knowing for client work.
- **Blocked generations are not billed.**
- **`amazon.nova-canvas-v1:0` and `amazon.nova-reel-v1:*` reached EOL on
  2026-09-30** and no longer exist. OpenAI's Sora 2 shut down 2026-09-24. This
  is why video generation sits behind the `VideoProvider` protocol in
  `providers/base.py` — assume today's model will be retired.
- **Resumable upload avoids the public-URL requirement entirely**, but is
  restricted to apps using Facebook Login for Business.
- **On Windows, `npx` must be resolved to `npx.CMD`**; a bare `npx` fails under
  `CreateProcess`.

## Verified output

Verified against the live **Vertex AI** (`veo-3.1-fast-generate-001`,
`us-central1`, no API key):

```
h264  720x1280  24fps  |  aac 48000Hz stereo  |   8.000000s   base clip, full pipeline
h264  720x1280  24fps  |  aac 48000Hz stereo  |  15.037000s   8s base + 7s extension
```

What that run established, none of it documented clearly:

- **Extension needs `duration_seconds=7` on Vertex.** The Gemini API wanted 8.
  Sending 8 fails with `Unsupported output video duration 8 seconds, supported
  durations are [7] for feature video_extension`.
- **An extension returns the combined clip** (base + extension, 15.037s), not
  just the new 7s, so nothing is stitched afterwards.
- **The extension continued the same scene** (same oven, loaf and board) when
  given the base clip's `gs://` URI.
- `person_generation="allow_all"` is accepted for text-to-video.

Not re-verified on Vertex: image-to-video (`VEO_IMAGE`) and native 1080p. The
1080x1920 / 8s shape was only seen on the earlier Gemini API. 15s needs 720p
because extension is 720p-only.

## Known rough edges

- **The critic rarely clears a threshold of 8** with Nova Pro — it is strict by
  design and most runs exit at `MAX_ITERATIONS` instead. Either lower
  `SCORE_THRESHOLD` to ~6, or set `MODEL_CRITIC` to a stronger model.
- **Vertex Veo quotas apply per project and region.** `veo-3.1-fast-generate-001`
  is GA, but the default quota is low; request an increase before running
  several reels at once.
- **A fresh `npx` MCP subprocess spawns per tool call**, not once per run.
  Costs a few seconds per call. Fixable with a persistent session if research
  latency starts to matter.
- **Bright Data occasionally returns a non-JSON response** for `search_engine`.
  The agent carries on, but the call is wasted.
- **Phase 2**: `providers/base.py` also defines an `ImageProvider` for carousels
  and single-image posts. It is not used yet.
