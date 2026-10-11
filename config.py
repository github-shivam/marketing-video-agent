"""Application configuration, loaded once from the environment.

Every tunable lives here so that nothing else in the codebase reads
``os.environ`` directly. Secrets come from ``.env`` (gitignored); only
``.env.example`` is ever committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from video_plan import VideoPlan, clamp_seconds, plan_for

load_dotenv()


def _str(name: str, default: str = "") -> str:
    """Read a string env var, treating whitespace-only as unset."""
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` if unset or malformed."""
    raw = _str(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` if unset or malformed."""
    raw = _str(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Anything but a known truthy word is False."""
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of configuration for one run."""

    # Safety
    dry_run: bool

    # Text models on Bedrock. Each node names its own; ``model`` is an optional
    # shared fallback and has no default.
    model: str
    bedrock_text_region: str
    model_temperature: float
    model_classify: str
    model_research: str
    model_writer: str
    model_critic: str
    model_dialogue: str
    model_converse: str

    # Script style
    script_language: str
    speech_pace_wpm: int

    # Writer/critic loop
    score_threshold: float
    max_iterations: int

    # Which video model the prompts are written for, and how long a video is
    # when the person does not say. Each run can ask for its own length.
    video_model: str
    video_seconds_default: int

    # Video (Veo 3.1 on Vertex AI, authenticated with Google Cloud ADC: no key)
    gcp_project: str
    gcp_location: str
    veo_gcs_bucket: str
    veo_model: str
    veo_resolution: str
    veo_timestamps: bool
    veo_aspect_ratio: str
    veo_person_generation: str
    veo_image: str
    veo_block_retries: int
    veo_poll_interval: int
    veo_timeout: int

    # Reading uploaded photos, video and audio. Gemini on Vertex AI, because it
    # takes all three natively and uses the same Google Cloud login as Veo.
    asset_model: str
    uploads_dir: Path

    # Research (Bright Data MCP)
    brightdata_api_token: str
    brightdata_groups: str
    brightdata_tools: str
    brightdata_rate_limit: str
    brightdata_max_retries: int

    # Upload strategy
    upload_strategy: str

    # Storage (only used by UPLOAD_STRATEGY=url)
    s3_bucket: str
    s3_prefix: str
    s3_region: str
    presign_expiry: int

    # Instagram Graph API
    ig_user_id: str
    ig_access_token: str
    ig_api_version: str
    ig_graph_host: str
    ig_poll_interval: int
    ig_poll_timeout: int

    # WhatsApp chat front end (Meta Cloud API)
    whatsapp_access_token: str
    whatsapp_phone_number_id: str
    whatsapp_app_secret: str
    whatsapp_verify_token: str
    whatsapp_api_version: str
    whatsapp_allowed_numbers: str
    whatsapp_max_runs_per_day: int

    # HTTP API
    api_key: str

    # Local
    checkpoint_db: str
    conversations_db: str
    output_dir: Path
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a Settings instance from the current environment."""
        return cls(
            dry_run=_bool("DRY_RUN", True),
            model=_str("MODEL"),
            bedrock_text_region=_str("BEDROCK_TEXT_REGION", "us-east-1"),
            model_temperature=_float("MODEL_TEMPERATURE", 0.7),
            model_classify=_str("MODEL_CLASSIFY"),
            model_research=_str("MODEL_RESEARCH"),
            model_writer=_str("MODEL_WRITER"),
            model_critic=_str("MODEL_CRITIC"),
            model_dialogue=_str("MODEL_DIALOGUE"),
            model_converse=_str("MODEL_CONVERSE"),
            script_language=_str("SCRIPT_LANGUAGE", "English"),
            speech_pace_wpm=_int("SPEECH_PACE_WPM", 175),
            score_threshold=_float("SCORE_THRESHOLD", 8.0),
            max_iterations=_int("MAX_ITERATIONS", 3),
            video_model=_str("VIDEO_MODEL", "veo").lower(),
            video_seconds_default=clamp_seconds(_int("VIDEO_SECONDS_DEFAULT", 15)),
            gcp_project=_str("GOOGLE_CLOUD_PROJECT"),
            gcp_location=_str("GOOGLE_CLOUD_LOCATION", "us-central1"),
            veo_gcs_bucket=_str("VEO_GCS_BUCKET"),
            veo_model=_str("VEO_MODEL", "veo-3.1-fast-generate-001"),
            veo_resolution=_str("VEO_RESOLUTION", "720p"),
            veo_timestamps=_bool("VEO_TIMESTAMPS", True),
            veo_aspect_ratio=_str("VEO_ASPECT_RATIO", "9:16"),
            veo_person_generation=_str("VEO_PERSON_GENERATION", "allow_all"),
            veo_image=_str("VEO_IMAGE"),
            veo_block_retries=_int("VEO_BLOCK_RETRIES", 2),
            veo_poll_interval=_int("VEO_POLL_INTERVAL", 10),
            veo_timeout=_int("VEO_TIMEOUT", 600),
            asset_model=_str("ASSET_MODEL", "gemini-2.5-flash"),
            uploads_dir=Path(_str("UPLOADS_DIR", "uploads")),
            brightdata_api_token=_str("BRIGHTDATA_API_TOKEN"),
            brightdata_groups=_str("BRIGHTDATA_GROUPS"),
            brightdata_tools=_str(
                "BRIGHTDATA_TOOLS",
                # Free-tier "Search & Extract" tools only; web_data_* is paid.
                "search_engine,search_engine_batch,scrape_as_markdown,"
                "scrape_batch,extract",
            ),
            brightdata_rate_limit=_str("BRIGHTDATA_RATE_LIMIT", "100/1h"),
            brightdata_max_retries=_int("BRIGHTDATA_MAX_RETRIES", 2),
            upload_strategy=_str("UPLOAD_STRATEGY", "resumable").lower(),
            s3_bucket=_str("S3_BUCKET"),
            s3_prefix=_str("S3_PREFIX", "instagram-agent"),
            s3_region=_str("S3_REGION", "us-east-1"),
            presign_expiry=_int("PRESIGN_EXPIRY", 3600),
            ig_user_id=_str("IG_USER_ID"),
            ig_access_token=_str("IG_ACCESS_TOKEN"),
            ig_api_version=_str("IG_API_VERSION", "v25.0"),
            ig_graph_host=_str("IG_GRAPH_HOST", "https://graph.facebook.com"),
            ig_poll_interval=_int("IG_POLL_INTERVAL", 5),
            ig_poll_timeout=_int("IG_POLL_TIMEOUT", 300),
            whatsapp_access_token=_str("WHATSAPP_ACCESS_TOKEN"),
            whatsapp_phone_number_id=_str("WHATSAPP_PHONE_NUMBER_ID"),
            whatsapp_app_secret=_str("WHATSAPP_APP_SECRET"),
            whatsapp_verify_token=_str("WHATSAPP_VERIFY_TOKEN"),
            whatsapp_api_version=_str("WHATSAPP_API_VERSION", "v25.0"),
            whatsapp_allowed_numbers=_str("WHATSAPP_ALLOWED_NUMBERS"),
            whatsapp_max_runs_per_day=_int("WHATSAPP_MAX_RUNS_PER_DAY", 10),
            api_key=_str("API_KEY"),
            checkpoint_db=_str("CHECKPOINT_DB", "checkpoints.sqlite"),
            conversations_db=_str("CONVERSATIONS_DB", "conversations.sqlite"),
            output_dir=Path(_str("OUTPUT_DIR", "output")),
            log_level=_str("LOG_LEVEL", "INFO"),
        )

    def plan(
        self, seconds: int | None = None, *, reference_images: bool = False
    ) -> VideoPlan:
        """The generations that make a video of ``seconds`` for this video model.

        Args:
            seconds: The requested length; ``VIDEO_SECONDS_DEFAULT`` when None.
            reference_images: True when photos of a person are sent to Veo, which
                needs an 8-second base clip.
        """
        return plan_for(
            seconds or self.video_seconds_default,
            self.video_model,
            fixed_base=reference_images,
        )

    #: Seconds of silence left at the end of each segment so the final word
    #: lands before the clip cuts. Delivery pace varies, and a sentence
    #: chopped mid-word is far worse than a moment of quiet.
    SPEECH_TAIL_SECONDS = 1.0

    def words_for(self, seconds: int) -> int:
        """Words that comfortably fit in ``seconds``, leaving a safety tail."""
        usable = max(1.0, seconds - self.SPEECH_TAIL_SECONDS)
        return max(1, round(self.speech_pace_wpm / 60 * usable))

    def words_ceiling(self, seconds: int) -> int:
        """Words that fill the whole clip. Beyond this, speech is cut off."""
        return max(1, round(self.speech_pace_wpm / 60 * seconds))

    @property
    def veo_live(self) -> bool:
        """True when Veo can really be called, i.e. generating will cost money.

        It needs a project and a bucket. Credentials come from Google Cloud
        Application Default Credentials, not from a setting.
        """
        return bool(self.gcp_project and self.veo_gcs_bucket)

    @property
    def whatsapp_configured(self) -> bool:
        """True when any WhatsApp setting is present, i.e. the feature is wanted."""
        return any(
            (
                self.whatsapp_access_token,
                self.whatsapp_phone_number_id,
                self.whatsapp_app_secret,
                self.whatsapp_verify_token,
            )
        )

    def whatsapp_missing(self) -> list[str]:
        """Names of the WhatsApp settings that must be set but are not."""
        required = {
            "WHATSAPP_ACCESS_TOKEN": self.whatsapp_access_token,
            "WHATSAPP_PHONE_NUMBER_ID": self.whatsapp_phone_number_id,
            "WHATSAPP_APP_SECRET": self.whatsapp_app_secret,
            "WHATSAPP_VERIFY_TOKEN": self.whatsapp_verify_token,
            "WHATSAPP_ALLOWED_NUMBERS": self.whatsapp_allowed_numbers,
        }
        return [name for name, value in required.items() if not value]

    def model_for(self, node: str) -> str:
        """Return the model ID for ``node``.

        Uses the node's own ``MODEL_<NODE>``, else the optional shared ``MODEL``.

        Raises:
            ValueError: If neither is set. There is deliberately no built-in
                default, so a missing model fails loudly rather than quietly
                running on one nobody chose.
        """
        own = {
            "classify": self.model_classify,
            "research": self.model_research,
            "writer": self.model_writer,
            "critic": self.model_critic,
            "dialogue": self.model_dialogue,
            "converse": self.model_converse,
        }.get(node, "")
        chosen = own or self.model
        if not chosen:
            raise ValueError(
                f"No text model configured for the {node!r} node. "
                f"Set MODEL_{node.upper()} (or MODEL as a shared default) in .env."
            )
        return chosen


settings = Settings.from_env()
