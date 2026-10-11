"""Veo 3.1 video generation on Vertex AI.

Shape of a run, set by the :class:`~video_plan.VideoPlan`: a base clip (4, 6 or
8 seconds), then as many extensions of exactly 7 seconds as the requested length
needs, producing a single continuous clip (8 + 7 = 15s by default).

Authentication is Google Cloud Application Default Credentials
(``gcloud auth application-default login``, or a service account when
deployed), the same way boto3 picks up an AWS profile. There is no API key.

Vertex writes every clip to a Cloud Storage bucket (``VEO_GCS_BUCKET``). An
extension reads its base clip from there, and the finished clip is downloaded
from there.

Constraints encoded here, from Google's documentation:

* Extension is 720p-only and requires an 8-second base.
* Text-to-video accepts ``person_generation="allow_all"`` only. (The
  ``allow_adult`` value is for image-to-video and is rejected here.)
* ``duration_seconds`` is an ``int`` in the Python SDK even though the REST
  API documents it as a string enum.
* Clips stay in the bucket until you delete them: put a lifecycle rule on it.
* Audio is requested explicitly with ``generate_audio``.
* Blocked generations are not billed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

from config import Settings
from video_plan import VideoPlan

logger = logging.getLogger(__name__)

#: Folder inside ``VEO_GCS_BUCKET`` that Vertex output is written under.
GCS_PREFIX = "marketing-agent"


class VeoError(RuntimeError):
    """Raised when a Veo generation fails, is blocked, or times out."""


class VeoRejectedError(VeoError):
    """Raised when Veo refuses the request itself, as invalid.

    Different from a block: nothing about the output was objectionable, the
    request just asked for something Veo will not do, such as a combination of
    settings it does not support.
    """


class VeoBlockedError(VeoError):
    """Raised when a safety filter blocked the generation.

    Google documents this as intermittent — "Veo 3.1 will *sometimes* block a
    video from generating because of safety filters or other processing issues
    with the audio" — and blocked generations are not billed, so retrying
    costs nothing.
    """


class VeoVideoProvider:
    """Generates a vertical reel with Veo 3.1 and returns the local MP4."""

    name = "veo"

    def __init__(self, settings: Settings) -> None:
        missing = [
            name
            for name, value in (
                ("GOOGLE_CLOUD_PROJECT", settings.gcp_project),
                ("VEO_GCS_BUCKET", settings.veo_gcs_bucket),
            )
            if not value
        ]
        if missing:
            raise VeoError(" and ".join(missing) + " must be set to use Veo")
        self.settings = settings
        #: Things the person should be told about the last video made.
        self.notes: list[str] = []

    def _client(self):
        """Build the client lazily so imports stay cheap in tests.

        The project and location are passed explicitly; credentials come from
        Application Default Credentials. No API key is ever used.
        """
        from google import genai

        return genai.Client(
            vertexai=True,
            project=self.settings.gcp_project,
            location=self.settings.gcp_location,
        )

    def _person_generation(self, *, with_image: bool) -> str:
        """Return the ``person_generation`` value for this call mode.

        The documented rule splits by input type::

            Text-to-video & Extension          -> "allow_all" only
            Image-to-video, Interpolation,
            Reference images                   -> "allow_adult" only

        Passing the wrong one is rejected, so it is derived per call rather
        than taken from a single setting.
        """
        return "allow_adult" if with_image else self.settings.veo_person_generation

    def _base_config(
        self,
        *,
        seconds: int = 8,
        with_image: bool = False,
        output_gcs_uri: str | None = None,
        references: list | None = None,
    ):
        """Config for the first (base) generation.

        Args:
            seconds: Length of the base clip: 4, 6 or 8. It must be 8 when the
                clip is going to be extended, or when reference images are used.
            with_image: True for any image input, which switches
                ``person_generation`` to ``allow_adult``.
            references: Photos of the subject for Veo to keep the look of.
        """
        from google.genai import types

        return types.GenerateVideosConfig(
            aspect_ratio=self.settings.veo_aspect_ratio,
            resolution=self.settings.veo_resolution,
            duration_seconds=seconds,
            person_generation=self._person_generation(with_image=with_image),
            number_of_videos=1,
            generate_audio=True,
            output_gcs_uri=output_gcs_uri,
            reference_images=references or None,
        )

    def _load_references(self, files: list[Path] | None):
        """Read photos of the subject into the form Veo takes them in.

        Veo accepts at most three. A file that has gone missing is skipped rather
        than failing a run that has already been paid for up to this point.
        """
        from google.genai import types

        out = []
        for path in list(files or [])[:3]:
            try:
                data = Path(path).read_bytes()
            except OSError:
                logger.warning("Reference photo %s is gone - skipping it", Path(path).name)
                continue
            out.append(
                types.VideoGenerationReferenceImage(
                    image=types.Image(image_bytes=data, mime_type="image/jpeg"),
                    reference_type=types.VideoGenerationReferenceType.ASSET,
                )
            )
        return out

    @staticmethod
    def _references_were_the_problem(exc: Exception) -> str | None:
        """Whether a failure means Veo will not take the reference photos.

        Returns a reason fit to tell the person, or None for any other failure,
        which must not be retried without the photos: running out of quota, say,
        will fail the same way again.
        """
        from google.genai import errors

        if isinstance(exc, VeoBlockedError):
            return "Veo's safety check blocked them"
        if isinstance(exc, VeoRejectedError):
            return "Veo does not accept them for this video"
        if isinstance(exc, errors.ClientError) and getattr(exc, "code", None) in (400, 403):
            return "Veo does not accept them for this video"
        return None

    def _load_first_frame(self):
        """Load the configured first-frame image, if any.

        Returns:
            A ``types.Image`` to animate, or ``None`` for text-to-video.

        Raises:
            VeoError: If a path is configured but missing.
        """
        if not self.settings.veo_image:
            return None

        from google.genai import types

        path = Path(self.settings.veo_image).expanduser()
        if not path.exists():
            raise VeoError(f"VEO_IMAGE does not exist: {path}")
        logger.info("Animating first frame from %s", path.name)
        return types.Image.from_file(location=str(path))

    def _extension_config(self, *, output_gcs_uri: str | None = None):
        """Config for an extension: always 720p and exactly 7s, on an 8s base.

        ``aspect_ratio`` must be passed explicitly and must match the source
        clip. Omitting it makes the API fall back to its own default of 16:9,
        which then conflicts with a 9:16 input and fails with::

            Aspect ratio of the input video must be 16:9, but got: 9:16

        That message reads like a restriction on vertical video, but it is
        really a mismatch between the requested output and the given input.
        """
        from google.genai import types

        return types.GenerateVideosConfig(
            aspect_ratio=self.settings.veo_aspect_ratio,
            resolution="720p",
            # Vertex rejects anything else, verified live: "Unsupported output
            # video duration 8 seconds, supported durations are [7] for
            # feature video_extension". (The Gemini API wanted 8 here.)
            duration_seconds=7,
            # Documented: "Text-to-video & Extension: 'allow_all' only".
            person_generation=self.settings.veo_person_generation,
            number_of_videos=1,
            generate_audio=True,
            output_gcs_uri=output_gcs_uri,
        )

    def _check_extension_support(self, plan: VideoPlan) -> None:
        """Reject an extension request that cannot succeed, before spending.

        Extension is documented as 720p-only and unavailable on Veo 3.1 Lite,
        so catch both here rather than after the base clip has been billed.

        Raises:
            VeoError: If the plan extends and the configuration cannot.
        """
        if plan.extensions <= 0:
            return
        if "lite" in self.settings.veo_model.lower():
            raise VeoError(
                f"{self.settings.veo_model} does not support video extension, "
                f"which a {plan.seconds}-second video needs. Use "
                "veo-3.1-fast-generate-001 or veo-3.1-generate-001, or ask for "
                "8 seconds or less."
            )
        if self.settings.veo_resolution != "720p":
            raise VeoError(
                "Veo extension is 720p-only, but VEO_RESOLUTION="
                f"{self.settings.veo_resolution}. Set VEO_RESOLUTION=720p, or "
                f"ask for 8 seconds or less (a {plan.seconds}-second video "
                "needs extension)."
            )

    async def _await_operation(self, client, operation):
        """Poll an operation until it finishes, or raise on timeout.

        Args:
            client: The Vertex AI client.
            operation: The operation returned by ``generate_videos``.

        Returns:
            The completed operation.

        Raises:
            VeoError: If the operation fails or exceeds ``VEO_TIMEOUT``.
        """
        deadline = time.monotonic() + self.settings.veo_timeout
        while not operation.done:
            if time.monotonic() > deadline:
                raise VeoError(
                    f"Veo generation timed out after {self.settings.veo_timeout}s"
                )
            await asyncio.sleep(self.settings.veo_poll_interval)
            operation = await client.aio.operations.get(operation)

        error = getattr(operation, "error", None)
        if error:
            # Code 3 is INVALID_ARGUMENT: Veo is refusing the request itself.
            code = error.get("code") if isinstance(error, dict) else None
            cls = VeoRejectedError if code == 3 else VeoError
            raise cls(f"Veo generation failed: {error}")

        videos = getattr(operation.response, "generated_videos", None)
        if not videos:
            # The response carries the Responsible-AI verdict; surface it
            # rather than guessing, because the remedy differs by reason.
            reasons = getattr(operation.response, "rai_media_filtered_reasons", None)
            count = getattr(operation.response, "rai_media_filtered_count", None)
            detail = "; ".join(reasons) if reasons else "no reason given"
            raise VeoBlockedError(
                f"Veo returned no video - blocked by a safety filter "
                f"(filtered={count}): {detail}. Blocked generations are not "
                f"billed."
            )
        return operation

    async def _generate_with_retry(self, client, label: str, **kwargs):
        """Run one generation, retrying when a safety filter blocks it.

        Blocked generations are not billed, so a retry is free. The block is
        documented as intermittent, and in practice the same prompt often
        succeeds on a second attempt.
        """
        attempts = max(1, self.settings.veo_block_retries + 1)
        last: VeoBlockedError | None = None
        for attempt in range(1, attempts + 1):
            operation = await client.aio.models.generate_videos(**kwargs)
            try:
                return await self._await_operation(client, operation)
            except VeoBlockedError as exc:
                last = exc
                if attempt < attempts:
                    logger.warning(
                        "%s blocked (attempt %d/%d), retrying - not billed",
                        label,
                        attempt,
                        attempts,
                    )
        assert last is not None
        raise last

    def _gcs_folder(self, run_folder: str, step: str) -> str:
        """Where Vertex should write one generation."""
        bucket = self.settings.veo_gcs_bucket
        return f"gs://{bucket}/{GCS_PREFIX}/{run_folder}/{step}/"

    async def _download_from_gcs(self, video, destination: Path) -> None:
        """Copy a finished Vertex clip from Cloud Storage to ``destination``.

        Args:
            video: The ``Video`` the operation returned; its ``uri`` is a
                ``gs://`` path. If Vertex sent the bytes inline instead, they
                are written directly.
            destination: Local file to write.

        Raises:
            VeoError: If the response carries neither a ``gs://`` URI nor bytes.
        """
        data = getattr(video, "video_bytes", None)
        if data:
            destination.write_bytes(data)
            return

        uri = getattr(video, "uri", None) or ""
        if not uri.startswith("gs://"):
            raise VeoError(f"Vertex returned neither bytes nor a gs:// URI: {uri!r}")
        bucket_name, _, blob_name = uri.removeprefix("gs://").partition("/")

        def download() -> None:
            # Blocking client, so it runs off the event loop.
            from google.cloud import storage

            client = storage.Client(project=self.settings.gcp_project)
            client.bucket(bucket_name).blob(blob_name).download_to_filename(
                str(destination)
            )

        logger.info("Downloading %s", uri)
        await asyncio.to_thread(download)

    async def generate(
        self,
        prompts: list[str],
        out_dir: Path,
        plan: VideoPlan | None = None,
        references: list[Path] | None = None,
    ) -> Path:
        """Generate the reel and return the path to the downloaded MP4.

        Args:
            prompts: One prompt per segment; the first is the base clip and each
                later prompt extends the same continuous shot by 7 seconds.
            out_dir: Directory to write the finished file into.
            plan: The generations to make. Defaults to the default length. The
                file returned is as long as the plan *generates*; trimming to
                the length the plan keeps is the caller's job.
            references: Photos of the person to feature, sent to Veo as subject
                references so it keeps their face. If Veo will not take them the
                video is made without them and ``notes`` says so.

        Returns:
            Path to the local MP4.

        Raises:
            VeoError: On too few prompts, a failure, or a timeout.
        """
        self.notes = []
        references = self._load_references(references)
        plan = plan or self.settings.plan(reference_images=bool(references))
        if not prompts:
            raise VeoError("at least one prompt is required")
        if len(prompts) < len(plan.segments):
            raise VeoError(
                f"a {plan.seconds}-second video needs {len(plan.segments)} "
                f"prompts, but only {len(prompts)} were given"
            )

        # Validate before generating: an extension request that is going to be
        # rejected must not cost a base clip first.
        self._check_extension_support(plan)

        client = self._client()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Photos the person just sent win over the one fixed in the settings.
        first_frame = None if references else self._load_first_frame()
        base_seconds = plan.segments[0].generate
        logger.info(
            "Veo base clip: %s %s %ss (%s), %d extension(s) to follow",
            self.settings.veo_model,
            self.settings.veo_resolution,
            base_seconds,
            "with reference photos" if references
            else "image-to-video" if first_frame else "text-to-video",
            plan.extensions,
        )
        # The prompt/image/video arguments are deprecated in favour of
        # `source`; removal is allowed from 2026-07-31 onwards.
        from google.genai import types

        # Vertex writes each clip under its own folder, so a run's base and
        # extensions never overwrite each other or another run.
        run_folder = f"{uuid.uuid4().hex[:10]}"

        operation = None
        if references:
            try:
                operation = await self._generate_with_retry(
                    client,
                    "base clip with reference photos",
                    model=self.settings.veo_model,
                    source=types.GenerateVideosSource(prompt=prompts[0]),
                    config=self._base_config(
                        seconds=base_seconds,
                        with_image=True,
                        output_gcs_uri=self._gcs_folder(run_folder, "base"),
                        references=references,
                    ),
                )
                logger.info("Veo took %d reference photo(s)", len(references))
            except Exception as exc:  # noqa: BLE001 - re-raised unless it was the photos
                reason = self._references_were_the_problem(exc)
                if reason is None:
                    raise
                logger.warning("Veo would not take the reference photos (%s): %s", reason, exc)
                self.notes.append(
                    f"Veo couldn't use your photos as references ({reason}), so the "
                    "person in the video comes from a description of them and may not "
                    "look exactly like them."
                )
        if operation is None:
            operation = await self._generate_with_retry(
                client,
                "base clip",
                model=self.settings.veo_model,
                source=types.GenerateVideosSource(prompt=prompts[0], image=first_frame),
                config=self._base_config(
                    seconds=base_seconds,
                    with_image=first_frame is not None,
                    output_gcs_uri=self._gcs_folder(run_folder, "base"),
                ),
            )

        for i in range(1, len(plan.segments)):
            prompt = prompts[i]
            logger.info("Veo extension %d (+7s)", i)
            operation = await self._generate_with_retry(
                client,
                f"extension {i}",
                model=self.settings.veo_model,
                source=types.GenerateVideosSource(
                    prompt=prompt,
                    video=operation.response.generated_videos[0].video,
                ),
                config=self._extension_config(
                    output_gcs_uri=self._gcs_folder(run_folder, f"ext{i}")
                ),
            )

        # The random part matters: two people's videos can finish in the same second,
        # and a name made from the time alone would have one overwrite the other.
        destination = out_dir / f"reel-{int(time.time())}-{uuid.uuid4().hex[:8]}.mp4"
        generated = operation.response.generated_videos[0]
        await self._download_from_gcs(generated.video, destination)
        if not destination.exists() or destination.stat().st_size == 0:
            raise VeoError(f"download produced no data at {destination}")

        logger.info(
            "Saved %s (%.1fMB)", destination, destination.stat().st_size / 1024 / 1024
        )
        return destination
