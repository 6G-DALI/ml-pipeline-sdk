"""Staging storage backed by the selected ZenML S3 artifact store connection."""

from contextlib import contextmanager
import os
import tempfile
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from requests import RequestException

from .catalogue import http_url
from .types import SDKError, safe_filename


@contextmanager
def storage_errors(message):
    try:
        yield
    except (BotoCoreError, ClientError, OSError, RequestException, ValueError):
        raise SDKError(message) from None


def s3_client(*, endpoint_url=None, **kwargs):
    if endpoint_url:
        http_url(endpoint_url)
    return boto3.client(
        "s3", endpoint_url=endpoint_url,
        config=Config(
            signature_version="s3v4", s3={"addressing_style": "path"},
            request_checksum_calculation="when_required", response_checksum_validation="when_required",
            connect_timeout=10, read_timeout=30, retries={"total_max_attempts": 3},
        ), **kwargs,
    )


class Staging:
    def __init__(self, client, bucket, signer=None):
        self.client = client
        self.signer = signer
        self.bucket = bucket

    def new_key(self, filename):
        return f"dali-staging/{uuid4().hex}/{safe_filename(filename)}"

    def uri(self, key):
        return f"s3://{self.bucket}/{key}"

    def ensure_push_signer(self):
        if self.signer is None:
            raise SDKError("Push mode requires a provider-reachable staging signer.")

    def presign(self, key, expires):
        with storage_errors("Could not sign a staging PUT using the stack credentials."):
            return self.signer.generate_presigned_url(
                "put_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires,
            )

    def verify(self, key):
        # A direct HEAD bypasses artifact-store directory caches.
        with storage_errors("Staged object is missing or unreadable. Check the staging bucket, endpoint and permissions."):
            result = self.client.head_object(Bucket=self.bucket, Key=key)
        return result["ContentLength"]

    def stage_download(self, key, response, timeout):
        # Spool to disk to provide bounded memory and a seekable source for S3 retries.
        with storage_errors("Could not download the signed dataset or write it to staging."):
            deadline = monotonic() + timeout
            with tempfile.TemporaryFile() as source:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if monotonic() >= deadline:
                        raise SDKError("Timed out streaming the signed dataset download.")
                    source.write(chunk)
                expected = response.headers.get("Content-Length")
                if expected is not None and source.tell() != int(expected):
                    raise SDKError("The signed dataset download ended before the expected byte count.")
                size = source.tell()
                source.seek(0)
                self.client.upload_fileobj(source, self.bucket, key)
            if self.verify(key) != size:
                raise SDKError("Staged dataset size does not match the downloaded byte count.")


def active_artifact_store():
    from zenml.client import Client

    return Client().active_stack.artifact_store


def staging_connection(*, push=False, bucket=None):
    store = active_artifact_store()
    if store is None or not store.path.startswith("s3://") or not hasattr(store, "get_credentials"):
        raise SDKError("The active ZenML stack needs an S3 artifact store for dataset staging.")
    bucket = bucket or os.environ.get("DALI_STAGING_BUCKET") or urlsplit(store.path).netloc
    if not bucket or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in bucket):
        raise SDKError("Configure a valid DALI_STAGING_BUCKET or stack S3 bucket.")
    # Resolve the same credentials and internal endpoint as ZenML's S3 store, including service connectors.
    try:
        access, secret, token, region = store.get_credentials()
        options = dict(store.config.client_kwargs or {})
    except Exception:
        raise SDKError("Could not resolve the active stack's S3 artifact store connection.") from None
    options.update({name: value for name, value in {
        "aws_access_key_id": access, "aws_secret_access_key": secret, "aws_session_token": token,
    }.items() if value is not None})
    options.setdefault("region_name", region or "us-east-1")
    with storage_errors("Could not initialize the active stack's S3 connection."):
        client = s3_client(**options)
        signer = None
        if push:
            public_endpoint = os.environ.get("DALI_STAGING_SIGN_ENDPOINT")
            signer = s3_client(**{**options, "endpoint_url": public_endpoint}) if public_endpoint else client
    return Staging(client, bucket, signer)


def open_staged(uri, size):
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise SDKError("DatasetFile requires a durable S3 URI.")
    staging = staging_connection(bucket=parsed.netloc)
    with storage_errors("Could not open the staged dataset using the active stack connection."):
        result = staging.client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    if result["ContentLength"] != size:
        result["Body"].close()
        raise SDKError("The staged dataset no longer matches its verified size.")
    return result["Body"]
