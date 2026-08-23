"""Strict Phase 5 state and response schemas."""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .notice_input import NoticeFields, NoticeIdentity


ResponseStatus = Literal["answered", "refused", "unidentified", "ambiguous", "error"]
EvidenceType = Literal["guidance", "notice"]
NoticeFieldName = Literal[
    "notice_code",
    "notice_date",
    "due_or_response_date",
    "amount",
    "reference_number",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResponseField(StrictModel):
    value: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    source_text: str | None


class ResponseNoticeFields(StrictModel):
    notice_code: ResponseField
    notice_date: ResponseField
    due_or_response_date: ResponseField
    amount: ResponseField
    reference_number: ResponseField

    @classmethod
    def from_notice_fields(cls, fields: NoticeFields) -> "ResponseNoticeFields":
        return cls.model_validate(fields.as_dict())


class Citation(StrictModel):
    citation_id: str = Field(min_length=1)
    notice_code: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    heading_path: list[str]
    source_url: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)

    @field_validator("source_url")
    @classmethod
    def validate_official_url(cls, value: str) -> str:
        if not value.startswith("https://www.irs.gov/"):
            raise ValueError("Citation source must be an official IRS HTTPS URL")
        return value


class GroundedClaim(StrictModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_type: EvidenceType
    citation_ids: list[str] = Field(default_factory=list)
    notice_field_names: list[NoticeFieldName] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_support(self) -> "GroundedClaim":
        if self.evidence_type == "guidance":
            if not self.citation_ids or self.notice_field_names:
                raise ValueError("Guidance claims require citations and no notice fields")
        elif not self.notice_field_names or self.citation_ids:
            raise ValueError("Notice claims require notice fields and no guidance citations")
        return self


class GroundedResponse(StrictModel):
    status: ResponseStatus
    notice_code: str | None
    answer: str
    claims: list[GroundedClaim]
    citations: list[Citation]
    notice_fields: ResponseNoticeFields

    @model_validator(mode="after")
    def validate_response_shape(self) -> "GroundedResponse":
        citation_ids = [citation.citation_id for citation in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Citation IDs must be unique")
        known = set(citation_ids)
        if any(not set(claim.citation_ids).issubset(known) for claim in self.claims):
            raise ValueError("A claim references an unknown citation")
        if self.status == "answered" and not self.claims:
            raise ValueError("Answered responses require at least one grounded claim")
        if self.status != "answered" and (self.claims or self.citations):
            raise ValueError("Non-answer responses cannot contain claims or citations")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Claim IDs must be unique")
        used_citations = {citation_id for claim in self.claims for citation_id in claim.citation_ids}
        if used_citations != known:
            raise ValueError("Every citation must support at least one claim")
        field_values = self.notice_fields.model_dump()
        for claim in self.claims:
            if claim.evidence_type == "notice" and any(
                field_values[name]["value"] is None for name in claim.notice_field_names
            ):
                raise ValueError("Notice claims cannot cite null notice fields")
        detected_code = self.notice_fields.notice_code.value
        if self.status == "answered" and detected_code is not None and self.notice_code != detected_code:
            raise ValueError("Response notice code differs from the deterministic identity")
        return self


class DraftClaim(StrictModel):
    """Provider-facing claim shape; source semantics are enforced by the app binder.

    The strict-output provider schema requires every field, but JSON Schema does
    not carry Pydantic's cross-field validators reliably across providers. The
    application therefore treats these values as untrusted candidates and uses
    only the fields appropriate to ``evidence_type`` after exact source checks.
    """

    statement: str = Field(min_length=1, max_length=500)
    evidence_type: EvidenceType
    evidence_quote: str | None = Field(max_length=700)
    support_chunk_ids: list[str] = Field(max_length=5)
    notice_field_names: list[NoticeFieldName] = Field(max_length=5)


class GenerationDraft(StrictModel):
    status: Literal["answer", "insufficient"]
    claims: list[DraftClaim] = Field(max_length=6)

    @model_validator(mode="after")
    def validate_status(self) -> "GenerationDraft":
        if self.status == "answer" and not self.claims:
            raise ValueError("Answer drafts require claims")
        if self.status == "insufficient" and self.claims:
            raise ValueError("Insufficient drafts cannot contain claims")
        return self


class ClaimJudgment(StrictModel):
    claim_id: str
    supported: bool
    explanation: str = Field(min_length=1, max_length=500)


class FaithfulnessJudgment(StrictModel):
    judgments: list[ClaimJudgment]


class NoticeLensState(TypedDict, total=False):
    notice_text: str
    notice_first_page: str
    notice_identity: NoticeIdentity
    notice_fields: NoticeFields
    question: str
    retrieved_documents: list[Document]
    evidence_sufficient: bool
    policy_refusal_reason: str | None
    answer: GroundedResponse
    status: ResponseStatus
    timings: dict[str, float]
