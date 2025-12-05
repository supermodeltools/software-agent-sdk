from typing import ClassVar

from pydantic import BaseModel, Field


class CriticResult(BaseModel):
    """A critic result is a score and a message."""

    THRESHOLD: ClassVar[float] = 0.5

    score: float = Field(
        description="A predicted probability of success between 0 and 1.",
        ge=0.0,
        le=1.0,
    )
    message: str | None = Field(description="An optional message explaining the score.")

    @property
    def success(self) -> bool:
        """Whether the agent is successful."""
        return self.score >= CriticResult.THRESHOLD
