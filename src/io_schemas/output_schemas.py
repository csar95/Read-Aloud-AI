from pydantic import BaseModel, Field


class FormattedPageText(BaseModel):
    text: str = Field(description="The text of the page formatted for TTS output.")
