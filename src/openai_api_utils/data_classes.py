from enum import Enum
from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Optional, Union


class PromptModel(BaseModel):
    system_msg: Optional[str] = None
    user_msg: str


class ImageModel(BaseModel):
    image: Image.Image
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("image")
    def validate_image(cls, v):
        if not isinstance(v, Image.Image):
            raise Exception("images should be a list of PIL.Image.Image objects.")
        return v


class ContextModel(BaseModel):
    images: Optional[List[ImageModel]] = []
    question: str
    answer: Optional[str] = ""


class MessageContentTypesEnum(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"


class ImageUrlDetailEnum(str, Enum):
    LOW = "low"
    HIGH = "high"
    AUTO = "auto"


class ImageUrlModel(BaseModel):
    url: str
    detail: ImageUrlDetailEnum


class MessageContentModel(BaseModel):
    type: MessageContentTypesEnum
    text: Optional[str] = None
    image_url: Optional[ImageUrlModel] = None


class MessageRoleEnum(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MessageModel(BaseModel):
    role: MessageRoleEnum
    content: Union[str, List[MessageContentModel]]
