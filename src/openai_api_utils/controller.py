import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from PIL import Image
from pydantic import BaseModel

from src.openai_api_utils.data_classes import (
    ContextModel,
    ImageModel,
    ImageUrlModel,
    ImageUrlDetailEnum,
    MessageContentModel,
    MessageContentTypesEnum,
    MessageModel,
    MessageRoleEnum,
    PromptModel,
)
from src.utils.custom_exceptions import OpenAIAPICallError


def _image_to_base64_data_uri(image: Image.Image) -> str:
    """
    Encode an image to a base64 string.

    Parameters
    ----------
    image: PIL.Image
        The image to encode.

    Raises
    ------
    TypeError: If the image is not a PIL image.
    ValueError: If the image mode is not RGB or RGBA.

    Returns
    -------
    str: The base64 string with the data URI format.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("The image must be a PIL image.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"


class OpenAIAPIController:
    def __init__(
        self,
        openai_client: OpenAI,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self.client = openai_client

    @staticmethod
    def _get_message_content(
        user_msg: str, images: Optional[List[ImageModel]] = []
    ) -> List[MessageContentModel]:
        """
        Get the message content of a given user message. If images are provided, the
        message content will include the images as well.

        Parameters
        ----------
        user_msg : str
            The user message.
        images : list of ImageModel, optional
            The images to send. The default is [].

        Returns
        -------
        List of MessageContentModel:
            The message content to send to append to a given message.

        """
        message_content = (
            [MessageContentModel(type=MessageContentTypesEnum.TEXT, text=user_msg)]
            if images
            else user_msg
        )
        for img in images:
            message_content.append(
                MessageContentModel(
                    type=MessageContentTypesEnum.IMAGE_URL,
                    image_url=ImageUrlModel(
                        url=_image_to_base64_data_uri(image=img.image),
                        detail=ImageUrlDetailEnum.HIGH,
                    ),
                )
            )

        return message_content

    @staticmethod
    def _build_messages(
        prompt: PromptModel,
        images: List[ImageModel],
        context: List[ContextModel],
    ) -> List[MessageModel]:
        """
        Build the pile of messages to send to the API. The messages are built based on
        the prompt, the images, and the context provided if any. The messages are built
        in the following order: system message, context, user message.

        Parameters
        ----------
        prompt : PromptModel
            The system and user messages.
        images : list of ImageModel
            The images to send along with the user message.
        context : list of ContextModel
            The messages that provide context to the conversation.

        Returns
        -------
        List of MessageModel:
            The pile of messages to send to the API.
        """
        messages = (
            [MessageModel(role=MessageRoleEnum.SYSTEM, content=prompt.system_msg)]
            if prompt.system_msg
            else []
        )

        for context_tuple in context:
            messages.append(
                MessageModel(
                    role=MessageRoleEnum.USER,
                    content=OpenAIAPIController._get_message_content(
                        user_msg=context_tuple.question, images=context_tuple.images
                    ),
                )
            )
            if context_tuple.answer != "" and context_tuple.answer is not None:
                messages.append(
                    MessageModel(
                        role=MessageRoleEnum.ASSISTANT,
                        content=context_tuple.answer,
                    )
                )

        messages.append(
            MessageModel(
                role=MessageRoleEnum.USER,
                content=OpenAIAPIController._get_message_content(
                    user_msg=prompt.user_msg, images=images
                ),
            )
        )

        return messages

    def send_request(
        self,
        prompt: Dict,
        response_format: Optional[Union[Dict, BaseModel]] = None,
        images: Optional[List[Image.Image]] = [],
        context: Optional[List[Tuple[List[Image.Image], str, str]]] = [],
        **kwargs,
    ) -> Tuple[ChatCompletion, float, int]:
        """
        Send a request to the OpenAI API to generate a response based on the given
        prompt, images, and conversation context.

        Parameters
        ----------
        prompt : dict
            The system and user messages.
        response_format : dict or Pydantic model, optional
            The format of the response. Options are {"type": "text"} or {"type":
            "json_object"} or {"type": "json_schema", "json_schema":
            {[SEE_API_REFERENCE]}}. In beta version you can also use a Pydantic model.
        images : list of Image.Image, optional
            The images to send along with the user message.
        context : list of tuple, optional
            A list of tuples to apply context to the conversation. For example, it could
            be use to apply few-shot learning. Each tuple would consist of images,
            question, and answer. Or it could be used to apply context about the images,
            without any predefined answer. The default is [].
        **kwargs
            Additional arguments to pass to the API. E.g., max_tokens=300.

        Raises
        ------
        OpenAIAPICallError:
            If the API call fails.

        Returns
        -------
        ChatCompletion:
            The response object from the API.
        float:
            The time taken by the API to generate the response.
        int:
            The number of retries taken by the API to generate the response.

        Examples
        --------
        >>> client = OpenAI(api_key="YOUR_API_KEY")
        >>> controller = OpenAIAPIController(openai_client=client)
        >>> prompt = {
        ...     "system_msg": "You are a travel agent assisting a customer with booking a flight.",
        ...     "user_msg": "I would like to book a flight to Paris."
        ... }
        >>> response_format = {"type": "text"}
        >>> image = Image.open("path/to/image.jpg")
        >>> few_shots = [
        ...     ([image], "What is the capital of France?", "The capital of France is Paris.")
        ... ]
        >>> response = controller.send_request(
        ...     prompt=prompt,
        ...     response_format=response_format,
        ...     images=[image],
        ...     context=few_shots,
        ...     max_tokens=300
        ... )
        """
        prompt = PromptModel(**prompt)
        images = [ImageModel(image=img) for img in images]
        context = [
            ContextModel(
                images=[ImageModel(image=img) for img in imgs],
                question=question,
                answer=answer,
            )
            for imgs, question, answer in context
        ]

        try:
            if (
                response_format
                and isinstance(response_format, type)
                and issubclass(response_format, BaseModel)
            ):
                response = self.client.beta.chat.completions.with_raw_response.parse(
                    model=self.model_name,
                    messages=OpenAIAPIController._build_messages(
                        prompt=prompt, images=images, context=context
                    ),
                    response_format=response_format,
                    **kwargs,
                )
            else:
                response = self.client.chat.completions.with_raw_response.create(
                    model=self.model_name,
                    messages=OpenAIAPIController._build_messages(
                        prompt=prompt, images=images, context=context
                    ),
                    response_format=response_format,
                    **kwargs,
                )
        except Exception as e:
            raise OpenAIAPICallError(
                openai_error=e.__class__.__name__, openai_error_message=str(e)
            )

        return (
            response.parse(),
            response.elapsed.total_seconds(),
            response.retries_taken,
        )
