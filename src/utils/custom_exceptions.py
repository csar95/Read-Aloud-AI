from src.utils.constants import SUPPORTED_FORMATS


class SystemException(Exception):
    pass


class OpenAIAPICallError(SystemException):
    def __init__(self, openai_error: str, openai_error_message: str):
        self.openai_error = openai_error
        self.openai_error_message = openai_error_message
        super().__init__(
            f"OpenAI API call failed with error: {openai_error}: {openai_error_message}"
        )


class OpenAIInvalidResponseFormatError(SystemException):
    def __init__(self):
        super().__init__("Non valid OpenAI API response format")


class UnsupportedFileFormatError(SystemException):
    def __init__(self):
        super().__init__(
            f'Unsupported media type. Only {", ".join(SUPPORTED_FORMATS)} files are supported.'
        )
