from pathlib import Path
from typing import List

from huggingface_hub.constants import HF_HUB_CACHE
from kokoro import KModel, KPipeline
import numpy as np
import torch

from src.utils.constants import TTS_MODEL_REPO_ID


class TTSModelClient:
    def __init__(self):
        self.model = self._load_model()
        self.pipeline = self._setup_pipeline()

    def _load_model(self):
        tts_model_path = str(next(Path(HF_HUB_CACHE).rglob("kokoro-v1_0.pth")))
        tts_model = KModel(repo_id=TTS_MODEL_REPO_ID, model=tts_model_path)
        return tts_model

    def _setup_pipeline(self):
        print(f"CUDA available: {torch.cuda.is_available()}")
        pipeline = KPipeline(
            lang_code="a",
            repo_id=TTS_MODEL_REPO_ID,
            model=self.model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return pipeline

    @staticmethod
    def _chunk_text(text: str, max_words: int = 50) -> List[str]:
        """
        Splits the text into chunks formed by sentences, ensuring that each chunk does not
        exceed the specified number of words.

        Parameters
        ----------
        text : str
            The text to be split into chunks.
        max_words : int
            The maximum number of words allowed in each chunk.

        Returns
        -------
        List[str]
            A list of text chunks, each containing a maximum of `max_words` words.
        """
        small_chunks_for_tts = []
        current_chunk = ""
        # Split the text into sentences using ". " as a delimiter and filter out empty strings
        for sentence in filter(lambda x: x != "", map(str.strip, text.split(". "))):

            current_chunk_words = current_chunk.split()
            sentence_words = sentence.split()

            if (
                current_chunk != ""
                and len(current_chunk_words) + len(sentence_words) > max_words
            ):
                small_chunks_for_tts.append(current_chunk.rstrip())
                current_chunk = ""

            current_chunk += sentence + (". " if sentence[-1] != "." else "")

        if current_chunk:
            small_chunks_for_tts.append(current_chunk.rstrip())

        return small_chunks_for_tts

    def text_to_speech(
        self, text: str, voice: str = None, speed: float = 1
    ) -> np.ndarray:
        """
        Converts the given text to speech using the TTS model.
        The text is split into smaller chunks to ensure that the model can process it
        efficiently. Each chunk is processed separately, and the resulting audio chunks
        are concatenated to form the final audio output.
        
        Parameters
        ----------
        text : str
            The text to be converted to speech.
        voice : str, optional
            The voice to be used for the TTS model. If not provided, a default voice
            will be used.
        speed : float, optional
            The speed of the speech. Default is 1 (normal speed).

        Returns
        -------
        np.ndarray
            The audio output as a NumPy array.
        """
        audios_from_text_between_pauses = []
        text_chunks = self._chunk_text(text=text)

        for small_text_chunk in text_chunks:
            audio_generator = self.pipeline(
                text=small_text_chunk, voice=voice, speed=speed
            )
            for graphemes, phonemes, audio_chunk in audio_generator:
                print(
                    f"++++ Processing audio chunk\n"
                    f" Number of words: {len(graphemes.split())}\n"
                    f" Graphemes: {graphemes}\n"
                    f" Phonemes: {phonemes}"
                )
                audios_from_text_between_pauses.append(audio_chunk)

        audios_from_text_between_pauses = np.concatenate(
            audios_from_text_between_pauses
        )

        return audios_from_text_between_pauses
