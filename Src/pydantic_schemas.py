"""
Schemas for structured output.
"""

###############################################################################
# Imports
from pydantic import BaseModel, Field
from typing import Optional


###############################################################################
# Pydantic Schemas
class Idioms(BaseModel):
    """Identified idioms in a sentence"""

    idioms: list[str] = Field(
        description="only the idioms in the sentence that are in figurative usage"
    )


class IdiomsCoT(BaseModel):
    """Identified idioms in a sentence"""

    sentence: str = Field(description="the sentence you were provided with")
    explanation: str = Field(
        description="the explanation of the idioms in the sentence and if the usage is figurative or literal"
    )
    idioms: list[str] = Field(
        description="only the idioms in the sentence that are in figurative usage"
    )


class IdiomsCoTCorrection(BaseModel):
    """Identified idioms in a sentence"""

    sentence: str = Field(description="the sentence you were provided")
    potential_idioms: list[str] = Field(
        description="the idioms in the sentence with figurative usage"
    )
    explanation: str = Field(
        description="the explanation of the idioms in the sentence and if the usage is figurative or literal"
    )

    idioms: list[str] = Field(
        description="only the idioms that are sure to be idiomatic, in a figurative usage"
    )


class IdiomsCoTGen(BaseModel):
    """Identified idioms in a sentence"""

    sentence: str = Field(description="the sentence you were provided")
    potential_idioms: list[str] = Field(
        description="the potential idioms in the sentence"
    )
    figurative_examples: list[str] = Field(
        description="3 generated examples of figurative usage of the idioms"
    )
    literal_examples: list[str] = Field(
        description="3 generated examples of literal usage of the idioms"
    )
    explanation: str = Field(
        description="the explanation of the idioms in the sentence and if the usage is figurative or literal"
    )
    idioms: list[str] = Field(
        description="only the idioms in the sentence that are in figurative usage"
    )


class IdiomsCoTBest(BaseModel):
    """Identified idioms in a sentence"""

    sentence: str = Field(description="the sentence you were provided")
    potential_idioms: list[str] = Field(
        description="the potential idioms in the sentence"
    )
    explanation: str = Field(
        description="the explanation of the idioms in the sentence and if the usage is figurative or literal"
    )
    idioms: list[str] = Field(
        description="one idiom only. The best idiom - the one you are absolutely sure that appears in figurative usage"
    )


class IdiomsCoTSynonym(BaseModel):
    """
    A schema for determining whether an idiom is truly used idiomatically.
    Replacement/paraphrasing is only relevant if idiomatic usage is confirmed.
    """

    sentence: str = Field(description="the sentence you were provided")
    potential_idioms: list[str] = Field(
        description="the potential idioms in the sentence"
    )
    selected_words: list[str] = Field(
        description="for each potential idiom, select one content word (not a function word)"
    )

    synonyms: list[Optional[str]] = Field(
        description="for each selected word, provide a synonym that replaces it"
    )

    new_phrases: list[Optional[str]] = Field(
        description="You must retain the idiom structure and only change ONE content word in the idiom phrase"
    )
    figurative_preserved: list[Optional[bool]] = Field(
        description="For each modified idiom, explain whether the figurative meaning is preserved, based on the full context"
    )
    explanation: str = Field(
        description="the explanation of the idioms in the sentence and if the usage is figurative or literal"
        "base this explanation on the context of the sentence, not just the idiom itself. "
    )
    idioms: list[str] = Field(
        description="only the idioms in the sentence that are in figurative usage"
    )


class MWEs(BaseModel):
    """Identified MWEs in a sentence."""

    mwes: list[str] = Field(
        description="only the MWEs in the sentence that are following the given definition"
    )


class MWEsCoT(BaseModel):
    """Identified MWEs in a sentence and using chain of thought."""

    sentence: str = Field(description="the sentence you were provided with")
    explanation: str = Field(
        description="the explanation of the WMEs in the sentence and why they follow the given definition"
    )
    mwes: list[str] = Field(
        description="only the MWEs in the sentence that are following the given definition"
    )


class VMWEs(BaseModel):
    """Identified VMWEs in a sentence."""

    vmwes: list[str] = Field(
        description="only the VMWEs in the sentence that are following the given definition"
    )


class VMWEsCoT(BaseModel):
    """Identified VMWEs in a sentence and using chain of thought."""

    sentence: str = Field(description="the sentence you were provided with")
    explanation: str = Field(
        description="the explanation of the VWMEs in the sentence and why they follow the given definition"
    )
    vmwes: list[str] = Field(
        description="only the VMWEs in the sentence that are following the given definition"
    )


###############################################################################


PYDANTIC_SCHEMAS = {
    "Idioms": Idioms,
    "IdiomsCoT": IdiomsCoT,
    "IdiomsCoTCorrection": IdiomsCoTCorrection,
    "IdiomsCoTGen": IdiomsCoTGen,
    "IdiomsCoTBest": IdiomsCoTBest,
    "IdiomsCoTSynonym": IdiomsCoTSynonym,
    "MWEs": MWEs,
    "MWEsCoT": MWEsCoT,
    "VMWEs": VMWEs,
    "VMWEsCoT": VMWEsCoT,
}
