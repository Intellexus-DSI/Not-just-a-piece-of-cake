"""
Schemas for structured output
"""

###############################################################################
# Imports
from typing_extensions import Annotated, TypedDict, Optional


###############################################################################
# Pydantic Schemas
class Idioms(TypedDict):
    """Identified idioms in a sentence."""

    idioms: Annotated[
        list[str], ..., "only the idioms in the sentence that are in figurative usage"
    ]


class IdiomsCoT(TypedDict):
    """Identified idioms in a sentence."""

    sentence: Annotated[str, ..., "the sentence you were provided with"]
    explanation: Annotated[
        str,
        ...,
        "the explanation of the idioms in the sentence and if the usage is figurative or literal",
    ]
    idioms: Annotated[
        list[str], ..., "only the idioms in the sentence that are in figurative usage"
    ]


class IdiomsCoTCorrection(TypedDict):
    """Identified idioms in a sentence."""

    sentence: Annotated[str, ..., "the sentence you were provided with"]
    potential_idioms: Annotated[
        list[str], ..., "the idioms in the sentence with figurative usage"
    ]
    explanation: Annotated[
        str,
        ...,
        "the explanation of the idioms in the sentence and if the usage is figurative or literal",
    ]
    idioms: Annotated[
        list[str],
        ...,
        "only the idioms that are sure to be idiomatic, in a figurative usage",
    ]


class IdiomsCoTGen(TypedDict):
    """Identified idioms in a sentence."""

    sentence: Annotated[str, ..., "the sentence you were provided with"]
    potential_idioms: Annotated[list[str], ..., "the potential idioms in the sentence"]
    figurative_examples: Annotated[
        list[str], ..., "3 generated examples of figurative usage of the idioms"
    ]
    literal_examples: Annotated[
        list[str], ..., "3 generated examples of literal usage of the idioms"
    ]
    explanation: Annotated[
        str,
        ...,
        "the explanation of the idioms in the sentence and if the usage is figurative or literal",
    ]
    idioms: Annotated[
        list[str], ..., "only the idioms in the sentence that are in figurative usage"
    ]


class IdiomsCoTBest(TypedDict):
    """Identified idioms in a sentence."""

    sentence: Annotated[str, ..., "the sentence you were provided with"]
    potential_idioms: Annotated[list[str], ..., "the potential idioms in the sentence"]
    explanation: Annotated[
        str,
        ...,
        "the explanation of the idioms in the sentence and if the usage is figurative or literal",
    ]
    idioms: Annotated[
        list[str],
        ...,
        "one idiom only. The best idiom - the one you are absolutely sure that appears in figurative usage",
    ]


class IdiomsCoTSynonym(TypedDict):
    """
    A schema for determining whether an idiom is truly used idiomatically.
    Replacement/paraphrasing is only relevant if idiomatic usage is confirmed.
    """

    sentence: Annotated[str, ..., "the sentence you were provided"]
    potential_idioms: Annotated[list[str], ..., "the potential idioms in the sentence"]
    selected_words: Annotated[
        list[Optional[str]],
        ...,
        "for each potential idiom, select one content word (not a function word)",
    ]
    synonyms: Annotated[
        list[Optional[str]],
        ...,
        "for each selected word, provide a synonym that replaces it",
    ]
    new_phrases: Annotated[
        list[Optional[str]],
        ...,
        "You must retain the idiom structure and only change ONE content word in the idiom phrase",
    ]
    figurative_preserved: Annotated[
        list[Optional[bool]],
        ...,
        "For each modified idiom, explain whether the figurative meaning is preserved, based on the full context",
    ]
    explanation: Annotated[
        str,
        ...,
        "the explanation of the idioms in the sentence and if the usage is figurative or literal"
        "base this explanation on the context of the sentence, not just the idiom itself. ",
    ]
    idioms: Annotated[
        list[str], ..., "only the idioms in the sentence that are in figurative usage"
    ]


class MWEs(TypedDict):
    """Identified MWEs in a sentence."""

    mwes: Annotated[
        list[str],
        ...,
        "only the MWEs in the sentence that are following the given definition",
    ]


class MWEsCoT(TypedDict):
    """Identified MWEs in a sentence and using chain of thought."""

    sentence: Annotated[str, ..., "the sentence you were provided with"]
    explanation: Annotated[
        str,
        ...,
        "the explanation of the WMEs in the sentence and why they follow the given definition",
    ]
    mwes: Annotated[
        list[str],
        ...,
        "only the MWEs in the sentence that are following the given definition",
    ]


class VMWEs(TypedDict):
    """Identified VMWEs in a sentence."""

    mwes: Annotated[
        list[str],
        ...,
        "only the VMWEs in the sentence that are following the given definition",
    ]


class VMWEsCoT(TypedDict):
    """Identified VMWEs in a sentence and using chain of thought."""

    sentence: Annotated[str, ..., "the sentence you were provided with"]
    explanation: Annotated[
        str,
        ...,
        "the explanation of the VWMEs in the sentence and why they follow the given definition",
    ]
    mwes: Annotated[
        list[str],
        ...,
        "only the VMWEs in the sentence that are following the given definition",
    ]


TYPED_SCHEMAS = {
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
