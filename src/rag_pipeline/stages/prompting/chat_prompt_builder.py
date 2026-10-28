from __future__ import annotations

from rag_pipeline.models import Prompt, Question, RetrievedChunk
from rag_pipeline.stages.prompting.interface import PromptBuilder


CHAT_SYSTEM_MESSAGE = (
    "As an advanced reading comprehension assistant, your task is to analyze "
    "text passages and corresponding questions meticulously. Your response "
    'starts after "Thought: ", where you will methodically reason about the '
    'answer. Conclude with "Answer: " to present a concise, definitive response, '
    "devoid of additional elaborations."
)

# A fixed demonstration mirrors the message structure used by the reference
# prompt. The current question is always sent as a new user message after it.
CHAT_EXAMPLE_USER = (
    "Wikipedia Title: The Last Horse\n"
    "The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"
    "Wikipedia Title: Southampton\n"
    "The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research."
    "Wikipedia Title: Stanton Township, Champaign County, Illinois\n"
    "Stanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units."
    "Wikipedia Title: Neville A. Stanton\n"
    'Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n'
    "Wikipedia Title: Finding Nemo\n"
    "Finding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $94 million Box office $940.3 million\n\n"
    "Question: When was Neville A. Stanton\'s employer founded?\n"
    "Thought: "
)

CHAT_EXAMPLE_ASSISTANT = (
    "The employer of Neville A. Stanton is University of Southampton. "
    "The University of Southampton was founded in 1862.\n"
    "Answer: 1862."
)


class ChatPromptBuilder(PromptBuilder):
    """Build the fixed few-shot chat prompt used by the Ollama chat answerer."""
    def __init__(self, config: dict):
        return

    def build(self, question: Question, chunks: list[RetrievedChunk]) -> Prompt:
        context = format_chat_context(chunks)
        final_user_message = f"{context}\n\nQuestion: {question.text}\nThought: "
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_MESSAGE},
            {"role": "user", "content": CHAT_EXAMPLE_USER},
            {"role": "assistant", "content": CHAT_EXAMPLE_ASSISTANT},
            {"role": "user", "content": final_user_message},
        ]
        return Prompt(
            text=final_user_message,
            context=context,
            question=question,
            references=chunks,
            messages=messages,
        )


def format_chat_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks in the title-and-passage format of the example."""
    parts: list[str] = []
    for item in chunks:
        metadata = item.chunk.metadata
        idx = metadata.get("chunk_index", item.chunk.id)
        parts.append(f"Wikipedia Title: {idx}:{item.chunk.text}")
    return "\n\n".join(parts)
