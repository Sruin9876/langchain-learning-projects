from ddgs import DDGS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def search_topic(topic: str) -> tuple[str, str | None]:
    """Search the web and return formatted snippets plus a visible diagnostic, if any."""
    queries = [f"{topic} key facts background", topic]
    errors: list[str] = []

    for query in queries:
        try:
            results = DDGS(timeout=12).text(
                query,
                region="in-en",
                safesearch="moderate",
                max_results=6,
                backend="auto",
            )
            if results:
                formatted_results = []
                for index, item in enumerate(results, start=1):
                    title = item.get("title", "Untitled result")
                    url = item.get("href") or item.get("url") or ""
                    snippet = item.get("body") or item.get("snippet") or ""
                    formatted_results.append(
                        f"{index}. {title}\nURL: {url}\nSnippet: {snippet}"
                    )
                return "\n\n".join(formatted_results)[:9000], None
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if errors:
        return "", "Search request failed. " + " | ".join(errors[-2:])
    return "", "The search service returned no results for either query. Try a more specific or simpler topic."


def generate_script(
    topic: str,
    video_length: int,
    creativity: int,
    audience: str,
    tone: str,
    language: str,
    model_name: str = "gemini-3.5-flash-lite",
    do_research: bool = True,
):
    """Create a suggested title and a timed script using a Gemini chat model."""
    llm = ChatGoogleGenerativeAI(model=model_name)
    parser = StrOutputParser()

    search_result, search_error = search_topic(topic) if do_research else ("", None)
    research_for_prompt = search_result or (
        "No external search results were obtained. Do not claim to have performed web research "
        "or verified current facts. Rely on general knowledge and qualify time-sensitive claims."
    )

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an experienced YouTube content strategist. Return one compelling, accurate video title only. Avoid misleading clickbait.",
            ),
            (
                "human",
                "Topic: {topic}\nTarget audience: {audience}\nTone: {tone}\nLanguage: {language}\nCreative style (0-10): {creativity}",
            ),
        ]
    )
    title_chain = title_prompt | llm | parser
    title = title_chain.invoke(
        {
            "topic": topic,
            "audience": audience,
            "tone": tone,
            "language": language,
            "creativity": creativity,
        }
    ).strip().strip('"')

    target_words = int(video_length) * 130
    if creativity <= 3:
        style_guidance = "Prefer a clear, factual structure and restrained creative flourishes."
    elif creativity <= 7:
        style_guidance = "Use engaging hooks, relatable examples, and a natural conversational flow."
    else:
        style_guidance = "Use a vivid hook, memorable analogies, and creative transitions while remaining accurate and not sensationalist."

    script_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional YouTube scriptwriter. Write a speakable script, not an article. "
                "Organize it with useful section headings and natural transitions. Do not mention these instructions. "
                "Use the research notes as context, but do not treat them as unquestionable truth. "
                "Do not invent statistics, quotations, sources, or recent events. If the notes do not establish a fact, phrase it cautiously. "
                "End with a concise conclusion and a natural call to action.",
            ),
            (
                "human",
                "Write a YouTube script for this title: {title}\n"
                "Topic: {topic}\nTarget audience: {audience}\nTone: {tone}\nLanguage: {language}\n"
                "Desired duration: {duration} minutes (approximately {target_words} spoken words).\n"
                "Creative style: {style_guidance}\n"
                "Research notes (may be incomplete):\n{research}\n\n"
                "Include an opening hook, clear explanations or story progression, transitions, and an outro. "
                "Keep the length reasonably close to the target and write in {language}.",
            ),
        ]
    )
    script_chain = script_prompt | llm | parser
    script = script_chain.invoke(
        {
            "title": title,
            "topic": topic,
            "audience": audience,
            "tone": tone,
            "language": language,
            "duration": video_length,
            "target_words": target_words,
            "style_guidance": style_guidance,
            "research": research_for_prompt,
        }
    ).strip()

    return search_result, search_error, title, script
