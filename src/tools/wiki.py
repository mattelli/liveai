"""Wikipedia search tool (port of smolagents v1.26.0 ``WikipediaSearchTool``).

Searches Wikipedia through its public MediaWiki API and returns page
summaries. No API key required; a custom ``User-Agent`` is mandatory per
Wikimedia's API policy (https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy).
"""
import json

getLogger = __import__("logging").getLogger

from src.common import dump_func_call

_USER_AGENT = "live-assistant/1.0 (local personal assistant)"

_SEARCH_URL = "https://{lang}.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"


@dump_func_call
def wikipedia_search(
    query: str,
    language: str = "en",
    max_results: int = 3,
    timeout_s: float = 10.0,
) -> str:
    """
    Search Wikipedia and return the summary of the top matching articles.

    Args:
        query: The search terms.
        language: Wikipedia language code (default "en").
        max_results: Number of articles to summarize (1-10).
        timeout_s: HTTP timeout for the two API calls per article.

    Returns:
        For each article: title, URL and a short summary extract, separated
        by blank lines. "No results" line when nothing matched; an "Error:"
        line on HTTP/network failure (never raises to the model).
    """
    query = str(query or "").strip()
    if not query:
        return "Error: empty Wikipedia search query"
    try:
        max_results = max(1, min(10, int(max_results)))
    except (TypeError, ValueError):
        max_results = 3
    language = str(language or "en").strip() or "en"
    import requests  # lazy: keeps `import src.tools` cheap (package pattern)

    headers = {"User-Agent": _USER_AGENT}
    try:
        search_resp = requests.get(
            _SEARCH_URL.format(lang=language),
            params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": max_results,
            },
            headers=headers, timeout=timeout_s,
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])
    except Exception as exc:
        return f"Error: Wikipedia search failed: {exc}"
    if not hits:
        return f"No Wikipedia articles found for {query!r}"

    blocks: list[str] = []
    for hit in hits[:max_results]:
        title = str(hit.get("title", "")).replace(" ", "_")
        try:
            summary_resp = requests.get(
                _SUMMARY_URL.format(lang=language, title=title),
                headers=headers, timeout=timeout_s,
            )
            summary_resp.raise_for_status()
            data = summary_resp.json()
            extract = str(data.get("extract", "")).strip()
            url = str(data.get("content_urls", {}).get("desktop", {})
                      .get("page", "")) or (
                f"https://{language}.wikipedia.org/wiki/{title}")
        except Exception as exc:
            blocks.append(f"# {hit.get('title', title)}\n"
                          f"(summary unavailable: {exc})")
            continue
        blocks.append(f"# {data.get('title', hit.get('title', title))}\n"
                      f"{url}\n\n{extract}")
    return "\n\n".join(blocks)
