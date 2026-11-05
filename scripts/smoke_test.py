"""Live capability checks: python -m scripts.smoke_test."""

from collections.abc import Callable

from pydantic import BaseModel

from src.config import get_llm, get_tavily_client


class CityAnswer(BaseModel):
    """Minimal structured output."""

    city: str


def get_time(city: str) -> str:
    """Return a dummy time for the supplied city."""
    return f"12:00 in {city}"


def check_answer() -> bool:
    """Require a nonempty model answer."""
    return bool(get_llm().invoke("Reply with hello.").content)


def check_tools() -> bool:
    """Require and execute a valid dummy tool call."""
    reply = get_llm().bind_tools([get_time], tool_choice="required").invoke(
        "Call get_time for Lisbon."
    )
    return any(
        call["name"] == "get_time"
        and call["args"].get("city", "").casefold() == "lisbon"
        and bool(get_time(**call["args"]))
        for call in reply.tool_calls
    )


def check_structured() -> bool:
    """Validate function-calling output with Pydantic."""
    result = get_llm().with_structured_output(
        CityAnswer, method="function_calling"
    ).invoke("Return the city Lisbon.")
    return isinstance(result, CityAnswer) and result.city.casefold() == "lisbon"


def check_search() -> bool:
    """Require at least one Lisbon museum result."""
    return bool(
        get_tavily_client().search("Lisbon museums", max_results=3).get("results")
    )


def check_extract() -> bool:
    """Require extracted museum page text."""
    result = get_tavily_client().extract(urls=["https://www.maat.pt/en"])
    return any(
        item.get("raw_content", "").strip() for item in result.get("results", [])
    )


def main() -> int:
    """Print all results without exposing provider exceptions or secrets."""
    checks: list[tuple[str, Callable[[], bool]]] = [
        ("Nebius answers", check_answer),
        ("Nebius TOOL CALLING", check_tools),
        ("Nebius STRUCTURED OUTPUT", check_structured),
        ("Tavily search", check_search),
        ("Tavily extract", check_extract),
    ]
    failed = False
    for label, check in checks:
        try:
            ok = bool(check())
        except Exception:  # noqa: BLE001 - report all checks without leaking secrets
            ok = False
        print(f"{'PASS' if ok else 'FAIL'}: {label}")
        if not ok:
            failed = True
            if label in ("Nebius TOOL CALLING", "Nebius STRUCTURED OUTPUT"):
                print("Pick a NEBIUS_MODEL that supports function calling")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
