"""Stage 3: Response Generation — produce plain language output from eligibility results."""

from navigator.models import UserProfile, BenefitsResponse, ReadingLevel
from navigator.ollama_client import OllamaClient
from navigator.prompts import get_response_prompt, RESPONSE_DISCLAIMER
from navigator.config import SUPPORTED_LANGUAGES


class ResponseGenerator:
    """Generate plain-language benefits guidance from eligibility results."""

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()

    def generate(self, response: BenefitsResponse, profile: UserProfile) -> str:
        """Generate a plain-language response for the user.

        Args:
            response: The BenefitsResponse from the eligibility engine.
            profile: The user's profile (for reading level and language).
        """
        context = self._format_context(response, profile)
        language = SUPPORTED_LANGUAGES.get(profile.language, "English")
        system_prompt = get_response_prompt(
            reading_level=profile.reading_level.value,
            language=language,
        )

        return self.client.chat(context, system_prompt=system_prompt)

    def generate_stream(self, response: BenefitsResponse, profile: UserProfile):
        """Stream a plain-language response token by token.

        Yields partial response strings (cumulative).
        """
        context = self._format_context(response, profile)
        language = SUPPORTED_LANGUAGES.get(profile.language, "English")
        system_prompt = get_response_prompt(
            reading_level=profile.reading_level.value,
            language=language,
        )

        accumulated = ""
        for token in self.client.chat_stream(context, system_prompt=system_prompt):
            accumulated += token
            yield accumulated

    def _format_context(self, response: BenefitsResponse, profile: UserProfile) -> str:
        """Format eligibility results as structured context for the LLM."""
        parts = []

        # User situation summary
        parts.append("## User Situation")
        parts.append(f"- Income: ${profile.income:,.0f}/year" if profile.income else "- Income: Unknown")
        parts.append(f"- Household size: {profile.household_size}")
        parts.append(f"- County: {profile.county or 'Unknown'}")
        parts.append(f"- Employment: {profile.employment_status or 'Unknown'}")
        if profile.dependents:
            ages = ", ".join(str(d.age) for d in profile.dependents)
            parts.append(f"- Dependents: {len(profile.dependents)} (ages: {ages})")
        if profile.fpl_percentage:
            parts.append(f"- Federal Poverty Level: {profile.fpl_percentage}%")

        # Eligible programs
        parts.append("\n## Eligible Programs")
        high_priority = [r for r in response.eligible_programs if r.priority == "high"]
        normal_priority = [r for r in response.eligible_programs if r.priority != "high"]

        if high_priority:
            parts.append("\n### HIGH PRIORITY")
            for r in high_priority:
                parts.append(f"\n**{r.program_name}**")
                parts.append(f"- Category: {r.category}")
                parts.append(f"- Reason: {r.reason}")
                if r.estimated_benefit:
                    parts.append(f"- Estimated benefit: {r.estimated_benefit}")
                parts.append(f"- Confidence: {r.confidence}")
                parts.append(f"- Source: {r.source}")

        if normal_priority:
            parts.append("\n### ALSO CHECK")
            for r in normal_priority:
                parts.append(f"\n**{r.program_name}**")
                parts.append(f"- Reason: {r.reason}")
                if r.estimated_benefit:
                    parts.append(f"- Estimated benefit: {r.estimated_benefit}")
                parts.append(f"- Source: {r.source}")

        # Documents
        if response.documents_needed:
            parts.append("\n## Documents to Gather")
            for doc in response.documents_needed:
                parts.append(f"- {doc}")

        # Application portals
        if response.application_groups:
            parts.append("\n## Where to Apply")
            for portal, programs in response.application_groups.items():
                parts.append(f"- **{portal}**: {', '.join(programs)}")

        # Disclaimer
        parts.append(f"\n## Disclaimer\n{RESPONSE_DISCLAIMER}")

        return "\n".join(parts)
