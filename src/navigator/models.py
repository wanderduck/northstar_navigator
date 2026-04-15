"""Data models for the Navigator application."""

from enum import Enum
from pydantic import BaseModel, Field


class ReadingLevel(str, Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    DETAILED = "detailed"


# FPL thresholds for 2026 (HHS Poverty Guidelines)
# https://aspe.hhs.gov/topics/poverty-economic-mobility/poverty-guidelines
FPL_2026_BASE = 15_650  # 1 person
FPL_2026_PER_ADDITIONAL = 5_380  # each additional person


def fpl_threshold(household_size: int) -> int:
    """Return the Federal Poverty Level for a given household size (2026, 48 contiguous states)."""
    if household_size < 1:
        household_size = 1
    return FPL_2026_BASE + FPL_2026_PER_ADDITIONAL * (household_size - 1)


class Dependent(BaseModel):
    age: int | None = None
    relationship: str = "child"


class UserProfile(BaseModel):
    income: float | None = None
    household_size: int = 1
    county: str | None = None
    state: str = "MN"
    employment_status: str | None = None
    dependents: list[Dependent] = Field(default_factory=list)
    age: int | None = None
    is_veteran: bool = False
    is_disabled: bool = False
    citizenship_status: str = "citizen"
    concerns: list[str] = Field(default_factory=list)
    language: str = "en"
    reading_level: ReadingLevel = ReadingLevel.STANDARD

    @property
    def fpl_percentage(self) -> float | None:
        """Calculate income as a percentage of the Federal Poverty Level."""
        if self.income is None:
            return None
        threshold = fpl_threshold(self.household_size)
        return round((self.income / threshold) * 100, 1)

    @property
    def has_children(self) -> bool:
        return any(d.age < 18 for d in self.dependents)

    @property
    def has_young_children(self) -> bool:
        """Children under 5 (relevant for WIC, Head Start)."""
        return any(d.age < 5 for d in self.dependents)


class Program(BaseModel):
    id: str
    name: str
    category: str  # food, cash, health, housing, energy, employment, childcare, emergency
    jurisdiction: str  # "federal", "state:MN", "county:ramsey", "cap:caprw"
    description: str
    eligibility_summary: str = ""
    application_url: str = ""
    application_portal: str = ""  # e.g., "MNbenefits", "MNsure", "uimn.org"
    phone: str = ""
    source: str = ""
    documents_needed: list[str] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    program_id: str
    program_name: str
    eligible: bool | None = None  # None = unknown/need more info
    confidence: str = "medium"  # "high", "medium", "low"
    reason: str = ""
    estimated_benefit: str = ""
    source: str = ""
    priority: str = "normal"  # "high", "normal", "low"
    category: str = ""


class BenefitsResponse(BaseModel):
    eligible_programs: list[EligibilityResult] = Field(default_factory=list)
    documents_needed: list[str] = Field(default_factory=list)
    application_groups: dict[str, list[str]] = Field(default_factory=dict)
    follow_up_questions: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an informational tool, not legal advice. "
        "Eligibility determinations are unofficial estimates. "
        "Always verify with the relevant agency before applying. "
        "Program rules change -- information last verified April 2026."
    )
