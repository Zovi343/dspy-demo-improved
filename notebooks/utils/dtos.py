from typing import Literal

from pydantic import BaseModel, Field

class Merger(BaseModel):
    article_id: int | None = Field(default=None)
    company_1: str | None = Field(description="First company in the merger")
    company_1_ticker: list[str] | None = Field(description="Stock ticker of first company")
    company_2: str | None = Field(description="Second company in the merger")
    company_2_ticker: list[str] | None = Field(description="Stock ticker of second company")
    merged_entity: str | None = Field(description="Name of merged entity")
    deal_amount: str | None = Field(description="Total monetary amount of the deal")
    deal_currency: Literal["USD", "CAD", "AUD", "Unknown"] = Field(
        description="Currency of the merger deal"
    )
    article_type: Literal["merger"] = "merger"


class Acquisition(BaseModel):
    article_id: int | None = Field(default=None)
    parent_company: str | None = Field(description="Parent company in the acquisition")
    parent_company_ticker: list[str] | None = Field(description="Stock ticker of parent company")
    child_company: str | None = Field(description="Child company in the acquisition")
    child_company_ticker: list[str] | None = Field(description="Stock ticker of child company")
    deal_amount: str | None = Field(description="Total monetary amount of the deal")
    deal_currency: Literal["USD", "CAD", "AUD", "Unknown"] = Field(
        description="Currency of the acquisition deal"
    )
    article_type: Literal["acquisition"] = "acquisition"


class Other(BaseModel):
    article_id: int | None = Field(default=None)
    article_type: Literal["other"] = "other"