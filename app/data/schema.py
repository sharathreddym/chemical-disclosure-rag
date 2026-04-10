"""SQLite schema for the cosmetics chemicals dataset."""

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chemicals_in_cosmetics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    CDPHId INTEGER,
    ProductName TEXT,
    CSFId TEXT,
    CSF TEXT,
    CompanyId INTEGER,
    CompanyName TEXT,
    BrandName TEXT,
    PrimaryCategoryId INTEGER,
    PrimaryCategory TEXT,
    SubCategoryId INTEGER,
    SubCategory TEXT,
    CasId INTEGER,
    CasNumber TEXT,
    ChemicalId INTEGER,
    ChemicalName TEXT,
    InitialDateReported TEXT,
    MostRecentDateReported TEXT,
    DiscontinuedDate TEXT,
    ChemicalCreatedAt TEXT,
    ChemicalUpdatedAt TEXT,
    ChemicalDateRemoved TEXT,
    ChemicalCount INTEGER
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_company ON chemicals_in_cosmetics(CompanyName);",
    "CREATE INDEX IF NOT EXISTS idx_brand ON chemicals_in_cosmetics(BrandName);",
    "CREATE INDEX IF NOT EXISTS idx_chemical ON chemicals_in_cosmetics(ChemicalName);",
    "CREATE INDEX IF NOT EXISTS idx_cas ON chemicals_in_cosmetics(CasNumber);",
    "CREATE INDEX IF NOT EXISTS idx_category ON chemicals_in_cosmetics(PrimaryCategory);",
    "CREATE INDEX IF NOT EXISTS idx_subcategory ON chemicals_in_cosmetics(SubCategory);",
    "CREATE INDEX IF NOT EXISTS idx_cdphid ON chemicals_in_cosmetics(CDPHId);",
    "CREATE INDEX IF NOT EXISTS idx_chemicalid ON chemicals_in_cosmetics(ChemicalId);",
    "CREATE INDEX IF NOT EXISTS idx_discontinued ON chemicals_in_cosmetics(DiscontinuedDate);",
]

# Column metadata for agents to understand the schema
COLUMN_DESCRIPTIONS = {
    "CDPHId": "California Department of Public Health product ID",
    "ProductName": "Label name of the cosmetic/personal care product",
    "CSFId": "Cosmetic Safety Filing ID",
    "CSF": "Cosmetic Safety Filing description",
    "CompanyId": "Numeric ID of the company/manufacturer",
    "CompanyName": "Name of the company/manufacturer on the product label",
    "BrandName": "Product brand name",
    "PrimaryCategoryId": "Numeric ID of the primary product category",
    "PrimaryCategory": "Primary product category (e.g., Makeup Products, Hair Care Products)",
    "SubCategoryId": "Numeric ID of the product subcategory",
    "SubCategory": "Product subcategory (e.g., Lip Color, Hair Shampoos)",
    "CasId": "Internal CAS registry ID",
    "CasNumber": "Chemical Abstracts Service registry number (e.g., 13463-67-7)",
    "ChemicalId": "Internal chemical ID",
    "ChemicalName": "Name of the reported hazardous chemical ingredient",
    "InitialDateReported": "Date the chemical was first reported (MM/DD/YYYY)",
    "MostRecentDateReported": "Most recent reporting date (MM/DD/YYYY)",
    "DiscontinuedDate": "Date the product was discontinued (MM/DD/YYYY), empty if still active",
    "ChemicalCreatedAt": "Date the chemical record was created",
    "ChemicalUpdatedAt": "Date the chemical record was last updated",
    "ChemicalDateRemoved": "Date the chemical was removed from the product, empty if still present",
    "ChemicalCount": "Number of reported chemicals for this product",
}

SCHEMA_SUMMARY = """
Table: chemicals_in_cosmetics
Contains records of hazardous chemicals reported in cosmetic products sold in California.
Each row represents a product-chemical pair (one product can have multiple chemicals).

Key columns for querying:
- ProductName, CompanyName, BrandName: product identification
- ChemicalName, CasNumber: chemical identification
- PrimaryCategory, SubCategory: product classification
- InitialDateReported, MostRecentDateReported, DiscontinuedDate: lifecycle dates
- CDPHId, ChemicalId, CasId: record IDs for citation

Date format: MM/DD/YYYY (stored as text)
"""
