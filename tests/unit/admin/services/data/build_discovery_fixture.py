"""Regenerate `discovery_index_multi_country.xlsx`.

The fixture is committed as a workbook rather than built in memory: it is the one end-to-end
input for the discovery write path, exercising the real template headers and the reference-area
axis over a realistic mix of records. `test_discovery_upload.py` still builds its workbooks in
memory, because those tests are about parsing edge cases rather than about a plausible file.

Run from this directory:

    python build_discovery_fixture.py

Every record is fictional. URLs use the reserved `example.com` domain and no row is a real
published dataset, so the fixture cannot be mistaken for onboarding content. Agency names are
real organizations because the agency is a filter axis and has to look like one.
"""

import openpyxl

DST = "discovery_index_multi_country.xlsx"

HEADERS = [
    "Reference area / country",
    "Regional coverage",
    "Excluded regional values",
    "Agency / organization",
    "Dataset ID",
    "Dataset name",
    "Description",
    "Dataset URL",
    "Time coverage",
    "Frequency coverage",
    "Indicators coverage (incl. units of measure)",
    "Relevant indicators not present in the dataset",
]

# reference_area, regional_coverage, excluded_regional_values, agency, dataset_id, name,
# description, url, time_coverage, frequency_coverage, indicators_coverage, missing_indicators
ROWS: list[list[str]] = [
    [
        "Japan (JPN)",
        "None",
        "",
        "Bank of Japan (BOJ)",
        "JP_MONETARY_BASE",
        "Monetary Base",
        "Monetary base and its components: banknotes in circulation, coins in circulation, and "
        "current account balances held at the central bank. Averages of daily figures.",
        "https://example.com/jp/monetary-base",
        "From 1970-01 to present (latest available 2026-06)",
        "Monthly; Annual",
        "monetary base (in JPY billions); banknotes in circulation (in JPY billions); current "
        "account balances (in JPY billions)",
        "consumer prices; trade in goods; gross domestic product, GDP",
    ],
    [
        "Japan (JPN)",
        "None",
        "",
        "Statistics Bureau of Japan (SBJ)",
        "JP_CPI",
        "Consumer Price Index",
        "Consumer price index, all items and by major group, including all items less fresh "
        "food. Published as an index and as year-on-year change.",
        "https://example.com/jp/cpi",
        "From 1970-01 to present (latest available 2026-07)",
        "Monthly; Annual",
        "consumer prices (as an index (2020 = 100), as year-on-year % change); consumer prices "
        "excluding fresh food (as an index (2020 = 100))",
        "producer prices; wages; monetary base",
    ],
    [
        "Japan (JPN)",
        "None",
        "",
        "Bank of Japan (BOJ)",
        "JP_FX",
        "Foreign Exchange Rates",
        "Spot exchange rates against major currencies and the nominal and real effective "
        "exchange rate indices.",
        "https://example.com/jp/fx",
        "From 1980-01 to present (latest available 2026-07)",
        "Business daily; Monthly",
        "spot exchange rate (in JPY per USD, in JPY per EUR); nominal effective exchange rate "
        "(as an index (2020 = 100))",
        "gross domestic product, GDP; consumer prices; interest rates",
    ],
    [
        "Canada (CAN)",
        "provinces, territories",
        "Nunavut and Yukon are absent",
        "Statistics Canada (StatCan)",
        "CA_LABOUR_FORCE",
        "Labour Force Survey",
        "Labour force characteristics by province and territory: employment, unemployment, the "
        "unemployment rate, and the participation rate. Seasonally adjusted and unadjusted.",
        "https://example.com/ca/labour-force",
        "From 1976-01 to present (latest available 2026-07)",
        "Monthly; Annual",
        "unemployment rate (as %); employment (persons, thousands); labour force participation "
        "rate (as %)",
        "job vacancies; union membership",
    ],
    [
        "Canada (CAN)",
        "provinces",
        "Quebec and Ontario are absent",
        "Statistics Canada (StatCan)",
        "CA_RETAIL",
        "Retail Trade by Province",
        "Retail trade sales by province and by retail subsector, at current prices, seasonally "
        "adjusted and unadjusted.",
        "https://example.com/ca/retail",
        "From 1991-01 to present (latest available 2026-06)",
        "Monthly; Annual",
        "retail sales (in current CAD millions); retail sales excluding motor vehicles (in "
        "current CAD millions)",
        "e-commerce sales; wholesale trade",
    ],
    [
        "Brazil (BRA)",
        "None",
        "",
        "Brazilian Institute of Geography and Statistics (IBGE)",
        "BR_CPI",
        "Broad National Consumer Price Index",
        "Broad national consumer price index covering households in the main metropolitan "
        "areas, as an index and as monthly and year-on-year change.",
        "https://example.com/br/cpi",
        "From 1979-12 to present (latest available 2026-07)",
        "Monthly; Annual",
        "consumer prices (as an index (2020 = 100), as year-on-year % change)",
        "producer prices; gross domestic product, GDP",
    ],
    [
        "India (IND)",
        "states, union territories",
        "",
        "Reserve Bank of India (RBI)",
        "IN_MONEY",
        "Money and Banking Statistics",
        "Monetary aggregates and banking indicators: reserve money, broad money and its "
        "components, aggregate deposits, and bank credit to the commercial sector.",
        "https://example.com/in/money",
        "From 1970-04 to present (latest available 2026-06)",
        "Monthly; Annual",
        "broad money M3 (in INR billions); reserve money (in INR billions); bank credit (in INR "
        "billions)",
        "consumer prices; balance of payments",
    ],
    [
        "Malaysia (MYS)",
        "None",
        "",
        "Bank Negara Malaysia (BNM)",
        "MY_MONETARY",
        "Monetary and Financial Statistics",
        "Broad money and its components, banking system deposits, and loans outstanding by "
        "sector.",
        "https://example.com/my/monetary",
        "From 1990-01 to present (latest available 2026-06)",
        "Monthly",
        "broad money M3 (in MYR millions); narrow money M1 (in MYR millions); loans outstanding "
        "(in MYR millions)",
        "consumer prices; labour force",
    ],
    [
        "Indonesia (IDN)",
        "None",
        "",
        "Bank Indonesia (BI)",
        "ID_BROAD_MONEY",
        "Broad Money and its Affecting Factors",
        "Broad money (M2) and components, and the factors affecting broad money: net foreign "
        "assets, net claims on central government, and claims on the private sector.",
        "https://example.com/id/broad-money",
        "From 1989-01 to present (latest available 2026-06)",
        "Monthly",
        "broad money M2 (in IDR billions); narrow money M1 (in IDR billions); quasi money (in "
        "IDR billions); net foreign assets (in IDR billions)",
        "consumer prices; policy interest rates",
    ],
    [
        "Germany (DEU)",
        "states",
        "",
        "Federal Statistical Office of Germany (Destatis)",
        "DE_NATIONAL_ACCOUNTS",
        "National Accounts",
        "Gross domestic product by output, expenditure, and income approach, at current prices "
        "and chain-linked volumes, with state-level annual aggregates.",
        "https://example.com/de/national-accounts",
        "From 1991 to present (latest available 2026-Q1)",
        "Quarterly; Annual",
        "gross domestic product, GDP (in current EUR billions, as year-on-year % change); gross "
        "value added (in current EUR billions)",
        "consumer prices; labour force",
    ],
    [
        "Indonesia (IDN); Malaysia (MYS); Thailand (THA); Philippines (PHL); Viet Nam (VNM); "
        "Singapore (SGP)",
        "None",
        "",
        "ASEAN Secretariat (ASEAN)",
        "ASEAN_MACRO",
        "Key Macroeconomic Indicators",
        "Comparable macroeconomic indicators for member states: gross domestic product at "
        "current prices and in growth terms, consumer price inflation, and population.",
        "https://example.com/asean/macro",
        "From 2005 to present (latest available 2025)",
        "Annual",
        "gross domestic product, GDP (in current USD billions, as year-on-year % change); "
        "consumer prices (as year-on-year % change); population (persons, millions)",
        "sub-national breakdowns; monthly indicators",
    ],
    [
        "Indonesia (IDN); ASEAN member states",
        "None",
        "",
        "ASEAN Secretariat (ASEAN)",
        "ID_MIXED_SCOPE",
        "Trade Facilitation Indicators",
        "Trade facilitation indicators for one reporting member and the grouping as a whole, "
        "compiled from national submissions.",
        "https://example.com/asean/trade-facilitation",
        "From 2010 to present (latest available 2025)",
        "Annual",
        "customs clearance time (in days); documentary compliance cost (in current USD)",
        "merchandise trade values; tariff rates",
    ],
    [
        "Euro area",
        "None",
        "",
        "Statistical Office of the European Union (Eurostat)",
        "EA_HICP",
        "Harmonised Index of Consumer Prices",
        "Harmonised index of consumer prices for the euro area aggregate, as an index and as "
        "annual rate of change, with breakdowns by classification group.",
        "https://example.com/ea/hicp",
        "From 1996-01 to present (latest available 2026-07)",
        "Monthly; Annual",
        "consumer prices (as an index (2015 = 100), as year-on-year % change); consumer prices "
        "excluding energy and food (as an index (2015 = 100))",
        "national accounts; labour force; sub-national data",
    ],
    [
        "World",
        "None",
        "",
        "International Monetary Fund (IMF)",
        "WORLD_AGGREGATES",
        "Selected World and Regional Aggregates",
        "World and regional aggregates for output growth, consumer price inflation, and current "
        "account balances, with projections alongside outturns.",
        "https://example.com/world/aggregates",
        "From 1980 to 2031 (projections from 2026)",
        "Annual",
        "gross domestic product, GDP (as year-on-year % change); consumer prices (as "
        "year-on-year % change); current account balance (as % of GDP)",
        "sub-national data; monthly indicators; country-level detail",
    ],
    [
        "",
        "None",
        "",
        "Bank for International Settlements (BIS)",
        "BIS_CREDIT",
        "Credit to the Non-Financial Sector",
        "Credit to the non-financial sector across reporting economies: credit to the private "
        "non-financial sector, to households, and to non-financial corporations.",
        "https://example.com/bis/credit",
        "From 1940-Q4 to present (latest available 2026-Q1)",
        "Quarterly",
        "credit to the private non-financial sector (as % of GDP); credit to households (as % "
        "of GDP)",
        "consumer prices; labour force",
    ],
    [
        "Japan (JPN); Germany (DEU); partner countries: China; United States; European Union",
        "None",
        "",
        "International Monetary Fund (IMF)",
        "TRADE_BILATERAL",
        "Bilateral Merchandise Trade, Selected Reporters",
        "Bilateral merchandise trade for selected reporting economies against their partner "
        "countries and partner groups, as reported and as derived estimates.",
        "https://example.com/imf/bilateral-trade",
        "From 1960-01 to present (latest available 2026-05)",
        "Monthly; Quarterly; Annual",
        "merchandise exports free on board (in current USD millions); merchandise imports cost, "
        "insurance and freight (in current USD millions)",
        "trade in services; trade by commodity",
    ],
]


def main() -> None:
    workbook = openpyxl.Workbook()
    instructions = workbook.active
    assert instructions is not None
    # The template ships an `Instructions` sheet before `Datasets`; the parser picks the sheet
    # by name, so its presence is part of what the fixture exercises.
    instructions.title = "Instructions"
    instructions["A1"] = "Fixture for the discovery write path. Every record is fictional."

    sheet = workbook.create_sheet("Datasets")
    sheet.append(HEADERS)
    for row in ROWS:
        if len(row) != len(HEADERS):
            raise ValueError(f"Row {row[4]!r} has {len(row)} columns, expected {len(HEADERS)}")
        sheet.append(row)

    workbook.save(DST)
    print(f"Wrote {DST}: {len(ROWS)} records.")


if __name__ == "__main__":
    main()
