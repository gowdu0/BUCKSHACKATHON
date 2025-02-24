#!/usr/bin/env python3
# source .venv/bin/activate

import pandas as pd
import numpy as np

def clean_account_level_data(input_file, output_file):
    """
    Reads the account-level CSV, removes rows where STM == 1, drops only the STM column,
    leaves Season and Account Number in the final dataset (but does not compute stats on them).
    For AvgSpend, removes the '$' sign to make it numeric.
    
    Then prints and saves summary stats for these columns only:
      Numeric: SingleGameTickets, PartialPlanTickets, GroupTickets,
               AvgSpend, GamesAttended, DistanceToArena, BasketballPropensity
      Categorical: FanSegment, SocialMediaEngagement
    
    Fills missing numeric columns with the mean, missing categorical columns
    with the alphabetical "median," and saves both the cleaned CSV and a CSV
    of summary stats.
    """

    # 1. Load the CSV file into a pandas DataFrame.
    df = pd.read_csv(input_file)
    print(f"\n[INFO] Loaded data from '{input_file}'.")

    # 2. Remove rows where STM == 1 (i.e., season ticket members we don't want to analyze).
    #    If STM is stored as integer:
    df = df[df["STM"] != 1]
    #    If instead STM was stored as string "1", do: df = df[df["STM"] != "1"]

    # 3. Drop only the STM column; keep Season and Account Number in the final dataset.
    df.drop(columns=["STM"], inplace=True, errors="ignore")

    # 4. Clean the 'AvgSpend' column by removing any '$' sign and converting to float.
    #    (Do this only if AvgSpend exists in the dataset.)
    if "AvgSpend" in df.columns:
        df["AvgSpend"] = (
            df["AvgSpend"]
            .astype(str)                        # ensure string type
            .str.replace("$", "", regex=False)  # remove the $ symbol
            .str.replace(",", "", regex=False)  # remove commas if any
            .astype(float)                      # convert to float
        )

    # -----------------------------------------------------------------
    # NEW STEP: Read seat-level data, compute distinct Games + total tickets
    # -----------------------------------------------------------------
    try:
        seat_df = pd.read_csv("Prompt1SeatLevel.csv")  # or your seat-level filename

        # 1) Drop duplicates so each (AccountNumber, Game) is unique
        seat_unique = seat_df[["AccountNumber", "Game"]].drop_duplicates()

        # 2) SumGamesAttended = distinct games per account
        sum_games_series = (
            seat_unique.groupby("AccountNumber").size().rename("SumGamesAttended")
        )

        # 3) TotNumTicketsPurchased = total rows (tickets) per account
        tot_tickets_series = seat_df.groupby("AccountNumber").size().rename("TotNumTicketsPurchased")

        # 4) Combine into one DataFrame for merging
        seat_stats = pd.concat([sum_games_series, tot_tickets_series], axis=1)

        # 5) Merge these columns into df
        df = df.merge(seat_stats, how="left", on="AccountNumber")

        # 6) Fill NaN with 0 where seat data is missing
        df["SumGamesAttended"] = df["SumGamesAttended"].fillna(0).astype(int)
        df["TotNumTicketsPurchased"] = df["TotNumTicketsPurchased"].fillna(0).astype(int)

        print("\n[INFO] Appended SumGamesAttended (distinct games) "
              "and TotNumTicketsPurchased from seat-level data.\n")
    except FileNotFoundError:
        print("\n[WARNING] 'Prompt1SeatLevel.csv' not found. Cannot calculate SumGamesAttended or TotNumTicketsPurchased.\n")
        df["SumGamesAttended"] = 0
        df["TotNumTicketsPurchased"] = 0

    # ------------------------------------------------------------
    # 5. Define the columns we actually want to compute stats on.
    #    (Leave "Season" and "Account Number" in df but skip them here)
    # ------------------------------------------------------------
    numeric_cols = [
        "SingleGameTickets", 
        "PartialPlanTickets", 
        "GroupTickets", 
        "AvgSpend", 
        "GamesAttended", 
        "DistanceToArena", 
        "BasketballPropensity"
    ]
    categorical_cols = [
        "FanSegment",
        "SocialMediaEngagement"
    ]

    # ------------------------------------------------------------
    # 6. Print summary statistics for these columns to the console
    # ------------------------------------------------------------
    print("\n===== SUMMARY STATISTICS FOR NUMERIC COLUMNS =====")
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n--- Column: {col} ---")
            print(df[col].describe())  # count, mean, std, min, quartiles, max
    
    print("\n===== SUMMARY STATISTICS FOR CATEGORICAL COLUMNS =====")
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n--- Column: {col} ---")
            print(df[col].describe())  # count, unique, top, freq

            # Also print min, max, median (alphabetically), total count for each category
            cat_series = df[col].dropna()
            unique_sorted = sorted(cat_series.unique())
            total_count = cat_series.count()

            cat_min = min(unique_sorted) if unique_sorted else "N/A"
            cat_max = max(unique_sorted) if unique_sorted else "N/A"
            if unique_sorted:
                median_idx = (len(unique_sorted) - 1) // 2
                cat_median = unique_sorted[median_idx]
            else:
                cat_median = "N/A"

            print(f"  -> MIN Category (alphabetical): {cat_min}")
            print(f"  -> MAX Category (alphabetical): {cat_max}")
            print(f"  -> MEDIAN Category (alphabetical): {cat_median}")
            print(f"  -> TOTAL non-null COUNT: {total_count}")

    # ------------------------------------------------------------
    # 7. SAVE summary stats (only for these columns) to a CSV:
    #    'accountLevelSummaryStats.csv'
    # ------------------------------------------------------------
    
    # 7a. Numeric columns: Use the standard describe() approach
    numeric_summary = df[[c for c in numeric_cols if c in df.columns]].describe().T
    numeric_summary["ColumnType"] = "Numeric"

    # Make the index a column (so we can merge with categorical summary later)
    numeric_summary.reset_index(inplace=True)
    numeric_summary.rename(columns={"index": "Column"}, inplace=True)

    # Add placeholder columns for categorical-like fields (Category, Frequency, etc.)
    # so the final CSV can include both numeric and categorical in a single file
    for col_to_add in ["Category", "Frequency", "MinCategory", "MaxCategory", "MedianCategory"]:
        numeric_summary[col_to_add] = None

    # 7b. Categorical columns: Build a custom summary
    cat_data = []
    for col in categorical_cols:
        if col in df.columns:
            # Count how many times each category appears
            freq_series = df[col].value_counts(dropna=False)
            
            # Sort the unique categories alphabetically
            unique_sorted = sorted(freq_series.index)
            if len(unique_sorted) > 0:
                cat_min = unique_sorted[0]            # alphabetically first
                cat_max = unique_sorted[-1]           # alphabetically last
                median_index = (len(unique_sorted) - 1) // 2
                cat_median = unique_sorted[median_index]
            else:
                # If no values, set them to None or "N/A"
                cat_min, cat_max, cat_median = None, None, None
            
            # For each distinct category in this column, record frequency + min/max/median
            for category, freq in freq_series.items():
                cat_data.append({
                    "Column": col,
                    "Category": category,
                    "Frequency": freq,
                    "MinCategory": cat_min,
                    "MaxCategory": cat_max,
                    "MedianCategory": cat_median,
                    "ColumnType": "Categorical"
                })

    # Convert our categorical list-of-dicts into a DataFrame
    categorical_summary = pd.DataFrame(cat_data)

    # For a consistent final CSV, add placeholder numeric columns 
    # (so numeric and categorical summaries have the same column set)
    for col_to_add in ["count","mean","std","min","25%","50%","75%","max"]:
        categorical_summary[col_to_add] = None

    # 7c. Combine numeric and categorical summaries into one DataFrame
    combined_summary = pd.concat([numeric_summary, categorical_summary], ignore_index=True)

    # 7d. Save summary to CSV
    summary_file = "accountLevelSummaryStats.csv"
    combined_summary.to_csv(summary_file, index=False)
    print(f"\n[INFO] Summary statistics saved to '{summary_file}'.\n")
    # ------------------------------------------------------------
    # 8. Fill missing values:
    #    - Numeric => mean
    #    - Categorical => alphabetical 'median'
    # ------------------------------------------------------------
    print("[INFO] Filling missing numeric values with column means...\n")
    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
    
    print("[INFO] Filling missing categorical values with 'median' category (alphabetically)...\n")
    for col in categorical_cols:
        if col in df.columns:
            unique_cats = df[col].dropna().unique()
            unique_cats.sort()
            if len(unique_cats) > 0:
                median_index = len(unique_cats) // 2
                median_cat = unique_cats[median_index]
                df[col].fillna(median_cat, inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

    # ------------------------------------------------------------
    # 9. Save the cleaned dataset (which still contains Season and Account Number)
    #    into a new CSV file
    # ------------------------------------------------------------
    df.to_csv(output_file, index=False)
    print(f"[INFO] Cleaned data saved to '{output_file}'.")


# Optional main method to run directly:
if __name__ == "__main__":
    input_csv = "Prompt1AccountLevel.csv"   # Change to your actual input CSV file
    output_csv = "AccountLevelCleaned.csv"  # Desired output CSV file
    clean_account_level_data(input_csv, output_csv)
