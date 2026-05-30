import pandas as pd

docs = []

# -------------------------
# CROPS
# -------------------------
crops = pd.read_csv("data/raw/crops.csv")

for _, row in crops.iterrows():

    text = (
        f"In {row['Area']} during {row['Year']}, "
        f"{row['Item']} had {row['Element']} "
        f"of {row['Value']} {row['Unit']}."
    )

    docs.append(text)

# -------------------------
# FERTILIZERS
# -------------------------
fert = pd.read_csv("data/raw/fertilizers.csv")

for _, row in fert.iterrows():

    text = (
        f"In {row['Area']} during {row['Year']}, "
        f"{row['Item']} agricultural use "
        f"was {row['Value']} {row['Unit']}."
    )

    docs.append(text)

# -------------------------
# PESTICIDES
# -------------------------
pest = pd.read_csv("data/raw/pesticides.csv")

for _, row in pest.iterrows():

    text = (
        f"In {row['Area']} during {row['Year']}, "
        f"{row['Item']} usage was "
        f"{row['Value']} {row['Unit']}."
    )

    docs.append(text)

# -------------------------
# TEMPERATURE
# -------------------------
temp = pd.read_csv("data/raw/temperature.csv")

for _, row in temp.iterrows():

    text = (
        f"In {row['Area']} during {row['Year']} "
        f"for {row['Months']}, "
        f"temperature change was "
        f"{row['Value']} {row['Unit']}."
    )

    docs.append(text)

# -------------------------
# SAVE ALL DOCS
# -------------------------
with open("data/processed/farming_docs.txt", "w", encoding="utf-8") as f:

    for d in docs:
        f.write(d + "\n")

print(f"Saved {len(docs)} farming documents")