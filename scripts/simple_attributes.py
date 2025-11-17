import pandas as pd

# Load the board game data
games_df = pd.read_csv("bgg_games_data.csv")

# Load the mapping files
simple_mech_df = pd.read_csv("simple_mechanics.csv")  # Columns: original_mechanic,simple_mechanic
simple_cat_df = pd.read_csv("simple_category.csv")    # Columns: original_category,simple_category

# Convert mappings to dictionaries for faster lookup
mech_map = dict(zip(simple_mech_df['mechanics'], simple_mech_df['simple_mechanics']))
cat_map = dict(zip(simple_cat_df['category'], simple_cat_df['simple_category']))

# Function to map semicolon-delimited list to simplified values with single space after semicolon
def map_semicolon_list(original_list, mapping_dict):
    if pd.isna(original_list) or original_list.strip() == "":
        return ""
    items = [item.strip() for item in original_list.split(';')]
    simple_items = [mapping_dict.get(item, item) for item in items]
    # Remove duplicates and maintain order
    seen = set()
    simple_items_unique = [x for x in simple_items if not (x in seen or seen.add(x))]
    # Join with '; ' (semicolon + single space)
    return '; '.join(simple_items_unique)

# Apply mapping to mechanics and category columns
games_df['simple_mechanics'] = games_df['mechanics'].apply(lambda x: map_semicolon_list(x, mech_map))
games_df['simple_category'] = games_df['category'].apply(lambda x: map_semicolon_list(x, cat_map))

# Keep required columns, including original columns
result_df = games_df[['bgg_id', 'name', 'mechanics', 'simple_mechanics', 'category', 'simple_category']]

# Save the resulting file
result_df.to_csv("game_simple_attributes.csv", index=False)
