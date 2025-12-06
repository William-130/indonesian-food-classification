"""
Nutrition Matcher - Clean implementation with exact class name matching.
Maps the 35 Indonesian food classes to their nutrition data from CSV.
No fuzzy matching - perfect 1:1 correspondence with dataset classes.
"""
import pandas as pd
import os


class NutritionMatcher:
    """Matches food class names to nutrition information using exact matching."""
    
    def __init__(self, csv_path="Dataset ML/nutrition_35_foods.csv"):
        """Initialize the nutrition matcher."""
        if not os.path.isabs(csv_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.dirname(os.path.dirname(current_dir))
            csv_path = os.path.join(workspace_root, csv_path)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Nutrition CSV not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"Loaded nutrition data: {len(self.df)} foods")
        
        self.class_names = [
            "asinan-jakarta", "ayam-betutu", "ayam-bumbu-rujak", "ayam-goreng-lengkuas",
            "bika-ambon", "bir-pletok", "bubur-manado", "cendol", "es-dawet", "gado-gado",
            "gudeg", "gulai-ikan-mas", "keladi", "kerak-telor", "klappertart", "kolak",
            "kue-lumpur", "kunyit-asam", "laksa-bogor", "lumpia-semarang", "mie-aceh",
            "nagasari", "nasi-goreng-kampung", "papeda", "pempek-palembang", "rawon-surabaya",
            "rendang", "rujak-cingur", "sate-ayam-madura", "sate-lilit", "sate-maranggi",
            "soerabi", "soto-ayam-lamongan", "soto-banjar", "tahu-telur"
        ]
        
        self.nutrition_map = {}
        for idx, class_name in enumerate(self.class_names):
            if idx >= len(self.df):
                continue
            row = self.df.iloc[idx]
            self.nutrition_map[class_name] = {
                "name": row["name"],
                "calories": float(row["calories"]),
                "protein": float(row["protein"]),
                "fat": float(row["fat"]),
                "carbohydrate": float(row["carbohydrate"])
            }
        print(f"Built nutrition mapping for {len(self.nutrition_map)} classes")
    
    def get_nutrition(self, class_name):
        """Get nutrition information for a food class."""
        return self.nutrition_map.get(class_name)
    
    def get_all_foods(self):
        """Get list of all available food class names."""
        return list(self.nutrition_map.keys())


if __name__ == "__main__":
    matcher = NutritionMatcher()
    print(f"\nTotal foods: {len(matcher.get_all_foods())}")
    for idx, food in enumerate(matcher.get_all_foods()[:5], 1):
        nut = matcher.get_nutrition(food)
        print(f"{idx}. {food} -> {nut['name']} ({nut['calories']:.0f} kcal)")
