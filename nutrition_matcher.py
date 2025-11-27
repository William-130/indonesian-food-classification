"""
Nutrition Matcher - Match food classification results with nutrition database
Using TheFuzz for fuzzy string matching
"""

import pandas as pd
from thefuzz import fuzz, process
import json


class NutritionMatcher:
    def __init__(self, csv_path='Dataset ML/nutrition.csv'):
        """
        Initialize nutrition matcher with CSV database
        
        Args:
            csv_path: Path to nutrition.csv file
        """
        self.df = pd.read_csv(csv_path)
        self.food_names = self.df['name'].tolist()
        
        # 35 class names from our model
        self.class_names = [
            'ayam-goreng', 'ayam-pop', 'batagor', 'bubur-ayam', 'dadih',
            'dendeng-batokok', 'gado-gado', 'gulai-ikan', 'gulai-tambusu',
            'gulai-tunjang', 'kue-sapik', 'laksa', 'lakso', 'lemang', 'lontong-sayur',
            'martabak-har', 'mi-celor', 'mie-aceh', 'nasi-kapau', 'nasi-uduk',
            'otak-otak', 'palai-rinuak', 'pempek', 'perkedel', 'rendang',
            'sate-padang', 'serabi', 'seblak', 'soto', 'tahu-petis',
            'tekwan', 'telur-gulung', 'telur-puyuh', 'ayam-bakar', 'soto-betawi'
        ]
        
        # Create mapping cache
        self._create_mapping()
    
    def _create_mapping(self):
        """Create fuzzy matching mapping between class names and nutrition database"""
        self.nutrition_map = {}
        
        print("🔍 Creating nutrition mapping with fuzzy matching...\n")
        
        for class_name in self.class_names:
            # Clean class name for matching
            search_term = class_name.replace('-', ' ')
            
            # Find best match using fuzzy matching
            best_match = process.extractOne(
                search_term,
                self.food_names,
                scorer=fuzz.token_set_ratio
            )
            
            if best_match:
                matched_name, score = best_match[0], best_match[1]
                
                # Get nutrition data
                nutrition_data = self.df[self.df['name'] == matched_name].iloc[0]
                
                self.nutrition_map[class_name] = {
                    'matched_name': matched_name,
                    'match_score': score,
                    'calories': float(nutrition_data['calories']),
                    'proteins': float(nutrition_data['proteins']),
                    'fat': float(nutrition_data['fat']),
                    'carbohydrate': float(nutrition_data['carbohydrate']),
                    'image_url': nutrition_data['image']
                }
                
                print(f"✓ {class_name:20s} → {matched_name:30s} (score: {score}%)")
            else:
                print(f"✗ {class_name:20s} → No match found")
                self.nutrition_map[class_name] = None
        
        print(f"\n✅ Mapped {len([v for v in self.nutrition_map.values() if v])} / {len(self.class_names)} classes")
    
    def get_nutrition(self, class_name):
        """
        Get nutrition information for a detected class
        
        Args:
            class_name: Class name from model prediction
            
        Returns:
            dict: Nutrition information or None
        """
        return self.nutrition_map.get(class_name)
    
    def format_nutrition_text(self, class_name):
        """
        Format nutrition info as readable text for display
        
        Args:
            class_name: Class name from model prediction
            
        Returns:
            str: Formatted nutrition text
        """
        nutrition = self.get_nutrition(class_name)
        
        if not nutrition:
            return "No nutrition data"
        
        text = (
            f"{nutrition['matched_name']}\n"
            f"Cal: {nutrition['calories']:.0f} kcal\n"
            f"Protein: {nutrition['proteins']:.1f}g\n"
            f"Fat: {nutrition['fat']:.1f}g\n"
            f"Carbs: {nutrition['carbohydrate']:.1f}g"
        )
        
        return text
    
    def save_mapping(self, output_path='nutrition_mapping.json'):
        """Save mapping to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.nutrition_map, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Mapping saved to {output_path}")


if __name__ == "__main__":
    # Test the matcher
    print("=" * 60)
    print("NUTRITION MATCHER - Fuzzy String Matching Test")
    print("=" * 60 + "\n")
    
    matcher = NutritionMatcher()
    
    # Save mapping
    matcher.save_mapping()
    
    # Test some examples
    print("\n" + "=" * 60)
    print("EXAMPLE QUERIES")
    print("=" * 60 + "\n")
    
    test_classes = ['rendang', 'gado-gado', 'soto', 'nasi-uduk', 'pempek']
    
    for class_name in test_classes:
        print(f"\n📍 {class_name.upper()}:")
        print("-" * 40)
        print(matcher.format_nutrition_text(class_name))
