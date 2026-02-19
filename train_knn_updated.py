"""
KNN Model Training for Gesture Recognition - UPDATED
New gesture set: no drag/fist, new pinky gestures
Trains on collected data with 80/20 split
"""

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Configuration
DATA_DIR = 'training_data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# NEW GESTURE DEFINITIONS
GESTURES = {
    0: "PALM",
    1: "V_SIGN",
    2: "MIDDLE_ONLY",
    3: "INDEX_ONLY",
    4: "JOINED_FINGERS",
    5: "PINKY_ONLY",      # NEW: Scroll down / Volume down
    6: "PINKY_THUMB",     # NEW: Scroll up / Volume up
    7: "THUMBS_UP",
    8: "THUMBS_DOWN",
    9: "ROCK_SIGN"
}

class GestureModelTrainer:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        """Load all CSV files from training data directory"""
        print("\n" + "=" * 60)
        print("LOADING TRAINING DATA")
        print("=" * 60)
        
        csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {DATA_DIR}/")
            return False
        
        print(f"\nFound {len(csv_files)} data file(s):")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
        
        # Load and combine all data
        all_data = []
        for csv_file in csv_files:
            data = np.loadtxt(csv_file, delimiter=',')
            all_data.append(data)
            print(f"  ✓ Loaded {len(data)} samples from {os.path.basename(csv_file)}")
        
        combined_data = np.vstack(all_data)
        
        X = combined_data[:, :-1]
        y = combined_data[:, -1]
        
        print(f"\n📊 Total samples loaded: {len(X)}")
        print(f"   Features per sample: {X.shape[1]}")
        
        print("\n📈 Samples per gesture:")
        for label in range(10):
            count = np.sum(y == label)
            if count > 0:
                print(f"   {GESTURES[label]:15s}: {count:3d} samples")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets (80/20)"""
        print("\n" + "=" * 60)
        print("SPLITTING DATA (80% Train / 20% Test)")
        print("=" * 60)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Training set: {len(self.X_train)} samples")
        print(f"✓ Test set:     {len(self.X_test)} samples")
        
        return True
    
    def train(self):
        """Train KNN model"""
        print("\n" + "=" * 60)
        print(f"TRAINING KNN MODEL (k={self.n_neighbors})")
        print("=" * 60)
        
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(self.X_train, self.y_train)
        
        print("\n✓ Model trained successfully!")
        
        return True
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")
        
        print("\n📊 Per-Gesture Accuracy:")
        for label in range(10):
            mask = self.y_test == label
            if np.sum(mask) > 0:
                class_accuracy = accuracy_score(
                    self.y_test[mask],
                    y_pred[mask]
                )
                bar = "█" * int(class_accuracy * 20) + "░" * (20 - int(class_accuracy * 20))
                print(f"   {GESTURES[label]:15s}: {class_accuracy * 100:.1f}%  {bar}")
        
        print("\n🔢 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        
        print("\n   ", end="")
        for i in range(10):
            if np.sum(self.y_test == i) > 0:
                print(f"{i:3d}", end=" ")
        print()
        
        for i in range(10):
            if np.sum(self.y_test == i) > 0:
                print(f"{i:2d} ", end="")
                for j in range(10):
                    if np.sum(self.y_test == j) > 0:
                        print(f"{cm[i][j]:3d}", end=" ")
                print(f"  ({GESTURES[i]})")
        
        print("\n📋 Detailed Classification Report:")
        print(classification_report(
            self.y_test,
            y_pred,
            target_names=[GESTURES[i] for i in range(10)],
            zero_division=0
        ))
        
        return accuracy
    
    def save_model(self, filename='gesture_knn_model_updated.pkl'):
        """Save trained model to file"""
        filepath = os.path.join(MODEL_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n💾 Model saved: {filepath}")
        return filepath
    
    def run_training_pipeline(self):
        """Complete training pipeline"""
        print("\n" + "=" * 60)
        print("KNN GESTURE RECOGNITION TRAINING")
        print("NEW GESTURE SET - NO DRAG/FIST")
        print("=" * 60)
        
        result = self.load_data()
        if not result:
            return False
        
        X, y = result
        
        self.split_data(X, y)
        self.train()
        accuracy = self.evaluate()
        self.save_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\n✓ Final Accuracy: {accuracy * 100:.2f}%")
        print(f"✓ Model saved in: {MODEL_DIR}/")
        print("\n🆕 NEW GESTURES:")
        print("   5: PINKY_ONLY   - Scroll/Volume down")
        print("   6: PINKY_THUMB  - Scroll/Volume up")
        print("\n❌ REMOVED:")
        print("   - FIST (drag)")
        print("   - PINCH (scroll)")
        print("\n📝 Next: Run 'python main_knn_updated.py'")
        print("=" * 60 + "\n")
        
        return True

def main():
    trainer = GestureModelTrainer(n_neighbors=5)
    
    try:
        trainer.run_training_pipeline()
    except FileNotFoundError:
        print("\n❌ Error: No training data found!")
        print("Please run 'python collect_data_updated.py' first to collect data.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
