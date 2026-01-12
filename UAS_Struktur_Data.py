# -*- coding: utf-8 -*-
"""
Program Diagnosis COVID-19 dengan Decision Tree
Compatible dengan Google Colab
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class COVID19DiagnosisSystem:
    def __init__(self):
        """Inisialisasi sistem diagnosis COVID-19"""
        self.model = None
        self.features = [
            'Demam', 'Batuk', 'Sesak_Napas', 'Kelelahan',
            'Sakit_Tenggorokan', 'Sakit_Kepala',
            'Usia_Diatas_60', 'Penyakit_Bawaan'
        ]
        self._create_dataset()
        self._train_model()
    
    def _create_dataset(self):
        """Membuat dataset gejala COVID-19"""
        # Dataset yang lebih komprehensif
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'Demam': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Batuk': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'Sesak_Napas': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'Kelelahan': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'Sakit_Tenggorokan': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Sakit_Kepala': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'Usia_Diatas_60': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'Penyakit_Bawaan': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
        
        # Menentukan target (COVID-19) berdasarkan aturan tertentu
        df = pd.DataFrame(data)
        
        # Aturan sederhana untuk menentukan COVID-19
        conditions = (
            (df['Demam'] == 1) & 
            (df['Batuk'] == 1) & 
            ((df['Sesak_Napas'] == 1) | (df['Kelelahan'] == 1))
        ) | (
            (df['Demam'] == 1) & 
            (df['Sesak_Napas'] == 1) & 
            (df['Usia_Diatas_60'] == 1)
        )
        
        df['COVID-19'] = np.where(conditions, 1, 0)
        
        # Menambahkan beberapa noise untuk realisme
        noise_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
        df.loc[noise_indices, 'COVID-19'] = 1 - df.loc[noise_indices, 'COVID-19']
        
        self.dataset = df
        print(f"Dataset dibuat dengan {len(df)} sampel")
        print(f"Kasus positif: {df['COVID-19'].sum()} ({df['COVID-19'].mean():.1%})")
    
    def _train_model(self):
        """Melatih model Decision Tree"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        
        X = self.dataset[self.features]
        y = self.dataset['COVID-19']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluasi
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Decision Tree telah dilatih")
        print(f"Akurasi pada data test: {accuracy:.2%}")
        
        # Feature importance
        print("\nPentingnya setiap fitur:")
        for feature, importance in zip(self.features, self.model.feature_importances_):
            print(f"  {feature}: {importance:.3f}")
    
    def diagnose(self, symptoms):
        """
        Mendiagnosis berdasarkan gejala
        
        Args:
            symptoms (dict): Dictionary gejala pasien
        
        Returns:
            dict: Hasil diagnosis dan rekomendasi
        """
        # Konversi gejala ke format yang sesuai
        input_data = []
        for feature in self.features:
            input_data.append(symptoms.get(feature, 0))
        
        input_array = np.array(input_data).reshape(1, -1)
        
        # Prediksi
        prediction = self.model.predict(input_array)[0]
        probability = self.model.predict_proba(input_array)[0]
        
        # Interpretasi hasil
        result = {
            'diagnosis': 'POSITIF COVID-19' if prediction == 1 else 'NEGATIF COVID-19',
            'confidence': float(probability[prediction]),
            'probability_positive': float(probability[1]),
            'probability_negative': float(probability[0]),
            'recommendation': ''
        }
        
        # Rekomendasi berdasarkan hasil
        if prediction == 1:
            result['recommendation'] = (
                "Segera lakukan tes PCR/antigen dan isolasi mandiri. "
                "Hubungi fasilitas kesehatan terdekat. "
                "Gunakan masker dan hindari kontak dengan orang lain."
            )
        else:
            result['recommendation'] = (
                "Tetap jaga kesehatan dan patuhi protokol kesehatan. "
                "Lakukan tes jika gejala memberat atau kontak dengan positif COVID-19."
            )
        
        return result
    
    def show_tree_diagram(self):
        """Menampilkan diagram decision tree"""
        try:
            from sklearn.tree import plot_tree
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(20, 12))
            plot_tree(self.model, 
                     feature_names=self.features,
                     class_names=['Negatif', 'Positif'],
                     filled=True, 
                     rounded=True,
                     fontsize=10,
                     max_depth=3)  # Batasi depth untuk kejelasan
            
            plt.title("Decision Tree untuk Diagnosis COVID-19", fontsize=16)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Tidak dapat menampilkan diagram: {e}")
    
    def interactive_diagnosis(self):
        """Diagnosis interaktif dari input pengguna"""
        print("\n" + "="*50)
        print("SISTEM DIAGNOSIS COVID-19")
        print("="*50)
        print("\nMasukkan gejala pasien (1 = Ya, 0 = Tidak):\n")
        
        symptoms = {}
        for feature in self.features:
            while True:
                try:
                    value = int(input(f"Apakah mengalami {feature.replace('_', ' ')}? (1/0): "))
                    if value in [0, 1]:
                        symptoms[feature] = value
                        break
                    else:
                        print("Masukkan 1 untuk Ya atau 0 untuk Tidak")
                except:
                    print("Input tidak valid. Masukkan 1 atau 0")
        
        print("\n" + "-"*50)
        result = self.diagnose(symptoms)
        
        print(f"\nHASIL DIAGNOSIS:")
        print(f"Status: {result['diagnosis']}")
        print(f"Tingkat Kepercayaan: {result['confidence']:.1%}")
        print(f"Probabilitas Positif: {result['probability_positive']:.1%}")
        print(f"Probabilitas Negatif: {result['probability_negative']:.1%}")
        print(f"\nREKOMENDASI:")
        print(result['recommendation'])
        
        return result

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi sistem
    print("Menginisialisasi Sistem Diagnosis COVID-19...")
    system = COVID19DiagnosisSystem()
    
    # Tampilkan dataset
    print("\nContoh Data Training:")
    print(system.dataset.head())
    
    # Tampilkan diagram tree (opsional)
    show_diagram = input("\nTampilkan diagram decision tree? (y/n): ").lower()
    if show_diagram == 'y':
        system.show_tree_diagram()
    
    # Diagnosis interaktif
    while True:
        result = system.interactive_diagnosis()
        
        continue_diagnosis = input("\nLakukan diagnosis lagi? (y/n): ").lower()
        if continue_diagnosis != 'y':
            print("\nTerima kasih telah menggunakan sistem diagnosis COVID-19!")
            print("Tetap patuhi protokol kesehatan dan jaga kesehatan!")
            break
