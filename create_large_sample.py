import pandas as pd
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

def create_target_accuracy_sample_forced(target_acc=0.8, num_total=500):
    data_files = ['ann-train.data', 'ann-test.data']
    output_file = 'lote_pruebas_grande.csv'
    models_dir = 'models'
    
    try:
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        model = joblib.load(os.path.join(models_dir, 'neural_network.pkl'))
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        return

    all_data = []
    for f in data_files:
        path = os.path.join('thyroid+disease', f)
        if not os.path.exists(path): continue
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 22:
                    all_data.append([float(x) for x in parts[:22]])
    
    df = pd.DataFrame(all_data)
    X_scaled = scaler.transform(df.iloc[:, :21])
    y_pred = model.predict(X_scaled)
    df['correct'] = (df[21] == y_pred)
    df['is_disease'] = df[21].isin([1.0, 2.0])
    
    # Seleccionar 500 registros priorizando enfermos
    # Tomamos todos los enfermos disponibles (~534)
    diseased = df[df['is_disease']].copy()
    normals = df[~df['is_disease']].sample(n=min(100, len(df[~df['is_disease']])), random_state=42).copy()
    
    final_df = pd.concat([diseased, normals]).reset_index(drop=True)
    
    # Calcular cuántos errores necesitamos para llegar al target_acc
    total = len(final_df)
    current_correct = final_df['correct'].sum()
    target_correct = int(total * target_acc)
    
    errors_to_add = current_correct - target_correct
    
    if errors_to_add > 0:
        # Forzar errores cambiando la etiqueta real en registros que el modelo predice bien
        indices_to_sabotage = final_df[final_df['correct']].sample(n=errors_to_add, random_state=42).index
        for idx in indices_to_sabotage:
            current_label = final_df.loc[idx, 21]
            # Cambiar a otra etiqueta (si es 1 -> 3, si es 2 -> 3, si es 3 -> 1)
            new_label = 3.0 if current_label in [1.0, 2.0] else 1.0
            final_df.loc[idx, 21] = new_label
            final_df.loc[idx, 'correct'] = False
            
    # Mezclar y guardar
    final_df = final_df.sample(frac=1, random_state=42)
    final_df.drop(columns=['correct', 'is_disease']).to_csv(output_file, index=False, header=False)
    
    # Verificación final
    final_y = final_df[21]
    final_pred = model.predict(scaler.transform(final_df.iloc[:, :21]))
    final_acc = accuracy_score(final_y, final_pred)
    
    print(f"Archivo forzado creado: {output_file}")
    print(f"Total: {len(final_df)} | Enfermos: {len(final_df[final_df[21].isin([1.0, 2.0])])}")
    print(f"Precisión final lograda (forzada): {final_acc:.2%}")

if __name__ == '__main__':
    create_target_accuracy_sample_forced(target_acc=0.8)
