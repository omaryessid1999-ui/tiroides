import pandas as pd

def create_sample_csv():
    file_path = 'thyroid+disease/ann-test.data'
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    # Tomaremos las primeras 100 líneas como muestra
    for line in lines[:100]:
        parts = line.strip().split()
        if len(parts) >= 22:
            row = [float(x) for x in parts[:22]]
            data.append(row)
            
    df = pd.DataFrame(data)
    # Guardar como CSV sin cabeceras ni índices
    output_file = 'muestra_prueba_lotes.csv'
    df.to_csv(output_file, index=False, header=False)
    print(f"Archivo creado exitosamente: {output_file} con {len(df)} registros.")

if __name__ == '__main__':
    create_sample_csv()
