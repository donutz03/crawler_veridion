import pandas as pd
#e nevoie de pyarrow : pip install pyarrow

def citeste_parquet(fisier_parquet):
    try:
        df = pd.read_parquet(fisier_parquet)

        print(f"Fișierul conține {len(df)} înregistrări și {len(df.columns)} coloane.")
        print(f"Coloanele din fișier: {df.columns.tolist()}")

        print("\nConținutul fișierului:")
        print(df)

        return df
    except Exception as e:
        print(f"Eroare la citirea fișierului: {e}")
        return None

#idei: imaginile nu sunt incarcate doar cu tagul img
# pot fi si cu tag a poate
# sau pot fi setate din JS/CSS ca background-image (il gasesc la computed styles) si atasate
# unui alt element, cum ar fi un div, span, etc
# atunci eu cat dupa url(link imagine) in CSS -> computed. Si il iau de acolo
# intrebarea e cum gasesc fisierul CSS bun
# Mai mult, poate sunt mai multe img, cum ma asigur ca ceea ce descarc e un logo si nu altceva

if __name__ == "__main__":
    fisier_parquet = "logos.snappy.parquet"
    citeste_parquet(fisier_parquet)

#python logo_clustering.py --input-dir extracted_logos --output-dir cluster_results