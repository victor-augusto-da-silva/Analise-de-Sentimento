import os
import re
import shutil
import unicodedata
import pandas as pd
import numpy as np
import glob
import gc
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Ativa o progresso do pandas
tqdm.pandas()

# IMPORTANTE: Biblioteca externa de temas
try:
    from temas import temas as TEMAS_DICT
except ImportError:
    print("❌ Erro: Arquivo 'temas.py' não encontrado.")
    TEMAS_DICT = {}

# ----------------- Configurações -----------------
pasta_origem_dir = r"C:"

NEGATIVOS = ["pessimo", "horrivel", "ruim", "odioso", "nao recomendo", "decepcionante", "finjo", "jamais", "detesto", "pior"]
POSITIVOS = ["excelente", "otimo", "fantastico", "maravilhoso", "adorei", "muito bom", "super recomendo", "parabens", "recomendo"]

# ----------------- Funções Otimizadas -----------------

def limpar_texto(txt):
    # 1. Checa se é nulo primeiro (sem usar 'if not')
    if pd.isna(txt):
        return ""
    
    # 2. Converte para string e remove espaços
    txt_str = str(txt).strip()
    
    # 3. Se a string estiver vazia, retorna vazio
    if txt_str == "":
        return ""
    
    # Agora segue o baile com a limpeza
    txt_lower = txt_str.lower()
    
    # Normalização e remoção de acentos
    nfkd = unicodedata.normalize("NFKD", txt_lower)
    txt_sem_acento = "".join([c for c in nfkd if not unicodedata.combining(c)])
    
    # Limpeza de caracteres especiais
    cleaned = re.sub(r'[^\w\s]', ' ', txt_sem_acento).strip()
    return cleaned

def regra_sentimento(txt_limpo):
    """Recebe o texto já limpo para economizar processamento"""
    if not txt_limpo: return "Neutro"
    if any(neg in txt_limpo for neg in NEGATIVOS): return "Detrator"
    if any(pos in txt_limpo for pos in POSITIVOS): return "Promotor"
    return "Neutro"

def regra_topico(txt_limpo):
    """Recebe o texto já limpo para economizar processamento"""
    if not txt_limpo: return "N / E"
    for tema, keywords in TEMAS_DICT.items():
        if any(re.search(rf"\b{re.escape(kw)}\b", txt_limpo) for kw in keywords):
            return tema
    return "N / E"

def processar_tudo_em_um(texto):
    limpo = limpar_texto(texto)
    sentimento = regra_sentimento(limpo)
    topico = regra_topico(limpo)
    return pd.Series([limpo, sentimento, topico])

def classificar_sentimento_nota(nota):
    try:
        if pd.isna(nota): return "Neutro"
        valor = float(nota)
        if valor <= 2: return "Detrator"
        if valor >= 4: return "Promotor"
        return "Neutro"
    except:
        return "Neutro"

# ----------------- Core de Treino e Processamento -----------------

def treinar_modelos(df_treino):
    print("\n" + "="*60)
    print("TREINANDO MODELOS (LOGISTIC REGRESSION)")
    print("="*60)
    
    X = df_treino["texto_limpo"].fillna("")
    
    # Modelo Sentimento - Adicionado n_jobs=-1 para usar todos os núcleos do PC
    y_s = df_treino["sentimento_regra"]
    model_s = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=10000), 
                             LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
    model_s.fit(X, y_s)
    
    # Modelo Tópico
    df_t = df_treino[df_treino["topico_regra"] != "N / E"]
    if not df_t.empty:
        model_t = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), 
                                 LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
        model_t.fit(df_t["texto_limpo"], df_t["topico_regra"])
    else:
        model_t = None
    
    return model_s, model_t

def processar_arquivo(caminho_arquivo, model_s, model_t):
    df = pd.read_parquet(caminho_arquivo)
    
    # Aplicando a lógica unificada (Substitui os 3 apply individuais)
    # Isso percorre a coluna 'comentario' apenas UMA vez
    df[["texto_limpo", "sentimento_regra", "topico_regra"]] = df["comentario"].progress_apply(processar_tudo_em_um)
    
    df["sentimento_nota_pura"] = df["avaliacao"].apply(classificar_sentimento_nota)
    
    # Predição ML
    mask_util = df["texto_limpo"].apply(lambda x: len(str(x).split()) >= 1)
    df["sentimento_ml"] = "Neutro"
    df["topico_ml"] = "N / E"
    
    if any(mask_util):
        df.loc[mask_util, "sentimento_ml"] = model_s.predict(df.loc[mask_util, "texto_limpo"])
        if model_t:
            df.loc[mask_util, "topico_ml"] = model_t.predict(df.loc[mask_util, "texto_limpo"])
    
    # Hierarquia Combo
    def escolher_sentimento_final(row):
        if not mask_util[row.name]: return row["sentimento_nota_pura"]
        if row["sentimento_regra"] != "Neutro": return row["sentimento_regra"]
        if row["sentimento_ml"] != "Neutro": return row["sentimento_ml"]
        return row["sentimento_nota_pura"]

    df["sentimento_combo"] = df.apply(escolher_sentimento_final, axis=1)
    df["topico_combo"] = df.apply(lambda r: r["topico_regra"] if r["topico_regra"] != "N / E" else r["topico_ml"], axis=1)
    
    # Limpeza para salvar
    cols_drop = ["texto_limpo", "sentimento_nota_pura", "sentimento_ml", "topico_ml"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    
    return df

# ----------------- Pipeline de Execução -----------------

def run_graduacao_integrado():
    pattern = os.path.join(pasta_origem_dir, "df_evento_disc_ccom*.parquet")
    arquivos = sorted(glob.glob(pattern))
    
    if not arquivos:
        print("❌ Nenhum arquivo Parquet encontrado.")
        return

    print(f"📂 Encontrados {len(arquivos)} arquivos. Iniciando leitura e amostragem...")

    try:
        lista_dfs = []
        for a in tqdm(arquivos, desc="Carregando arquivos para treino"):
            # Lemos apenas as colunas necessárias para o treino para economizar RAM
            temp_df = pd.read_parquet(a, columns=["comentario"])
            lista_dfs.append(temp_df)
        
        df_full = pd.concat(lista_dfs, ignore_index=True)
        del lista_dfs # Libera memória imediatamente
        gc.collect()

        # ESTRATÉGIA DE AMOSTRAGEM: 55M é impossível em RAM comum.
        # 300k a 500k é o "sweet spot" para treino de texto. 
        if len(df_full) > 500000:
            print(f"⚖️ Aplicando amostragem: Reduzindo de {len(df_full)} para 500.000 linhas.")
            df_full = df_full.sample(n=500000, random_state=42)

        print("\n⚙️ Processando regras de treino (Unificado)...")
        # Aplica limpeza e regras em um único passo
        df_full[["texto_limpo", "sentimento_regra", "topico_regra"]] = df_full["comentario"].progress_apply(processar_tudo_em_um)
        
        model_s, model_t = treinar_modelos(df_full)
        
        # Limpa o df de treino para dar espaço ao processamento real
        del df_full
        gc.collect()

    except Exception as e:
        print(f"❌ Falha na fase de treino: {e}")
        return

    # Etapa 2: Processamento Individual
    backup_map = {}
    try:
        for caminho in arquivos:
            print(f"\n🚀 Processando: {os.path.basename(caminho)}")
            
            backup_path = caminho + ".bak"
            shutil.copy2(caminho, backup_path)
            backup_map[caminho] = backup_path
            
            df_processado = processar_arquivo(caminho, model_s, model_t)
            df_processado.to_parquet(caminho, index=False)
            
            # Limpa o DF atual da memória antes do próximo arquivo
            del df_processado
            gc.collect()

        # Remove backups se tudo ok
        for b in backup_map.values():
            if os.path.exists(b): os.remove(b)
        
        print("\n" + "="*60)
        print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*60)

    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")

if __name__ == "__main__":
    run_graduacao_integrado()