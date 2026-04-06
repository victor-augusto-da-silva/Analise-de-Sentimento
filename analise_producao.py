import os
import re
import shutil
import unicodedata
import pandas as pd
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Config
pasta_origem_dir = os.path.join(os.path.expanduser("~"), "Downloads") #caminho do seu arquivo

NEGATIVOS = ["péssimo", "horrível", "ruim", "odioso", "não recomendo", "decepcionante", "finjo", "jamais", "detesto", "pior", "deus me defenda"]
POSITIVOS = ["excelente", "ótimo", "fantástico", "maravilhoso", "adorei", "muito bom", "super recomendo", "parabéns"]


def safe_text(txt):
    if txt is None or pd.isna(txt):
        return ""
    return str(txt).strip()


def remove_acento(txt):
    txt = safe_text(txt)
    nfkd = unicodedata.normalize("NFKD", txt)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower()


def classificar_topico(texto):
    txt = remove_acento(texto)
    if not txt:
        return "N / E"

    ruido = ["zzz", "xxx", "..."]
    if any(r in txt for r in ruido):
        return "N / E"
    if txt == "nao":
        return "N / E"

    temas = {
        "Professor / Conteúdo": [
            "professor", "coordenador", "explica", "didatica", "postura", "aula", "aulas", 
            "ensino", "paciencia", "atencioso", "materia", "material", "grade", "aprendizado", 
            "ler", "apostila", "slides", "pdf", "leitura", "livro", "prova", "exercicio","conteudo", "assunto", "tema", "topico", "assunto", "dificil de entender", "dificil de aprender", "dificil de acompanhar", "dificil de seguir", "dificil de compreender", "dificil de assimilar", "dificil de captar", "dificil de absorver", "dificil de pegar", "dificil de entender", "dificil de aprender", "dificil de acompanhar", "dificil de seguir", "dificil de compreender", "dificil de assimilar", "dificil de captar", "dificil de absorver", "dificil de pegar","dificil de entender", "dificil de aprender", "dificil de acompanhar", "dificil de seguir", "dificil de compreender", "dificil de assimilar", "dificil de captar", "dificil de absorver", "dificil de pegar", "facil de entender", "facil de aprender", "facil de acompanhar", "facil de seguir", "facil de compreender", "facil de assimilar", "facil de captar", "facil de absorver", "facil de pegar"],
        
        "Elogio": [
            "gostei", "amei", "excelente", "otimo", "perfeito", "parabens", "interessante", 
            "motivador", "divertido", "top", "bom", "show","maravilhoso", "incrivel", "fantastico", "adorei", "muito bom", "super recomendo","parabéns","recomendo", "satisfeito", "satisfacao", "satisfaz", "satisfazendo", "satisfez", "satisfeito", "satisfeitos", "satisfeita", "satisfeitas"],
        
        "Erro - BUG": [
            "travou", "acesso", "bug", "estabilidade", "app", "aplicativo", "link", 
            "moodle", "blackboard", "lag", "conexao", "lentidao", "carregamento", "erro","falha", "instabilidade", "problema tecnico", "problema de acesso", "problema de conexao", "problema de carregamento", "problema de link", "problema de app", "problema de aplicativo", "travamento", "travou", "lag", "lentidao","problema tecnico", "problema de acesso", "problema de conexao", "problema de carregamento", "problema de link", "problema de app", "problema de aplicativo"],
        
        "Audio": [
            "audio", "som", "baixo", "ruido", "chiado", "mudo", "microfone", "escutar", "ouvir","volume","alto", "claridade", "distancia", "eco", "interferencia", "falar", "voz", "fala", "dificil de ouvir", "dificil de escutar", "dificil de entender"],
        
        "Conteúdo diferente da Aula": [
            "diferente", "divergente", "nao condiz", "outro assunto", "errado", "trocado","conteudo diferente", "conteudo divergente", "conteudo nao condiz", "conteudo outro assunto", "conteudo errado", "conteudo trocado","assunto diferente", "assunto divergente", "assunto nao condiz", "assunto outro assunto", "assunto errado", "assunto trocado","topico diferente", "topico divergente", "topico nao condiz", "topico outro assunto", "topico errado", "topico trocado"],
        
        "Qualidade do Vídeo": [
            "video", "imagem", "resolucao", "baixa qualidade", "pixelado", "borrado", "escuro","qualidade do video", "qualidade da imagem", "resolucao baixa", "video pixelado", "video borrado", "video escuro", "imagem pixelada", "imagem borrada", "imagem escura", "resolucao baixa", "qualidade ruim", "qualidade pessima","qualidade do video", "qualidade da imagem", "resolucao baixa", "video pixelado", "video borrado", "video escuro", "imagem pixelada", "imagem borrada", "imagem escura", "resolucao baixa", "qualidade ruim", "qualidade pessima"],
        
        "Edição": [
            "edicao", "corte", "montagem", "transicao", "legenda", "erro de edicao","edicao ruim", "edicao pessima", "corte ruim", "corte pessimo", "montagem ruim", "montagem pessima", "transicao ruim", "transicao pessima", "legenda ruim", "legenda pessima", "erro de edicao", "erro de edição","erro de edicao", "erro de edição"],
        
        "Musica: Abertura / Fechamento": [
            "musica", "trilha", "abertura", "fechamento", "vinheta", "intro", "volume da musica","musica ruim", "musica pessima", "trilha ruim", "trilha pessima", "abertura ruim", "abertura pessima", "fechamento ruim", "fechamento pessimo", "vinheta ruim", "vinheta pessima", "intro ruim", "intro pessima", "volume da musica ruim", "volume da musica alto", "volume da musica baixo","musica ruim", "musica pessima", "trilha ruim", "trilha pessima", "abertura ruim", "abertura pessima", "fechamento ruim", "fechamento pessimo", "vinheta ruim", "vinheta pessima", "intro ruim", "intro pessima", "volume da musica ruim", "volume da musica alto", "volume da musica baixo"],
        
        "Tutor": [
            "tutor", "tutoria", "monitor", "ajuda", "suporte acadêmico", "duvida", "atendimento", "resposta", "rapidez", "eficaz","tutor ruim", "tutor pessimo", "tutoria ruim", "tutoria pessima", "monitor ruim", "monitor pessimo", "ajuda ruim", "ajuda pessima", "suporte academico ruim", "suporte academico pessimo", "duvida sem resposta", "duvida nao respondida", "atendimento ruim", "atendimento pessimo", "resposta ruim", "resposta pessima", "rapidez ruim", "rapidez pessima", "eficaz ruim", "eficaz pessimo","tutor ruim", "tutor pessimo", "tutoria ruim", "tutoria pessima", "monitor ruim", "monitor pessimo", "ajuda ruim", "ajuda pessima", "suporte academico ruim", "suporte academico pessimo", "duvida sem resposta", "duvida nao respondida", "atendimento ruim", "atendimento pessimo", "resposta ruim", "resposta pessima", "rapidez ruim", "rapidez pessima", "eficaz ruim", "eficaz pessimo"],
        }

    for tema, palavras in temas.items():
        for p in palavras:
            if re.search(rf"\b{re.escape(p)}\b", txt):
                return tema

    if any(word in txt for word in ["gostei", "amei", "excelente", "otimo", "otima", "muito bom", "maravilhoso"]):
        return "Engajamento"
    if any(word in txt for word in ["ruim", "péssimo", "horrivel", "chato", "pior"]):
        return "Engajamento"

    return "N / E"


def classificar_sentimento_hibrido(texto, nota=None):
    txt = safe_text(texto).lower()

    if any(neg in txt for neg in NEGATIVOS):
        return "Detrator"
    if any(pos in txt for pos in POSITIVOS):
        return "Promotor"

    if nota is not None and pd.notna(nota):
        try:
            valor = float(nota)
            if valor <= 2:
                return "Detrator"
            if valor >= 4:
                return "Promotor"
            if valor == 3:
                return "Neutro"
        except Exception:
            pass

    if not txt.strip():
        return "Neutro"

    return "Neutro"


def treinar_modelos(df_treino, texto_col="comentario", nota_col="avaliacao"):
    """Treina os modelos de Sentimento e Topico com dados agregados"""
    
    print("\n" + "="*60)
    print("TREINO DOS MODELOS")
    print("="*60)
    
    # Preparar dados de treino
    X = df_treino[texto_col].fillna("")
    
    # Treinar modelo de Sentimento
    print("\n>>> Treinando modelo de Sentimento...")
    y_sentimento = df_treino["true_sentiment"] if "true_sentiment" in df_treino.columns else df_treino["sentimento_regra"]
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_sentimento, stratify=y_sentimento, random_state=42, test_size=0.2)
    modelo_sentimento = make_pipeline(TfidfVectorizer(max_features=15000, ngram_range=(1,2)), LogisticRegression(max_iter=1500, class_weight="balanced"))
    modelo_sentimento.fit(X_train_s, y_train_s)
    
    pred_sentimento = modelo_sentimento.predict(X_test_s)
    print("\n=== Resultado: Modelo Sentimento (test) ===")
    print(classification_report(y_test_s, pred_sentimento, digits=4))
    
    # Treinar modelo de Topico
    print("\n>>> Treinando modelo de Topico...")
    y_topico = df_treino["topico_regra"]
    
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X, y_topico, stratify=y_topico, random_state=42, test_size=0.2)
    modelo_topico = make_pipeline(TfidfVectorizer(max_features=15000, ngram_range=(1,2)), LogisticRegression(max_iter=1500, class_weight="balanced"))
    modelo_topico.fit(X_train_t, y_train_t)
    
    pred_topico = modelo_topico.predict(X_test_t)
    print("\n=== Resultado: Modelo Topico (test) ===")
    print(classification_report(y_test_t, pred_topico, digits=4))
    
    return modelo_sentimento, modelo_topico


def processar_arquivo(caminho_arquivo, modelo_sentimento, modelo_topico, texto_col="comentario", nota_col="avaliacao"):
    """Aplica classificações a um arquivo e retorna dataframe atualizado"""
    
    print(f"\n>>> Processando: {os.path.basename(caminho_arquivo)}")
    
    # Carregar arquivo
    df = pd.read_parquet(caminho_arquivo)
    
    # Aplicar regras
    df["topico_regra"] = df[texto_col].apply(classificar_topico)
    df["sentimento_regra"] = df.apply(lambda r: classificar_sentimento_hibrido(r[texto_col], r.get(nota_col)), axis=1)
    
    # Aplicar modelos ML
    X = df[texto_col].fillna("")
    df["sentimento_ml"] = modelo_sentimento.predict(X)
    df["topico_ml"] = modelo_topico.predict(X)
    
    # Aplicar combos
    df["sentimento_combo"] = df.apply(lambda r: r["sentimento_regra"] if r["sentimento_regra"] != "Neutro" else r["sentimento_ml"], axis=1)
    df["topico_combo"] = df.apply(lambda r: r["topico_regra"] if r["topico_regra"] != "N / E" else r["topico_ml"], axis=1)
    
    print(f"   ✓ {len(df)} linhas processadas")
    
    return df


def rollback_arquivos(backup_map):
    """Restaura todos os arquivos a partir do backup"""
    print("\n--- Iniciando rollback dos arquivos modificados ---")
    for original, backup in backup_map.items():
        try:
            shutil.copy2(backup, original)
            print(f"   ✓ Restaurado: {os.path.basename(original)}")
        except Exception as e:
            print(f"   ❌ Falha ao restaurar {os.path.basename(original)}: {e}")
    print("--- Rollback concluído ---\n")


def run_producao():
    """Pipeline de produção: encontra e atualiza todos os arquivos df_evento_disc_ccom"""
    
    print("\n" + "="*60)
    print("PIPELINE PRODUÇÃO - CLASSIFICAÇÃO DE DISCIPLINAS")
    print("="*60)
    print(f"Pasta: {pasta_origem_dir}")
    
    # Encontrar arquivos
    pattern = os.path.join(pasta_origem_dir, "df_evento_disc_ccom*.parquet")
    arquivos = sorted(glob.glob(pattern))
    
    if not arquivos:
        print("❌ Nenhum arquivo df_evento_disc_ccom*.parquet encontrado!")
        return
    
    print(f"\n✓ Encontrados {len(arquivos)} arquivo(s):")
    for arq in arquivos:
        print(f"  - {os.path.basename(arq)}")
    
    # Etapa 1: Agregar dados para treino
    print("\n" + "="*60)
    print("ETAPA 1: AGREGAÇÃO PARA TREINO")
    print("="*60)
    
    dfs_lista = []
    for caminho in arquivos:
        try:
            df = pd.read_parquet(caminho)
            if "comentario" not in df.columns:
                print(f"   ⚠ Pulando {os.path.basename(caminho)}: coluna 'comentario' não encontrada")
                continue
            dfs_lista.append(df)
            print(f"  ✓ Carregado: {os.path.basename(caminho)} ({len(df)} linhas)")
        except Exception as e:
            print(f"  ❌ Erro ao carregar {os.path.basename(caminho)}: {e}")
    
    if not dfs_lista:
        print("❌ Nenhum arquivo válido para processar!")
        return
    
    df_completo = pd.concat(dfs_lista, ignore_index=True)
    print(f"\n✓ Dados agregados: {len(df_completo)} linhas totais")
    
    # Aplicar regras base para agregado
    df_completo["topico_regra"] = df_completo["comentario"].apply(classificar_topico)
    df_completo["sentimento_regra"] = df_completo.apply(lambda r: classificar_sentimento_hibrido(r["comentario"], r.get("avaliacao")), axis=1)
    df_completo["true_sentiment"] = df_completo["sentimento_regra"]
    
    # Etapa 2: Treinar modelos
    modelo_sentimento, modelo_topico = treinar_modelos(df_completo)
    
    # Etapa 3: Aplicar a todos os arquivos
    print("\n" + "="*60)
    print("ETAPA 3: APLICAÇÃO NOS ARQUIVOS")
    print("="*60)
    
    backup_map = {}
    try:
        for caminho in arquivos:
            backup_caminho = f"{caminho}.bak"
            shutil.copy2(caminho, backup_caminho)
            backup_map[caminho] = backup_caminho

            df_processado = processar_arquivo(caminho, modelo_sentimento, modelo_topico)

            # Sobrescrever arquivo original
            df_processado.to_parquet(caminho, index=False)
            print(f"   ✓ Arquivo salvo com sucesso")

    except Exception as e:
        print(f"   ❌ Erro crítico durante o processamento: {e}")
        rollback_arquivos(backup_map)
        print("❌ Processo abortado. Arquivos originais restaurados com backup.")
        return

    # Remover backups somente se tudo deu certo
    for backup in backup_map.values():
        try:
            os.remove(backup)
        except Exception:
            pass

    # Relatório final
    print("\n" + "="*60)
    print("✓ PROCESSO CONCLUÍDO COM SUCESSO")
    print("="*60)
    print("\nColunas adicionadas aos arquivos:")
    print("  - topico_regra, topico_ml, topico_combo")
    print("  - sentimento_regra, sentimento_ml, sentimento_combo")
    print("\nOs arquivos foram atualizados in-place (sem criar novos arquivos)")


if __name__ == "__main__":
    run_producao()
