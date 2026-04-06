import os
import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Config
pasta_origem_dir = os.path.join(os.path.expanduser("~"), "Downloads")  #caminho do seu arquivo 
pasta_saida = os.path.join(os.path.expanduser("~"), "Downloads")
arquivo_piloto = "df_evento_disc_ccom_jan_2026.parquet"

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


def run_pipeline():
    path = os.path.join(pasta_origem_dir, arquivo_piloto)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_parquet(path)
    texto_col = "comentario"
    nota_col = "avaliacao"

    if texto_col not in df.columns:
        raise ValueError(f"Coluna não encontrada: {texto_col}")

    df["topico_regra"] = df[texto_col].apply(classificar_topico)
    df["sentimento_regra"] = df.apply(lambda r: classificar_sentimento_hibrido(r[texto_col], r.get(nota_col)), axis=1)

    # Se não houver uma coluna manual de verdade (true_sentiment), atribuir via regra para treino inicial
    if "true_sentiment" not in df.columns:
        df["true_sentiment"] = df["sentimento_regra"]

    # ML: treinar modelo de sentimento com base anotada (true_sentiment)
    X = df[texto_col].fillna("")
    y = df["true_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    model = make_pipeline(TfidfVectorizer(max_features=15000, ngram_range=(1,2)), LogisticRegression(max_iter=1500, class_weight="balanced"))

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    print("\n=== Relatorio ML sentiment (test) ===")
    print(classification_report(y_test, pred_test, digits=4))

    df["sentimento_ml"] = model.predict(X)
    df["sentimento_combo"] = df.apply(lambda r: r["sentimento_regra"] if r["sentimento_regra"] != "Neutro" else r["sentimento_ml"], axis=1)

    # ML: treinar modelo de topico com base em topico_regra, para evitar N / E indevido
    X_topico = df[texto_col].fillna("")
    y_topico = df["topico_regra"]
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_topico, y_topico, stratify=y_topico, random_state=42, test_size=0.2)
    topico_model = make_pipeline(TfidfVectorizer(max_features=15000, ngram_range=(1,2)), LogisticRegression(max_iter=1500, class_weight="balanced"))

    topico_model.fit(X_train_t, y_train_t)
    topico_pred_test = topico_model.predict(X_test_t)
    print("\n=== Relatorio ML topico (test) ===")
    print(classification_report(y_test_t, topico_pred_test, digits=4))

    df["topico_ml"] = topico_model.predict(X_topico)
    df["topico_combo"] = df.apply(lambda r: r["topico_regra"] if r["topico_regra"] != "N / E" else r["topico_ml"], axis=1)

    # Export Excel com primeira amostra de 5000 linhas (sem true_sentiment, inclui novos campos)
    amostra = df.head(5000)[[texto_col, nota_col, "topico_regra", "topico_ml", "topico_combo", "sentimento_regra", "sentimento_ml", "sentimento_combo"]].copy()
    for col_dt in amostra.select_dtypes(include=['datetimetz']).columns:
        amostra[col_dt] = amostra[col_dt].dt.tz_convert(None)
    amostra.to_excel(os.path.join(pasta_saida, "validacao_piloto_sentimento.xlsx"), index=False)

    output_base = os.path.join(pasta_saida, "tabela_sentimento_topico")

    df_out2 = df[[texto_col, nota_col, "topico_regra", "topico_ml", "topico_combo", "sentimento_regra", "sentimento_ml", "sentimento_combo", "true_sentiment"]].copy()
    for col_dt in df_out2.select_dtypes(include=['datetimetz']).columns:
        df_out2[col_dt] = df_out2[col_dt].dt.tz_convert(None)
    df_out2.to_excel(f"{output_base}_ml.xlsx", index=False)

    print(f"Dados gravados:")
    print(f" - {output_base}_ml.xlsx")


if __name__ == "__main__":
    run_pipeline()