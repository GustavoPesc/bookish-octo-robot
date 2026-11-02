"""Previsao experimental da faixa da proxima glicemia.

O modelo usa as cinco ultimas medicoes de glicose para prever a faixa da
seguinte. A divisao de treino e teste e feita por paciente para evitar que
janelas do mesmo paciente aparecam nos dois conjuntos.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
	classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PASTA_DADOS = Path(__file__).parent / "Diabetes-Data"
TAMANHO_JANELA = 5
FRACAO_TESTE = 0.2
SEMENTE_ALEATORIA = 42
CODIGOS_GLICEMIA = {48, 57, 58, 59, 60, 61, 62, 63, 64}
CLASSES_GLICEMIA = ["baixa", "alvo", "alta"]
CONFIGURACOES_MLP = [
	("Pequena", (16,)),
	("Base", (32, 16)),
	("Ampla", (64, 32)),
]

def carregar_sequencias() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	#Le arquivos e monta exemplos para treinar e avaliar o modelo.
	entradas: list[np.ndarray] = []
	faixas_alvo: list[str] = []
	ids_pacientes: list[str] = []

	# Cada arquivo data-NN representa um paciente. Ordenamos para que as janelas acompanhem a sequencia temporal daquele paciente.
	arquivos_pacientes = sorted(
		caminho
		for caminho in PASTA_DADOS.glob("data-*")
		if caminho.is_file() and caminho.name[5:].isdigit()
	)

	# Os arquivos tem quatro colunas tabuladas: data, hora, codigo do evento e valor.
	nomes_colunas = ["data", "hora", "codigo", "valor"]
	for arquivo_paciente in arquivos_pacientes:
		registros = pd.read_csv(
			arquivo_paciente,
			sep="\t",
			header=None,
			names=nomes_colunas,
			dtype="string",
			on_bad_lines="skip",
		)
		codigos = pd.to_numeric(registros["codigo"], errors="coerce")
		registros_glicemia = registros.loc[
			codigos.isin(CODIGOS_GLICEMIA)
		].copy()
		registros_glicemia["valor"] = pd.to_numeric(
			registros_glicemia["valor"], errors="coerce"
		)
		registros_glicemia["data_hora"] = pd.to_datetime(
			registros_glicemia["data"] + " " + registros_glicemia["hora"],
			format="%m-%d-%Y %H:%M",
			errors="coerce",
		)

		# Valores nao numericos, datas invalidas e leituras iguais a zero nao entram no modelo.
		linhas_validas = (
			registros_glicemia["valor"].notna()
			& registros_glicemia["data_hora"].notna()
			& (registros_glicemia["valor"] > 0)
		)
		registros_glicemia = registros_glicemia.loc[linhas_validas].sort_values(
			"data_hora"
		)
		valores_glicemia = registros_glicemia["valor"].to_numpy(dtype=float)
		if len(valores_glicemia) <= TAMANHO_JANELA:
			continue

		# Cada janela tem cinco entradas e a leitura seguinte como alvo. A operacao vetorizada e aplicada separadamente a cada paciente.
		janelas = np.lib.stride_tricks.sliding_window_view(
			valores_glicemia, TAMANHO_JANELA + 1
		)
		entradas.append(janelas[:, :-1])
		faixas_alvo.extend(
			np.select(
				[janelas[:, -1] < 70, janelas[:, -1] <= 180],
				["baixa", "alvo"],
				default="alta",
			)
		)
		ids_pacientes.extend([arquivo_paciente.name] * len(janelas))

	return (
		np.concatenate(entradas),
		np.asarray(faixas_alvo),
		np.asarray(ids_pacientes),
	)

def criar_modelo(arquitetura: tuple[int, ...]):
	"""Cria a MLP, padronizando os dados de treino dentro do pipeline."""
	return make_pipeline(
		StandardScaler(),
		MLPClassifier(
			hidden_layer_sizes=arquitetura,
			activation="relu",
			solver="adam",
			alpha=0.001,
			max_iter=500,
			early_stopping=False,
			random_state=SEMENTE_ALEATORIA,
		),
	)

def calcular_metricas(
	classes_reais: np.ndarray, previsoes: np.ndarray
) -> tuple[float, float, pd.DataFrame]:
	"""Retorna acuracia, F1 macro e tabela de metricas."""
	dados = classification_report(
		classes_reais,
		previsoes,
		labels=CLASSES_GLICEMIA,
		output_dict=True,
		zero_division=0,
	)
	acuracia = dados.pop("accuracy")
	f1_macro = dados["macro avg"]["f1-score"]
	dados.pop("macro avg")
	dados.pop("weighted avg")
	relatorio = pd.DataFrame.from_dict(dados, orient="index").rename(
		columns={
			"precision": "Precisao",
			"recall": "Revocacao",
			"f1-score": "Pontuacao F1",
			"support": "Quantidade de exemplos",
		},
	)
	relatorio["Quantidade de exemplos"] = relatorio[
		"Quantidade de exemplos"
	].astype(int)
	return acuracia, f1_macro, relatorio.round(3)


def executar_modelo() -> None:
	"""Compara tres MLPs e avalia a melhor arquitetura no teste reservado."""
	entradas, faixas_alvo, ids_pacientes = carregar_sequencias()
	indices_desenvolvimento, indices_teste = next(
		GroupShuffleSplit(
			n_splits=1, test_size=FRACAO_TESTE, random_state=SEMENTE_ALEATORIA
		).split(entradas, faixas_alvo, groups=ids_pacientes)
	)
	indices_treino_relativos, indices_validacao_relativos = next(
		GroupShuffleSplit(
			n_splits=1, test_size=0.25, random_state=SEMENTE_ALEATORIA + 1
		).split(
			entradas[indices_desenvolvimento],
			faixas_alvo[indices_desenvolvimento],
			groups=ids_pacientes[indices_desenvolvimento],
		)
	)
	indices_treino = indices_desenvolvimento[indices_treino_relativos]
	indices_validacao = indices_desenvolvimento[indices_validacao_relativos]
	entradas_treino = entradas[indices_treino]
	faixas_treino = faixas_alvo[indices_treino]
	entradas_validacao = entradas[indices_validacao]
	classes_validacao = faixas_alvo[indices_validacao]
	pesos_treino = compute_sample_weight(class_weight="balanced", y=faixas_treino)
	resultados = []

	for numero, (nome, arquitetura) in enumerate(CONFIGURACOES_MLP, start=1):
		modelo = criar_modelo(arquitetura)
		modelo.fit(
			entradas_treino,
			faixas_treino,
			mlpclassifier__sample_weight=pesos_treino,
		)
		previsoes = modelo.predict(entradas_validacao)
		acuracia, f1_macro, relatorio = calcular_metricas(
			classes_validacao, previsoes
		)
		resultados.append((f1_macro, nome, arquitetura, acuracia))

		# Cada janela reune a matriz, a curva de perda e as metricas do modelo.
		figura, eixos = plt.subplots(
			1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1.5]}
		)
		visualizacao = ConfusionMatrixDisplay.from_predictions(
			classes_validacao,
			previsoes,
			labels=CLASSES_GLICEMIA,
			values_format="d",
			cmap="Blues",
			ax=eixos[0],
		)
		visualizacao.im_.colorbar.set_label("Quantidade")
		eixos[0].set(
			title="Matriz de confusao",
			xlabel="Faixa prevista",
			ylabel="Faixa real",
		)
		classificador = modelo.named_steps["mlpclassifier"]
		eixos[1].plot(classificador.loss_curve_, color="darkorange")
		eixos[1].set(
			title="Perda no treinamento",
			xlabel="Iteracao",
			ylabel="Funcao de perda",
		)
		eixos[1].grid(alpha=0.25)
		eixos[2].axis("off")
		eixos[2].set_title(f"Metricas de validacao | Acuracia: {acuracia:.3f}")
		tabela = eixos[2].table(
			cellText=relatorio.values,
			rowLabels=relatorio.index,
			colLabels=relatorio.columns,
			cellLoc="center",
			loc="center",
		)
		tabela.auto_set_font_size(False)
		tabela.set_fontsize(8)
		tabela.scale(1, 1.5)
		figura.suptitle(f"Experimento {numero}: MLP {nome} {arquitetura}")
		figura.tight_layout()

	_, melhor_nome, melhor_arquitetura, _ = max(
		resultados, key=lambda resultado: resultado[0]
	)

	# Reajusta a melhor arquitetura com treino e validacao; o teste so e usado
	# nesta avaliacao final.
	indices_treino_validacao = np.concatenate(
		[indices_treino, indices_validacao]
	)
	modelo_final = criar_modelo(melhor_arquitetura)
	pesos_finais = compute_sample_weight(
		class_weight="balanced", y=faixas_alvo[indices_treino_validacao]
	)
	modelo_final.fit(
		entradas[indices_treino_validacao],
		faixas_alvo[indices_treino_validacao],
		mlpclassifier__sample_weight=pesos_finais,
	)
	previsoes_teste = modelo_final.predict(entradas[indices_teste])
	acuracia_teste, _, relatorio_teste = calcular_metricas(
		faixas_alvo[indices_teste], previsoes_teste
	)
	comparacao = pd.DataFrame(
		[
			(nome, arquitetura, acuracia)
			for _, nome, arquitetura, acuracia in resultados
		],
		columns=["Modelo", "Camadas", "Acuracia"],
	).round(3)
	figura_resumo, eixos_resumo = plt.subplots(1, 2, figsize=(14, 5))
	for eixo, tabela, titulo in (
		(eixos_resumo[0], comparacao, "Comparacao na validacao"),
		(
			eixos_resumo[1],
			relatorio_teste,
			f"Teste final | MLP {melhor_nome} {melhor_arquitetura} | Acuracia: {acuracia_teste:.3f}",
		),
	):
		eixo.axis("off")
		eixo.set_title(titulo)
		visual = eixo.table(
			cellText=tabela.values,
			rowLabels=tabela.index,
			colLabels=tabela.columns,
			cellLoc="center",
			loc="center",
		)
		visual.auto_set_font_size(False)
		visual.set_fontsize(9)
		visual.scale(1, 1.5)
	figura_resumo.suptitle("Resumo dos experimentos")
	figura_resumo.tight_layout()
	plt.show()

if __name__ == "__main__":
	executar_modelo()