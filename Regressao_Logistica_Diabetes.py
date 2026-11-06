"""Previsao experimental da faixa da próxima glicemia com regressão logistica."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from Rede_Neural_Diabetes import (
	CLASSES_GLICEMIA,
	FRACAO_TESTE,
	SEMENTE_ALEATORIA,
	calcular_metricas,
	carregar_sequencias,
)

MAX_ITERACOES = 2000
FATOR_PESO_ALVO = 1.2


def calcular_pesos_classe(classes_treino: np.ndarray) -> dict[str, float]:
	"""Aumenta o peso da classe alvo apos calcular os pesos balanceados."""
	pesos_balanceados = compute_class_weight(
		class_weight="balanced",
		classes=np.asarray(CLASSES_GLICEMIA),
		y=classes_treino,
	)
	pesos = dict(zip(CLASSES_GLICEMIA, pesos_balanceados))
	pesos["alvo"] *= FATOR_PESO_ALVO
	return pesos


def criar_modelo(pesos_classe: dict[str, float]):
	return make_pipeline(
		StandardScaler(),
		LogisticRegression(
			class_weight=pesos_classe,
			max_iter=MAX_ITERACOES,
			random_state=SEMENTE_ALEATORIA,
		),
	)


def mostrar_resultados(
	titulo: str,
	classes_reais: np.ndarray,
	previsoes: np.ndarray,
	acuracia: float,
	relatorio: pd.DataFrame,
) -> None:
	"""Exibe a matriz de confusao e as metricas por faixa."""
	figura, eixos = plt.subplots(
		1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.5]}
	)
	visualizacao = ConfusionMatrixDisplay.from_predictions(
		classes_reais,
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
	eixos[1].axis("off")
	eixos[1].set_title(f"Metricas | Acuracia: {acuracia:.3f}")
	tabela = eixos[1].table(
		cellText=relatorio.values,
		rowLabels=relatorio.index,
		colLabels=relatorio.columns,
		cellLoc="center",
		loc="center",
	)
	tabela.auto_set_font_size(False)
	tabela.set_fontsize(9)
	tabela.scale(1, 1.5)
	figura.suptitle(titulo)
	figura.tight_layout()


def executar_modelo() -> None:
	"""Treina, valida e testa a regressao logistica sem misturar pacientes."""
	entradas, faixas_alvo, ids_pacientes = carregar_sequencias()
	indices_desenvolvimento, indices_teste = next(
		GroupShuffleSplit(
			n_splits=1,
			test_size=FRACAO_TESTE,
			random_state=SEMENTE_ALEATORIA,
		).split(entradas, faixas_alvo, groups=ids_pacientes)
	)
	indices_treino_relativos, indices_validacao_relativos = next(
		GroupShuffleSplit(
			n_splits=1,
			test_size=0.25,
			random_state=SEMENTE_ALEATORIA + 1,
		).split(
			entradas[indices_desenvolvimento],
			faixas_alvo[indices_desenvolvimento],
			groups=ids_pacientes[indices_desenvolvimento],
		)
	)
	indices_treino = indices_desenvolvimento[indices_treino_relativos]
	indices_validacao = indices_desenvolvimento[indices_validacao_relativos]

	pesos_treino = calcular_pesos_classe(faixas_alvo[indices_treino])
	modelo = criar_modelo(pesos_treino)
	modelo.fit(entradas[indices_treino], faixas_alvo[indices_treino])
	previsoes_validacao = modelo.predict(entradas[indices_validacao])
	acuracia_validacao, _, relatorio_validacao = calcular_metricas(
		faixas_alvo[indices_validacao], previsoes_validacao
	)
	mostrar_resultados(
		"Regressao logistica | Validacao",
		faixas_alvo[indices_validacao],
		previsoes_validacao,
		acuracia_validacao,
		relatorio_validacao,
	)

	indices_treino_validacao = np.concatenate(
		[indices_treino, indices_validacao]
	)
	pesos_treino_validacao = calcular_pesos_classe(
		faixas_alvo[indices_treino_validacao]
	)
	modelo_final = criar_modelo(pesos_treino_validacao)
	modelo_final.fit(
		entradas[indices_treino_validacao],
		faixas_alvo[indices_treino_validacao],
	)
	previsoes_teste = modelo_final.predict(entradas[indices_teste])
	acuracia_teste, _, relatorio_teste = calcular_metricas(
		faixas_alvo[indices_teste], previsoes_teste
	)
	mostrar_resultados(
		"Regressao logistica | Teste final",
		faixas_alvo[indices_teste],
		previsoes_teste,
		acuracia_teste,
		relatorio_teste,
	)
	plt.show()


if __name__ == "__main__":
	executar_modelo()