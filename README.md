# Previsao de glicemia

Programa experimental que usa as cinco medicoes anteriores de glicose para
prever a faixa da medicao seguinte: **baixa**, **alvo** ou **alta**. Ha dois
programas: um compara tres arquiteturas de MLP e o outro treina uma regressao
logistica. Ambos avaliam o modelo em um conjunto de teste reservado.

## O que precisa ser instalado

- **Python 3.10 ou superior**.
- **Git**, somente se for baixar o projeto com `git clone`.
- As bibliotecas Python `numpy`, `pandas`, `matplotlib` e `scikit-learn`.
  O `scikit-learn` deve ser versao **1.7 ou superior**, pois o codigo usa
  pesos de amostra no treinamento da MLP.

O conjunto de dados `Diabetes-Data` ja esta incluido no repositorio. Portanto,
quem clonar ou baixar o projeto completo nao precisa baixar os dados a
parte. A pasta deve permanecer ao lado de `Rede_Neural_Diabetes.py`.

## 1. Baixar o projeto

Com Git instalado, abra um terminal e execute:

```bash
git clone https://github.com/GustavoPesc/bookish-octo-robot.git
cd bookish-octo-robot
```

Tambem e possivel baixar o projeto como ZIP pelo GitHub e extrair todos os
arquivos. Nesse caso, abra o terminal na pasta extraida.

Confira se a estrutura esta completa:

```text
bookish-octo-robot/
|-- Rede_Neural_Diabetes.py
`-- Diabetes-Data/
    |-- data-01
    |-- ...
    `-- data-70
```

## 2. Criar um ambiente Python

Um ambiente virtual separa as bibliotecas deste projeto das instalacoes do
resto do computador.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
```

### macOS ou Linux

```bash
python3 -m venv .venv
```

## 3. Instalar as bibliotecas

Execute os comandos na pasta do projeto.

### Windows (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib "scikit-learn>=1.7"
```

### macOS ou Linux

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install numpy pandas matplotlib "scikit-learn>=1.7"
```

Esses comandos baixam as dependencias necessarias. Nao e necessario instalar
separadamente as bibliotecas padrao do Python usadas pelo programa.

## 4. Executar

Execute o modelo desejado a partir da pasta do projeto. Ambos usam o mesmo
conjunto de dados e a mesma divisao por paciente; a regressao logistica tambem
usa as funcoes de preparacao de dados de `Rede_Neural_Diabetes.py`, portanto
esse arquivo deve permanecer na pasta mesmo ao executar o outro programa.

### Windows (PowerShell)

```powershell
.\.venv\Scripts\python.exe Rede_Neural_Diabetes.py
```

Para executar a regressao logistica:

```powershell
.\.venv\Scripts\python.exe Regressao_Logistica_Diabetes.py
```

### macOS ou Linux

```bash
./.venv/bin/python Rede_Neural_Diabetes.py
```

Para executar a regressao logistica:

```bash
./.venv/bin/python Regressao_Logistica_Diabetes.py
```

O treinamento pode levar alguns minutos. Os programas abrem graficos com
matrizes de confusao e metricas; o da MLP tambem mostra a curva de perda e
compara as arquiteturas. Feche as janelas de graficos para encerrar a execucao.
Os resultados sao exibidos na tela e nao sao salvos automaticamente em arquivos.

## Problemas comuns

- **A pasta `Diabetes-Data` nao foi encontrada:** baixe ou extraia o repositorio
  completo e confirme que essa pasta esta no mesmo nivel do arquivo
  `Rede_Neural_Diabetes.py`. Nao basta copiar somente o script.
- **`No module named ...`:** confira se as bibliotecas foram instaladas no
  ambiente virtual e execute o programa usando o Python dentro de `.venv`,
  conforme os comandos acima.
- **Versao antiga do scikit-learn:** atualize com
  `python -m pip install --upgrade "scikit-learn>=1.7"` usando o Python do
  ambiente virtual.
