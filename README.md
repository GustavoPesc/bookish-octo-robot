# Previsão de glicemia

Programa experimental que usa as cinco medições anteriores de glicose para
prever a faixa da medição seguinte: **baixa**, **alvo** ou **alta**. Há dois
programas: um compara três arquiteturas de MLP e o outro treina uma regressão
logística. Ambos avaliam o modelo em um conjunto de teste reservado.

## O que precisa ser instalado

- **Python 3.10 ou superior**.
- **Git**, somente se for baixar o projeto com `git clone`.
- As bibliotecas Python `numpy`, `pandas`, `matplotlib` e `scikit-learn`.
  O `scikit-learn` deve ser versão **1.7 ou superior**, pois o código usa
  pesos de amostra no treinamento da MLP.

O conjunto de dados `Diabetes-Data` já está incluído no repositório. Portanto,
quem clonar ou baixar o projeto completo não precisa baixar os dados à
parte. A pasta deve permanecer ao lado de `Rede_Neural_Diabetes.py`.

## 1. Baixar o projeto

Com Git instalado, abra um terminal e execute:

```bash
git clone https://github.com/GustavoPesc/bookish-octo-robot.git
cd bookish-octo-robot
```

Também é possível baixar o projeto como ZIP pelo GitHub e extrair todos os
arquivos. Nesse caso, abra o terminal na pasta extraída.

Confira se a estrutura está completa:

```text
bookish-octo-robot/
|-- Rede_Neural_Diabetes.py
`-- Diabetes-Data/
    |-- data-01
    |-- ...
    `-- data-70
```

## 2. Criar um ambiente Python

Um ambiente virtual separa as bibliotecas deste projeto das instalações do
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

Esses comandos baixam as dependências necessárias. Não é necessário instalar
separadamente as bibliotecas padrão do Python usadas pelo programa.

## 4. Executar

Execute o modelo desejado a partir da pasta do projeto. Ambos usam o mesmo
conjunto de dados e a mesma divisão por paciente; a regressão logística também
usa as funções de preparação de dados de `Rede_Neural_Diabetes.py`, portanto
esse arquivo deve permanecer na pasta mesmo ao executar o outro programa.

### Windows (PowerShell)

```powershell
.\.venv\Scripts\python.exe Rede_Neural_Diabetes.py
```

Para executar a regressão logística:

```powershell
.\.venv\Scripts\python.exe Regressao_Logistica_Diabetes.py
```

### macOS ou Linux

```bash
./.venv/bin/python Rede_Neural_Diabetes.py
```

Para executar a regressão logística:

```bash
./.venv/bin/python Regressao_Logistica_Diabetes.py
```

O treinamento pode levar alguns minutos. Os programas abrem gráficos com
matrizes de confusão e métricas; o da MLP também mostra a curva de perda e
compara as arquiteturas. Feche as janelas de gráficos para encerrar a execução.
Os resultados são exibidos na tela e não são salvos automaticamente em arquivos.

## Problemas comuns

- **A pasta `Diabetes-Data` não foi encontrada:** baixe ou extraia o repositório
  completo e confirme que essa pasta está no mesmo nível do arquivo
  `Rede_Neural_Diabetes.py`. Não basta copiar somente o script.
- **`No module named ...`:** confira se as bibliotecas foram instaladas no
  ambiente virtual e execute o programa usando o Python dentro de `.venv`,
  conforme os comandos acima.
- **Versão antiga do scikit-learn:** atualize com
  `python -m pip install --upgrade "scikit-learn>=1.7"` usando o Python do
  ambiente virtual.
