# Olá a todos!

Abaixo estão os itens com suas respectivas respostas:

## Item 1 - Sobre Storytelling e Apresentação

Aqui está o [meu vídeo de explicação](https://youtu.be/Wi1cg3hbybs).

### Arquitetura

![Print da arquitetura](./arquitetura.jpg)

#### Arquitetura antiga

![Print da arquitetura](./arq1.jpg)

Uma arquitetura que apresenta pontos negativos como complexidade de integração, custos elevados e escalabidade manual, o que pode ser um problema.

#### Arquitetura nova

![Print da arquitetura](./arq2.jpg)

A nova arquitura centraliza tudo na Dadosfera, utilizando através de API, pípeline e processamento próprio, mantendo apenas o envio para o S3 Bucket e ainda sendo possível a criação de dashboards.

## Item 2 - Sobre a Dadosfera

Os dados foram extraídos pelo link fornecido do dataset

![Print dataset](./dataset.jpg)

Porém ao tentar subir o arquivo para a plataforma da Dadosfera, verifiquei que existe a limitação de 250 mb e aceita somente Json, CSV e Parquet:

![Printlimitacao](./import.jpg)

Pensando em uma solução, elaborei um notebook jupyter para extrair os dados do URL do Dataset e salvar 150 registros em um .CSV, pensando na intenção de diminuir o tamanho do arquivo final e conseguir subir isso para a plataforma.
![Printcsv](/csv.jpg)

Com o arquivo gerado, tive exito na súbida do arquivo para a plataforma da Dadosfera.
![Printdados](/subset.jpg)

Arquivos disponibilizados na plataforma.
![Printdataset_case](/dataset_case.jpg)

## Item 3 - Sobre GenAI e LLMs

Nessa etapa era necessário gerar um novo arquivo, onde os dados obtidos do CSV fosse tratados e preparados para seguir esse modelo:

![Print_modelo](/modelo.jpg)

Seguindo a sugestão colocada diretamente na proposta, utilize uma LLM forncida pela própria HuggingFace que seria a plataforma responsável por hospedar o Dataset.

```python
import pandas as pd
import json
import re
from transformers import pipeline

# Caminho para o arquivo CSV e o arquivo de saída CSV
file_path = r"C:\Users\Moraes\subset_data.csv"
output_file_path = r"C:\Users\Moraes\products_with_features.csv"

# Carregar o modelo de perguntas e respostas da Hugging Face
qa_pipeline = pipeline("question-answering")


# Função para limpar e ajustar o campo Product Description
def clean_product_description(text):
    if not isinstance(text, str):
        return ""

    cleaned_text = re.sub(
        r"(?i)\bProduct\s*Description[:\-\—\s]*(.*)", r"\1", text
    ).strip()

    return cleaned_text


# Função para gerar features usando o modelo da Hugging Face
def generate_features_hf(title, text):
    text = clean_product_description(text)

    questions = {
        "category": "What is the category of the product?",
        "material": "What material is the product made of?",
        "receiver_design": "Describe the receiver design.",
        "hand_strap": "Does the product have a hand strap?",
        "RFID_technique": "What is the RFID technique used for?",
        "handmade": "Is the product handmade?",
        "stitching": "Describe the stitching of the product.",
        "card_slots": "Does the product have card slots?",
        "cosmetic_mirror": "Does the product have a cosmetic mirror?",
        "kickstand_function": "Does the product have a kickstand function?",
        "space_amplification": "Does the product have space amplification?",
        "color_options": "What are the color options for the product?",
        "compatibility": "What is the compatibility of the product?",
    }

    features = {}
    try:
        for key, question in questions.items():
            answer = qa_pipeline(
                question=question, context=f"Title: {title}\nDescription: {text}"
            )
            features[key] = answer["answer"]
    except Exception as e:
        features["error"] = str(e)

    return features


# Função para processar cada linha do DataFrame
def process_row(row):
    original_description = row["Product Description"]
    cleaned_description = clean_product_description(original_description)

    features = generate_features_hf(row["Titulo"], cleaned_description)

    row_data = {
        "Titulo": row["Titulo"],
        "Product Description": cleaned_description,
    }
    row_data.update(features)

    return row_data



data = pd.read_csv(file_path)
data.rename(columns={"title": "Titulo", "text": "Product Description"}, inplace=True)


processed_data = data.apply(process_row, axis=1).tolist()


processed_df = pd.DataFrame(processed_data)

processed_df.to_csv(output_file_path, index=False)

print(f"Arquivo CSV salvo como {output_file_path}.")

```

Nessa etapa além do uso da LLM para fazer a extração das perguntas com base no modelo cedido, também foi realizado uma limpeza nos dados através de um regex tendo em vista que, no campo de Product Description em alguns momentos era retornado duplicidade nos dados ficando por exemplo Product "Description: Product Description", com isso foi realizada um ajuste para ser preparado o código e higienizado para melhor atentar a precisão das necessidades dos dados.

![Printllm](./llm.jpg)

Arquivo tratado e hospedado na plataforma da Dadosfera.

## Item 4 - Sobre SQL e Python

Nessa etapa foi necessário executar uma Query SQL dentro da plataforma da Dadosfera com o propósito de categorizar os produtos contidos no Dataset, para isso foi necessário criar uma coleção.

![Printvisualization](./visualization.jpg)

Então foi executado uma query para gerar um resultado para encontrarmos os valores requisitados.

![Printquery](./Query%20Result.jpg)

Só quero deixar um comentário, que seguindo a documentação disponibilizada, não foi possível encontrar a fonte de dados diretamente na plataforma e para conseguir resolver essa situação, subi novamente o arquivo .CSV na plataforma para realizar a Query.

## Item 5 - Sobre Data Apps

Foi criado o ambiente com Streamlit integrado, e junto a isso também foi realizado uma análise dos dados anteriores utilizando EDA, onde é feito o upload do arquivo .CSV com os dados tratados nas etapas anteriores:

![Printeda](/EDA.jpg)

E com isso é possível verificar alguns pontos através do Streamlit que são reconhecidos através de insights utilizados no código Python utilizado:

```python
import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("EDA Streamlit App - Análise de Produtos")
    st.header("Upload seu arquivo CSV de dados")

    # Adicionando uma chave única para o file_uploader
    data_file = st.file_uploader("Upload CSV", type=["csv"], key="uploader_1")

    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Visualização dos dados:")
        st.write(data.head())

        # Listar as colunas disponíveis para diagnóstico
        st.write("Colunas disponíveis:", data.columns.tolist())

        st.sidebar.header("Opções de Visualização")
        plot_options = [
            "Distribuição de Categorias",
            "Materiais Mais Utilizados",
            "Produtos sem Descrição",
            "Características de Design",
            "Combinações Populares de Características"
        ]
        selected_plot = st.sidebar.selectbox("Escolha um tipo de análise", plot_options)

        # 1. Distribuição de Categorias
        if selected_plot == "Distribuição de Categorias":
            if 'category' in data.columns:
                st.write("Distribuição das Categorias de Produtos")
                counts = data['category'].value_counts()
                st.bar_chart(counts)
            else:
                st.error("Coluna 'category' não encontrada no DataFrame.")

        # 2. Materiais Mais Utilizados
        elif selected_plot == "Materiais Mais Utilizados":
            if 'material' in data.columns:
                st.write("Frequência de Materiais Utilizados")
                material_counts = data['material'].value_counts()
                st.bar_chart(material_counts)
            else:
                st.error("Coluna 'material' não encontrada no DataFrame.")

        # 3. Produtos sem Descrição
        elif selected_plot == "Produtos sem Descrição":
            if 'Product Description' in data.columns:
                st.write("Análise de Produtos com Descrições Ausentes")
                missing_descriptions = data['Product Description'].isna().sum()
                total_products = len(data)
                st.write(f"Número de produtos sem descrição: {missing_descriptions}")
                st.write(f"Porcentagem de produtos sem descrição: {missing_descriptions / total_products * 100:.2f}%")
            else:
                st.error("Coluna 'Product Description' não encontrada no DataFrame.")

        # 4. Características de Design
        elif selected_plot == "Características de Design":
            design_feature = st.sidebar.selectbox("Selecione a característica de design", ['hand_strap', 'RFID_technique'])
            if design_feature in data.columns:
                st.write(f"Distribuição da Característica de Design: {design_feature}")
                feature_counts = data[design_feature].value_counts()
                st.bar_chart(feature_counts)
            else:
                st.error(f"Coluna '{design_feature}' não encontrada no DataFrame.")

        # 5. Combinações Populares de Características
        elif selected_plot == "Combinações Populares de Características":
            if 'material' in data.columns and 'receiver_design' in data.columns:
                st.write("Combinações Populares de Materiais e Características de Design")
                combined = data.groupby(['material', 'receiver_design']).size().reset_index(name='counts')
                combined = combined.sort_values(by='counts', ascending=False).head(10)
                # Exibe a tabela e o gráfico sem definir índices adicionais
                st.write(combined)
                st.bar_chart(combined.pivot(index='material', columns='receiver_design', values='counts').fillna(0))
            else:
                st.error("Colunas 'material' ou 'receiver_design' não encontradas no DataFrame.")

if __name__ == "__main__":
    main()

```

![PrintVisualization](./visualizacao.jpg)

### Item Bonus - Sobre GenAI + Data Apps

Ao tentar realizar a etapa bonus para gerar imagens dos produtos através de IA com StreamLit, tive alguns impedimentos relacionados ao Billing, tentei fazer as execuções via máquina local utilizando algumas APIs como OpenAI, Replicate e HuggingFace, mas acabei batendo em limitações de custo e acabou me limitando na execução, cheguei a conseguir 1 resultado mas após isso comecei a tomar erros por limitação.

![Printbloqueio](./bloqueio.jpg)

Código usado

```python

import streamlit as st
import pandas as pd
import requests
import base64
from io import BytesIO

REPLICATE_API_KEY = '{token_key}'
REPLICATE_URL = 'https://api.replicate.com/v1/predictions'

def generate_image_replicate(prompt):
    headers = {
        'Authorization': f'Token {REPLICATE_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "version": "e13215e4806ebae1b7cde53a6d88beed7fe12d05db5f285eccf5bc3fb9d28f4d",
        "input": {
            "prompt": prompt
        }
    }

    response = requests.post(REPLICATE_URL, headers=headers, json=data)
    response.raise_for_status()

    prediction_id = response.json()["id"]

    result = None
    while True:
        response = requests.get(f"{REPLICATE_URL}/{prediction_id}", headers=headers)
        result = response.json()
        if result["status"] == "succeeded":
            break
        elif result["status"] == "failed":
            raise Exception("Falha na geração da imagem.")

    image_url = result["output"][0]
    image_response = requests.get(image_url)
    image_response.raise_for_status()

    return image_response.content

def main():
    st.title("Gerador de Apresentações de Produto")
    st.header("Faça upload do seu arquivo CSV com informações do produto")

    data_file = st.file_uploader("Upload CSV", type=["csv"], key="uploader_1")

    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Visualização dos dados do produto:")
        st.write(data.head())

        # Verificação de colunas essenciais presentes no CSV
        expected_columns = ['Titulo', 'Product Description', 'category', 'material',
                            'receiver_design', 'hand_strap', 'RFID_technique', 'handmade',
                            'stitching', 'card_slots', 'cosmetic_mirror', 'kickstand_function',
                            'space_amplification', 'color_options', 'compatibility']
        missing_columns = [col for col in expected_columns if col not in data.columns]

        if missing_columns:
            st.error(f"As seguintes colunas estão ausentes no CSV: {', '.join(missing_columns)}")
            st.stop()

        data.fillna('Não informado', inplace=True)

        # Selecionar o produto para gerar a apresentação
        st.sidebar.header("Selecione o Produto para Apresentação")
        product_index = st.sidebar.selectbox("Escolha o produto", data.index)

        if st.sidebar.button("Gerar Apresentação"):
            selected_product = data.iloc[product_index]
            product_info = {
                "Nome": selected_product['Titulo'],
                "Descrição": selected_product['Product Description'],
                "Categoria": selected_product['category'],
                "Material": selected_product['material'],
                "Design do receptor": selected_product['receiver_design']
            }

            # Gerar a imagem do produto
            image_prompt = f"Crie uma imagem representando um produto chamado '{product_info['Nome']}' com a descrição: {product_info['Descrição']}."
            st.write("Gerando imagem...")
            try:
                image_data = generate_image_replicate(image_prompt)
                image_filename = f"imagem_{product_info['Nome']}.png"

                # Mostrar a imagem gerada
                st.image(image_data, caption=f"Imagem de {product_info['Nome']}")
            except Exception as e:
                st.error(f"Erro ao gerar imagem: {e}")

if __name__ == "__main__":
    main()

```
