# 📄 Conversor de PDF para Markdown (SIAGOV)

Este projeto é uma ferramenta robusta desenvolvida para transformar documentos PDF em Markdown (.md) de forma inteligente, utilizando a biblioteca **Docling** para garantir alta fidelidade na extração de títulos, listas e tabelas.

## Funcionalidades

- **Extração de Tabelas:** Reconstrução precisa de tabelas para o formato Markdown.
- **Suporte OCR:** Consegue ler PDFs escaneados ou baseados em imagem.
- **Validação de Qualidade:**
  - **Métrica de Fidelidade:** Cálculo de similaridade entre o original e o gerado.
  - **Revisão Ortográfica:** Detecção de erros em Português (PT-BR).

## Tecnologias Utilizadas

- [Python 3.10+](https://www.python.org/)
- [Docling](https://github.com/DS4SD/docling) (Motor de conversão)
- [Streamlit](https://streamlit.io/) (Interface UI)
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy) (Métricas de similaridade)

## 🚀 Como Instalar e Usar

1. **Instale as dependências:**
   ```bash
   pip install streamlit docling fuzzywuzzy python-Levenshtein language_tool_python pyspellchecker PyPDF2