import os
import time
from pathlib import Path
import google.generativeai as genai
from pdf2image import convert_from_path

# 1. Configuração com a chave e o modelo novo
genai.configure(api_key="AIzaSyA2pG9TEhU7iaG47Uc1HbEIYESFqKdAOpk")
NOME_MODELO = 'gemini-3.1-flash-lite-preview' # O modelo que apareceu na sua lista

def process_pdf_with_gemini(pdf_path_str):
    pdf_path = Path(pdf_path_str)
    md_output_path = pdf_path.with_suffix('.md')
    
    print(f"--- Iniciando: {pdf_path.name} ---")
    
    # Converte PDF para imagens (300 DPI para garantir leitura de letras pequenas)
    pages = convert_from_path(pdf_path, dpi=300)
    
    model = genai.GenerativeModel(NOME_MODELO)
    full_markdown = []

    for i, page in enumerate(pages):
        print(f"Processando página {i+1}/{len(pages)}...")
        
        prompt = """
        Analise esta imagem de um documento oficial municipal.
        Converta todo o conteúdo para Markdown estruturado.
        - Mantenha o cabeçalho oficial.
        - Identifique Artigos, Parágrafos e Incisos usando a formatação correta (#, ##, >).
        - Se houver assinaturas ou carimbos, apenas indique [Assinatura] ou [Carimbo].
        - Não ignore notas de rodapé.
        Retorne apenas o Markdown puro.
        """
        
        try:
            # Chamada multimodal (Texto + Imagem)
            response = model.generate_content([prompt, page])
            full_markdown.append(response.text)
            
            # Pequena pausa para não estourar o limite de "Requisições por Minuto" (RPM)
            time.sleep(2) 
            
        except Exception as e:
            print(f"Erro na página {i+1}: {e}")
            full_markdown.append(f"\n> [ERRO NA PÁGINA {i+1}]\n")

    # Salva o arquivo final na mesma pasta
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(full_markdown))
        
    print(f"\nConcluído! Veja o resultado em: {md_output_path}")

if __name__ == "__main__":
    # Use o nome do arquivo exatamente como ele é (entre aspas)
    arquivo = r"1. Decreto nº 10.716 2024.pdf"
    process_pdf_with_gemini(arquivo)