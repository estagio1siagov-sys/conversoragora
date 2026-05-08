import os
import re
import logging
from typing import List

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import streamlit as st

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.1-flash-lite-preview"


def _build_gemini_model():
    api_key = (
        st.secrets.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Chave da API Gemini não encontrada. "
            "Defina em .streamlit/secrets.toml ou na variável GEMINI_API_KEY."
        )

    genai.configure(api_key=api_key)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        safety_settings=safety_settings
    )


class GeminiService:
    """
    Serviço único para conversão e validação de documentos legislativos.
    - convert()  → PDF (imagens) → Markdown + validação automática + score independente
    - validate() → PDF (imagens) + Markdown existente → score + detalhes + erros
    """

    def __init__(self, model):
        self.model = model
        logger.info(f"GeminiService pronto — modelo: {MODEL_NAME}")

    # ──────────────────────────────────────────────────────────────────────────
    # CONVERSÃO
    # ──────────────────────────────────────────────────────────────────────────

    def convert(self, pages: List) -> dict:
        """
        Converte PDF para Markdown e valida automaticamente com critérios independentes.
        Se score < 90, tenta reconverter focando nos erros encontrados.
        Retorna sempre o melhor resultado entre as tentativas.
        """
        result = self._convert_once(pages)
        if not result['success']:
            return result

        validation = self.validate(pages, result['markdown'])
        result['score']    = validation.get('score')
        result['errors']   = validation.get('errors', [])
        result['details']  = validation.get('details', [])
        result['attempts'] = 1

        score = validation.get('score') or 0

        if score >= 90:
            return result

        logger.info(f"Score {score}% abaixo de 90% — reconvertendo com foco nos erros.")
        errors_found  = validation.get('errors', [])
        details_found = validation.get('details', [])
        retry = self._convert_once(pages, errors_found + details_found)

        if not retry['success']:
            return result

        val_retry = self.validate(pages, retry['markdown'])
        retry['score']    = val_retry.get('score')
        retry['errors']   = val_retry.get('errors', [])
        retry['details']  = val_retry.get('details', [])
        retry['attempts'] = 2

        if (val_retry.get('score') or 0) >= score:
            logger.info(f"Retry melhorou: {score}% → {val_retry.get('score')}%")
            return retry

        logger.info(f"Retry não melhorou ({val_retry.get('score')}%). Mantendo primeira versão.")
        return result

    def _convert_once(self, pages: List, errors_to_fix: list = None) -> dict:
        """
        Faz uma única chamada de conversão.
        Se errors_to_fix for passado, o prompt foca em corrigir esses erros.
        """
        if errors_to_fix:
            errors_block = "\n".join(f"- {e}" for e in errors_to_fix)
            extra = f"""
ATENÇÃO — Esta é uma segunda tentativa. Uma conversão anterior cometeu os seguintes erros:
{errors_block}

Preste atenção redobrada nesses pontos e garanta que estejam corretos nesta versão.
"""
        else:
            extra = ""

        prompt = f"""Você é um especialista em digitalização de documentos jurídicos oficiais do Diário Oficial do Município de João Pessoa.
{extra}
SUA TAREFA:
Extraia TODO o conteúdo do documento e converta para Markdown estruturado e fiel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTRUTURA FÍSICA DO DOCUMENTO — LEIA COM ATENÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cada página física do PDF pode conter DUAS COLUNAS de texto lado a lado, representando duas seções ou decretos diferentes.

As assinaturas aparecem ROTACIONADAS 90° nas laterais da página, entre as colunas ou nas bordas. Cada assinatura lateral pertence logicamente à seção/decreto mais próximo a ela.

REGRA DE POSICIONAMENTO DE ASSINATURAS:
- Identifique a qual seção ou decreto cada assinatura lateral pertence
- Posicione a assinatura ao FINAL da seção correspondente no Markdown
- NUNCA consolide todas as assinaturas no final do documento
- Se uma página tem duas seções, cada seção termina com sua própria assinatura

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS DE EXTRAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ESTRUTURA: Preserve a hierarquia exata — Título, Ementa, Artigos (Art.), Parágrafos (§), Incisos (I, II...), Alíneas (a, b, c...)

2. COLUNAS: Processe da esquerda para a direita. Se houver duas colunas, processe a coluna esquerda completamente antes de iniciar a coluna direita. Use um separador entre seções:
   ---

3. TABELAS: Converta TODAS as tabelas para formato Markdown (| col | col |). Preserve todos os valores numéricos, códigos orçamentários e descrições sem alteração alguma.

4. CABEÇALHO: Inclua número do diário oficial, data e página se visíveis.

5. VALORES: Transcreva valores numéricos E por extenso exatamente como estão — mesmo que haja inconsistência entre o numeral e o extenso no documento original, transcreva ambos fielmente.

6. ASSINATURAS: Ao final de cada seção/decreto, inclua o bloco:
   > [Assinatura] NOME COMPLETO
   > Cargo

   Se houver verificação digital, inclua no mesmo bloco:
   > [Verificação Digital]
   > Código: XXXX-XXXX-XXXX-XXXX
   > Assinado por: NOME COMPLETO (CPF XXX.XXX.XXX-XX) em DD/MM/AAAA HH:MM (GMT-HH:MM)
   > Papel: <papel>
   > Emitido por: <autoridade certificadora>
   > Link: https://...

7. FIDELIDADE: Não resuma, não omita, não interprete — transcreva fielmente.

8. RUÍDOS: Ignore marcas d'água, carimbos de protocolo e artefatos de digitalização.

9. DUPLICAÇÕES: Nunca repita artigos, parágrafos ou seções — cada elemento aparece UMA única vez.

RESPONDA ESTRITAMENTE neste formato:

MARKDOWN:
<markdown completo do documento>
"""

        try:
            response = self.model.generate_content([prompt] + pages)
            return self._parse_convert_response(response.text)
        except Exception as e:
            logger.error(f"Erro na conversão Gemini: {e}", exc_info=True)
            return {'markdown': '', 'score': None, 'errors': [], 'details': [], 'success': False, 'error': str(e)}

    # ──────────────────────────────────────────────────────────────────────────
    # VALIDAÇÃO
    # ──────────────────────────────────────────────────────────────────────────

    def validate(self, pages: List, markdown: str) -> dict:
        """
        Valida o Markdown contra o PDF original com critérios rigorosos.
        Retorna APENAS score, detalhes e erros — sem correções, sem reescrita.
        """
        prompt = f"""Você é um auditor RIGOROSO de integridade de documentos jurídicos oficiais.

Você está recebendo as imagens do documento original e um Markdown gerado a partir dele.
Sua única função é medir a fidelidade com MÁXIMO RIGOR — NÃO corrija, NÃO reescreva nada.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTRUTURA FÍSICA DO DOCUMENTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cada página física pode ter DUAS COLUNAS lado a lado com assinaturas rotacionadas 90° nas laterais.
As assinaturas laterais pertencem logicamente à seção mais próxima.
No Markdown correto, cada assinatura deve aparecer ao FINAL da seção correspondente — não consolidadas no final do documento.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGRAS DE PONTUAÇÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IGNORE ao avaliar:
- Carimbos de protocolo, marcas d'água, cabeçalhos/rodapés repetitivos de página
- Ausência de representação gráfica de QR codes e barras laterais (elementos visuais não representáveis em Markdown)

PENALIZAÇÕES OBRIGATÓRIAS:
- Tabela AUSENTE no Markdown mas presente no PDF → desconte 30 pontos
- Tabela INCOMPLETA ou com valores errados → desconte 15 pontos
- Bloco de assinaturas AUSENTE → desconte 15 pontos
- Assinaturas consolidadas no final quando deveriam estar distribuídas por seção → desconte 10 pontos
- Código de verificação digital AUSENTE quando o PDF claramente o contém → desconte 10 pontos
- Artigo ou parágrafo AUSENTE → desconte 10 pontos por ocorrência
- Valor numérico divergente do extenso no mesmo trecho (ex: "25%" mas escrito "vinte por cento") → desconte 8 pontos por ocorrência
- Valor monetário ou código orçamentário ERRADO → desconte 5 pontos por ocorrência
- Nome próprio, data ou número de decreto ERRADO → desconte 3 pontos por ocorrência

PESOS DOS CRITÉRIOS:
- (30%) TABELAS: presença, estrutura e todos os valores
- (25%) ARTIGOS: numeração, hierarquia e texto completo
- (20%) IDENTIFICAÇÃO: número do ato, data, ementa
- (15%) ASSINATURAS: presença, posicionamento correto e código de verificação se aplicável
- (10%) NOMES, DATAS E VALORES: grafias e numerais corretos

REGRAS ABSOLUTAS:
- Documento sem tabelas quando o original tem tabelas NUNCA pode ter score acima de 70
- Documento sem assinaturas quando o original tem assinaturas NUNCA pode ter score acima de 85
- Assinaturas consolidadas incorretamente = erro de posicionamento, não apenas de presença
- Reporte inconsistências entre numeral e extenso mesmo que sejam do documento original — indique que o Markdown transcreveu fielmente o erro do original
- Se apenas parte dos decretos tem verificação digital, penalize só a ausência dos que claramente a possuem
- Seja honesto e rigoroso — scores inflados prejudicam a integridade do sistema

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMATO DE RESPOSTA (sem texto fora dele)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCORE: <número de 0 a 100>

DETALHES:
- Tabelas: <OK | PARCIAL | AUSENTE> — <observação específica>
- Artigos: <OK | PARCIAL | AUSENTE/ERRADO> — <observação específica>
- Identificação: <OK | PARCIAL | ERRADO> — <observação específica>
- Assinaturas: <OK | PARCIAL | AUSENTE> — <observação específica>
- Nomes e Datas: <OK | PARCIAL | ERRADO> — <observação específica>

ERROS:
- <descrição objetiva do erro, ex: "Art. 9º inciso I: consta 'vinte por cento' mas numeral indica 25%">
(ou "Nenhum erro encontrado." se o markdown estiver correto)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MARKDOWN PARA AUDITAR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{markdown}
"""

        try:
            response = self.model.generate_content([prompt] + pages)
            return self._parse_validate_response(response.text)
        except Exception as e:
            logger.error(f"Erro na validação Gemini: {e}", exc_info=True)
            return self._safe_validate_result(str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # PARSERS
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_convert_response(self, text: str) -> dict:
        try:
            markdown = self._extract_block(text, "MARKDOWN")
            if not markdown:
                markdown = text.strip()
            return {'markdown': markdown, 'score': None, 'errors': [], 'details': [], 'success': True, 'error': None}
        except Exception as e:
            logger.error(f"Erro ao parsear conversão: {e}")
            return {'markdown': text, 'score': None, 'errors': [], 'details': [], 'success': True, 'error': None}

    def _parse_validate_response(self, text: str) -> dict:
        try:
            score   = self._extract_score(text)
            details = self._extract_details(text)
            errors  = self._extract_errors(text)
            return {'score': score, 'details': details, 'errors': errors, 'success': True, 'error': None}
        except Exception as e:
            logger.error(f"Erro ao parsear validação: {e}")
            return {'score': None, 'details': [], 'errors': [], 'success': False, 'error': str(e)}

    def _safe_validate_result(self, error: str) -> dict:
        return {'score': None, 'details': [], 'errors': [], 'success': False, 'error': error}

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_score(self, text: str):
        match = re.search(r'SCORE:\s*(\d+)', text)
        if match:
            return min(max(int(match.group(1)), 0), 100)
        return None

    def _extract_block(self, text: str, label: str) -> str:
        pattern = rf'{label}:\s*\n(.*?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return re.sub(r'```markdown\n?|```', '', match.group(1)).strip()
        return ""

    def _extract_details(self, text: str) -> list:
        block = self._extract_block(text, "DETALHES")
        if not block:
            return []
        return [
            line.lstrip("- ").strip()
            for line in block.splitlines()
            if line.strip().startswith("-")
        ]

    def _extract_errors(self, text: str) -> list:
        block = self._extract_block(text, "ERROS")
        if not block or "nenhum erro" in block.lower():
            return []
        return [
            line.lstrip("- ").strip()
            for line in block.splitlines()
            if line.strip().startswith("-")
        ]