import os
import warnings
import logging
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from pdf2image import convert_from_path
from rapidfuzz import process, fuzz

from gemini_service import GeminiService, _build_gemini_model

st.set_page_config(
    page_title="Conversor Legislativo",
    page_icon="⚖️",
    layout="wide"
)


@st.cache_resource(show_spinner="⏳ Conectando ao Gemini...")
def get_service():
    model = _build_gemini_model()
    return GeminiService(model)


def _normalize(name: str) -> str:
    import re
    name = name.lower()
    name = re.sub(r'^[\d\s._-]+', '', name)
    name = re.sub(r'[._\-\s]+', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _match_files(pdf_files, md_files, threshold=70):
    pdf_map  = {Path(f.name).stem: f for f in pdf_files}
    md_map   = {Path(f.name).stem: f for f in md_files}
    pdf_norm = {stem: _normalize(stem) for stem in pdf_map}
    md_norm  = {stem: _normalize(stem) for stem in md_map}

    matched   = []
    used_pdfs = set()
    used_mds  = set()

    for pdf_stem, pdf_norm_name in pdf_norm.items():
        md_stems = list(md_norm.keys())
        md_norms = list(md_norm.values())
        if not md_stems:
            break
        result = process.extractOne(
            pdf_norm_name, md_norms, scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= threshold:
            best_idx     = md_norms.index(result[0])
            best_md_stem = md_stems[best_idx]
            if best_md_stem not in used_mds:
                matched.append((pdf_map[pdf_stem], md_map[best_md_stem], result[1]))
                used_pdfs.add(pdf_stem)
                used_mds.add(best_md_stem)

    pdf_only = [pdf_map[s] for s in pdf_map if s not in used_pdfs]
    md_only  = [md_map[s]  for s in md_map  if s not in used_mds]
    return matched, pdf_only, md_only


def main():
    # ── Pré-carregamento na inicialização ─────────────────────────────────────
    if "ready" not in st.session_state:
        st.session_state.ready = False

    if not st.session_state.ready:
        st.title("⚖️ Conversor Legislativo")
        st.markdown("---")
        with st.status("🚀 Inicializando sistema...", expanded=True) as status:
            st.write("⏳ Conectando ao Gemini...")
            get_service()
            st.write("✅ Gemini conectado e pronto.")
            status.update(label="✅ Sistema pronto!", state="complete")
        st.session_state.ready = True
        st.rerun()

    service = get_service()
    # ─────────────────────────────────────────────────────────────────────────

    st.title("⚖️ Conversor Legislativo")
    st.caption("Conversão e validação de documentos legislativos via Gemini")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configurações")
        st.info(
            "**Fluxo:**\n"
            "- **Converter:** PDF → Gemini → Markdown + validação automática\n"
            "- **Validar:** PDF + Markdown → Gemini → Score + Erros"
        )
        st.markdown("---")
        st.caption("Modelo: `gemini-3.1-flash-lite-preview`")

    tab_convert, tab_validate = st.tabs(["📄 Converter PDF", "🔍 Validar Markdown"])

    # ══════════════════════════════════════════════════════════════════════════
    # ABA 1: CONVERTER
    # ══════════════════════════════════════════════════════════════════════════
    with tab_convert:
        st.subheader("Converter PDF para Markdown")
        st.caption("O Gemini extrai o conteúdo e valida automaticamente. Se necessário, reconverte com foco nos erros.")

        uploaded_pdfs = st.file_uploader(
            "Selecione os arquivos PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploader_convert"
        )

        if uploaded_pdfs and st.button("🚀 Converter", key="btn_convert"):
            for uploaded_file in uploaded_pdfs:
                st.markdown(f"### 📄 {uploaded_file.name}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    with st.status("Processando...", expanded=True) as status:
                        st.write("📸 Renderizando páginas do PDF...")
                        pages = convert_from_path(tmp_path, dpi=200)
                        st.write(f"✅ {len(pages)} página(s) renderizada(s).")

                        st.write("🤖 Convertendo com Gemini...")
                        result = service.convert(pages)

                        if not result['success'] and not result['markdown']:
                            st.error(f"Erro: {result.get('error')}")
                            status.update(label="Falha", state="error")
                            continue

                        attempts = result.get('attempts', 1)
                        if attempts == 2:
                            st.write("🔁 Score abaixo de 90% — reconvertido com foco nos erros.")
                        st.write(f"✅ Concluído em {attempts} tentativa(s).")
                        status.update(label="✅ Concluído!", state="complete")

                    _render_convert_results(
                        uploaded_file.name,
                        result['markdown'],
                        result.get('score'),
                        len(pages),
                        result.get('errors', []),
                        result.get('details', [])
                    )

                except Exception as e:
                    st.error(f"Erro inesperado: {e}")
                finally:
                    if tmp_path.exists():
                        os.unlink(tmp_path)

    # ══════════════════════════════════════════════════════════════════════════
    # ABA 2: VALIDAR
    # ══════════════════════════════════════════════════════════════════════════
    with tab_validate:
        st.subheader("Validar Markdown existente")
        st.caption("Suba múltiplos PDFs e Markdowns. O sistema cruza pelo nome e avisa arquivos órfãos.")

        col_a, col_b = st.columns(2)

        with col_a:
            pdf_files = st.file_uploader(
                "📎 PDFs originais",
                type=["pdf"],
                accept_multiple_files=True,
                key="uploader_val_pdf"
            )

        with col_b:
            md_files = st.file_uploader(
                "📝 Arquivos Markdown (.md)",
                type=["md", "txt"],
                accept_multiple_files=True,
                key="uploader_val_md"
            )

        if pdf_files or md_files:
            matched, pdf_only, md_only = _match_files(
                pdf_files or [], md_files or []
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Pares encontrados", len(matched))
            c2.metric("⚠️ PDFs sem Markdown", len(pdf_only))
            c3.metric("⚠️ Markdowns sem PDF", len(md_only))

            if matched:
                with st.expander("✅ Pares detectados"):
                    for pdf_f, md_f, sim in matched:
                        st.write(f"- `{pdf_f.name}` ←→ `{md_f.name}` (similaridade: {sim:.0f}%)")

            if pdf_only:
                with st.expander(f"⚠️ PDFs órfãos ({len(pdf_only)}) — sem Markdown correspondente"):
                    for f in pdf_only:
                        st.write(f"- `{f.name}`")

            if md_only:
                with st.expander(f"⚠️ Markdowns órfãos ({len(md_only)}) — sem PDF correspondente"):
                    for f in md_only:
                        st.write(f"- `{f.name}`")

            if not matched:
                st.warning("Nenhum par PDF + Markdown correspondente foi encontrado.")
            else:
                if st.button("🔍 Validar pares", key="btn_validate"):
                    for pdf_file, md_file, _ in matched:
                        st.markdown(f"### 📄 {pdf_file.name}")

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(pdf_file.getvalue())
                            tmp_path = Path(tmp.name)

                        try:
                            markdown_existente = md_file.read().decode("utf-8")

                            with st.status("Validando...", expanded=True) as status:
                                st.write("📸 Renderizando páginas do PDF...")
                                pages = convert_from_path(tmp_path, dpi=200)
                                st.write(f"✅ {len(pages)} página(s).")

                                st.write("🤖 Auditando com Gemini...")
                                result = service.validate(pages, markdown_existente)

                                if not result['success'] and result['score'] is None:
                                    st.error(f"Erro: {result.get('error')}")
                                    status.update(label="Falha", state="error")
                                    continue

                                status.update(label="✅ Validação concluída!", state="complete")

                            _render_validate_results(
                                pdf_file.name,
                                result.get('score'),
                                len(pages),
                                result.get('details', []),
                                result.get('errors', [])
                            )

                        except Exception as e:
                            st.error(f"Erro inesperado em {pdf_file.name}: {e}")
                        finally:
                            if tmp_path.exists():
                                os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE RENDERIZAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def _render_convert_results(filename: str, markdown: str, score, num_pages: int, errors: list, details: list):
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if score is not None:
            delta_color = "normal" if score >= 90 else "off"
            st.metric("Fidelidade", f"{score}%", delta_color=delta_color)
        else:
            st.metric("Fidelidade", "N/A")

    with col2:
        st.metric("Páginas", num_pages)

    with col3:
        safe_name = Path(filename).stem.replace(" ", "_")
        st.download_button(
            label="📥 Baixar Markdown",
            data=markdown,
            file_name=f"{safe_name}.md",
            mime="text/markdown",
            key=f"dl_{safe_name}_{num_pages}"
        )

    if score is not None and score < 90:
        st.warning("⚠️ Fidelidade abaixo de 90%. Verifique o documento manualmente.")

    if details:
        with st.expander("📊 Detalhes por critério"):
            for d in details:
                if "OK" in d:
                    st.success(f"✅ {d}")
                elif "PARCIAL" in d:
                    st.warning(f"⚠️ {d}")
                else:
                    st.error(f"❌ {d}")

    if errors:
        with st.expander(f"🚨 Erros encontrados ({len(errors)})"):
            for e in errors:
                st.error(f"• {e}")
    else:
        if score is not None:
            st.success("✅ Nenhum erro encontrado.")

    with st.expander("👁️ Prévia do Markdown"):
        st.markdown(markdown)


def _render_validate_results(filename: str, score, num_pages: int, details: list, errors: list):
    col1, col2 = st.columns([1, 1])

    with col1:
        if score is not None:
            delta_color = "normal" if score >= 90 else "off"
            st.metric("Fidelidade", f"{score}%", delta_color=delta_color)
        else:
            st.metric("Fidelidade", "N/A")

    with col2:
        st.metric("Páginas", num_pages)

    if score is not None and score < 90:
        st.warning("⚠️ Fidelidade abaixo de 90%. Considere reconverter este documento.")

    if details:
        with st.expander("📊 Detalhes por critério"):
            for d in details:
                if "OK" in d:
                    st.success(f"✅ {d}")
                elif "PARCIAL" in d:
                    st.warning(f"⚠️ {d}")
                else:
                    st.error(f"❌ {d}")

    if errors:
        with st.expander(f"🚨 Erros encontrados ({len(errors)})"):
            for e in errors:
                st.error(f"• {e}")
    else:
        st.success("✅ Nenhum erro encontrado.")


if __name__ == "__main__":
    main()