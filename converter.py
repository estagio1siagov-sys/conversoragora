import os
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import logging

try:
    from docling.document_converter import DocumentConverter, FormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
    )
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
except ImportError:
    raise ImportError("Docling não está instalado. Execute: pip install docling")

logger = logging.getLogger(__name__)


def _build_converter() -> DocumentConverter:
    """
    Constrói o DocumentConverter uma única vez.
    Chamado apenas pelo cache do Streamlit.
    """
    ocr_options = RapidOcrOptions(
        lang=["pt"],
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.do_table_structure = True

    pdf_format_option = FormatOption(
        pipeline_cls=StandardPdfPipeline,
        backend=PyPdfiumDocumentBackend,
        pipeline_options=pipeline_options
    )

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: pdf_format_option}
    )


class PDFToMarkdownConverter:
    """
    Converte PDF para Markdown usando Docling + RapidOCR.
    Recebe o DocumentConverter já inicializado para evitar recarregar modelos.
    """

    def __init__(self, converter: DocumentConverter):
        self.converter = converter
        self._setup_logging()
        logger.info("PDFToMarkdownConverter pronto.")

    def _setup_logging(self):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def convert(self, pdf_path: str) -> Optional[Dict]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            logger.error(f"Arquivo não encontrado: {pdf_path}")
            return None

        try:
            logger.info(f"Convertendo: {pdf_path.name}")
            start = datetime.now()

            result = self.converter.convert(str(pdf_path))

            elapsed = (datetime.now() - start).total_seconds()
            markdown = result.document.export_to_markdown()
            num_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 0

            logger.info(f"Concluído: {num_pages} pág. em {elapsed:.1f}s")

            return {
                'success': True,
                'pdf_path': str(pdf_path),
                'markdown': markdown,
                'num_pages': num_pages,
                'elapsed_seconds': round(elapsed, 1),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro em {pdf_path.name}: {e}", exc_info=True)
            return {
                'success': False,
                'pdf_path': str(pdf_path),
                'error': str(e)
            }