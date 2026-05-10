import base64
import io
import os
import traceback
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

import click
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.settings import settings

app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


LLM_SERVICE_MAP = {
    "gemini": "marker.services.gemini.GoogleGeminiService",
    "vertex": "marker.services.vertex.GoogleVertexService",
    "ollama": "marker.services.ollama.OllamaService",
    "claude": "marker.services.claude.ClaudeService",
    "openai": "marker.services.openai.OpenAIService",
    "azure_openai": "marker.services.azure_openai.AzureOpenAIService",
}

LLM_SERVICE_OPTIONS = list(LLM_SERVICE_MAP.keys())


EXTRA_LLM_PARAMS_MAP = {
    "gemini": [("api_key", "gemini_api_key")],
    "vertex": [
        ("api_key", "vertex_project_id"),
    ],
    "ollama": [
        ("base_url", "ollama_base_url"),
        ("model", "ollama_model"),
        ("auth_header", "ollama_auth_header"),
    ],
    "claude": [("api_key", "claude_api_key"), ("model", "claude_model_name")],
    "openai": [("api_key", "openai_api_key"), ("model", "openai_model")],
    "azure_openai": [
        ("api_key", "azure_api_key"),
        ("base_url", "azure_endpoint"),
        ("model", "deployment_name"),
    ],
}


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20",
            example=None,
        ),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"
    use_llm: bool = Field(default=False, description="Use llm")
    llm_service: Annotated[
        str,
        Field(
            description=f"The LLM service to use.  Must be one of {list(LLM_SERVICE_MAP.keys())}."
        ),
    ] = "ollama"
    base_url: Annotated[Optional[str], Field(description="The base URL to use.")] = None
    api_key: Annotated[Optional[str], Field(description="The API key to use.")] = None
    model: Annotated[Optional[str], Field(description="The model to use.")] = None
    auth_header: Annotated[
        Optional[str], Field(description="The auth header to use.")
    ] = None

    gemini_api_key: SkipJsonSchema[Optional[str]] = None
    vertex_project_id: SkipJsonSchema[Optional[str]] = None
    ollama_base_url: SkipJsonSchema[Optional[str]] = None
    ollama_model: SkipJsonSchema[Optional[str]] = None
    ollama_auth_header: SkipJsonSchema[Optional[str]] = None
    claude_api_key: SkipJsonSchema[Optional[str]] = None
    claude_model_name: SkipJsonSchema[Optional[str]] = None
    openai_api_key: SkipJsonSchema[Optional[str]] = None
    openai_model: SkipJsonSchema[Optional[str]] = None
    azure_api_key: SkipJsonSchema[Optional[str]] = None
    azure_endpoint: SkipJsonSchema[Optional[str]] = None
    deployment_name: SkipJsonSchema[Optional[str]] = None


async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html", "chunks"], (
        "Invalid output format"
    )
    try:
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
            settings.OUTPUT_ENCODING
        )

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
    llm_service: Optional[Literal[LLM_SERVICE_OPTIONS]] = Form(default=None),  # ty:ignore[invalid-type-form]
    base_url: Optional[str] = Form(default=None),
    api_key: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
    auth_header: Optional[str] = Form(default=None),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
        llm_service=llm_service,
        base_url=base_url,
        api_key=api_key,
        model=model,
        auth_header=auth_header,
    )
    # Get the extra LLM params from the extra LLM params map
    llm_service = params.llm_service
    if llm_service in LLM_SERVICE_MAP:
        for param_name, param_value in EXTRA_LLM_PARAMS_MAP.get(llm_service, []):
            setattr(params, param_value, getattr(params, param_name, None))

        params.use_llm = True
        params.llm_service = LLM_SERVICE_MAP[llm_service]

    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )
