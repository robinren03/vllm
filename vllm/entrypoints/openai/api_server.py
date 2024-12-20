import asyncio
import importlib
import inspect
import re
import signal
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional, Set
import json

import fastapi
import uvicorn
from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest, ErrorResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

engine: AsyncLLMEngine
engine_args: AsyncEngineArgs
openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding
openai_serving_tokenization: OpenAIServingTokenization

active_exit = False

logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


router = APIRouter()


def mount_metrics(app: fastapi.FastAPI):
    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app())
    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile('^/metrics(?P<path>.*)$')
    app.routes.append(metrics_route)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)

@router.get("/future")
async def get_memory_future() -> Response:
    """Get future."""
    return JSONResponse(content={"gpu_guess":engine.engine.gpu_cache_guess}, status_code=200)

@router.get("/exit")  
async def exit() -> Response:  
    """Exit the engine after responding to the request."""  
    global active_exit
    active_exit = True
    asyncio.create_task(schedule_exit())  
    return Response(content="Shutting down...", status_code=200)  

async def schedule_exit(delay: float = 0.5) -> None:  
    """Schedules the application to exit after a short delay."""  
    import os, signal
    await asyncio.sleep(delay)  # 等待足够的时间来发送HTTP响应  
    os.kill(os.getpid(), signal.SIGINT)  
    await asyncio.sleep(delay * 10)
    os.kill(os.getpid(), signal.SIGKILL)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    generator = await openai_serving_tokenization.create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, TokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest):
    generator = await openai_serving_tokenization.create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, DetokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/models")
async def show_available_models():
    models = await openai_serving_completion.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    if request.request_stop:
        await openai_serving_chat.remove_session(request.session_id)
        return Response(content="good")
    
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        return JSONResponse(content=generator.model_dump())


def build_app(args):
    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


async def build_server(
    args,
    llm_engine: Optional[AsyncLLMEngine] = None,
    **uvicorn_kwargs,
) -> uvicorn.Server:
    app = build_app(args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    global engine, engine_args

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.OPENAI_API_SERVER))

    model_config = await engine.get_model_config()

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    global openai_serving_chat
    global openai_serving_completion
    global openai_serving_embedding
    global openai_serving_tokenization

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )
    openai_serving_embedding = OpenAIServingEmbedding(
        engine,
        model_config,
        served_model_names,
        request_logger=request_logger,
    )
    openai_serving_tokenization = OpenAIServingTokenization(
        engine,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )
    app.root_path = args.root_path

    logger.info("Available routes are:")
    for route in app.routes:
        if not hasattr(route, 'methods'):
            continue
        methods = ', '.join(route.methods)
        logger.info("Route: %s, Methods: %s", route.path, methods)

    config = uvicorn.Config(
        app,
        host=None,
        port=args.port,
        log_level=args.uvicorn_log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    return uvicorn.Server(config), served_model_names

REGISTER_MACHINE_URL = "/register_machine"
UNREGISTER_MACHINE_URL = "/unregister_machine"

async def run_server(args, llm_engine=None, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    server, local_name = await build_server(
        args,
        llm_engine,
        **uvicorn_kwargs,
    )

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    import os, requests

    if args.host is None:
        model_url = "http://localhost"
    else:
        model_url = args.host

    model_url += f":{args.port}"
    router_api = os.environ.get("SARS_ROUTER_API", None)
    while (router_api is not None):
        if args.global_name is None:
            model_name = local_name[0]
        else:
            model_name = args.global_name
        
        num_gpu = args.tensor_parallel_size * args.pipeline_parallel_size

        response = requests.post(router_api + REGISTER_MACHINE_URL, json={
            "source_url": model_url, 
            "model": model_name, 
            "local_name": local_name[0], 
            "gpu_cost": num_gpu,
            "total_gpu_blocks": engine.engine.cache_config.num_gpu_blocks})
        
        if response.status_code != 200:
            print(f"Failed to register machine {model_url}")
        else:
            print(f"Registered machine {model_url} for model {model_name}")
            break

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        if (router_api is not None and not active_exit):
            response = requests.post(router_api + UNREGISTER_MACHINE_URL, json={
                "source_url": model_url, 
                "model": model_name, 
                "local_name": local_name[0], 
                "gpu_cost": num_gpu,
                "total_gpu_blocks": engine.engine.cache_config.num_gpu_blocks})
            if response.status_code != 200:
                print(f"Failed to unregister machine {model_url}")
            else:
                print(f"Unregistered machine {model_url} for model {model_name}")
        
        with open("metrics.txt", "w") as f:
            metrics = {
                "metrics_1": engine.engine.metrics_1,
                "metrics_2": engine.engine.metrics_2
            }
            json.dump(metrics, f)
            
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
    except asyncio.CancelledError:
        print("Gracefully stopping http server")
        await server.shutdown()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    asyncio.run(run_server(args))
