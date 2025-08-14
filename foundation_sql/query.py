import inspect
import os
import functools
from pathlib import Path
from typing import Any, Callable, Optional, Type, get_type_hints

from pydantic import BaseModel
from sqlalchemy import func

from foundation_sql.prompt import SQLPromptGenerator, FunctionSpec
from foundation_sql.gen import SQLGenerator
from foundation_sql.cache import SQLTemplateCache
from foundation_sql import db
from typing import Callable, Dict, Optional

from importlib import resources as impresources

DEFAULT_SYSTEM_PROMPT = impresources.read_text('foundation_sql.prompts', 'prompts.md')


class SQLQueryDecorator:
    """
    Advanced decorator for generating and executing SQL queries with comprehensive features.
    
    Supports:
    - Dynamic SQL template generation
    - Configurable LLM backend
    - Persistent template caching
    - Robust error handling and regeneration
    
    Attributes:
        name (Optional[str]): Custom name for SQL template
        regen (Optional[bool]): SQL template regeneration strategy
        config (SQLGeneratorConfig): Configuration for SQL generation
    """
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        regen: Optional[bool] = None,
        repair: Optional[int] = 0,
        schema: Optional[str] = None,
        schema_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        system_prompt_path: Optional[str] = None,
        cache_dir: Optional[str] = '__sql__',
        db_url: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,

        auto_schema: bool = False,
        schema_regen: Optional[bool] = None,
        schema_validate: bool = True,
        nested_strategy: str = "tables",
        table_name: Optional[str] = None,
        schema_cache_namespace: Optional[str] = None,
        schema_models: Optional[list] = None,
    ):
        """
        Initialize the SQL query decorator.
        
        Args:
            name (Optional[str]): Custom name for SQL file/folder. 
                                  Defaults to function name.
            regen (Optional[bool]): SQL template regeneration strategy.
            config (Optional[SQLGeneratorConfig]): Custom configuration 
                                                   for SQL generation.
        """
        self.name = name
        self.regen = regen
        self.cache_dir = cache_dir
        self.schema = schema or self._load_file(schema_path) if schema_path else None
        if system_prompt or system_prompt_path:
            self.system_prompt = system_prompt or self._load_file(system_prompt_path)
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

        self.db_url = db_url
        if not self.db_url:
            raise ValueError(f"Database URL not provided either through constructor or environment variable")
        
        # Initialize cache and SQL generator
        self.cache = SQLTemplateCache(cache_dir=cache_dir)

        self.sql_generator = SQLGenerator(
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        self.repair = repair
        # Auto-schema config
        self.auto_schema = auto_schema
        self.schema_regen = schema_regen if schema_regen is not None else regen
        self.schema_validate = schema_validate
        self.nested_strategy = nested_strategy
        self.table_name = table_name
        self.schema_cache_namespace = schema_cache_namespace
        self.schema_models = schema_models

    @staticmethod
    def _load_file(path: str) -> str:
        """
        Load file content from path.
        
        Returns:
            str: File content
        """
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")

        with open(path, 'r') as f:
            return f.read()
        
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation for SQL query generation and execution.
        
        Provides a comprehensive workflow for:
        - Extracting function context
        - Generating SQL templates
        - Executing queries
        - Handling errors and regeneration
        
        Args:
            func (Callable): Function to be decorated
        
        Returns:
            Callable: Wrapped function with SQL generation and execution logic
        """
        # Namespace template cache by module to avoid cross-module name collisions
        default_template_name = f"{func.__module__}.{func.__name__}.sql"
        template_name = self.name or default_template_name
        fn_spec = FunctionSpec(func)

        # Resolve schema to be used for query prompt. Prefer explicit schema passed at init.
        explicit_schema = self.schema

        # Helper: infer primary Pydantic model from function signature or provided overrides
        def _infer_model_class():
            if self.schema_models:
                for m in self.schema_models:
                    try:
                        if inspect.isclass(m) and issubclass(m, BaseModel):
                            return m
                    except Exception:
                        continue
            # Fallback: first BaseModel-annotated parameter
            for param in fn_spec.signature.parameters.values():
                if param.annotation is not param.empty:
                    try:
                        if inspect.isclass(param.annotation) and issubclass(param.annotation, BaseModel):
                            return param.annotation
                    except Exception:
                        continue
            return None

        # Helper: build schema cache key for this function/model
        def _schema_cache_key(model_cls: Optional[Type[BaseModel]]):
            parts = [p for p in [self.schema_cache_namespace, 'schema', func.__module__, func.__name__] if p]
            # Prefer deterministic table/model based key to maximize reuse across functions when model repeats
            if model_cls is not None:
                tbl = self.table_name or f"{model_cls.__name__.lower()}s"
                parts.append(tbl)
            return "__".join(parts) + ".sql"

        # Generate and optionally apply schema when auto_schema is enabled and no explicit schema provided
        auto_schema_text: Optional[str] = None

        def _ensure_schema_generated_and_applied():
            nonlocal auto_schema_text
            if explicit_schema is not None or not self.auto_schema:
                return

            model_cls = _infer_model_class()
            if model_cls is None:
                # No model found; nothing to do
                return

            cache_key = _schema_cache_key(model_cls)

            # Detect DB backend from URL for better prompt guidance
            def _detect_backend(url: Optional[str]) -> Optional[str]:
                if not url:
                    return None
                u = url.lower()
                if u.startswith('sqlite'):
                    return 'sqlite'
                if u.startswith('postgres') or u.startswith('postgresql'):
                    return 'postgres'
                if u.startswith('mysql'):
                    return 'mysql'
                return None

            db_backend = _detect_backend(self.db_url)
            # Use schema_regen policy separate from query regen
            should_regen = bool(self.schema_regen)

            if not should_regen and self.cache.exists(cache_key):
                auto_schema_text = self.cache.get(cache_key)
            else:
                # Build prompt using SQLPromptGenerator with function spec
                schema_prompt_gen = SQLPromptGenerator(fn_spec, cache_key, self.system_prompt, None)
                prompt = schema_prompt_gen.generate_schema_prompt_from_function(
                    nested_strategy=self.nested_strategy,
                    table_name=self.table_name,
                    db_backend=db_backend,
                )
                auto_schema_text = self.sql_generator.generate_sql(prompt)
                self.cache.set(cache_key, auto_schema_text)

            # Apply/validate schema against DB if configured
            if self.schema_validate and self.db_url and auto_schema_text:
                attempt = 0
                while attempt <= (self.repair or 0):
                    try:
                        database = db.get_db(self.db_url)
                        database.init_schema(schema_sql=auto_schema_text)
                        break
                    except Exception:
                        attempt += 1
                        if attempt > (self.repair or 0):
                            raise
                        # Clear and regenerate
                        if self.cache.exists(cache_key):
                            self.cache.clear(cache_key)
                        schema_prompt_gen = SQLPromptGenerator(fn_spec, cache_key, self.system_prompt, None)
                        prompt = schema_prompt_gen.generate_schema_prompt_from_function(
                            nested_strategy=self.nested_strategy,
                            table_name=self.table_name,
                            db_backend=db_backend,
                        )
                        auto_schema_text = self.sql_generator.generate_sql(prompt)
                        self.cache.set(cache_key, auto_schema_text)

        # Ensure schema is ready (if auto_schema is enabled)
        _ensure_schema_generated_and_applied()

        # Choose schema for query prompt
        effective_schema = explicit_schema or auto_schema_text

        prompt_generator = SQLPromptGenerator(
            fn_spec,
            template_name,
            self.system_prompt,
            effective_schema
        )

        def sql_gen(kwargs: Dict[str, Any], error: Optional[str] = None, prev_template: Optional[str] = None):
            if self.regen or not self.cache.exists(template_name) or error:
                prompt = prompt_generator.generate_prompt(kwargs, error, prev_template)
                sql_template = self.sql_generator.generate_sql(prompt)
                self.cache.set(template_name, sql_template)
            else:
                sql_template = self.cache.get(template_name)
            return sql_template

        @functools.wraps(func)
        def wrapper(**kwargs: Any) -> Any:
            error, sql_template = None, None
            # Generate and execute with optional repair loop
            attempts = 0
            max_attempts = (self.repair or 0) + 1
            while attempts < max_attempts:
                # Generate SQL (with error context if retrying)
                sql_template = sql_gen(kwargs, error, sql_template)
                try:
                    result_data = db.run_sql(self.db_url, sql_template, **kwargs)
                    break
                except Exception as e:
                    attempts += 1
                    error = str(e)
                    if attempts >= max_attempts:
                        raise

            if fn_spec.wrapper == 'list':
                parsed_result = [
                    db.parse_query_to_pydantic(row, fn_spec.return_type)
                    for row in result_data.all()
                ]
            elif isinstance(result_data, int):
                parsed_result = result_data
            else:
                first_row = result_data.first()
                parsed_result = db.parse_query_to_pydantic(first_row, fn_spec.return_type) if first_row else None

            return parsed_result

        return wrapper